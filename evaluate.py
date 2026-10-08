"""Evaluation of an ONNX time series encoder with linear (ridge) probes, on a GPU if there is one, else on CPU.

    python evaluate.py model.onnx path/to/evaldata [--out results.json]

The script is self-contained so it can be uploaded as a challenge evaluator. ``evaldata`` is the unzipped
evaluation data; the runner starts the script from a temp directory, so pass it as an absolute path.

Problems with the submitted model are reported as {"error": ...} on stdout, which the leaderboard shows.

Prints {"score": <float>, "score_task1": <float>, ...} as JSON on stdout; the detailed report goes to stderr.

Each task in the data is a set of items, each item one or more windows of CONTEXT values:
    embed every window, mean-pool the windows of each item, fit a ridge probe on the items where `test` is
    False, and take the macro one-vs-rest ROC AUC on the rest
    task score = (AUC - floor) / (reference - floor): 0 at an untrained encoder, 1 at the reference model
Score = weighted mean of the task scores (weights and anchors come with the data)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BATCH, CHANNELS, CONTEXT, MAX_DIM = 64, 1, 1024, 2048
ALPHAS = (1.0, 10.0, 100.0, 1000.0)  # ridge penalty, chosen per class by leave-one-out on the fitted items
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"


class SubmissionError(Exception):
    """A problem with the submitted model, shown to the participant."""


def load_model(path):
    if hasattr(ort, "preload_dlls"):  # find CUDA/cuDNN from the nvidia-* pip packages, no LD_LIBRARY_PATH needed
        ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(path, opts, providers=[CUDA, CPU])
    except Exception as e:
        raise SubmissionError(f"Could not load the ONNX model: {e}") from None
    if sess.get_providers()[0] != CUDA:
        # On CPU, one single-threaded batch per core in parallel (see embed) is 2-4x faster than one batch at a
        # time across all cores when there are many cores, since a batch of 64 windows rarely keeps many threads
        # busy; on 2 cores the two are equal.
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, opts, providers=[CPU])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        raise SubmissionError("ONNX model must have exactly one input and one output")
    if ins[0].type != "tensor(float)":
        raise SubmissionError(f"Model input must be float32, not {ins[0].type}")
    shape = ins[0].shape
    if len(shape) != 3 or any(isinstance(g, int) and g != w for g, w in zip(shape, (BATCH, CHANNELS, CONTEXT))):
        raise SubmissionError(f"Model input must accept shape ({BATCH}, {CHANNELS}, {CONTEXT}); it declares {shape}")
    return sess


def embed(sess, windows):
    """[n, CONTEXT] -> [n, D]. Always runs full batches of BATCH; the last one is zero-padded and trimmed.

    On a GPU the batches run one after another; on CPU, one per core in parallel.
    """
    name = sess.get_inputs()[0].name

    def run(i):
        block = np.asarray(windows[i : i + BATCH], dtype=np.float32)
        x = np.zeros((BATCH, CONTEXT), np.float32)
        x[: len(block)] = block
        try:
            z = sess.run(None, {name: x.reshape(BATCH, CHANNELS, CONTEXT)})[0]
        except Exception as e:
            raise SubmissionError(f"Model failed on float32 input of shape {(BATCH, CHANNELS, CONTEXT)}: {e}") from None
        if not isinstance(z, np.ndarray) or z.ndim != 2 or z.shape[0] != BATCH:
            raise SubmissionError(f"Embedding shape {getattr(z, 'shape', None)}, expected ({BATCH}, D)")
        return z[: len(block)]

    starts = range(0, len(windows), BATCH)
    if sess.get_providers()[0] == CPU:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        with ThreadPoolExecutor(cores) as pool:
            results = list(pool.map(run, starts))
    else:
        results = map(run, starts)
    out, dim = [], None
    for z in results:
        if dim is None:
            dim = z.shape[1]
            if not 1 <= dim <= MAX_DIM:
                raise SubmissionError(f"Embedding width {dim}, expected between 1 and {MAX_DIM}")
        elif z.shape[1] != dim:
            raise SubmissionError("Model returned inconsistent embedding widths")
        out.append(z.astype(np.float32))
    z = np.concatenate(out)
    if not np.isfinite(z).all():
        raise SubmissionError("Embeddings contain NaN or infinite values")
    return z


def pool(z, item, n_items):
    """Mean of the window embeddings of each item."""
    total = np.zeros((n_items, z.shape[1]), np.float64)
    np.add.at(total, item, z)
    return (total / np.maximum(np.bincount(item, minlength=n_items), 1)[:, None]).astype(np.float32)


def probe_auc(x, labels, test):
    """Standardize, one ridge per class on +/-1 targets, macro one-vs-rest ROC AUC on the test items."""
    fit = ~test
    scaler = StandardScaler().fit(x[fit])
    targets = np.where(labels[fit], 1.0, -1.0)
    ridge = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(scaler.transform(x[fit]), targets)
    scores = ridge.predict(scaler.transform(x[test]))
    y = labels[test]
    aucs = [roc_auc_score(y[:, c], scores[:, c]) for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(aucs))


def evaluate(model_path, data):
    t0 = time.time()
    sess = load_model(model_path)
    print(f"running on {sess.get_providers()[0]}", file=sys.stderr, flush=True)
    tasks = json.load(open(os.path.join(data, "tasks.json")))["tasks"]
    feats, res = {}, {}
    for n, task in enumerate(tasks, start=1):
        folder = os.path.join(data, f"task{n:02d}")
        labels = np.load(os.path.join(folder, "labels.npy"))
        test = np.load(os.path.join(folder, "test.npy"))
        keep = np.load(os.path.join(folder, "keep.npy")) if os.path.exists(os.path.join(folder, "keep.npy")) else None
        w = task["windows"]
        if w not in feats:
            windows = np.load(os.path.join(data, "windows", f"{w}.npy"), mmap_mode="r")
            item = np.load(os.path.join(data, "windows", f"{w}_item.npy"))
            feats[w] = pool(embed(sess, windows), item, len(labels))
        x = feats[w]
        if keep is not None:
            x, labels, test = x[keep], labels[keep], test[keep]
        auc = probe_auc(x, labels, test)
        score = (auc - task["floor"]) / max(task["reference"] - task["floor"], 1e-6)
        res[f"task{n}"] = {"auc": auc, "score": score, "weight": task["weight"]}
        print(f"task{n:<3d} auc {auc:.4f}  score {score:7.4f}  [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    weights = np.array([r["weight"] for r in res.values()])
    score = float(np.dot(weights, [r["score"] for r in res.values()]) / weights.sum())
    return {"score": score, "tasks": res, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ap.add_argument("data", help="directory of the unzipped evaluation data")
    ap.add_argument("--out", default=None, help="also write the full results JSON here")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.data, "tasks.json")):
        sys.exit(f"Evaluation data not found at {a.data!r}")
    try:
        r = evaluate(a.model, a.data)
    except SubmissionError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    print(f"score {r['score']:.4f}  time {r['seconds']}s", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print(json.dumps({"score": round(r["score"], 6),
                      **{f"score_{k}": round(v["score"], 6) for k, v in r["tasks"].items()}}))
