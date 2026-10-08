"""Check a submission before uploading it.

    python test_submission.py model.onnx

Runs the evaluator's own loading and embedding code (evaluate.py, next to this file) on random inputs, so a
model that passes here loads and runs the same way in the official evaluation. Checks:

    1. loads in onnxruntime: one float32 input accepting (64, 1, 1024), one output    enforced by the evaluator
    2. returns finite (64, D) embeddings, 1 <= D <= 2048                              enforced by the evaluator
    3. ONNX opset >= 17
    4. deterministic: the same input gives the same embedding
    5. each window is encoded independently of the others in its batch
    6. finite embeddings for raw inputs at very different scales and offsets (the evaluator does not normalize)
    7. throughput on this machine, and the full evaluation's time at that rate

Exits 0 if every check passes, 1 otherwise. Like the evaluation, it runs on a GPU if there is one, else on CPU.
"""

import argparse
import sys
import time

import numpy as np

from evaluate import BATCH, CONTEXT, SubmissionError, embed, load_model

OPSET_MIN = 17
SUITE_WINDOWS = 158_078  # windows embedded by the full evaluation
rng = np.random.default_rng(0)


def report(ok, message):
    print(f"[{' ok ' if ok else 'FAIL'}] {message}")
    return ok


def opset(path):
    try:
        import onnx
    except ImportError:
        return None
    model = onnx.load(path, load_external_data=False)
    return max((i.version for i in model.opset_import if i.domain in ("", "ai.onnx")), default=None)


def check(path):
    try:
        sess = load_model(path)
    except SubmissionError as e:
        return report(False, str(e))
    provider = sess.get_providers()[0]
    report(True, f"loads, one float32 input accepting ({BATCH}, 1, {CONTEXT}); running on {provider}")

    x = rng.standard_normal((BATCH, CONTEXT)).astype(np.float32)
    try:
        z = embed(sess, x)
    except SubmissionError as e:
        return report(False, str(e))
    ok = report(True, f"embeddings {z.shape}, finite")

    version = opset(path)
    if version is None:
        print("[warn] could not read the opset (pip install onnx to check it)")
    else:
        ok &= report(version >= OPSET_MIN, f"opset {version} (minimum {OPSET_MIN})")

    same = np.array_equal(z, embed(sess, x))
    ok &= report(same, "deterministic" if same else
                 "not deterministic: the same input gave different embeddings (dropout left on at export?)")

    perm = rng.permutation(BATCH)
    independent = np.allclose(embed(sess, x[perm]), z[perm], atol=1e-4, rtol=1e-3)
    ok &= report(independent, "each window encoded independently of its batch" if independent else
                 "embeddings depend on the other windows in the batch (batch norm exported in training mode?)")

    for label, probe in (("x 1e-3", x * 1e-3), ("x 1e3", x * 1e3), ("+ 1000", x + 1000)):
        try:
            embed(sess, probe)
        except SubmissionError as e:
            ok = report(False, f"input {label}: {e}. Normalize inputs inside your model.")
            break
    else:
        report(True, "finite embeddings for inputs scaled by 1e-3 and 1e3 and offset by 1000")

    windows = rng.standard_normal((BATCH * 20, CONTEXT)).astype(np.float32)
    embed(sess, windows[:BATCH])
    start = time.perf_counter()
    embed(sess, windows)
    rate = len(windows) / (time.perf_counter() - start)
    print(f"[info] {rate:,.0f} windows/s on {provider}: the full evaluation ({SUITE_WINDOWS:,} windows) would take "
          f"~{SUITE_WINDOWS / rate / 60:.1f} min on this machine")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ok = check(ap.parse_args().model)
    print("\nAll checks passed." if ok else "\nSome checks failed; fix them before submitting.")
    sys.exit(0 if ok else 1)
