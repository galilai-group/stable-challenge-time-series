Pretrain a time series encoder, self-supervised, on the training corpus. It is scored on held-out time series
tasks from several domains, which stay private, by how well a linear probe on its frozen embeddings solves them.

> **Submission at a glance**
>
> - **Format:** a single `.onnx` file. Not a PyTorch checkpoint, zip, or weights without the architecture.
> - **Size cap: 20 MiB per file.** Larger uploads are rejected. Check with `ls -lh model.onnx`.
> - **Limit:** up to 5 submissions per day.
> - **Only your latest submission counts:** the leaderboard uses your most recent successfully evaluated model,
>   not your best. A worse resubmission replaces a better one.
> - **Leaderboard updates:** the leaderboard is refreshed periodically (likely every few days), not immediately
>   after each submission.
> - **Before uploading:** `uv run test_submission.py model.onnx` must pass. The submission page repeats these
>   checks on your file and won't upload a file that fails them.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` creates
`.venv` from [pyproject.toml](pyproject.toml), with the exact versions pinned in `uv.lock`.

```bash
git clone https://github.com/galilai-group/stable-challenge-time-series.git
cd stable-challenge-time-series
uv sync                                                    # create .venv with all dependencies
unzip corpus.zip -d corpus                                 # the training data (see below)
uv run example_train.py --corpus corpus --out model.onnx   # train and export a small example encoder
uv run test_submission.py model.onnx                       # check a model before submitting it
```

Use `uv run` in front of any command (e.g. `uv run python my_training.py`), or activate the environment with
`source .venv/bin/activate`. Add packages you need for training with `uv add <package>`. On Linux, PyTorch and
onnxruntime are installed with CUDA 12 support.

## Training data

**Download: `corpus.zip` (8.9 GB) from [Google Drive](https://drive.google.com/file/d/1ld4mifcgwaBZin7cGBNCYm7xXH2T90U6/view?usp=sharing).** Unzip it to get one file, `corpus.npz`, with two arrays:

| Array | dtype, shape | Content |
| --- | --- | --- |
| `values` | float32, [2,228,606,111] | every series, concatenated |
| `offsets` | int64, [1,762,339] | series `i` is `values[offsets[i]:offsets[i+1]]` |

That is all there is: 1,762,338 univariate series in random order, raw (unnormalized, unpadded) and of widely
varying lengths, with no labels, timestamps, sampling rates or sources.
[corpus_loader.py](corpus_loader.py) memory-maps the file (a plain `np.load(...)["values"]` would read all 8.9 GB
into memory) and yields shuffled batches of 1-D float32 arrays:

```python
import corpus_loader
for batch in corpus_loader.batches("corpus", batch_size=64):
    ...  # a list of 1-D float32 arrays of different lengths
```

**Rules.** The corpus is the only training data allowed: no other datasets and no pretrained weights. It carries
no labels.

## Example

[example_train.py](example_train.py) trains a small encoder with [stable-pretraining](https://github.com/galilai-group/stable-pretraining)
and exports it. The encoder (`SeriesViT`) is stable-pretraining's `ViT` run on each series as a 1 x 1024 image
with 1 x 32 patches, after per-series normalization; the objective (predict whether a series' next step goes up,
stays or goes down) is deliberately simple. It is a starting point that shows what a submission must get right,
not a strong baseline.

## Model format

An ONNX file (opset 17 or newer), an encoder only:

- **input**: float32, shape exactly `[64, 1, 1024]`: a batch of 64 univariate windows of 1,024 values
- **output**: float32, shape `[64, D]` with `D <= 2048`, all finite

Exactly one input and one output; their names do not matter. Export in eval mode with a static batch of 64.

**Normalize inside your model.** Windows reach the model as stored, at very different scales and offsets across
tasks, and the evaluator does not rescale them. Series shorter than 1,024 values are padded; longer ones are cut
into windows whose embeddings are averaged.

## Evaluation

[evaluate.py](evaluate.py) runs on a CPU or GPU with onnxruntime:

```bash
python evaluate.py model.onnx path/to/eval_data      # prints {"score": ..., "score_task1": ..., ...}
```

The official evaluation data is private, but you can run `evaluate.py` on any labeled time series data of your
own arranged in the same layout (see the docstring at the top of [evaluate.py](evaluate.py)). It embeds each series
with your frozen model and scores how well a linear probe on those embeddings solves each task. You don't need
evaluation data to check that your model will work with the evaluator: `test_submission.py` (below) does that.

## Validate submission

Check your model before uploading it:

```bash
uv run test_submission.py path/to/model.onnx
```

It loads and runs the model with the evaluator's own code and explains how to fix any problem it finds. The checks
are listed at the top of [test_submission.py](test_submission.py). Only submit once validation passes.

The submission page runs the same checks again in your browser when you choose the file, except the speed
estimate, and only uploads files that pass. A file it rejects is not uploaded and doesn't count toward your daily
attempts. Run `test_submission.py` locally anyway: it is quicker to iterate on, estimates the evaluation time, and is
the reference if the two ever disagree.
[example_submission/model.onnx](example_submission/model.onnx) is a model that passes.

Once model is ready submit your model at https://galilai-group.github.io/stable-challenge-time-series/ 
