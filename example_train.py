"""Worked example: train a small encoder on the corpus and export it.

    unzip corpus.zip -d corpus
    python example_train.py --corpus corpus --out model.onnx

A deliberately simple objective: hide each series' last value and classify
the step into it from everything before -- up, same or down. The encoder
reads the first ``len - 1`` points, a linear head predicts the three classes,
and only the encoder is exported.

This is a starting point, not a strong baseline. It shows what every
submission has to get right: a forward that matches the contract,
normalization *inside* the model (the evaluator feeds raw values), and a clean
ONNX export.

The encoder is stable-pretraining's (``spt``) image ``ViT``, run on each series
as a ``1 x CONTEXT`` image with ``1 x 32`` patches (see :class:`SeriesViT`).
Install with:

    pip install stable-pretraining onnx onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining.backbone import ViT

sys.path.insert(0, str(Path(__file__).parent))
import corpus_loader  # noqa: E402
from evaluate import BATCH, CHANNELS, CONTEXT  # noqa: E402

OPSET = 17  # minimum the evaluator accepts

UP, SAME, DOWN = 0, 1, 2


class SeriesViT(nn.Module):
    """spt's ``ViT`` on time series: ``(B, CHANNELS, CONTEXT) -> (B, embed_dim)``.

    Each series is z-scored over time (the evaluator feeds raw values, whose
    scale spans prices, counts and waveforms), then encoded as a
    ``1 x CONTEXT`` image cut into ``1 x patch_size`` patches, mean-pooled.
    Inputs must be exactly ``CONTEXT`` long; pad shorter series first.
    """

    def __init__(self, patch_size=32, embed_dim=256, depth=4, num_heads=4):
        super().__init__()
        self.norm = nn.InstanceNorm1d(CHANNELS)
        self.vit = ViT(
            img_size=(1, CONTEXT),
            patch_size=(1, patch_size),
            in_chans=CHANNELS,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            class_token=False,
            global_pool="avg",
        )

    def forward(self, x):
        return self.vit(self.norm(x).unsqueeze(2))


class NextStep(torch.utils.data.Dataset):
    """Every corpus series as ``(first len - 1 points, direction of the last)``.

    Inputs are left-padded with their first value to ``CONTEXT``, as the
    evaluator pads short windows; missing values become 0 after the padding.
    """

    def __init__(self, root):
        self.get, self.n = corpus_loader.load(root)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        series = self.get(i)[-CONTEXT:]  # audio clips run longer than CONTEXT
        history, last = series[:-1], series[-1]
        step = last - history[-1]
        label = SAME if step == 0 or not np.isfinite(step) else (UP if step > 0 else DOWN)
        x = np.pad(history, (CONTEXT - len(history), 0), mode="edge")
        return {"x": torch.from_numpy(np.nan_to_num(x)).float()[None], "label": label}


def next_step(self, batch, stage):
    """Forward for ``spt.Module``: classify the hidden step from the embedding."""
    embedding = self.backbone(batch["x"])
    logits = self.head(embedding)
    loss = F.cross_entropy(logits, batch["label"])
    accuracy = (logits.argmax(-1) == batch["label"]).float().mean()
    self.log_dict({f"{stage}/loss": loss, f"{stage}/acc": accuracy}, prog_bar=True)
    return {"loss": loss, "embedding": embedding}


def export(backbone, out: Path) -> None:
    """Export the encoder in eval mode with the contract's static shape."""
    backbone.eval().to("cpu")
    torch.onnx.export(
        backbone,
        (torch.randn(BATCH, CHANNELS, CONTEXT),),
        str(out),
        input_names=["x"],
        output_names=["embedding"],
        opset_version=OPSET,
        dynamo=False,
    )
    print(f"wrote {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("model.onnx"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    backbone = SeriesViT(patch_size=32, embed_dim=args.embed_dim)
    module = spt.Module(
        backbone=backbone,
        head=torch.nn.Linear(args.embed_dim, 3),
        forward=next_step,
        optim={"optimizer": {"type": "AdamW", "lr": 3e-4}},
    )
    loader = torch.utils.data.DataLoader(
        NextStep(args.corpus), batch_size=BATCH, shuffle=True, num_workers=args.workers
    )
    trainer = pl.Trainer(
        max_steps=args.steps,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
    )
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loader))()

    export(module.backbone, args.out)
    print(f"\nNow check it:  python test_submission.py {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
