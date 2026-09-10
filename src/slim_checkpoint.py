#!/usr/bin/env python3
"""Strip a training checkpoint down to what inference and lineage need.

A training checkpoint carries everything required to *resume* a run: the raw
model, the EMA model, optimizer, LR scheduler, grad scaler, the feature-KD
adapter and the EMA step counter. Only the EMA weights are ever used for
inference -- every loader in this repository (evaluate.py, export.py,
kd_cache.py, and train.py when it loads a teacher) reads `ema["module"]` first --
so a published checkpoint keeps those, in the same nesting, plus the small
metadata records the evaluation lineage reads. The YOLO26n student shrinks from
~43 MB to ~10 MB and the RT-DETR-R50vd teacher from ~655 MB to ~170 MB.

The result holds only tensors and plain Python values, so it also loads with
`torch.load(..., weights_only=True)`. Every tensor is checked bit-for-bit
against the source before the output is kept.

Usage:

    python src/slim_checkpoint.py weights/.../yolo_manual_best.pth out/yolo_manual_best.pth
"""

import argparse
import os
import sys

import torch

# Provenance and bookkeeping that evaluate.py's lineage reads, or that tells a
# reader what the weights are. Anything else is resume-only state.
KEEP_METADATA = (
    "epoch",
    "best_map",
    "training_regime",
    "label_budget",
    "run_provenance",
    "validation_manifest",
)


def inference_state_dict(checkpoint: dict) -> dict:
    """The weights inference uses: EMA when present, else the raw model."""
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model
    raise ValueError("checkpoint holds neither ema['module'] nor a model state dict")


def slim_checkpoint(src: str, dst: str) -> dict:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{src} is not a checkpoint dictionary")

    state_dict = inference_state_dict(checkpoint)
    slim = {"ema": {"module": state_dict}}
    slim.update({k: checkpoint[k] for k in KEEP_METADATA if k in checkpoint})

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(slim, dst)

    # Reload the way a cautious user would, and prove nothing changed.
    reloaded = torch.load(dst, map_location="cpu", weights_only=True)["ema"]["module"]
    if reloaded.keys() != state_dict.keys():
        raise RuntimeError(f"{dst}: tensor names differ from the source")
    for name, tensor in state_dict.items():
        if not torch.equal(reloaded[name], tensor):
            raise RuntimeError(f"{dst}: tensor {name!r} differs from the source")

    dropped = sorted(set(checkpoint) - set(slim))
    return {
        "src_mb": os.path.getsize(src) / 2**20,
        "dst_mb": os.path.getsize(dst) / 2**20,
        "tensors": len(state_dict),
        "kept": sorted(set(slim) - {"ema"}),
        "dropped": dropped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", help="training checkpoint (.pth)")
    parser.add_argument("dst", help="output path for the slim checkpoint")
    args = parser.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        parser.error("refusing to overwrite the source checkpoint")

    report = slim_checkpoint(args.src, args.dst)
    print(
        f"{args.dst}: {report['src_mb']:.1f} MB -> {report['dst_mb']:.1f} MB, "
        f"{report['tensors']} tensors verified\n"
        f"  kept:    ema.module, {', '.join(report['kept']) or '(no metadata)'}\n"
        f"  dropped: {', '.join(report['dropped']) or '(nothing)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
