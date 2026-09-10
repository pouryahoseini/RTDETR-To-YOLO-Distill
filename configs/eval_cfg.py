from .base_cfg import *

# ─────────────────────────────────────────────────────────────────────────────
# Visualization Settings
# ─────────────────────────────────────────────────────────────────────────────
visualization_config = {
    "num_random_frames": 10,
    "num_worst_offenders": 10,
    "num_best_frames": 10,
    "conf_threshold": 0.25,   # Confidence threshold for drawing prediction boxes
}

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Settings
# ─────────────────────────────────────────────────────────────────────────────
eval_config = {
    # Batch 1 models real-time frame-by-frame edge inference and keeps latency
    # directly comparable across PyTorch, ONNX Runtime, and TensorRT.
    "batch_size": 1,

    # Untimed inference passes run before the timed loop. Without them, cuDNN
    # autotuning, lazy kernel loading and the ONNX Runtime / TensorRT
    # first-inference setup are all charged to the first latency sample, which
    # inflates the mean badly on the small subsets used by the INT8 accuracy
    # gate. 0 disables warmup.
    "warmup_batches": 3,

    "conf_threshold": 0.001,
    "test_sets": ["test_baseline"],
    "conditions": ["clean"],
    "output_dir": "runs/eval",

    # Score every N-th image. Used by visualize.py; evaluate.py takes its own
    # from `--subsample`, defaulting to the training config's entry.
    "test_subsample_interval": 1,
}

# `num_workers` and `pin_memory` are deliberately absent: evaluation reads them
# from the training config so a single change moves both, and they do not affect
# a measured number. Batch size does -- batch 1 is what keeps latency comparable
# across PyTorch, ONNX Runtime and TensorRT -- so it is set here and nowhere
# else.
