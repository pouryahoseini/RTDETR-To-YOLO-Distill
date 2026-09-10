import os
from .base_cfg import *

# Automatic Ultralytics training accepts one square `imgsz`. The shared
# 736x1280 shape belongs to the rectangular manual/RT-DETR pipelines; keeping it
# here makes the automatic backend fail before training begins.
input_height = 992
input_width = 992

# ─────────────────────────────────────────────────────────────────────────────
# YOLO Configuration (Ultralytics Format) - Auto Mode Specifics
# ─────────────────────────────────────────────────────────────────────────────
yolo_train_config = {
    "model_yaml_path": "yolo26n.yaml",
    "pretrained_model_path": os.path.join(weights_dir, "yolo26n.pt"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared Training Settings
# ─────────────────────────────────────────────────────────────────────────────
shared_train_config = {
    "seed": 42,
    "epochs": 50,
    "early_stopping_patience": 5,
    "precision": "bf16",  # Options: "fp16", "bf16", "fp32"
}

# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Settings
# ─────────────────────────────────────────────────────────────────────────────
dataloader_config = {
    "batch_size": 16,
}

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Settings
# ─────────────────────────────────────────────────────────────────────────────
augmentation = {
    "mosaic": {
        "close_mosaic_epochs": 5,
    },
}
