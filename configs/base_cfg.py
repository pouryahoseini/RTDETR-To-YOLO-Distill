import os

# ─────────────────────────────────────────────────────────────────────────────
# Dataset split settings
# ─────────────────────────────────────────────────────────────────────────────
crowded_scene_iou_threshold = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# Label-budget splits
# ─────────────────────────────────────────────────────────────────────────────
# Whole-sequence subsets of each source manifest, used to measure how much of
# the annotation set a teacher actually needs before its predictions can stand
# in for the labels.
#
# Splits are drawn over *sequences*, never over frames. VisDrone images are
# sampled from drone flights, and a frame-level split -- every other image, say
# -- leaves 99.5% of the held-out images belonging to a flight the teacher
# already trained on, so "held out" would mean nothing.
#
# Sequences are drawn uniformly at random and accumulated until the requested
# fraction of *images* is reached. Selecting by size instead (largest flights
# first) reaches the same image count with 13 sequences rather than ~100,
# collapsing the split onto a handful of locations: measured mean sequence
# length 261 against the dataset's own 31.
#
# How a frame's sequence is recovered from its file name is a structural fact
# about each dataset, not a setting: it lives with the rest of that dataset's
# layout in `get_dataset_config()` in src/prepare_dataset.py. Only the split
# design is configured here.
#
# Each entry writes `<split>_<name>.json` for both train.json and val.json,
# holding the labels this budget bought. The remainder is the source manifest
# minus that file and is not written separately. Image and annotation IDs are
# carried over unchanged, so one teacher cache built over the full manifests
# serves every budget derived from them.
label_budget_splits = {
    "half":    {"fraction": 0.5,  "seed": 42},
    "quarter": {"fraction": 0.25, "seed": 42},
    "eighth":  {"fraction": 0.125, "seed": 42},
}

# Which annotation budget this run trains and validates on.
#   None        the full annotation set
#   "half"      the 50% split
#   "quarter"   the 25% split
#   "eighth"    the 12.5% split
#
# Validation follows the budget; the test set never does. A smaller annotation
# budget really does mean fewer labelled images to select a checkpoint on, so
# holding validation at 100% would quietly spend labels the budget says were
# never bought. The test set is the instrument every arm is scored against and
# stays fixed, which is what makes the arms comparable at all.
#
# A teacher-only student is the one arm that reads two different budgets: it
# trains over *all* images (it consumes no annotations, so nothing limits which
# images it sees) while validating on the same reduced split as its control.
# Its default training manifest is `full_train_json`; an explicit teacher-only
# `train_json` can still narrow the unlabeled pool when an ablation requires it.
label_budget = "eighth"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset path configurations
# ─────────────────────────────────────────────────────────────────────────────
dataset_name = "VisDrone"
processed_annotations_dir = "data/processed_annotations"
weights_dir = "weights"

# Derived artifacts that are rebuildable from a checkpoint plus the dataset, and
# so are never worth keeping or publishing. Separated from `weights_dir` so that
# directory holds only results -- checkpoints and deployable exports -- and one
# ignore rule covers everything here.
#
# Each cache below is keyed by a fingerprint of what produced it and refuses a
# mismatch rather than serving a stale entry, so deleting any of it costs only
# the time to recompute.
cache_dir = "cache"
teacher_cache_dir = os.path.join(cache_dir, "teacher")
calibration_cache_dir = os.path.join(cache_dir, "calibration")

# Derived annotation paths (used by train.py)
_USE_CONFIGURED_BUDGET = object()


# Splits an annotation budget applies to. Test sets are excluded by design.
BUDGETED_SPLITS = frozenset({"train", "val"})


def budget_manifest(
    split: str,
    *,
    budget=_USE_CONFIGURED_BUDGET,
    budget_splits=None,
    annotations_dir: str | None = None,
) -> str:
    """Resolve one split's manifest for the configured annotation budget.

    The name is validated against `label_budget_splits`, which catches a typo
    without touching the disk. Whether the file has been generated is left to
    the loader: `prepare_dataset.py` imports this module in order to *write*
    those files, so an existence check here could not be satisfied on a fresh
    checkout.

    The name is public because ``from .base_cfg import *`` skips
    underscore-prefixed names, and training provenance calls this resolver
    through ``configs.train_cfg``. The optional arguments let a caller resolve
    a copied or overridden config from *that config's* current values rather
    than closing over this module's import-time values.
    """
    # Test sets are deliberately absent: they are the fixed instrument every
    # arm is scored against, so budgeting one would make the arms incomparable.
    # Asking for one is a caller bug, so it fails here rather than returning a
    # path to a file that will never be generated.
    if split not in BUDGETED_SPLITS:
        raise ValueError(
            f"budget_manifest({split!r}) is not a budgeted split; valid splits are "
            f"{sorted(BUDGETED_SPLITS)}. Test sets are never budgeted -- name one "
            f"of eval_config['test_sets'] and read it from "
            f"{processed_annotations_dir}/<name>.json instead."
        )

    selected_budget = label_budget if budget is _USE_CONFIGURED_BUDGET else budget
    available_budgets = label_budget_splits if budget_splits is None else budget_splits
    output_dir = processed_annotations_dir if annotations_dir is None else annotations_dir

    if selected_budget is None:
        return os.path.join(output_dir, f"{split}.json")
    if selected_budget not in available_budgets:
        raise ValueError(
            f"label_budget={selected_budget!r} is not defined in "
            f"label_budget_splits (have: {sorted(available_budgets)})."
        )
    return os.path.join(output_dir, f"{split}_{selected_budget}.json")


def _budget_manifest(split: str) -> str:
    """Private alias kept for configs that reference the underscored name."""
    return budget_manifest(split)


train_json = budget_manifest("train")
val_json   = budget_manifest("val")

# The complete upstream source manifests, independent of `label_budget`.
# Teacher-only training and teacher caches use these so one cache covers every
# smaller budget derived from the same source. Predictions remain keyed by the
# image IDs preserved in each derived manifest.
full_train_json = budget_manifest("train", budget=None)
full_val_json   = budget_manifest("val", budget=None)

# ─────────────────────────────────────────────────────────────────────────────
# Object class mapping
# VisDrone dataset class mapping
# ─────────────────────────────────────────────────────────────────────────────
category_mapping = {
    0: {"id": 0, "name": "pedestrian",      "abb": "P",  "supercategory": "human"},
    1: {"id": 1, "name": "people",          "abb": "L", "supercategory": "human"},
    2: {"id": 2, "name": "bicycle",         "abb": "2", "supercategory": "vehicle"},
    3: {"id": 3, "name": "car",             "abb": "C",  "supercategory": "vehicle"},
    4: {"id": 4, "name": "van",             "abb": "V",  "supercategory": "vehicle"},
    5: {"id": 5, "name": "truck",           "abb": "T",  "supercategory": "vehicle"},
    6: {"id": 6, "name": "tricycle",        "abb": "3", "supercategory": "vehicle"},
    7: {"id": 7, "name": "awning-tricycle", "abb": "A",  "supercategory": "vehicle"},
    8: {"id": 8, "name": "bus",             "abb": "B",  "supercategory": "vehicle"},
    9: {"id": 9, "name": "motor",           "abb": "M",  "supercategory": "vehicle"},
}
num_classes = len(category_mapping)

# ─────────────────────────────────────────────────────────────────────────────
# Model / Target framework
#   "rtdetr"       → boxes output as [cx, cy, w, h]  (0-1 normalised)
#   "yolo_ultra"   → boxes output as [cx, cy, w, h]  (0-1 normalised, YOLO style via Ultralytics)
#   "yolo_manual"  → same as yolo_ultra, uses manual training loop
# ─────────────────────────────────────────────────────────────────────────────
model_format = "yolo_manual"   # one of: "rtdetr" | "yolo_ultra" | "yolo_manual"

# NMS IoU threshold for duplicate suppression. Unused for e2e models (YOLO26, RT-DETR).
nms_iou_threshold = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# General Input Resolution
# ─────────────────────────────────────────────────────────────────────────────
input_height = 736 # 992 for Yolo auto
input_width  = 1280 # 992 for Yolo auto
