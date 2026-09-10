"""Offline teacher-prediction cache for response-based knowledge distillation.

Running the teacher on the final training image is wrong in two ways at once.
It sees a four-image mosaic composite it was never trained on, and it sees that
composite after cutout, rain, blur and colour jitter have corrupted it -- so the
"knowledge" being distilled is the teacher's guess about an out-of-distribution
picture. Running it live on the pre-mosaic sources instead costs ~418 ms/step at
736x1280 with a 4-image mosaic (measured, RTX 3090), which roughly doubles step
time for a 100-epoch run.

Neither is necessary. The teacher is frozen, so its prediction for a given
source image never changes: it can be computed once, offline, on the clean
unaugmented image, and looked up thereafter.

Boxes are stored **normalized to the original image**, which makes them
invariant to whatever resize the training pipeline applies. That lets the
dataloader append them to the very same box list the ground truth travels in,
so mosaic placement, the window crop and the horizontal flip are applied to
teacher and ground-truth boxes by the same Albumentations call -- the pooling
across mosaic sources falls out for free, and there is no hand-written warp to
drift out of sync with the augmentation code.

The cache is keyed by image id and tagged with the teacher weights and input
resolution it was built at; `TeacherCache.load` refuses a cache built for a
different teacher rather than silently distilling stale predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import numpy as np
import torch


# Bump this whenever the meaning or serialization of a cached prediction
# changes. Version 2 invalidated caches created before the builder switched
# from raw model weights to the EMA teacher. Version 3 binds every cache entry
# to the manifest identity (file name and dimensions) of its image instead of
# trusting a COCO integer ID on its own.
CACHE_SCHEMA_VERSION = 3
CACHE_STATE_POLICY = "prefer_ema_module"
VALID_STATE_SOURCES = {"ema.module", "model", "checkpoint"}


def _image_identity(image: dict) -> str:
    """Stable identity for one manifest image, independent of annotations."""
    try:
        file_name = os.path.normpath(str(image["file_name"]).replace("\\", "/"))
        width = int(image["width"])
        height = int(image["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Every cached image must define file_name, integer width and integer "
            f"height; invalid image record: {image!r}."
        ) from exc
    canonical = json.dumps(
        {"file_name": file_name, "width": width, "height": height},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _select_teacher_state(checkpoint) -> tuple[object, str]:
    """Select teacher weights according to the cache's versioned policy."""
    if isinstance(checkpoint, dict):
        ema = checkpoint.get("ema")
        if isinstance(ema, dict) and ema.get("module") is not None:
            return ema["module"], "ema.module"
        if checkpoint.get("model") is not None:
            return checkpoint["model"], "model"
    return checkpoint, "checkpoint"


def _weights_fingerprint(weights_path: str) -> str:
    """Cryptographic identity of the complete checkpoint byte stream.

    Sampling only the head and tail can miss changed tensors in a same-size
    checkpoint, causing a retrained teacher to reuse stale predictions. A full
    streaming hash is slower, but cache correctness depends on every byte and
    the checkpoint is read only at cache/run setup rather than per batch.
    """
    digest = hashlib.sha256()
    with open(weights_path, "rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: str) -> None:
    """Publish a torch payload atomically after flushing it to stable storage."""
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = None
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if fd is not None:
            os.close(fd)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def default_cache_path(cache_dir: str, split: str, weights_path: str | None = None) -> str:
    """Path a split's cache is written to and read from.

    The teacher's fingerprint is part of the file name when `weights_path` is
    given, so caches for different teachers coexist instead of overwriting one
    another. A label-budget experiment trains a second teacher on a fraction of
    the annotations; with a single fixed path, building its cache would clobber
    the full-annotation teacher's, and each arm would then correctly refuse the
    other's cache and rebuild it -- ping-ponging one 6,471-image teacher pass
    per switch.

    Note what does *not* belong in this path: which training manifest is in use.
    Predictions are keyed by image id and the splits preserve those ids, so one
    cache built over the full manifest already covers every subset of it. Only
    the teacher changes what a cached prediction means.

    Args:
        cache_dir: Directory holding this teacher's caches, e.g.
            ``cfg.teacher_cache_dir``.
        split: ``"train"`` or ``"val"``.
        weights_path: Teacher checkpoint. Omitted reproduces the unqualified
            path, which older caches were written to.

    Returns:
        str: Cache path.
    """
    if weights_path is None:
        return os.path.join(cache_dir, f"{split}.pt")
    fingerprint = _weights_fingerprint(weights_path)
    return os.path.join(cache_dir, f"{split}.{fingerprint}.pt")


def ensure_teacher_caches(
    cache_dir: str,
    weights_path: str,
    train_needed: bool = True,
    val_needed: bool = False,
) -> None:
    """Ensure required teacher caches exist on disk; auto-build via subprocess if missing."""
    missing = []
    if train_needed and not os.path.exists(default_cache_path(cache_dir, "train", weights_path)):
        missing.append("train")
    if val_needed and not os.path.exists(default_cache_path(cache_dir, "val", weights_path)):
        missing.append("val")

    if not missing:
        return

    import subprocess
    import sys

    split_arg = "both" if len(missing) == 2 else missing[0]
    kd_cache_script = os.path.abspath(__file__)
    print(
        f"[Auto-Cache] Required teacher cache missing for {missing}. "
        f"Building '{split_arg}' cache via {kd_cache_script}..."
    )
    subprocess.run(
        [
            sys.executable,
            kd_cache_script,
            "--split",
            split_arg,
            "--weights",
            weights_path,
        ],
        check=True,
    )
    print(f"[Auto-Cache] Teacher cache build complete for '{split_arg}'.")


class TeacherCache:
    """Read-only lookup of cached teacher predictions, keyed by image id."""

    def __init__(self, boxes: dict[int, np.ndarray], logits: dict[int, np.ndarray], meta: dict):
        self.boxes = boxes
        self.logits = logits
        self.meta = meta
        self.image_identities = {
            int(image_id): identity
            for image_id, identity in meta["image_identities"].items()
        }

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def load(
        cls,
        path: str,
        weights_path: str,
        input_hw: tuple[int, int],
        *,
        expected_num_classes: int,
        requested_min_confidence: float,
        expected_cache_sha256: str | None = None,
    ) -> "TeacherCache":
        """Load a cache, refusing incompatible teacher predictions.

        Args:
            path: Cache file written by `build_teacher_cache`.
            weights_path: Teacher checkpoint this run will distil from.
            input_hw: ``(height, width)`` this run trains at.
            expected_num_classes: Classification width expected by the student.
            requested_min_confidence: Lowest confidence the caller may request
                from this cache. It cannot be below the builder's storage floor.

        Returns:
            TeacherCache: The loaded cache.

        Raises:
            FileNotFoundError: If the cache has not been built.
            ValueError: If it was built for a different teacher or resolution.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Teacher cache not found at {path}. Build it first with:\n"
                f"    python src/kd_cache.py --split train\n"
                f"Response KD reads its teacher predictions from this file."
            )
        # Hash and deserialize the same open inode. If another process
        # atomically publishes a replacement at `path`, this loader still has a
        # precise identity for the tensors it actually consumed.
        cache_digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(8 << 20):
                cache_digest.update(chunk)
            cache_sha256 = cache_digest.hexdigest()
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        if (
            expected_cache_sha256 is not None
            and cache_sha256 != expected_cache_sha256
        ):
            raise ValueError(
                f"Teacher cache at {path} changed while the run was being "
                f"resolved (expected {expected_cache_sha256}, loaded "
                f"{cache_sha256}). Retry after the cache build finishes."
            )
        meta = payload["meta"]

        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Teacher cache at {path} uses schema version "
                f"{meta.get('schema_version')!r}, but version "
                f"{CACHE_SCHEMA_VERSION} is required. Rebuild it."
            )
        if meta.get("state_policy") != CACHE_STATE_POLICY:
            raise ValueError(
                f"Teacher cache at {path} used state policy "
                f"{meta.get('state_policy')!r}, but {CACHE_STATE_POLICY!r} is "
                "required. Rebuild it."
            )
        if meta.get("state_source") not in VALID_STATE_SOURCES:
            raise ValueError(
                f"Teacher cache at {path} has an invalid state source "
                f"{meta.get('state_source')!r}. Rebuild it."
            )

        boxes = {int(image_id): value for image_id, value in payload["boxes"].items()}
        logits = {int(image_id): value for image_id, value in payload["logits"].items()}
        raw_identities = meta.get("image_identities")
        if not isinstance(raw_identities, dict):
            raise ValueError(
                f"Teacher cache at {path} does not record per-image identities. "
                "Rebuild it before using it with a training manifest."
            )
        identities = {int(image_id): value for image_id, value in raw_identities.items()}
        if set(boxes) != set(logits) or set(boxes) != set(identities):
            raise ValueError(
                f"Teacher cache at {path} has inconsistent box, logit or image-"
                "identity keys. Rebuild the corrupt/incomplete cache."
            )

        expected = _weights_fingerprint(weights_path)
        if meta.get("weights_fingerprint") != expected:
            raise ValueError(
                f"Teacher cache at {path} was built from a different checkpoint "
                f"(cache {meta.get('weights_fingerprint')}, current {expected}). "
                f"Rebuild it, or the run will distil stale predictions."
            )
        if tuple(meta.get("input_hw", ())) != tuple(input_hw):
            raise ValueError(
                f"Teacher cache at {path} was built at {meta.get('input_hw')}, "
                f"but this run trains at {tuple(input_hw)}. Rebuild it."
            )
        cached_num_classes = meta.get("num_classes")
        if cached_num_classes != int(expected_num_classes):
            raise ValueError(
                f"Teacher cache at {path} has num_classes={cached_num_classes!r}, "
                f"but this run expects {int(expected_num_classes)}. Rebuild it "
                "with the current category mapping and teacher head."
            )
        cached_floor_raw = meta.get("min_confidence")
        if cached_floor_raw is None:
            raise ValueError(
                f"Teacher cache at {path} does not record min_confidence. "
                "Rebuild it before sweeping the training threshold."
            )

        def validated_confidence(value, description: str) -> float:
            if isinstance(value, (bool, str, bytes)):
                raise ValueError(
                    f"{description} must be a finite numeric value in [0, 1], "
                    f"got {value!r}."
                )
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{description} must be a finite numeric value in [0, 1], "
                    f"got {value!r}."
                ) from exc
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(
                    f"{description} must be a finite numeric value in [0, 1], "
                    f"got {value!r}."
                )
            return normalized

        cached_floor = validated_confidence(
            cached_floor_raw, f"Teacher cache at {path} min_confidence"
        )
        requested_floor = validated_confidence(
            requested_min_confidence, "requested_min_confidence"
        )
        if requested_floor < cached_floor:
            raise ValueError(
                f"Teacher cache at {path} stores predictions only at confidence "
                f">= {cached_floor:g}, but this run requests "
                f"{requested_floor:g}. Rebuild the cache with "
                "--min-confidence at or below the requested threshold."
            )
        meta = {
            **meta,
            "image_identities": identities,
            "artifact_path": os.path.abspath(path),
            "artifact_sha256": cache_sha256,
        }
        return cls(boxes, logits, meta)

    def require_coverage(self, images: list[dict], source: str) -> None:
        """Require cached predictions for the exact selected manifest images."""
        missing: list[int] = []
        mismatched: list[int] = []
        for image in images:
            image_id = int(image["id"])
            if image_id not in self.boxes:
                missing.append(image_id)
            elif self.image_identities.get(image_id) != _image_identity(image):
                mismatched.append(image_id)

        if not missing and not mismatched:
            return
        problems = []
        if missing:
            problems.append(
                f"{len(missing)} of {len(images)} selected image IDs are missing "
                f"(examples: {missing[:5]})"
            )
        if mismatched:
            problems.append(
                f"{len(mismatched)} of {len(images)} image identities differ "
                f"despite matching IDs (examples: {mismatched[:5]})"
            )
        raise ValueError(
            f"Teacher cache does not cover {source}: {'; '.join(problems)}. "
            "Preserve image IDs, file names and dimensions in derived manifests, "
            "or rebuild the cache for the current dataset."
        )

    def get(self, image_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Cached ``(boxes, logits)`` for one image.

        Returns:
            tuple: ``boxes`` is (Q, 4) normalized ``cxcywh`` against the original
            image; ``logits`` is (Q, num_classes) raw teacher logits. Both are
            empty arrays when the image is absent from the cache.
        """
        boxes = self.boxes.get(int(image_id))
        if boxes is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        return boxes, self.logits[int(image_id)]


def write_teacher_pseudo_annotations(
    cache_path: str,
    source_json: str,
    output_json: str,
    conf_threshold: float,
    weights_path: str,
    input_hw: tuple[int, int],
    num_classes: int,
    expected_cache_sha256: str | None = None,
    labeled_image_ids: frozenset[int] | None = None,
) -> dict:
    """Write a COCO manifest whose annotations are the teacher's predictions.

    Used to validate a teacher-only run without consulting human labels. The
    images and categories are copied from ``source_json``. If ``labeled_image_ids``
    is None, every annotation is discarded and replaced by cached teacher boxes
    above ``conf_threshold``. If ``labeled_image_ids`` is provided (semi-supervised),
    GT annotations are preserved for those images, and only the remaining images
    receive teacher boxes.

    The result is a file rather than an in-memory filter on purpose. Checkpoint
    selection is the last place a teacher-only run still touches annotations,
    so the exact target set it was selected against should be inspectable
    afterwards instead of being reconstructed from config.

    Note what the resulting metric is. Scoring the student against these
    targets measures *agreement with the teacher*, not accuracy: the teacher
    recovers roughly 73% of val ground truth at conf 0.4, and its own errors
    are scored as correct. The number is usable for ranking epochs within one
    run and is not comparable to a ground-truth mAP from any other run.

    Args:
        cache_path: Teacher cache for the split being validated.
        source_json: COCO manifest supplying images and categories.
        output_json: Path to write.
        conf_threshold: Lowest teacher confidence to keep.
        weights_path: Teacher checkpoint, for the cache's fingerprint check.
        input_hw: Training resolution, for the cache's resolution check.
        num_classes: Student classification width, for the cache's class check.
        expected_cache_sha256: Optional content identity resolved for the
            output filename. A concurrent cache replacement is rejected rather
            than writing different targets under that name.
        labeled_image_ids: Optional set of image IDs to keep GT labels for.

    Returns:
        dict: Summary counts for logging.
    """
    cache = TeacherCache.load(
        cache_path,
        weights_path,
        input_hw,
        expected_num_classes=num_classes,
        requested_min_confidence=conf_threshold,
        expected_cache_sha256=expected_cache_sha256,
    )

    with open(source_json, "r") as f:
        source = json.load(f)

    images = source.get("images", [])
    cache.require_coverage(images, source_json)
    
    # Pre-index existing GT when mixing
    gt_annotations_by_image = {}
    if labeled_image_ids is not None:
        for ann in source.get("annotations", []):
            img_id = int(ann["image_id"])
            if img_id in labeled_image_ids:
                gt_annotations_by_image.setdefault(img_id, []).append(ann)

    annotations = []
    empty_images = 0
    next_id = 1
    for img in images:
        img_id = int(img["id"])
        if labeled_image_ids is not None and img_id in labeled_image_ids:
            # Transfer GT annotations for this image, renumbering their IDs
            for ann in gt_annotations_by_image.get(img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = next_id
                annotations.append(new_ann)
                next_id += 1
            if not gt_annotations_by_image.get(img_id, []):
                empty_images += 1
            continue

        width, height = float(img["width"]), float(img["height"])
        boxes, logits = cache.get(img_id)
        if boxes.shape[0] == 0:
            empty_images += 1
            continue

        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        confidence = scores.max(axis=-1)
        classes = scores.argmax(axis=-1)
        keep = confidence > conf_threshold
        if not keep.any():
            empty_images += 1
            continue

        for box, cls in zip(boxes[keep].astype(np.float32), classes[keep]):
            cx, cy, bw, bh = (float(v) for v in box)
            # Cached boxes are normalized cxcywh against the original image.
            abs_w, abs_h = bw * width, bh * height
            x = (cx - bw / 2.0) * width
            y = (cy - bh / 2.0) * height
            # Clip to the frame: a teacher box may extend past the edge, and
            # pycocotools would otherwise score area it cannot match against.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1 = min(width, x + abs_w)
            y1 = min(height, y + abs_h)
            if x1 <= x0 or y1 <= y0:
                continue
            annotations.append({
                "id": next_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": float((x1 - x0) * (y1 - y0)),
                "iscrowd": 0,
            })
            next_id += 1

    desc = (
        "Mixed GT and teacher pseudo-annotations"
        if labeled_image_ids is not None else
        f"Teacher pseudo-annotations from {os.path.basename(cache_path)}"
    )
    output = {
        "info": {
            "description": (
                f"{desc} at conf>{conf_threshold}."
            ),
            "source_manifest": source_json,
            "teacher_weights": weights_path,
            "conf_threshold": float(conf_threshold),
        },
        "images": images,
        "categories": source.get("categories", []),
        "annotations": annotations,
    }
    output_dir = os.path.dirname(os.path.abspath(output_json)) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_json)}.", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(fd, "w") as f:
            fd = None
            json.dump(output, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_json)
    finally:
        if fd is not None:
            os.close(fd)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return {
        "path": output_json,
        "cache_sha256": cache.meta["artifact_sha256"],
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": empty_images,
    }


@torch.no_grad()
def build_teacher_cache(
    split: str = "train",
    weights_path: str | None = None,
    top_k: int = 300,
    min_confidence: float = 0.05,
    batch_size: int = 4,
) -> str:
    """Run one teacher once over a whole split and store its predictions.

    Each image is loaded and resized exactly as the non-mosaic training path
    resizes it -- the same anisotropic resize a mosaic tile also gets -- so the
    teacher sees the clean, in-distribution picture. Its boxes are then stored
    normalized, which makes them independent of that resize.

    The split is always cached in full, whatever `label_budget` is set to. A
    cache is a lookup table keyed by image id, and the budget manifests preserve
    those ids, so one full cache serves every budget; building at a budget would
    instead produce a cache that quietly lacks entries as soon as the budget
    moved, and the coverage check would reject it at the start of the next run.

    Args:
        split: ``"train"`` or ``"val"``.
        weights_path: Teacher checkpoint to run. Defaults to the KD teacher,
            which is the one a subsequent distillation run will look for. Pass
            it explicitly to cache any other teacher -- the path a cache is
            written to carries that checkpoint's fingerprint, so caching a
            second teacher adds a file rather than replacing one.
        top_k: Keep only this many queries per image, ranked by confidence.
            RT-DETR emits exactly 300, so the default keeps everything above
            `min_confidence` and truncates nothing. 100 was measured to clip
            real signal: 94% of val images hit that cap, and on 62 of them
            every retained query still exceeded the 0.5 KD threshold (weakest
            retained reached 0.73), so usable detections were being dropped --
            on the crowded frames specifically, which is where VisDrone is
            hardest. Storage is ~8 KB/image, so there is nothing to save by
            trimming.
        min_confidence: Drop queries below this before ranking. Kept low so the
            KD threshold remains tunable at training time without rebuilding.
        batch_size: Images per teacher forward.

    Returns:
        str: Path the cache was written to.
    """
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import configs.train_cfg as cfg
    from models import build_rtdetr_model
    from dataloader import ObjectDetectionDataset, get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path is None:
        weights_path = cfg.teacher["weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {weights_path}")
    print(f"teacher: {weights_path} (fingerprint {_weights_fingerprint(weights_path)})")

    teacher = build_rtdetr_model(
        variant=cfg.teacher["variant"],
        pretrained_backbone=False,
        num_classes=cfg.num_classes,
    )
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # EMA weights first, exactly as `train.py` selects them. Caching the raw
    # `model` weights instead would distil a different teacher from the one the
    # live path and the evaluation numbers use.
    state, state_source = _select_teacher_state(checkpoint)
    state = state.state_dict() if hasattr(state, "state_dict") else state
    teacher.load_state_dict({k: v for k, v in state.items() if k in teacher.state_dict()}, strict=True)
    teacher.to(device).eval()

    # Deliberately the full manifest, not cfg.train_json/cfg.val_json -- those
    # follow `label_budget`, and a cache must not.
    annotation_file = cfg.full_train_json if split == "train" else cfg.full_val_json
    # is_training=False gives the deterministic resize path: no mosaic, no
    # appearance augmentation. That is the whole point -- a clean weak view.
    dataset = ObjectDetectionDataset(
        annotation_file=annotation_file,
        transforms=get_val_transforms(seed=int(cfg.shared_train_config.get("seed", 42))),
        is_training=False,
        seed=int(cfg.shared_train_config.get("seed", 42)),
        # A cache is shared by all label budgets. It must cover the complete
        # source manifest even when an experiment subsamples train/validation.
        # is_training remains false so inputs still use the deterministic,
        # clean validation transform rather than training augmentation.
        subsample_interval=1,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_config.get("num_workers", 8),
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b]),
    )

    boxes_out: dict[int, np.ndarray] = {}
    logits_out: dict[int, np.ndarray] = {}

    from tqdm import tqdm

    for images, targets in tqdm(loader, desc=f"caching teacher [{split}]"):
        preds = teacher(images.to(device))
        pred_logits = preds["pred_logits"].float()
        pred_boxes = preds["pred_boxes"].float()
        confidence = pred_logits.sigmoid().max(dim=-1).values

        for i, target in enumerate(targets):
            image_id = int(target["image_id"].item())
            keep = confidence[i] >= min_confidence
            idx = keep.nonzero(as_tuple=True)[0]
            if idx.numel() > top_k:
                order = confidence[i][idx].topk(top_k).indices
                idx = idx[order]
            # fp16 halves the file; teacher logits do not need more precision
            # than that, and they are cast back to fp32 at use.
            boxes_out[image_id] = pred_boxes[i][idx].cpu().numpy().astype(np.float16)
            logits_out[image_id] = pred_logits[i][idx].cpu().numpy().astype(np.float16)

    path = default_cache_path(cfg.teacher_cache_dir, split, weights_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_torch_save(
        {
            "boxes": boxes_out,
            "logits": logits_out,
            "meta": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "state_policy": CACHE_STATE_POLICY,
                "state_source": state_source,
                "weights_fingerprint": _weights_fingerprint(weights_path),
                "weights_path": weights_path,
                "input_hw": (cfg.input_height, cfg.input_width),
                "num_classes": cfg.num_classes,
                "top_k": top_k,
                "min_confidence": min_confidence,
                "split": split,
                "image_identities": {
                    image_id: _image_identity(dataset.coco.imgs[image_id])
                    for image_id in boxes_out
                },
            },
        },
        path,
    )
    total = sum(v.shape[0] for v in boxes_out.values())
    print(
        f"cached {total:,} predictions over {len(boxes_out):,} images -> {path} "
        f"({os.path.getsize(path) / 1e6:.1f} MB)"
    )
    return path


if __name__ == "__main__":
    import argparse
    import sys as _sys

    _sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import configs.train_cfg as _cfg

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", default="both", choices=["train", "val", "both"],
        help="Which split to cache. Both are needed for a distillation run -- "
             "train for the KD targets, val for teacher-scored validation -- so "
             "both is the default; name one to rebuild just that half.",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Teacher checkpoint to cache. Defaults to the configured teacher "
             f"({_cfg.teacher['weights']}). Caches are written per "
             "teacher fingerprint, so a second teacher adds a file rather than "
             "replacing one.",
    )
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    splits = ("train", "val") if args.split == "both" else (args.split,)
    for _split in splits:
        build_teacher_cache(
            split=_split,
            weights_path=args.weights,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            batch_size=args.batch_size,
        )
