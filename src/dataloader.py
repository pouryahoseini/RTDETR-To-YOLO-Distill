import os
import sys
import random
import warnings
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from evaluate import resolve_image_path

# Suppress warning from albumentations about divide by zero which is handled but still shows up
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    module="albumentations.augmentations.dropout.functional",
    message="invalid value encountered in divide"
)

# Import configurations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import configs.train_cfg as cfg


# --- Transformations ---
def build_hard_negative_block(hn_cfg: dict) -> list[A.BasicTransform]:
    """
    Assembles the list of sub-transforms for the hard-negative OneOf block.
    Only sub-transforms whose 'enabled' flag is True are included.
    Returns an empty list if no sub-transforms are active.

    Args:
        hn_cfg (dict): Hard-negative augmentation configuration.

    Returns:
        list[A.BasicTransform]: List of Albumentations transforms.
    """

    candidates = []

    # Random rain augmentation
    if hn_cfg.get("random_rain", {}).get("enabled", False):
        rr = hn_cfg["random_rain"]
        candidates.append(
            A.RandomRain(
                brightness_coefficient=rr["brightness_coefficient"],
                drop_width=rr["drop_width"],
                blur_value=rr["blur_value"],
                p=1.0,
            )
        )

    # Color jitter augmentation
    if hn_cfg.get("color_jitter_night", {}).get("enabled", False):
        cj = hn_cfg["color_jitter_night"]
        candidates.append(
            A.ColorJitter(
                brightness=tuple(cj["brightness"]),
                contrast=tuple(cj["contrast"]),
                p=1.0,
            )
        )

    # Motion blur augmentation
    if hn_cfg.get("motion_blur", {}).get("enabled", False):
        mb = hn_cfg["motion_blur"]
        candidates.append(A.MotionBlur(blur_limit=mb["blur_limit"], p=1.0))

    return candidates


def get_train_transforms(
    mosaic: bool = False, seed: int | None = None, stage: str = "all"
) -> A.Compose:
    """
    Build training-time augmentation pipeline.

    Args:
        mosaic: Whether this pipeline receives mosaic samples. `_load_mosaic`
            already jitters the scale and crops to the model input size, so the
            geometry block is skipped -- applying LSJ on top would jitter the
            scale twice. Everything else (photometric, flip, cutout,
            normalisation) is shared.
        seed: Seed for the Compose's own generator. Albumentations 2.x gives
            each Compose an independently seeded RNG, so `random.seed()` and
            `np.random.seed()` do not reach it. Workers re-seed at run time via
            `_worker_init_fn`; this only fixes the `num_workers=0` case.

    Returns:
        A.Compose: Training-time augmentation pipeline.
    """

    if stage not in {"all", "geometric", "appearance"}:
        raise ValueError(
            f"stage must be 'all', 'geometric' or 'appearance'; got {stage!r}."
        )

    # Get augmentation configuration from config.py
    aug = cfg.augmentation

    # Resolve normalization for the active model format
    norm = aug["normalize"][cfg.model_format]

    # `stage` splits this pipeline so the teacher can be shown a *weak view*:
    # the same geometry as the student, without the appearance corruption. A
    # teacher asked to label a rain-streaked, cutout-punched image is guessing,
    # and its guess is what gets distilled. Running the two stages in sequence
    # reproduces "all" except that photometric distortion then lands after the
    # geometry rather than before it -- a per-pixel op either way, differing
    # only in whether resize interpolation sees jittered pixels. That ordering
    # also matches YOLOv8's, which applies HSV to the final image.
    want_geometry = stage in {"all", "geometric"}
    want_appearance = stage in {"all", "appearance"}
    # The geometric stage hands a NumPy image to the appearance stage, so only
    # the stage that finishes the sample converts to a tensor.
    want_output = stage != "geometric"

    # Global photometric distortion
    transforms = []
    pd = aug.get("photometric_distort", {})
    if want_appearance and pd.get("enabled", False):
        transforms.append(
            A.ColorJitter(
                brightness=pd.get("brightness", 0.4),
                contrast=pd.get("contrast", 0.4),
                saturation=pd.get("saturation", 0.7),
                hue=pd.get("hue", 0.015),
                p=pd.get("p", 1.0),
            )
        )

    # Resolution augmentation (Large Scale Jitter). Mosaic samples arrive
    # already scaled and cropped to the input size by `_load_mosaic`, so LSJ is
    # skipped for them unless `lsj.apply_on_mosaic` asks for both, in which case
    # the two jitters compose (mosaic scale x LSJ scale) and LSJ pads rather
    # than pulling in more canvas, since the mosaic has already been cropped.
    lsj = aug.get("lsj", {})
    lsj_active = want_geometry and lsj.get("enabled", False) and (
        not mosaic or lsj.get("apply_on_mosaic", False)
    )
    if lsj_active:
        transforms.extend([
            A.RandomScale(scale_limit=lsj.get("scale_limit", [-0.9, 1.0]), p=lsj.get("p", 1.0)),
            A.PadIfNeeded(
                min_height=cfg.input_height,
                min_width=cfg.input_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                position="random",
                p=1.0,
            ),
            A.RandomCrop(
                height=cfg.input_height,
                width=cfg.input_width,
                p=1.0,
            ),
        ])
    elif mosaic or not want_geometry:
        # Already at the model input size; adding a resize here would be a no-op
        pass
    else:
        # Standard Resize
        transforms.append(
            A.Resize(height=cfg.input_height, width=cfg.input_width)
        )

    # Horizontal flip augmentation
    hf = aug.get("horizontal_flip", {})
    if want_geometry and hf.get("enabled", False):
        transforms.append(A.HorizontalFlip(p=hf["p"]))

    # Hard negative augmentations
    hn = aug.get("hard_negative", {})
    if want_appearance and hn.get("enabled", False):
        sub_transforms = build_hard_negative_block(hn)
        if sub_transforms:
            transforms.append(A.OneOf(sub_transforms, p=hn["p"]))

    # Random erasing / cutout
    cutout = aug.get("cutout", {})
    if want_appearance and cutout.get("enabled", False):
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(cutout.get("min_holes", 1), cutout.get("max_holes", 8)),
                hole_height_range=(cutout.get("min_hole_size", 8), cutout.get("max_hole_size", 32)),
                hole_width_range=(cutout.get("min_hole_size", 8), cutout.get("max_hole_size", 32)),
                fill=114,
                fill_mask=None,
                p=cutout.get("p", 0.5),
            )
        )

    # Normalization + tensor conversion
    if want_output:
        transforms += [
            A.Normalize(mean=norm["mean"], std=norm["std"], max_pixel_value=255.0),
            ToTensorV2(),
        ]

    # Set minimum visibility for bounding boxes
    # min_visibility mirrors the lsj config so both stay in sync
    min_vis = aug.get("lsj", {}).get("min_visibility", 0.1)

    # The appearance stage moves no pixels between coordinates, so it carries no
    # bbox_params: boxes were already resolved by the geometric stage, and
    # attaching a processor here would only demand label fields it cannot use.
    if stage == "appearance":
        return A.Compose(transforms, seed=seed)

    # Return the augmentation pipeline with COCO bbox parameters
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels", "original_areas"],
            min_visibility=min_vis,
            min_area=1.0,  # Drop zero-area boxes
            check_each_transform=True,  # Filter boxes after each transform to prevent downstream division by zero
        ),
        seed=seed,
    )


def get_val_transforms(seed: int | None = None) -> A.Compose:
    """
    Build validation-time augmentation pipeline.

    Args:
        seed: Seed for the Compose's own generator. The val pipeline is
            deterministic anyway; accepted for symmetry with the train builder.

    Returns:
        A.Compose: Validation-time augmentation pipeline.
    """

    # Get augmentation configuration from config.py
    aug = cfg.augmentation

    # Resolve normalization for the active model format
    norm = aug["normalize"][cfg.model_format]

    # Resize + normalize only
    return A.Compose(
        [
            A.Resize(height=cfg.input_height, width=cfg.input_width),
            A.Normalize(mean=norm["mean"], std=norm["std"], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels", "original_areas"],
            check_each_transform=True,
        ),
        seed=seed,
    )


def get_geometric_fallback_transforms(seed: int | None = None) -> A.Compose:
    """Deterministic resize fallback that preserves the staged-view contract.

    The ordinary validation fallback finishes preprocessing by normalizing and
    converting to a CHW tensor. That is correct for the unsplit training
    pipeline, but not for the weak-view path: after the geometric stage returns,
    the image is still forked into the teacher output stage and the student's
    appearance stage. This fallback therefore returns an unnormalized HWC
    NumPy image while applying the same deterministic resize to its boxes.

    Args:
        seed: Accepted for symmetry with the other Compose builders.

    Returns:
        A.Compose: Resize-only preprocessing with COCO-box handling.
    """
    return A.Compose(
        [A.Resize(height=cfg.input_height, width=cfg.input_width)],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels", "original_areas"],
            check_each_transform=True,
        ),
        seed=seed,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own augmentation stream, per epoch.

    Two problems this fixes. First, on Linux the workers are forked, so they
    inherit one `random` state: without re-seeding, worker 0 and worker 1 draw
    the *same* mosaic partner indices, scales and crop positions, just applied
    to different images -- augmentation diversity collapses by roughly the
    worker count. Second, Albumentations 2.x seeds each `A.Compose` from its own
    generator at construction, which `set_seed()` cannot reach.

    `torch.initial_seed()` is `base_seed + worker_id`, where PyTorch redraws
    `base_seed` from the main process generator each time the loader is
    iterated. Deriving from it therefore gives streams that differ per worker
    and per epoch, yet are reproducible from the global seed.

    Args:
        worker_id (int): Index of the worker being initialised.
    """

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    info = torch.utils.data.get_worker_info()
    if info is None:
        return

    # Offset per pipeline so the Composes do not run identical streams. Keep
    # the staged weak-view pipelines here as well: DataLoader workers are
    # forked after dataset construction and would otherwise inherit identical
    # Albumentations generators for geometry and appearance augmentation.
    pipeline_names = (
        "transforms",
        "mosaic_transforms",
        "fallback_transforms",
        "geometric_transforms",
        "mosaic_geometric_transforms",
        "geometric_fallback_transforms",
        "appearance_transforms",
        "output_transforms",
    )
    for offset, name in enumerate(pipeline_names):
        pipeline = getattr(info.dataset, name, None)
        if pipeline is not None and hasattr(pipeline, "set_random_seed"):
            pipeline.set_random_seed((seed + offset) % (2 ** 32))


def get_output_transforms() -> A.Compose:
    """Normalization and tensor conversion only, with no augmentation.

    Finishes the teacher's weak view: the geometric stage has already put the
    image in the student's exact frame, and nothing further should touch it.
    """
    norm = cfg.augmentation["normalize"][cfg.model_format]
    return A.Compose([
        A.Normalize(mean=norm["mean"], std=norm["std"], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def _teacher_weak_view_wanted() -> bool:
    """True when feature KD should see a geometry-only view of the sample.

    Feature KD distils the composite the student actually sees, so it cannot use
    the offline response cache -- but there is no reason for the teacher to see
    the cutout holes, rain streaks and colour jitter on top. Those corrupt the
    features being distilled without adding anything: the teacher was never
    trained on them, so what it encodes there is noise.
    """
    kd = getattr(cfg, "kd_config", {})
    if not kd.get("enabled", False):
        return False
    feature = kd.get("feature_based", {})
    return bool(feature.get("enabled", False) and feature.get("teacher_clean_view", True))


def _teacher_cache_wanted() -> bool:
    """True when a response-based KD method will need cached teacher predictions.

    Feature KD is deliberately excluded: it distils the composite the student
    actually sees, so it uses the live teacher, not the cache.
    """
    kd = getattr(cfg, "kd_config", {})
    if not kd.get("enabled", False):
        return False
    return bool(
        kd.get("pseudo_label", {}).get("enabled", False)
        or kd.get("pseudo_label_soft", {}).get("enabled", False)
    )


def apply_transforms_with_retry(
    transforms,
    fallback_transforms,
    image: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    min_visibility: float,
    max_trials: int,
):
    """Apply a random crop pipeline, falling back to deterministic preprocessing.

    Albumentations computes bbox visibility in the transformed coordinate system,
    so its ``min_visibility`` setting is the correct place to enforce how much of
    a box survives a crop. Comparing transformed pixel area with the raw box area
    is scale-dependent and is not an IoU or visibility measurement.
    """
    original_areas = [w * h for _, _, w, h in boxes]
    bbox_processor = transforms.processors.get("bboxes")
    bbox_params = bbox_processor.params if bbox_processor is not None else None
    baseline_visibility = float(bbox_params.min_visibility) if bbox_params is not None else 0.0
    requested_visibility = max(baseline_visibility, float(min_visibility))

    if bbox_params is not None:
        bbox_params.min_visibility = requested_visibility
    try:
        for _ in range(max(1, int(max_trials))):
            transformed = transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels,
                original_areas=original_areas,
            )
            # Empty source images do not need a surviving foreground box. For
            # annotated images, retry until at least one box meets the sampled
            # visibility requirement.
            if not boxes or transformed["bboxes"]:
                return (
                    transformed["image"],
                    list(transformed["bboxes"]),
                    list(transformed["class_labels"]),
                )
    finally:
        if bbox_params is not None:
            bbox_params.min_visibility = baseline_visibility

    if fallback_transforms is None:
        raise RuntimeError("No deterministic preprocessing fallback is configured.")

    fallback = fallback_transforms(
        image=image,
        bboxes=boxes,
        class_labels=labels,
        original_areas=original_areas,
    )
    return (
        fallback["image"],
        list(fallback["bboxes"]),
        list(fallback["class_labels"]),
    )

# --- Box format conversion ---
def coco_to_yolo(boxes_coco: list[list[float]], img_h: int, img_w: int) -> torch.Tensor:
    """
    Convert COCO-format bounding boxes to YOLO format:
    [x_min, y_min, w, h] (pixels) → [cx, cy, w, h] (0-1 normalised).

    Args:
        boxes_coco: List of bounding boxes in COCO format ([x_min, y_min, w, h]).
        img_h: Height of the image.
        img_w: Width of the image.

    Returns:
        torch.Tensor: Tensor of shape (N, 4) in [cx, cy, w, h] normalised to [0, 1].
    """

    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_h}x{img_w}.")

    # If no boxes, return empty tensor
    if not boxes_coco:
        return torch.zeros((0, 4), dtype=torch.float32)

    # Convert COCO format to YOLO format
    t = torch.as_tensor(boxes_coco, dtype=torch.float32)
    cx = (t[:, 0] + t[:, 2] / 2) / img_w
    cy = (t[:, 1] + t[:, 3] / 2) / img_h
    w  = t[:, 2] / img_w
    h  = t[:, 3] / img_h

    return torch.stack([cx, cy, w, h], dim=1)


def get_box_converter(model_format: str):
    """
    Get a bounding box converter function based on the model format.
    
    Args:
        model_format: Model format (e.g., "rtdetr", "detr", "yolo_ultra").
        
    Returns:
        function: Bounding box converter function.
    """
    
    # Map model format to a bounding box converter function
    box_format_map = {
        "rtdetr":      coco_to_yolo,
        "detr":        coco_to_yolo,
        "yolo":        coco_to_yolo,
        "yolo_ultra":  coco_to_yolo,
        "yolo_manual": coco_to_yolo,
    }

    # Get the bounding box converter function
    key = model_format.lower()
    if key not in box_format_map:
        raise ValueError(
            f"Unknown MODEL_FORMAT '{model_format}'. "
            f"Supported values: {list(box_format_map)}"
        )
    return box_format_map[key]


# --- Dataset class ---
class ObjectDetectionDataset(Dataset):
    """
    COCO-format dataset.

    Args:
        annotation_file: Path to a COCO-format JSON annotation file.
        transforms: An albumentations Compose pipeline (or None).
        is_training: Whether the dataset is for training.
        seed: Seed for the Compose pipelines built here. Only takes effect with
            `num_workers=0`; workers re-seed via `_worker_init_fn`.
        use_ground_truth: Whether annotation boxes may enter samples. False
            purges them before augmentation, which prevents crop/mosaic choices
            from depending on labels during teacher-only training.
        labeled_image_ids: When not None, ground truth is loaded only for images
            whose ID is in this set.  Used in semi-supervised mode where the
            dataset covers the full image pool but only a budget subset has
            labels.  Ignored when ``use_ground_truth`` is False.
        enable_teacher_cache: Explicit response-cache override. None preserves
            the legacy global-config behavior.
        emit_teacher_view: Explicit feature-KD view override. None preserves
            the legacy global-config behavior.
        teacher_cache_conf_threshold: Lowest teacher confidence that the cache
            must inject. None preserves the legacy per-method calculation.
        require_empty_annotations: When ground truth is disabled, reject a
            manifest containing annotations instead of purging it.
        subsample_interval: Explicit frame interval. None selects the configured
            train/validation interval from `is_training`; cache construction
            passes 1 while retaining the clean `is_training=False` transforms.
    """

    def __init__(
        self,
        annotation_file: str,
        transforms: A.Compose | None = None,
        is_training: bool = True,
        seed: int | None = None,
        use_ground_truth: bool = True,
        labeled_image_ids: frozenset[int] | None = None,
        enable_teacher_cache: bool | None = None,
        emit_teacher_view: bool | None = None,
        teacher_cache_conf_threshold: float | None = None,
        require_empty_annotations: bool = False,
        subsample_interval: int | None = None,
    ):
        self.annotation_file = annotation_file
        self.use_ground_truth = bool(use_ground_truth)
        self.labeled_image_ids = labeled_image_ids
        self.coco = COCO(annotation_file)
        annotation_count = len(self.coco.anns)
        if not self.use_ground_truth:
            if require_empty_annotations and annotation_count:
                raise ValueError(
                    "Teacher-only training requires an annotation-free manifest, "
                    f"but {annotation_file!r} contains {annotation_count} annotations. "
                    "Use the same images/categories/image IDs with annotations=[]."
                )
            # COCO parses the JSON during construction. Remove every annotation
            # index immediately so no later dataset path can accidentally read
            # GT, even if a future refactor bypasses `_load_image_and_boxes`.
            self.coco.anns.clear()
            self.coco.imgToAnns = {image_id: [] for image_id in self.coco.imgs}
            self.coco.catToImgs = {category_id: [] for category_id in self.coco.cats}
            self.coco.dataset["annotations"] = []
        self.ignored_ground_truth_count = (
            annotation_count if not self.use_ground_truth else 0
        )
        self.image_ids = sorted(
            self.coco.imgs, key=lambda image_id: self.coco.imgs[image_id].get("file_name", "")
        )
        
        # Apply dataset subsampling if configured
        interval_key = "frame_subsample_interval" if is_training else "val_subsample_interval"
        if subsample_interval is None:
            subsample_interval = int(cfg.dataloader_config.get(interval_key, 1))
        else:
            subsample_interval = int(subsample_interval)
            interval_key = "subsample_interval"
        if subsample_interval <= 0:
            raise ValueError(
                f"dataloader_config[{interval_key!r}] must be positive, "
                f"got {subsample_interval}."
            )
        if subsample_interval > 1:
            from experiment_artifacts import subsample_image_ids

            self.image_ids = subsample_image_ids(
                self.image_ids, subsample_interval, label=f"dataloader_config[{interval_key!r}]"
            )
            dataset_type = "training" if is_training else "validation"
            print(f"Subsampled {dataset_type} dataset: using 1/{subsample_interval} frames. Total frames: {len(self.image_ids)}")

        # A standard detection loss has no per-region "unknown" state. Mixing
        # a labeled source with an unlabeled source would make the unlabeled
        # pixels implicit background whenever the composite reaches the GT
        # loss. Keep Mosaic and MixUp within one supervision partition so the
        # sample-level ``has_gt`` contract stays sound.
        self._composition_indices: dict[bool, list[int]] | None = None
        if self.use_ground_truth and self.labeled_image_ids is not None:
            self._composition_indices = {
                True: [
                    i for i, image_id in enumerate(self.image_ids)
                    if image_id in self.labeled_image_ids
                ],
                False: [
                    i for i, image_id in enumerate(self.image_ids)
                    if image_id not in self.labeled_image_ids
                ],
            }

        self.transforms = transforms
        # Offset the seeds so the three pipelines do not run identical streams
        self.fallback_transforms = (
            get_val_transforms(seed=None if seed is None else seed + 2)
            if is_training and transforms is not None else None
        )
        # Mosaic samples are already scaled and cropped to the input size by
        # _load_mosaic, so they use a pipeline with no geometry block. The
        # shared fallback stays valid: its resize is a no-op at that size.
        self.mosaic_transforms = (
            get_train_transforms(mosaic=True, seed=None if seed is None else seed + 1)
            if is_training and transforms is not None else None
        )
        # Staged pipelines for the teacher's weak view. The geometric stage is
        # run once and its output forked: the teacher gets it normalized as-is,
        # the student gets the appearance stage on top. Sharing one geometric
        # call is what keeps the two views pixel-aligned, which feature KD needs
        # -- a flip applied to one and not the other would silently distil
        # mirrored features.
        teacher_view_wanted = (
            _teacher_weak_view_wanted()
            if emit_teacher_view is None
            else bool(emit_teacher_view)
        )
        self.emit_teacher_view = (
            is_training and transforms is not None and teacher_view_wanted
        )
        if self.emit_teacher_view:
            self.geometric_transforms = get_train_transforms(
                seed=None if seed is None else seed + 3, stage="geometric"
            )
            self.mosaic_geometric_transforms = get_train_transforms(
                mosaic=True, seed=None if seed is None else seed + 4, stage="geometric"
            )
            self.appearance_transforms = get_train_transforms(
                seed=None if seed is None else seed + 5, stage="appearance"
            )
            self.output_transforms = get_output_transforms()
            self.geometric_fallback_transforms = get_geometric_fallback_transforms(
                seed=None if seed is None else seed + 6
            )
        else:
            self.geometric_transforms = None
            self.mosaic_geometric_transforms = None
            self.appearance_transforms = None
            self.output_transforms = None
            self.geometric_fallback_transforms = None

        self.box_convert = get_box_converter(cfg.model_format)
        self.aug = cfg.augmentation
        self.is_training = is_training
        
        # Used to turn off mosaic and mixup in the final epochs
        self.use_complex_augs = True

        # Response KD reads teacher predictions from an offline cache rather
        # than running the teacher on the augmented composite. Loaded only for
        # training: validation never distils.
        self.teacher_cache = None
        self._kd_inject_threshold = 0.0
        teacher_cache_wanted = (
            _teacher_cache_wanted()
            if enable_teacher_cache is None
            else bool(enable_teacher_cache)
        )
        if is_training and teacher_cache_wanted:
            from kd_cache import TeacherCache, default_cache_path

            # Resolve the lowest confidence any active response objective can
            # consume before loading, so the cache can prove it actually stored
            # predictions down to that floor.
            if teacher_cache_conf_threshold is None:
                thresholds = [
                    kd.get("teacher_conf_threshold", 0.0)
                    for key in ("pseudo_label", "pseudo_label_soft")
                    for kd in [cfg.kd_config.get(key, {})]
                    if kd.get("enabled", False)
                ]
                self._kd_inject_threshold = min(thresholds) if thresholds else 0.0
            else:
                self._kd_inject_threshold = float(teacher_cache_conf_threshold)

            self.teacher_cache = TeacherCache.load(
                default_cache_path(
                    cfg.teacher_cache_dir, "train", cfg.teacher["weights"]
                ),
                cfg.teacher["weights"],
                (cfg.input_height, cfg.input_width),
                expected_num_classes=cfg.num_classes,
                requested_min_confidence=self._kd_inject_threshold,
            )
            # Hybrid response KD consumes cached predictions just as
            # teacher-only training does. Validate both coverage and manifest
            # image identity before either regime starts an expensive run.
            selected_images = [self.coco.imgs[image_id] for image_id in self.image_ids]
            self.teacher_cache.require_coverage(
                selected_images, f"training manifest {annotation_file!r}"
            )
            print(
                f"[Dataset] Teacher cache loaded: {len(self.teacher_cache):,} images "
                f"(injecting boxes above conf {self._kd_inject_threshold})."
            )
        self.close_epochs = self.aug.get("mosaic", {}).get("close_mosaic_epochs", 15)

        # Map configured COCO category IDs to contiguous model class indices.
        sorted_cat_ids = sorted(cfg.category_mapping.keys())
        self.coco_to_model_cls = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}

    def update_aug_schedule(self, epoch: int, max_epochs: int):
        """
        Check if current epoch is within the 'close_mosaic_epochs' from the end of training.
        If so, mosaic and mixup augmentations are disabled.

        Args:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
        """
        
        # During the last 'close_epochs' epochs, disable mosaic and mixup
        if max_epochs - epoch <= self.close_epochs:
            if self.use_complex_augs:
                print(f"[Dataset] Epoch {epoch}: Disabling Mosaic and MixUp for the final epochs.")
                self.use_complex_augs = False


    def _sample_composition_index(self, has_gt: bool) -> int:
        """Sample a Mosaic/MixUp source from the same supervision partition."""

        if self._composition_indices is not None:
            candidates = self._composition_indices[bool(has_gt)]
            if candidates:
                return random.choice(candidates)
        return random.randint(0, len(self.image_ids) - 1)
    # Methods whose KD can be switched off per augmentation, in the order the
    # target dict reports them.
    KD_METHODS = (("feature", "feature_based"),
                  ("hard", "pseudo_label"),
                  ("soft", "pseudo_label_soft"))

    @staticmethod
    def _kd_disable_flags(applied_augs: set) -> dict:
        """Per-method KD disable flags for the augmentations this sample used.

        A single global switch cannot express the policy the three methods
        actually need, because they are not equally sensitive. Hard KD is
        GT-anchored and uses teacher confidence only to select, so mixup cannot
        corrupt it. Feature KD distils the very composite the student sees.
        Soft KD is the exception: its confidence term distils an absolute
        probability the teacher measured on the clean, unmixed source, which no
        longer describes an object contributing a fraction of the blended
        pixels.

        Args:
            applied_augs: Names of the composition augmentations that fired for
                this sample, from {"mosaic", "mixup"}.

        Returns:
            dict: ``{"feature": bool, "hard": bool, "soft": bool}``, True where
            that method must skip the sample.
        """
        kd = cfg.kd_config
        fallback = kd.get("disable_on_augs", [])
        flags = {}
        for name, key in ObjectDetectionDataset.KD_METHODS:
            excluded = kd.get(key, {}).get("disable_on_augs", fallback)
            flags[name] = bool(applied_augs & set(excluded))
        return flags

    def _lsj_active(self, is_mosaic: bool) -> bool:
        """
        Whether the LSJ block runs for a sample, and therefore whether the
        pipeline built for it ends in a crop that crop-retries can influence.

        Args:
            is_mosaic: Whether the sample came from `_load_mosaic`.

        Returns:
            bool: True when LSJ applies to this sample.
        """

        lsj = self.aug.get("lsj", {})
        if not lsj.get("enabled", False):
            return False
        return (not is_mosaic) or bool(lsj.get("apply_on_mosaic", False))

    def class_instance_counts(self) -> np.ndarray:
        """
        Count annotated instances per model class index over the whole split.

        Counts follow the same filtering as `_load_image_and_boxes` (crowd
        annotations and degenerate boxes dropped) so the frequencies match what
        training actually sees. Used to build inverse-frequency class weights.

        Returns:
            np.ndarray: Float array of length `cfg.num_classes`.
        """

        if not self.use_ground_truth:
            raise RuntimeError(
                "Ground-truth class counts are unavailable when "
                "use_ground_truth=False. Teacher-only training must not derive "
                "class weights from annotations."
            )

        counts = np.zeros(cfg.num_classes, dtype=np.float64)
        ids_to_count = self.labeled_image_ids if self.labeled_image_ids is not None else self.image_ids
        for img_id in ids_to_count:
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
                if ann.get("iscrowd", 0):
                    continue
                cls_idx = self.coco_to_model_cls.get(ann["category_id"])
                if cls_idx is None:
                    continue
                _, _, w, h = ann["bbox"]
                if w > 0 and h > 0:
                    counts[cls_idx] += 1
        return counts

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """

        return len(self.image_ids)

    # Ground-truth labels are class indices in [0, num_classes). Cached teacher
    # boxes are appended to the very same box list, tagged with a label of
    # `num_classes + slot`, where `slot` indexes this sample's growing logits
    # list. Riding the ground truth's list is what makes the mosaic work: the
    # placement, window crop and horizontal flip are applied to teacher and GT
    # boxes by one Albumentations call, so predictions from all four sources
    # land pooled in the composite's frame with no warping code at all.
    TEACHER_LABEL_OFFSET_ATTR = "num_classes"

    def _append_teacher_boxes(
        self,
        image_id: int,
        image_width: int,
        image_height: int,
        boxes: list,
        labels: list,
        logit_sink: list,
    ) -> None:
        """Append this image's cached teacher boxes, in COCO pixel format."""
        if self.teacher_cache is None:
            return
        t_boxes, t_logits = self.teacher_cache.get(image_id)
        if t_boxes.shape[0] == 0:
            return

        # The cache deliberately stores every query so the threshold stays
        # tunable without a rebuild, but only boxes that will clear it are worth
        # sending through Albumentations -- a 4-image mosaic would otherwise
        # push 1,200 teacher boxes through the bbox transforms per sample to
        # discard most of them in the loss a moment later.
        if self._kd_inject_threshold > 0.0:
            conf = 1.0 / (1.0 + np.exp(-t_logits.astype(np.float32)))
            keep = conf.max(axis=1) > self._kd_inject_threshold
            if not keep.any():
                return
            t_boxes, t_logits = t_boxes[keep], t_logits[keep]

        # Stored normalized against the original image, so they survive any
        # resize the caller has already applied.
        cx = t_boxes[:, 0].astype(np.float32) * image_width
        cy = t_boxes[:, 1].astype(np.float32) * image_height
        bw = t_boxes[:, 2].astype(np.float32) * image_width
        bh = t_boxes[:, 3].astype(np.float32) * image_height
        x_min = cx - bw / 2.0
        y_min = cy - bh / 2.0

        for j in range(t_boxes.shape[0]):
            w, h = float(bw[j]), float(bh[j])
            if w < 1.0 or h < 1.0:
                continue
            boxes.append([float(x_min[j]), float(y_min[j]), w, h])
            labels.append(cfg.num_classes + len(logit_sink))
            logit_sink.append(t_logits[j].astype(np.float32))

    def _load_image_and_boxes(
        self, idx: int, logit_sink: list | None = None
    ) -> tuple[np.ndarray, list[list[float]], list[int], int, bool]:
        """
        Load an image and its corresponding bounding boxes and labels.

        Args:
            idx: Index of the image to load.

        Returns:
            tuple[np.ndarray, list[list[float]], list[int], int, bool]: Tuple
            containing:
                - Image in RGB format.
                - List of bounding boxes in COCO format.
                - List of labels.
                - Image ID.
                - Whether this image carried ground-truth annotations.
        """
        
        # Load image and annotations
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = resolve_image_path(img_info["file_name"], self.annotation_file)
        if img_path is None:
            raise FileNotFoundError(f"Could not resolve image path for {img_info['file_name']} in {self.annotation_file}")
            
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ground truth is read only when this dataset was explicitly allowed to
        # use it. In teacher-only mode this branch is skipped before any crop or
        # composition augmentation can observe annotation geometry.
        #
        # In semi-supervised mode (labeled_image_ids is set), GT is per-image
        # gated: only images whose ID is in the labeled set receive annotations.
        # Unlabeled images contribute zero GT boxes and rely on the teacher cache
        # for supervision.
        has_gt = False
        if self.use_ground_truth:
            if self.labeled_image_ids is not None and img_id not in self.labeled_image_ids:
                # Unlabeled partition — skip GT for this image.
                annotations = ()
            else:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                annotations = self.coco.loadAnns(ann_ids)
                has_gt = True
        else:
            annotations = ()

        # Extract bounding boxes and labels
        boxes, labels = [], []
        image_height, image_width = image.shape[:2]
        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue
            category_id = ann["category_id"]
            if category_id not in self.coco_to_model_cls:
                raise ValueError(
                    f"Annotation {ann.get('id', '<unknown>')} uses category_id "
                    f"{category_id}, which is absent from config.category_mapping."
                )

            x_min, y_min, w, h = map(float, ann["bbox"])
            x_max = max(0.0, min(float(image_width), x_min + w))
            y_max = max(0.0, min(float(image_height), y_min + h))
            x_min = max(0.0, min(float(image_width), x_min))
            y_min = max(0.0, min(float(image_height), y_min))
            w = x_max - x_min
            h = y_max - y_min

            if w > 0 and h > 0:
                boxes.append([x_min, y_min, w, h])
                labels.append(self.coco_to_model_cls[category_id])

        if logit_sink is not None:
            self._append_teacher_boxes(
                img_id, image_width, image_height, boxes, labels, logit_sink
            )

        return image, boxes, labels, img_id, has_gt

    def _sample_mosaic_window(
        self,
        canvas: np.ndarray,
        boxes: list[list[float]],
        labels: list[int],
        min_visibility: float,
        max_trials: int,
    ) -> tuple[np.ndarray, list[list[float]], list[int]]:
        """
        Jitter the scale and cut the model input out of a mosaic canvas.

        Equivalent to `RandomScale -> RandomCrop(H, W)` but done in one step: a
        scale `r` is sampled, a `(H/r, W/r)` window is cropped at native
        resolution, and only that window is resized to `(H, W)`. Scaling the
        whole canvas first would materialise an intermediate up to `r^2` times
        the canvas area and then discard almost all of it -- at `r = 2` that is
        a 5120x5120 image for a 736x1280 output. Ultralytics avoids the same
        cost by folding the scale into the affine matrix of the single
        `warpAffine` that produces its output.

        Windows larger than the canvas would need padding, so `r` is clamped to
        keep the window inside it. With the configured range this binds only at
        the lower end, where the window is already the full canvas.

        Args:
            canvas: Mosaic canvas, `2H x 2W`.
            boxes: Canvas-space boxes in COCO format.
            labels: Labels parallel to `boxes`.
            min_visibility: Fraction of a box that must survive the crop for the
                window to be accepted. Boxes below the configured baseline are
                dropped either way.
            max_trials: Window positions to try before keeping the last one.

        Returns:
            tuple[np.ndarray, list[list[float]], list[int]]: Cropped and resized
            image, remapped boxes, and their labels.
        """

        H, W = cfg.input_height, cfg.input_width
        canvas_h, canvas_w = canvas.shape[:2]

        lo, hi = self.aug.get("mosaic", {}).get(
            "scale_limit", self.aug.get("lsj", {}).get("scale_limit", [-0.5, 1.0])
        )
        r = random.uniform(1.0 + lo, 1.0 + hi)
        r = max(r, H / canvas_h, W / canvas_w)
        win_h = min(int(round(H / r)), canvas_h)
        win_w = min(int(round(W / r)), canvas_w)

        baseline_visibility = float(self.aug.get("lsj", {}).get("min_visibility", 0.1))
        keep_visibility = max(baseline_visibility, float(min_visibility))
        areas = [bw * bh for _, _, bw, bh in boxes]

        def clip_to(x0: int, y0: int, threshold: float):
            kept_boxes, kept_labels = [], []
            for (bx, by, bw, bh), label, area in zip(boxes, labels, areas):
                nx1, ny1 = max(bx, x0), max(by, y0)
                nx2, ny2 = min(bx + bw, x0 + win_w), min(by + bh, y0 + win_h)
                cw, ch = nx2 - nx1, ny2 - ny1
                if cw < 1.0 or ch < 1.0:
                    continue
                if area > 0 and (cw * ch) / area < threshold:
                    continue
                kept_boxes.append([nx1 - x0, ny1 - y0, cw, ch])
                kept_labels.append(label)
            return kept_boxes, kept_labels

        x0 = y0 = 0
        kept_boxes, kept_labels = [], []
        for _ in range(max(1, int(max_trials))):
            x0 = random.randint(0, canvas_w - win_w)
            y0 = random.randint(0, canvas_h - win_h)
            kept_boxes, kept_labels = clip_to(x0, y0, keep_visibility)
            if kept_boxes or not boxes:
                break
        else:
            # No window met the sampled requirement; keep the last one under the
            # baseline threshold rather than returning an empty sample.
            kept_boxes, kept_labels = clip_to(x0, y0, baseline_visibility)

        window = canvas[y0: y0 + win_h, x0: x0 + win_w]
        if (win_h, win_w) != (H, W):
            window = cv2.resize(window, (W, H))
            sx, sy = W / win_w, H / win_h
            kept_boxes = [[bx * sx, by * sy, bw * sx, bh * sy]
                          for bx, by, bw, bh in kept_boxes]
        else:
            window = np.ascontiguousarray(window)

        return window, kept_boxes, kept_labels

    def _load_mosaic(
        self,
        idx: int,
        min_visibility: float = 0.0,
        max_trials: int = 1,
        logit_sink: list | None = None,
    ) -> tuple[np.ndarray, list[list[float]], list[int], int]:
        """
        Create a mosaic image by combining 4 different images.

        The canvas is `2H x 2W` with each source stretched to exactly the model
        input size, so a window of the input size covers a quarter of it -- one
        image's worth of objects, which is what Ultralytics' square
        `2*imgsz` canvas gives it. A square `2s x 2s` canvas built from
        `s = max(H, W)` would instead make the output window only `H*W / (2H)^2`
        of the canvas, starving each sample of labels when the input is wide and
        short. Scale jitter and the crop are applied here, by
        `_sample_mosaic_window`, so LSJ does not apply to mosaic samples.

        Args:
            idx: Index of the base image to load.
            min_visibility: Passed through to the window sampler.
            max_trials: Passed through to the window sampler.

        Returns:
            tuple[np.ndarray, list[list[float]], list[int], int]: Tuple containing:
                - Mosaic image in RGB format, at the configured input size.
                - List of bounding boxes in COCO format.
                - List of labels.
                - Image ID of the base image.
        """

        # Keep every source in the base image's supervision partition. A
        # standard detection loss cannot represent the other partition as an
        # unknown region inside a composite.
        base_has_gt = (
            self.use_ground_truth
            and self.labeled_image_ids is not None
            and self.image_ids[idx] in self.labeled_image_ids
        )
        indices = [idx] + [
            self._sample_composition_index(base_has_gt) for _ in range(3)
        ]

        # Tile size is the model input; the canvas holds four of them
        H, W = cfg.input_height, cfg.input_width
        canvas_h, canvas_w = H * 2, W * 2

        # Get center point of mosaic image
        xc = int(random.uniform(W * 0.5, W * 1.5))
        yc = int(random.uniform(H * 0.5, H * 1.5))

        # Create a gray mosaic canvas
        mosaic_img = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)

        # Initialize lists to store the bounding boxes and labels
        mosaic_boxes = []
        mosaic_labels = []
        base_img_id = None

        # Loop through the 4 images to combine them into the mosaic image
        any_source_has_gt = False
        for i, index in enumerate(indices):
            # Load the i-th image
            img, boxes, labels, img_id, source_has_gt = self._load_image_and_boxes(index, logit_sink)
            any_source_has_gt = any_source_has_gt or source_has_gt
            if i == 0:
                base_img_id = img_id

            # Stretch to the model input size -- the same anisotropic resize the
            # val pipeline applies, so mosaic tiles carry evaluation-time scale
            h, w = img.shape[:2]
            scale_x, scale_y = W / w, H / h
            img = cv2.resize(img, (W, H))
            nh, nw = H, W

            # Get the coordinates of the four quadrants in the mosaic image and the corresponding area in the current image
            img_x1, img_y1, img_x2, img_y2 = 0, 0, 0, 0
            if i == 0:  # top left
                x1, y1, x2, y2 = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                img_x1, img_y1 = nw - (x2 - x1), nh - (y2 - y1)
                img_x2, img_y2 = nw, nh
            elif i == 1:  # top right
                x1, y1, x2, y2 = xc, max(yc - nh, 0), min(xc + nw, canvas_w), yc
                img_x1, img_y1 = 0, nh - (y2 - y1)
                img_x2, img_y2 = x2 - x1, nh
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = max(xc - nw, 0), yc, xc, min(canvas_h, yc + nh)
                img_x1, img_y1 = nw - (x2 - x1), 0
                img_x2, img_y2 = nw, y2 - y1
            elif i == 3:  # bottom right
                x1, y1, x2, y2 = xc, yc, min(canvas_w, xc + nw), min(canvas_h, yc + nh)
                img_x1, img_y1 = 0, 0
                img_x2, img_y2 = x2 - x1, y2 - y1

            # Place the current image in the mosaic image
            mosaic_img[y1: y2, x1: x2] = img[img_y1: img_y2, img_x1: img_x2]

            # Calculate the offset of the current image in the mosaic image
            dx = x1 - img_x1
            dy = y1 - img_y1

            # Adjust the bounding boxes to the mosaic image coordinates
            for (bx, by, bw, bh), label in zip(boxes, labels):
                bx, bw = bx * scale_x, bw * scale_x
                by, bh = by * scale_y, bh * scale_y
                nx1 = max(x1, bx + dx)
                ny1 = max(y1, by + dy)
                nx2 = min(x2, bx + bw + dx)
                ny2 = min(y2, by + bh + dy)
                if nx2 > nx1 and ny2 > ny1:
                    mosaic_boxes.append([nx1, ny1, nx2 - nx1, ny2 - ny1])
                    mosaic_labels.append(label)

        # Jitter the scale and cut the model input out of the canvas in one step
        mosaic_img, mosaic_boxes, mosaic_labels = self._sample_mosaic_window(
            mosaic_img, mosaic_boxes, mosaic_labels, min_visibility, max_trials
        )
        return mosaic_img, mosaic_boxes, mosaic_labels, base_img_id, any_source_has_gt

    def _load_mixup(self, img1, boxes1, labels1, img_id1, has_gt1: bool, use_mosaic: bool = False, logit_sink: list | None = None) -> tuple[np.ndarray, list[list[float]], list[int], int, bool]:
        """
        Generate a mixup image by combining a random image from the dataset with the given image.
        
        Args:
            img1: First image in RGB format.
            boxes1: List of bounding boxes in COCO format.
            labels1: List of labels.
            img_id1: ID of the first image.
            use_mosaic: Whether `img1` is a mosaic canvas, in which case the
                partner image is mosaicked too so both share the same scale.

        Returns:
            tuple[np.ndarray, list[list[float]], list[int], int, bool]: Tuple containing:
                - Mixup image in RGB format.
                - List of bounding boxes in COCO format after mixup.
                - List of labels after mixup.
                - Image ID of the first image.
                - Whether either source image carried ground-truth annotations.
        """
        
        # Get the second image. When the base is a mosaic, mosaic the partner
        # too instead of stretching a single native image up to the 2s x 2s
        # canvas -- that stretch would inflate the partner's objects by ~2x.
        # Ultralytics does the same, running MixUp with the mosaic pipeline as
        # its `pre_transform` so both operands share one scale convention.
        idx2 = self._sample_composition_index(has_gt1)
        if use_mosaic:
            img2, boxes2, labels2, _, has_gt2 = self._load_mosaic(idx2, logit_sink=logit_sink)
        else:
            img2, boxes2, labels2, _, has_gt2 = self._load_image_and_boxes(idx2, logit_sink)

        # Resize the second image to the target size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img2 = cv2.resize(img2, (w1, h1))
        
        # Scale the second image bounding boxes
        scaled_boxes2 = []
        for bx, by, bw, bh in boxes2:
            scaled_boxes2.append([bx * w1 / w2, by * h1 / h2, bw * w1 / w2, bh * h1 / h2])
            
        # Generate the mixup image
        alpha = self.aug.get("mixup", {}).get("random_distribution_alpha", 8.0)
        beta = self.aug.get("mixup", {}).get("random_distribution_beta", 8.0)
        r = random.betavariate(alpha, beta)
        mix_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # Return the mixup image, combined bounding boxes, combined labels, and the base image ID
        return mix_img, boxes1 + scaled_boxes2, labels1 + labels2, img_id1, (has_gt1 or has_gt2)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Get the item at the given index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Tuple of image and target.
        """

        # Apply mosaic and mixup augmentations
        apply_mosaic = self.is_training and self.use_complex_augs and self.aug.get("mosaic", {}).get("enabled", False) and random.random() < self.aug.get("mosaic", {}).get("p", 0.5)
        apply_mixup = self.is_training and self.use_complex_augs and self.aug.get("mixup", {}).get("enabled", False) and random.random() < self.aug.get("mixup", {}).get("p", 0.15)
        
        # Which composition augmentations fired. Resolved per KD method at the
        # end, since the three differ in what they can tolerate.
        applied_augs: set = set()

        is_mosaic = self.is_training and apply_mosaic

        # Collects the teacher logits for every cached box injected below. A box's
        # label encodes its slot here, so the two stay paired through the whole
        # augmentation pipeline even as boxes are cropped away or reordered.
        logit_sink: list | None = [] if self.teacher_cache is not None else None

        # --- Make sure there is at least one valid box ---
        # Sample an additional minimum visible fraction once per image. Both
        # croppers use it: the mosaic window sampler and the LSJ block. It is
        # sampled up front because the mosaic crops during loading.
        lsj_cfg = self.aug.get("lsj", {})
        if self.is_training and (is_mosaic or lsj_cfg.get("enabled", False)):
            visibility_candidates = lsj_cfg.get(
                "min_visibility_candidates", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
            )
            max_crop_trials = lsj_cfg.get("max_crop_trials", 50)
            min_visibility = random.choice(visibility_candidates)
        else:
            max_crop_trials = 1
            min_visibility = 0.0

        if is_mosaic:
            image, boxes, labels, img_id, has_gt = self._load_mosaic(
                idx, min_visibility=min_visibility, max_trials=max_crop_trials,
                logit_sink=logit_sink,
            )
            applied_augs.add("mosaic")
        else:
            image, boxes, labels, img_id, has_gt = self._load_image_and_boxes(idx, logit_sink)

        if self.is_training and apply_mixup:
            image, boxes, labels, img_id, has_gt = self._load_mixup(
                image, boxes, labels, img_id, has_gt, use_mosaic=is_mosaic,
                logit_sink=logit_sink,
            )
            applied_augs.add("mixup")

        # Pre-process boxes to avoid Albumentations ValueError
        valid_raw_boxes = []
        valid_raw_labels = []
        img_h, img_w = image.shape[:2]
        
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            # Clamp to image boundaries safely
            x_min = max(0.0, min(float(x), img_w - 1.0))
            y_min = max(0.0, min(float(y), img_h - 1.0))
            x_max = max(0.0, min(float(x + w), float(img_w)))
            y_max = max(0.0, min(float(y + h), float(img_h)))
            
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            
            # Require at least 1 pixel to avoid rounding errors causing y_max <= y_min
            if w >= 1.0 and h >= 1.0:
                valid_raw_boxes.append([x, y, w, h])
                valid_raw_labels.append(label)
                
        raw_image, raw_boxes, raw_labels = image, valid_raw_boxes, valid_raw_labels
        if self.transforms:
            # Mosaic samples already went through their own scale-and-crop.
            # Retries here are only worth running when the selected pipeline
            # still contains a crop, i.e. when LSJ is active for this sample.
            use_mosaic_pipeline = is_mosaic and self.mosaic_transforms is not None
            pipeline_crops = self._lsj_active(is_mosaic)
            retry_kwargs = dict(
                image=raw_image,
                boxes=raw_boxes,
                labels=raw_labels,
                min_visibility=min_visibility if pipeline_crops else 0.0,
                max_trials=(max_crop_trials if (self.is_training and pipeline_crops) else 1),
            )
            if self.emit_teacher_view:
                # One geometric pass, then fork. The teacher's copy is taken
                # before any appearance op touches it, so both views share the
                # crop, the flip and the scale exactly.
                geom_pipeline = (
                    self.mosaic_geometric_transforms if use_mosaic_pipeline
                    else self.geometric_transforms
                )
                g_image, t_boxes, t_labels = apply_transforms_with_retry(
                    transforms=geom_pipeline,
                    fallback_transforms=self.geometric_fallback_transforms,
                    **retry_kwargs,
                )
                teacher_view = self.output_transforms(image=g_image)["image"]
                t_image = self.appearance_transforms(image=g_image)["image"]
            else:
                teacher_view = None
                t_image, t_boxes, t_labels = apply_transforms_with_retry(
                    transforms=self.mosaic_transforms if use_mosaic_pipeline else self.transforms,
                    fallback_transforms=self.fallback_transforms,
                    **retry_kwargs,
                )
        else:
            teacher_view = None
            t_image, t_boxes, t_labels = raw_image, raw_boxes, raw_labels

        # Set the final image, boxes, and labels
        image, boxes, labels = t_image, t_boxes, t_labels

        # Transforms are optional. Keep that public path collatable and
        # normalize boxes against the image dimensions actually returned,
        # rather than assuming the configured resize happened.
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(
                    f"Expected an HWC image array, got shape {image.shape}."
                )
            image = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError(
                "Dataset transforms must return a CHW tensor or an HWC NumPy array."
            )
        out_h, out_w = image.shape[-2:]

        # Split the teacher's boxes back out. They travelled the whole
        # augmentation pipeline inside `boxes`, so whatever geometry was applied
        # to the ground truth has been applied to them identically -- including
        # the mosaic placement that pooled four source images into this frame.
        kd_boxes: list = []
        kd_logits: list = []
        if logit_sink is not None and labels:
            gt_boxes, gt_labels = [], []
            for box, label in zip(boxes, labels):
                if label >= cfg.num_classes:
                    slot = int(label) - cfg.num_classes
                    if 0 <= slot < len(logit_sink):
                        kd_boxes.append(box)
                        kd_logits.append(logit_sink[slot])
                else:
                    gt_boxes.append(box)
                    gt_labels.append(label)
            boxes, labels = gt_boxes, gt_labels

        # Convert boxes to target model format
        boxes_tensor = self.box_convert(boxes, out_h, out_w)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary and return with image
        target = {
            "boxes":        boxes_tensor,
            "labels":       labels_tensor,
            "image_id":     torch.tensor([img_id]),
            "has_gt":       torch.tensor([has_gt], dtype=torch.bool),
        }

        kd_flags = self._kd_disable_flags(applied_augs)
        for _name, _ in self.KD_METHODS:
            target[f"disable_kd_{_name}"] = torch.tensor([kd_flags[_name]], dtype=torch.bool)
        # Retained as the union, so any consumer that has not been taught about
        # per-method flags still fails safe by skipping the sample entirely.
        target["disable_kd"] = torch.tensor([any(kd_flags.values())], dtype=torch.bool)

        if teacher_view is not None:
            target["teacher_view"] = teacher_view

        if logit_sink is not None:
            target["kd_boxes"] = self.box_convert(kd_boxes, out_h, out_w)
            target["kd_logits"] = (
                torch.as_tensor(np.stack(kd_logits), dtype=torch.float32)
                if kd_logits else torch.zeros((0, cfg.num_classes), dtype=torch.float32)
            )

        return image, target


def object_detection_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """
    Custom collate function for the object detection dataset.

    Args:
        batch: List of tuples of (image, target).

    Returns:
        tuple[torch.Tensor, list[dict]]: Tuple of stacked images and targets. 
        Image is of shape (B, C, H, W), while targets is a list of dictionaries.
    """

    # Separate images and targets
    images, targets = zip(*batch)

    # Stack images and return targets
    return torch.stack(images, dim=0), list(targets)


# --- Dataloaders ---
def build_dataloaders(
    train_json: str,
    val_json: str,
    *,
    train_use_ground_truth: bool = True,
    train_labeled_image_ids: frozenset[int] | None = None,
    train_enable_teacher_cache: bool | None = None,
    train_emit_teacher_view: bool | None = None,
    train_teacher_conf_threshold: float | None = None,
    require_empty_train_annotations: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_json (str): Path to the training annotations JSON file.
        val_json (str): Path to the validation annotations JSON file.
        train_use_ground_truth: Whether train annotations may enter samples.
            Validation always retains ground truth.
        train_labeled_image_ids: When not None, ground truth is loaded only
            for images whose ID is in this set.  Used in semi-supervised mode.
        train_enable_teacher_cache: Optional response-cache override for train.
        train_emit_teacher_view: Optional feature-KD view override for train.
        train_teacher_conf_threshold: Optional cache injection threshold.
        require_empty_train_annotations: Reject a labelled train manifest when
            train ground truth is disabled.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple of training and validation dataloaders.
    """

    seed = int(cfg.shared_train_config.get("seed", 42))

    # Create train and validation datasets
    train_dataset = ObjectDetectionDataset(
        annotation_file=train_json,
        transforms=get_train_transforms(seed=seed),
        is_training=True,
        seed=seed,
        use_ground_truth=train_use_ground_truth,
        labeled_image_ids=train_labeled_image_ids,
        enable_teacher_cache=train_enable_teacher_cache,
        emit_teacher_view=train_emit_teacher_view,
        teacher_cache_conf_threshold=train_teacher_conf_threshold,
        require_empty_annotations=require_empty_train_annotations,
    )

    val_dataset = ObjectDetectionDataset(
        annotation_file=val_json,
        transforms=get_val_transforms(seed=seed),
        is_training=False,
        seed=seed,
        use_ground_truth=True,
        enable_teacher_cache=False,
        emit_teacher_view=False,
    )

    # Shuffling draws from this generator, and PyTorch derives each worker's
    # per-epoch base seed from it too, so the whole augmentation stream is
    # reproducible from shared_train_config["seed"].
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    # Create train and validation dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader_config["batch_size"],
        shuffle=True,
        num_workers=cfg.dataloader_config["num_workers"],
        collate_fn=object_detection_collate_fn,
        pin_memory=cfg.dataloader_config["pin_memory"],
        worker_init_fn=_worker_init_fn,
        generator=shuffle_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader_config["batch_size"],
        shuffle=False,
        num_workers=cfg.dataloader_config["num_workers"],
        collate_fn=object_detection_collate_fn,
        pin_memory=cfg.dataloader_config["pin_memory"],
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


# --- Main ---
if __name__ == "__main__":
    # Quick test
    print(f"Testing DataLoader  [model_format={cfg.model_format}] ...")

    # Get train and validation JSON paths
    train_json = os.path.join(cfg.processed_annotations_dir, "train.json")
    val_json   = os.path.join(cfg.processed_annotations_dir, "val.json")

    # Build and test dataloaders
    if os.path.exists(train_json):
        train_dl, val_dl = build_dataloaders(train_json, val_json)

        images, targets = next(iter(train_dl))
        print(f"  Image tensor : {images.shape}")                # [B, C, H, W]
        print(f"  Targets      : {len(targets)} items")
        print(f"  Boxes[0]     : {targets[0]['boxes'].shape}")    # [num_boxes, 4]

        print("Dataloader OK.")
    else:
        print("Annotation JSONs not found. Skipping live test.")
