#!/usr/bin/env python3
import os
import sys
import json
import math
import random
import re
from pathlib import Path
import cv2
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs.base_cfg as cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(path: str) -> Path:
    """Resolve project-owned relative paths independently of the caller's CWD."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def ordered_categories(category_mapping: dict) -> list[tuple[int, dict]]:
    """Validate and return categories ordered by their dataset category ID."""
    if not category_mapping:
        raise ValueError("category_mapping must not be empty.")

    categories = []
    for raw_id, raw_category in category_mapping.items():
        category_id = int(raw_id)
        category = dict(raw_category)
        if int(category.get("id", category_id)) != category_id:
            raise ValueError(
                f"Category mapping key {category_id} disagrees with entry id "
                f"{category.get('id')!r}."
            )
        if not str(category.get("name", "")).strip():
            raise ValueError(f"Category {category_id} has no name.")
        category["id"] = category_id
        categories.append((category_id, category))
    categories.sort(key=lambda item: item[0])
    return categories


# --- Helper Functions ---
def calculate_iou(boxA: list, boxB: list) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def generate_occlusion_test_set(test_baseline_json: str,
                                output_occlusion_json: str,
                                iou_threshold: float):
    """Filters the test_baseline dataset to extract images containing heavy
    occlusion (IoU > threshold).

    Args:
        test_baseline_json (str): Path to the test_baseline COCO JSON.
        output_occlusion_json (str): Path to save the occlusion COCO JSON.
        iou_threshold (float): IoU threshold for filtering.
    """
    with project_path(test_baseline_json).open('r') as f:
        coco = json.load(f)
        
    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
        
    # Detect occluded images
    occluded_image_ids = set()
    for img_id, anns in anns_by_img.items():
        is_occluded = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if calculate_iou(anns[i]["bbox"], anns[j]["bbox"]) > iou_threshold:
                    is_occluded = True
                    break
            if is_occluded:
                break
        if is_occluded:
            occluded_image_ids.add(img_id)
            
    # Filter images and annotations
    occluded_images = [img for img in coco["images"] if img["id"] in occluded_image_ids]
    occluded_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in occluded_image_ids]
    
    # Create new COCO JSON with only occluded images
    occlusion_coco = {
        "info": coco.get("info", {}),
        "images": occluded_images,
        "annotations": occluded_annotations,
        "categories": coco["categories"]
    }
    
    # Save the occlusion COCO JSON
    output_path = project_path(output_occlusion_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(occlusion_coco, f)
        
    print(f"  Generated {output_occlusion_json} (Occlusion split) - Images: {len(occluded_images)}, Annotations: {len(occluded_annotations)}")


def group_images_by_sequence(images: list[dict], sequence_pattern: str) -> dict[str, list[int]]:
    """Group COCO image records by the sequence their file name belongs to.

    The pattern is matched against the *basename*: `file_name` carries a
    project-relative path, so a pattern anchored at the start of the string
    would never match the sequence field.

    Args:
        images: COCO ``images`` records.
        sequence_pattern: Regex whose first group captures the sequence id.

    Returns:
        dict[str, list[int]]: Image IDs per sequence id, each list sorted.

    Raises:
        ValueError: If the pattern has no capture group, or any file name fails
            to match. Falling back to a per-image sequence id would silently
            turn the sequence split into a frame split, which is the one thing
            it exists to avoid.
    """
    compiled = re.compile(sequence_pattern)
    if compiled.groups < 1:
        raise ValueError(
            f"sequence_id_pattern {sequence_pattern!r} has no capture group; "
            "the first group must capture the sequence id."
        )

    sequences = defaultdict(list)
    unmatched = []
    for image in images:
        match = compiled.search(os.path.basename(image["file_name"]))
        if match is None:
            unmatched.append(image["file_name"])
            continue
        sequences[match.group(1)].append(image["id"])

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} image name(s) do not match sequence_id_pattern "
            f"{sequence_pattern!r}, e.g. {unmatched[:3]}. Set the pattern for "
            "this dataset before generating sequence splits."
        )

    return {key: sorted(value) for key, value in sequences.items()}


def select_sequence_subset(sequence_sizes: dict[str, int], fraction: float,
                           seed: int) -> list[str]:
    """Draw whole sequences at random until `fraction` of the images is covered.

    The draw is a prefix of a uniform random permutation, so selection is
    independent of how long a sequence is. That independence is the point:
    accumulating the largest sequences first reaches a 50% image budget with 13
    of VisDrone's 208 sequences instead of roughly 100, which buys the requested
    image count at a fraction of the scene diversity.

    Shuffling starts from a *sorted* base order, so the result is a function of
    the seed and the sequence ids alone -- not of dict insertion order, and not
    of how much the global RNG was used elsewhere.

    Only the boundary is size-aware: the final sequence is dropped when
    stopping short of the target lands closer to it than overshooting. Without
    that, a 434-image flight drawn last takes a 50% request to 55%.

    Args:
        sequence_sizes: Image count per sequence id.
        fraction: Target share of images, in (0, 1) exclusive.
        seed: Seed for the permutation.
    Returns:
        list[str]: Selected sequence ids, sorted. Always a proper non-empty
        subset, so both sides of the split have at least one sequence.

    Raises:
        ValueError: If `fraction` is outside (0, 1), or there are fewer than two
            sequences to split.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction!r}.")
    if len(sequence_sizes) < 2:
        raise ValueError(
            f"Need at least 2 sequences to form a split, got {len(sequence_sizes)}."
        )

    order = sorted(sequence_sizes)
    random.Random(seed).shuffle(order)

    target = fraction * sum(sequence_sizes.values())
    selected: list[str] = []
    covered = 0
    for sequence_id in order:
        if covered >= target:
            break
        selected.append(sequence_id)
        covered += sequence_sizes[sequence_id]

    # Boundary adjustment, only while it leaves both sides non-empty.
    if len(selected) > 1:
        without_last = covered - sequence_sizes[selected[-1]]
        if abs(without_last - target) < abs(covered - target):
            selected.pop()

    if len(selected) == len(sequence_sizes):
        selected.pop()

    return sorted(selected)


def write_sequence_split(source: dict, image_ids: set[int], sequences: list[str],
                         output_json: str, provenance: dict) -> dict:
    """Write one side of a sequence split, preserving IDs and recording how it was made.

    Image and annotation IDs are carried over untouched. The teacher-prediction
    cache is keyed by image ID, so preserving them lets a single cache built
    over the full manifest serve every split derived from it.

    Args:
        source: The parsed source COCO manifest.
        image_ids: IDs to retain.
        sequences: Sequence ids retained, recorded in ``info``.
        output_json: Destination path.
        provenance: Fields describing the draw, merged into ``info``.

    Returns:
        dict: The ``info`` block that was written.
    """
    images = [image for image in source["images"] if image["id"] in image_ids]
    annotations = [
        annotation for annotation in source["annotations"]
        if annotation["image_id"] in image_ids
    ]

    info = {
        **provenance,
        "num_sequences": len(sequences),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "fraction_realized_images": round(len(images) / max(len(source["images"]), 1), 4),
        "fraction_realized_annotations": round(
            len(annotations) / max(len(source["annotations"]), 1), 4
        ),
        # The full list keeps the draw and nesting relationship auditable
        # without re-running preparation.
        "sequences": sequences,
    }

    output_path = project_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(
            {
                "info": info,
                "images": images,
                "annotations": annotations,
                "categories": source["categories"],
            },
            output_file,
        )
    return info


def generate_sequence_split(source_json: str, output_json: str, fraction: float,
                            seed: int, sequence_pattern: str) -> None:
    """Write a whole-sequence subset holding roughly `fraction` of the images.

    Only the retained side is written. Its complement is exactly the source
    manifest minus this one, and `info["sequences"]` lists what was kept, so the
    unwritten side carries no information the pair would not already have.

    Because whole sequences move together, a model trained on the result has
    seen none of the drone flights the remainder was drawn from.

    Args:
        source_json: Manifest to subset, e.g. ``train.json``.
        output_json: Destination manifest.
        fraction: Target share of images to retain, in (0, 1).
        seed: Seed for the sequence draw.
        sequence_pattern: Regex whose first group captures the sequence id.
    """
    with project_path(source_json).open("r") as source_file:
        source = json.load(source_file)

    sequences = group_images_by_sequence(source["images"], sequence_pattern)
    sizes = {key: len(value) for key, value in sequences.items()}
    selected = select_sequence_subset(sizes, fraction, seed)
    image_ids = {image_id for key in selected for image_id in sequences[key]}

    provenance = {
        "description": f"whole-sequence subset ({fraction:.0%} target, seed {seed})",
        "source": source_json,
        "fraction_requested": fraction,
        "seed": seed,
        "sequence_id_pattern": sequence_pattern,
    }

    info = write_sequence_split(
        source, image_ids, selected, output_json,
        provenance,
    )
    print(
        f"  Generated {output_json} - Sequences: {info['num_sequences']}, "
        f"Images: {info['num_images']} ({info['fraction_realized_images']:.1%}), "
        f"Annotations: {info['num_annotations']} "
        f"({info['fraction_realized_annotations']:.1%})"
    )


def generate_sequence_splits(processed_dir: str, splits: dict,
                             sequence_pattern: str,
                             source_name: str = "train.json") -> None:
    """Generate every configured label-budget split from one source manifest.

    Args:
        processed_dir: Directory holding the source manifest and receiving outputs.
        splits: ``{name: {"fraction": float, "seed": int}}``; each writes
            ``<source stem>_<name>.json``.
        sequence_pattern: Regex whose first group captures the sequence id.
        source_name: File name of the manifest to partition.
    """
    if not splits:
        return

    source_json = os.path.join(processed_dir, source_name)
    if not project_path(source_json).is_file():
        print(f"  Skipping sequence splits: {source_json} not found.")
        return

    stem = Path(source_name).stem
    print(f"Generating label-budget splits from {source_name}...")
    for name, spec in splits.items():
        generate_sequence_split(
            source_json=source_json,
            output_json=os.path.join(processed_dir, f"{stem}_{name}.json"),
            fraction=float(spec["fraction"]),
            seed=int(spec["seed"]),
            sequence_pattern=sequence_pattern,
        )


def write_yolo_yaml(output_yaml_path: str, dataset_dir: str, category_mapping: dict):
    """Write a portable YOLO dataset YAML with contiguous model class indices.

    COCO category IDs need not be contiguous. YOLO class indices always are, so
    the names table uses the ordered model index while conversion back to COCO
    restores the original category ID.
    """
    categories = ordered_categories(category_mapping)
    dataset_path = project_path(dataset_dir).resolve()
    output_path = project_path(output_yaml_path)

    lines = [
        "# Dataset configuration for YOLO (generated programmatically)",
        f"path: {json.dumps(str(dataset_path))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(categories)}",
        "",
        "names:",
    ]
    for model_index, (_, category) in enumerate(categories):
        lines.append(f"  {model_index}: {json.dumps(category['name'])}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  Generated YOLO dataset YAML at: {output_path}")


def yolo_to_coco(yolo_dir: str, split_name: str, output_json: str, category_mapping: dict):
    """Convert one YOLO split to COCO while validating and clipping boxes."""
    dataset_path = project_path(yolo_dir)
    images_dir = dataset_path / "images" / split_name
    labels_dir = dataset_path / "labels" / split_name
    if not images_dir.is_dir():
        print(f"  Skipping split '{split_name}': image directory not found.")
        return

    categories = ordered_categories(category_mapping)
    model_to_category = {
        model_index: category_id
        for model_index, (category_id, _) in enumerate(categories)
    }
    coco_format = {
        "info": {"description": f"{split_name} converted from YOLO"},
        "images": [],
        "annotations": [],
        "categories": [category for _, category in categories],
    }

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )

    annotation_id = 1
    for image_path in tqdm(image_paths, desc=f"Converting {split_name} to COCO"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  WARNING: could not decode {image_path}; skipping it.")
            continue

        height, width = image.shape[:2]
        try:
            file_name = str(image_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            file_name = str(image_path.resolve())

        # IDs must remain dense even when an unreadable image was skipped.
        stored_image_id = len(coco_format["images"]) + 1
        coco_format["images"].append({
            "id": stored_image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        with label_path.open("r") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                parts = line.split()
                if len(parts) < 5:
                    print(
                        f"  WARNING: ignoring malformed label {label_path}:{line_number}."
                    )
                    continue
                try:
                    model_class = int(parts[0])
                    cx, cy, box_width, box_height = map(float, parts[1:5])
                except ValueError:
                    print(
                        f"  WARNING: ignoring non-numeric label {label_path}:{line_number}."
                    )
                    continue

                values = (cx, cy, box_width, box_height)
                if (
                    model_class not in model_to_category
                    or not all(math.isfinite(value) for value in values)
                    or box_width <= 0
                    or box_height <= 0
                ):
                    continue

                left = max(0.0, min(float(width), (cx - box_width / 2) * width))
                top = max(0.0, min(float(height), (cy - box_height / 2) * height))
                right = max(0.0, min(float(width), (cx + box_width / 2) * width))
                bottom = max(0.0, min(float(height), (cy + box_height / 2) * height))
                absolute_width = right - left
                absolute_height = bottom - top
                if absolute_width <= 0 or absolute_height <= 0:
                    continue

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": stored_image_id,
                    "category_id": model_to_category[model_class],
                    "bbox": [left, top, absolute_width, absolute_height],
                    "area": absolute_width * absolute_height,
                    "iscrowd": 0,
                })
                annotation_id += 1

    output_path = project_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(coco_format, output_file)

    print(
        f"  Generated {output_path} - Images: {len(coco_format['images'])}, "
        f"Annotations: {len(coco_format['annotations'])}"
    )


# --- Generic Dataset Preparation Helpers ---
def download_ultralytics_dataset(dataset_yaml_name: str, target_dir: str):
    """Downloads a dataset using the Ultralytics API and moves it to the target directory.

    Args:
        dataset_yaml_name (str): Name of the dataset YAML (e.g. 'VisDrone.yaml').
        target_dir (str): Target directory to store the dataset.
    """
    target_path = project_path(target_dir)
    if target_path.exists():
        print(f"Dataset directory '{target_path}' already exists. Skipping download.")
        return

    try:
        from ultralytics import settings
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required only for automatic dataset downloads. "
            "Install it or place the dataset at the configured target path."
        ) from exc

    print(f"Dataset path '{target_path}' not found. Downloading via Ultralytics API...")

    # Override datasets_dir temporarily so the download lands under data/.
    datasets_dir = str(PROJECT_ROOT / "data")
    original_datasets_dir = settings['datasets_dir']
    settings.update({'datasets_dir': datasets_dir})
    try:
        check_det_dataset(dataset_yaml_name)
    finally:
        settings.update({'datasets_dir': original_datasets_dir})

    # Ultralytics unpacks into <datasets_dir>/<yaml stem>, which is usually
    # already `target_dir`. Derive it from the referenced directory,
    # so a dataset whose folder name differs from `target_dir` still ends up
    # where the rest of the pipeline looks for it.
    dataset_folder_name = os.path.splitext(dataset_yaml_name)[0]
    expected_download_dir = os.path.join(datasets_dir, dataset_folder_name)

    expected_path = Path(expected_download_dir)
    if expected_path.resolve() == target_path.resolve():
        return
    if expected_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.rename(expected_path, target_path)
    else:
        print(
            f"WARNING: Ultralytics reported success but neither '{target_path}' nor "
            f"'{expected_path}' exists. Check the download manually."
        )


def convert_yolo_to_coco_splits(dataset_dir: str, splits: list[tuple[str, str]], category_mapping: dict, output_dir: str):
    """Converts multiple YOLO split directories back into COCO JSON format.

    Args:
        dataset_dir (str): Path to the YOLO dataset.
        splits (list[tuple[str, str]]): List of tuples containing (split_name, output_filename).
        category_mapping (dict): A dictionary mapping class IDs to names.
        output_dir (str): Directory where the output COCO JSONs will be saved.
    """
    for split_dir, out_file in splits:
        out_path = os.path.join(output_dir, out_file)
        yolo_to_coco(dataset_dir, split_dir, out_path, category_mapping)


# --- VisDrone-Specific Post-Processing ---
def generate_visdrone_occlusion_test_set(processed_dir: str, iou_threshold: float):
    """VisDrone-specific post-processing to generate the specialized occlusion test set.

    Args:
        processed_dir (str): Path to the processed annotations directory.
        iou_threshold (float): IoU threshold for filtering crowded scenes.
    """
    print("Generating specialized occlusion test set for VisDrone...")
    test_baseline_json = os.path.join(processed_dir, "test_baseline.json")
    test_occlusion_json = os.path.join(processed_dir, "test_occlusion.json")
    generate_occlusion_test_set(test_baseline_json, test_occlusion_json, iou_threshold)


def visdrone_post_process(processed_dir: str, dataset_config: dict):
    """All VisDrone post-processing: the occlusion test set and label-budget splits.

    Args:
        processed_dir (str): Path to the processed annotations directory.
        dataset_config (dict): This dataset's entry from `get_dataset_config`.
    """
    generate_visdrone_occlusion_test_set(processed_dir, cfg.crowded_scene_iou_threshold)

    # Each official source is budgeted independently. Whole sequences stay
    # together within a source, while any overlap defined by the upstream
    # train/validation split remains intact. The fixed test set is unchanged.
    splits = getattr(cfg, "label_budget_splits", {})
    sequence_pattern = dataset_config["sequence_id_pattern"]
    generate_sequence_splits(
        processed_dir, splits, sequence_pattern, source_name="train.json"
    )
    generate_sequence_splits(
        processed_dir, splits, sequence_pattern, source_name="val.json"
    )


# --- Dataset Configuration Registry ---
def get_dataset_config(dataset_name: str) -> dict:
    """Returns dataset-specific configurations (download urls, paths, splits, and post-processing).

    Args:
        dataset_name (str): Name of the dataset (e.g., 'VisDrone').

    Returns:
        dict: Dataset configuration dictionary.
    """
    configs = {
        "VisDrone": {
            "dataset_yaml_name": "VisDrone.yaml",
            "dataset_dir": "data/VisDrone",
            "yaml_config_path": "data/visdrone.yaml",
            "splits_to_convert": [
                ("train", "train.json"),
                ("val", "val.json"),
                ("test", "test_baseline.json")
            ],
            # VisDrone-DET ships its frames as <sequence>_<frame>_<d|v>_<index>.jpg
            # and preparation preserves those names, so the leading field is the
            # drone flight an image was sampled from. Matched against the
            # basename; the first capture group is the sequence id.
            "sequence_id_pattern": r"^(\d{7})_",
            "post_process_fn": visdrone_post_process
        }
    }

    if dataset_name not in configs:
        raise ValueError(
            f"Dataset '{dataset_name}' is not configured in prepare_dataset.py. "
            f"Supported configurations: {list(configs.keys())}"
        )

    return configs[dataset_name]


# --- Main ---
if __name__ == "__main__":
    dataset = cfg.dataset_name
    print(f"Starting dataset preparation pipeline for: {dataset}...")

    # Load dataset-specific configuration dynamically
    ds_cfg = get_dataset_config(dataset)

    dataset_yaml_name = ds_cfg["dataset_yaml_name"]
    dataset_dir = ds_cfg["dataset_dir"]
    yaml_config_path = ds_cfg["yaml_config_path"]
    splits_to_convert = ds_cfg["splits_to_convert"]

    # Download dataset if missing
    download_ultralytics_dataset(dataset_yaml_name, dataset_dir)

    # Generate the local dataset YAML configuration programmatically
    write_yolo_yaml(yaml_config_path, dataset_dir, cfg.category_mapping)

    # Convert YOLO splits back to COCO format
    convert_yolo_to_coco_splits(
        dataset_dir, 
        splits_to_convert, 
        cfg.category_mapping, 
        cfg.processed_annotations_dir
    )

    # Run dataset-specific post-processing if defined
    post_proc = ds_cfg.get("post_process_fn")
    if post_proc is not None:
        post_proc(cfg.processed_annotations_dir, ds_cfg)

    print("Data Pipeline Complete.")
