"""Tests for portable, lossless YOLO-to-COCO dataset preparation."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from prepare_dataset import (  # noqa: E402
    download_ultralytics_dataset,
    generate_sequence_split,
    generate_sequence_splits,
    get_dataset_config,
    group_images_by_sequence,
    ordered_categories,
    select_sequence_subset,
    write_yolo_yaml,
    yolo_to_coco,
)

SEQUENCE_PATTERN = get_dataset_config("VisDrone")["sequence_id_pattern"]


CATEGORIES = {
    3: {"id": 3, "name": "car", "supercategory": "vehicle"},
    9: {"id": 9, "name": "motor: cycle", "supercategory": "vehicle"},
}


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((10, 20, 3), dtype=np.uint8))


def test_yolo_conversion_restores_category_ids_and_clips_boxes(tmp_path):
    dataset = tmp_path / "dataset"
    image_path = dataset / "images" / "val" / "frame.jpeg"
    _write_image(image_path)

    label_path = dataset / "labels" / "val" / "frame.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text(
        "0 0.5 0.5 0.5 0.5\n"
        "1 0.0 0.0 0.4 0.4\n"
        "not-a-label\n"
        "2 0.5 0.5 0.2 0.2\n"
        "0 nan 0.5 0.2 0.2\n"
    )
    output = tmp_path / "annotations.json"

    yolo_to_coco(str(dataset), "val", str(output), CATEGORIES)

    coco = json.loads(output.read_text())
    assert [category["id"] for category in coco["categories"]] == [3, 9]
    assert len(coco["images"]) == 1
    assert [annotation["category_id"] for annotation in coco["annotations"]] == [3, 9]
    assert coco["annotations"][0]["bbox"] == pytest.approx([5.0, 2.5, 10.0, 5.0])
    assert coco["annotations"][1]["bbox"] == pytest.approx([0.0, 0.0, 4.0, 2.0])


def test_yolo_conversion_keeps_negative_images_without_labels_dir(tmp_path):
    dataset = tmp_path / "dataset"
    _write_image(dataset / "images" / "test" / "negative.jpg")
    output = tmp_path / "annotations.json"

    yolo_to_coco(str(dataset), "test", str(output), CATEGORIES)

    coco = json.loads(output.read_text())
    assert len(coco["images"]) == 1
    assert coco["annotations"] == []


def test_yolo_yaml_uses_contiguous_indices_and_quotes_names(tmp_path):
    output = tmp_path / "dataset.yaml"
    write_yolo_yaml(str(output), str(tmp_path / "dataset"), CATEGORIES)

    text = output.read_text()
    assert "nc: 2" in text
    assert '  0: "car"' in text
    assert '  1: "motor: cycle"' in text
    assert "  3:" not in text
    assert "  9:" not in text


def test_category_mapping_rejects_disagreeing_entry_id():
    with pytest.raises(ValueError, match="disagrees"):
        ordered_categories({3: {"id": 4, "name": "car"}})



def test_existing_dataset_does_not_require_ultralytics(tmp_path):
    dataset = tmp_path / "existing"
    dataset.mkdir()
    download_ultralytics_dataset("unused.yaml", str(dataset))



def test_category_mapping_must_not_be_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        ordered_categories({})


# --- Sequence-disjoint label-budget splits ---------------------------------

def _sequence_manifest(sequence_sizes: dict[str, int], tmp_path: Path,
                       source_name: str = "train.json") -> Path:
    """Build a COCO manifest whose file names encode VisDrone sequence ids."""
    images, annotations = [], []
    for sequence_id, count in sequence_sizes.items():
        for frame in range(count):
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": (
                    f"data/VisDrone/images/{Path(source_name).stem}/"
                    f"{sequence_id}_{frame:05d}_d_{image_id:07d}.jpg"
                ),
                "width": 20,
                "height": 10,
            })
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [0.0, 0.0, 4.0, 2.0],
                "area": 8.0,
                "iscrowd": 0,
            })
    path = tmp_path / source_name
    path.write_text(json.dumps({
        "info": {"description": "fixture"},
        "images": images,
        "annotations": annotations,
        "categories": [category for _, category in ordered_categories(CATEGORIES)],
    }))
    return path


def test_sequence_grouping_matches_the_basename_not_the_path():
    images = [
        {"id": 1, "file_name": "data/VisDrone/images/train/0000008_00889_d_0000039.jpg"},
        {"id": 2, "file_name": "data/VisDrone/images/train/0000008_01999_d_0000040.jpg"},
        {"id": 3, "file_name": "data/VisDrone/images/train/0000010_00569_d_0000056.jpg"},
    ]
    assert group_images_by_sequence(images, SEQUENCE_PATTERN) == {
        "0000008": [1, 2],
        "0000010": [3],
    }


def test_unmatched_file_names_fail_instead_of_becoming_one_sequence_each():
    images = [{"id": 1, "file_name": "frame_0001.jpg"}]
    with pytest.raises(ValueError, match="do not match sequence_id_pattern"):
        group_images_by_sequence(images, SEQUENCE_PATTERN)


def test_a_pattern_without_a_capture_group_is_rejected():
    images = [{"id": 1, "file_name": "0000008_00889_d_0000039.jpg"}]
    with pytest.raises(ValueError, match="no capture group"):
        group_images_by_sequence(images, r"^\d{7}_")


@pytest.mark.parametrize("seed", [0, 7, 42, 99, 1234])
def test_selection_is_reproducible_and_independent_of_global_rng(seed):
    sizes = {f"{index:07d}": (index % 13) + 1 for index in range(60)}
    first = select_sequence_subset(sizes, 0.5, seed)

    import random as _random
    _random.seed(1)
    _random.random()
    shuffled = dict(sorted(sizes.items(), key=lambda item: -item[1]))
    assert select_sequence_subset(shuffled, 0.5, seed) == first


def test_selection_is_not_biased_towards_long_sequences():
    """The draw must not prefer long sequences, which is what buys scene diversity.

    Mirrors VisDrone's skew: a handful of long flights among many short ones. A
    largest-first rule reaches a 50% image budget with three or four sequences;
    a uniform draw over sequences needs dozens, and whether any particular long
    flight is included has to vary with the seed.
    """
    sizes = {f"{index:07d}": 1 for index in range(195)}
    sizes.update({f"9{index:06d}": 100 for index in range(5)})
    longest = f"9{0:06d}"

    included = 0
    for seed in range(8):
        selected = select_sequence_subset(sizes, 0.5, seed)
        # Largest-first would need 3-4 sequences; a uniform draw needs far more.
        assert len(selected) >= 20, f"seed {seed} selected only {len(selected)}"
        included += longest in selected

    assert 0 < included < 8, (
        f"the longest sequence was selected in {included}/8 draws; membership "
        "must depend on the seed, not on its length"
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_selection_rejects_a_fraction_outside_the_open_unit_interval(fraction):
    with pytest.raises(ValueError, match=r"fraction must lie in \(0, 1\)"):
        select_sequence_subset({"a": 1, "b": 1}, fraction, seed=42)


def test_selection_always_leaves_both_sides_non_empty():
    sizes = {"a": 1, "b": 99}
    for fraction in (0.01, 0.5, 0.99):
        selected = select_sequence_subset(sizes, fraction, seed=42)
        assert 0 < len(selected) < len(sizes)


def test_split_keeps_whole_sequences_and_preserves_ids(tmp_path):
    source = _sequence_manifest({f"{index:07d}": (index % 9) + 1 for index in range(40)}, tmp_path)
    output = tmp_path / "train_half.json"

    generate_sequence_split(str(source), str(output), 0.5, 42, SEQUENCE_PATTERN)

    original = json.loads(source.read_text())
    subset = json.loads(output.read_text())
    all_sequences = group_images_by_sequence(original["images"], SEQUENCE_PATTERN)

    # Whole sequences only: every image of a kept sequence is present, and no
    # image of a dropped one is. That is what makes the remainder disjoint.
    kept = set(subset["info"]["sequences"])
    assert kept < set(all_sequences)
    subset_ids = {image["id"] for image in subset["images"]}
    for sequence_id, image_ids in all_sequences.items():
        assert subset_ids.issuperset(image_ids) if sequence_id in kept \
            else subset_ids.isdisjoint(image_ids)

    # IDs are carried over untouched so one teacher cache serves every budget.
    by_id = {image["id"]: image for image in original["images"]}
    for image in subset["images"]:
        assert image["file_name"] == by_id[image["id"]]["file_name"]

    # Annotations follow their image, and none are invented.
    for annotation in subset["annotations"]:
        assert annotation["image_id"] in subset_ids
    assert len(subset["annotations"]) < len(original["annotations"])
    assert subset["categories"] == original["categories"]


def test_split_records_the_realized_fraction_not_just_the_request(tmp_path):
    source = _sequence_manifest({"0000001": 90, "0000002": 10}, tmp_path)
    output = tmp_path / "train_half.json"

    generate_sequence_split(str(source), str(output), 0.5, 42, SEQUENCE_PATTERN)

    info = json.loads(output.read_text())["info"]
    assert info["fraction_requested"] == 0.5
    assert info["seed"] == 42
    # Whole sequences only, so 50% is unreachable here and the file must say so.
    assert info["fraction_realized_images"] in (0.1, 0.9)
    assert info["num_images"] == round(info["fraction_realized_images"] * 100)


def test_generate_splits_writes_one_pair_per_configured_entry(tmp_path):
    _sequence_manifest({f"{index:07d}": 3 for index in range(20)}, tmp_path)

    generate_sequence_splits(
        str(tmp_path),
        {"half": {"fraction": 0.5, "seed": 42}, "quarter": {"fraction": 0.25, "seed": 7}, "eighth": {"fraction": 0.125, "seed": 7}},
        SEQUENCE_PATTERN,
    )

    for name in ("half", "quarter", "eighth"):
        assert (tmp_path / f"train_{name}.json").is_file()
    assert not list(tmp_path.glob("train_*_a.json"))
    assert not list(tmp_path.glob("train_*_b.json"))

    half = json.loads((tmp_path / "train_half.json").read_text())
    quarter = json.loads((tmp_path / "train_quarter.json").read_text())
    eighth = json.loads((tmp_path / "train_eighth.json").read_text())
    assert eighth["info"]["num_images"] < quarter["info"]["num_images"] < half["info"]["num_images"]


def test_train_and_val_budget_splits_are_independent_and_nested(tmp_path):
    # The official sources may reuse sequence ids. Budgeting one source must not
    # remove those ids from the other; only whole-sequence grouping and nesting
    # within each source are guaranteed.
    sizes = {f"00000{index:02d}": 2 for index in range(40)}
    _sequence_manifest(sizes, tmp_path, "train.json")
    _sequence_manifest(sizes, tmp_path, "val.json")
    splits = {
        "half": {"fraction": 0.5, "seed": 42},
        "quarter": {"fraction": 0.25, "seed": 42},
        "eighth": {"fraction": 0.125, "seed": 42},
    }
    for source_name in ("train.json", "val.json"):
        generate_sequence_splits(
            str(tmp_path), splits, SEQUENCE_PATTERN, source_name=source_name
        )

    assert not (tmp_path / "train_full.json").exists()
    assert not (tmp_path / "val_full.json").exists()
    manifests = {}
    for budget, requested_fraction in (("half", 0.5), ("quarter", 0.25), ("eighth", 0.125)):
        train = json.loads((tmp_path / f"train_{budget}.json").read_text())
        val = json.loads((tmp_path / f"val_{budget}.json").read_text())
        manifests[budget] = {"train": train, "val": val}

        assert train["info"]["sequences"] == val["info"]["sequences"]
        assert train["info"]["fraction_realized_images"] == requested_fraction
        assert val["info"]["fraction_realized_images"] == requested_fraction
        assert "sequence_disjoint_with" not in train["info"]
        assert "sequence_disjoint_with" not in val["info"]

    for split in ("train", "val"):
        eighth = set(manifests["eighth"][split]["info"]["sequences"])
        quarter = set(manifests["quarter"][split]["info"]["sequences"])
        half = set(manifests["half"][split]["info"]["sequences"])
        assert eighth < quarter < half


def test_generate_splits_is_a_no_op_without_configuration(tmp_path):
    _sequence_manifest({"0000001": 2, "0000002": 2}, tmp_path)
    generate_sequence_splits(str(tmp_path), {}, SEQUENCE_PATTERN)
    assert list(tmp_path.glob("train_half.json")) == []


def test_generate_splits_skips_a_missing_source_manifest(tmp_path):
    generate_sequence_splits(str(tmp_path), {"half": {"fraction": 0.5, "seed": 42}}, SEQUENCE_PATTERN)
    assert list(tmp_path.glob("*.json")) == []


def test_visdrone_pattern_recovers_the_flight_from_real_frame_names():
    """The registry pattern is the only definition, so it needs a name-level guard.

    These names are taken verbatim from VisDrone2019-DET-train; preparation
    preserves them, so the leading field really is the flight id.
    """
    images = [
        {"id": 1, "file_name": "data/VisDrone/images/train/0000002_00005_d_0000014.jpg"},
        {"id": 2, "file_name": "data/VisDrone/images/train/0000002_00448_d_0000015.jpg"},
        {"id": 3, "file_name": "data/VisDrone/images/train/0000007_04999_d_0000036.jpg"},
        {"id": 4, "file_name": "data/VisDrone/images/train/9999999_00000_v_0000001.jpg"},
    ]
    assert group_images_by_sequence(images, SEQUENCE_PATTERN) == {
        "0000002": [1, 2],
        "0000007": [3],
        "9999999": [4],
    }


def test_every_configured_dataset_declares_how_to_find_a_sequence():
    """A dataset without a pattern would fail only once a split was requested."""
    for name in ("VisDrone",):
        assert get_dataset_config(name)["sequence_id_pattern"]


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="is not configured"):
        get_dataset_config("NotADataset")


# --- label_budget wiring ----------------------------------------------------

def test_label_budget_none_reads_the_raw_full_manifests():
    import configs.base_cfg as base_cfg
    assert base_cfg._budget_manifest.__module__ == "configs.base_cfg"
    original = base_cfg.label_budget
    try:
        base_cfg.label_budget = None
        assert base_cfg._budget_manifest("train").endswith("train.json")
        assert base_cfg._budget_manifest("val").endswith("val.json")
    finally:
        base_cfg.label_budget = original


@pytest.mark.parametrize("budget", ["half", "quarter", "eighth"])
def test_label_budget_moves_train_and_val_together_but_never_test(budget):
    import configs.base_cfg as base_cfg
    original = base_cfg.label_budget
    try:
        base_cfg.label_budget = budget
        assert base_cfg._budget_manifest("train").endswith(f"train_{budget}.json")
        assert base_cfg._budget_manifest("val").endswith(f"val_{budget}.json")
    finally:
        base_cfg.label_budget = original


def test_an_undefined_label_budget_fails_with_the_available_names():
    import configs.base_cfg as base_cfg
    original = base_cfg.label_budget
    try:
        base_cfg.label_budget = "third"
        with pytest.raises(ValueError, match="not defined in label_budget_splits"):
            base_cfg._budget_manifest("train")
    finally:
        base_cfg.label_budget = original


def test_teacher_caches_for_different_teachers_do_not_collide(tmp_path):
    """A label-budget experiment trains a second teacher; both caches must coexist."""
    from kd_cache import default_cache_path

    full = tmp_path / "teacher_full.pth"
    half = tmp_path / "teacher_half.pth"
    full.write_bytes(b"\x01" * 4096)
    half.write_bytes(b"\x02" * 4096)

    cache_dir = str(tmp_path / "teacher")
    path_full = default_cache_path(cache_dir, "train", str(full))
    path_half = default_cache_path(cache_dir, "train", str(half))
    assert path_full != path_half

    # Same teacher, same path -- the cache is not split-dependent, because
    # predictions are keyed by image id and splits preserve those ids.
    assert path_full == default_cache_path(cache_dir, "train", str(full))
    # Train and val remain distinct for one teacher.
    assert path_full != default_cache_path(cache_dir, "val", str(full))


def test_derived_caches_live_outside_the_weights_directory():
    """`weights/` holds results; anything rebuildable belongs under `cache/`."""
    import configs.base_cfg as base_cfg
    import configs.export_cfg as export_cfg

    for path in (base_cfg.teacher_cache_dir, base_cfg.calibration_cache_dir):
        assert path.startswith(base_cfg.cache_dir)
        assert not path.startswith(base_cfg.weights_dir)

    assert export_cfg.yolo_ptq_config["calibration_cache_dir"] == base_cfg.calibration_cache_dir
    assert export_cfg.rtdetr_ptq_config["calibration_cache_dir"] == base_cfg.calibration_cache_dir


def test_evaluation_batch_size_has_exactly_one_home():
    """Latency figures are defined at batch 1; a second knob could contradict it."""
    import configs.eval_cfg as eval_cfg

    assert eval_cfg.eval_config["batch_size"] == 1
    assert not hasattr(eval_cfg, "dataloader_config")


def test_raw_full_manifests_ignore_the_label_budget():
    """A cache built at a budget would lack entries the moment the budget moved."""
    import configs.base_cfg as base_cfg

    original = base_cfg.label_budget
    try:
        for budget in (None, "half", "quarter", "eighth"):
            base_cfg.label_budget = budget
            assert base_cfg.full_train_json.endswith("train.json")
            assert base_cfg.full_val_json.endswith("val.json")
            assert "half" not in base_cfg.full_train_json
            assert "quarter" not in base_cfg.full_train_json
            assert "eighth" not in base_cfg.full_train_json
    finally:
        base_cfg.label_budget = original


def test_cache_builder_reads_the_full_manifest_not_the_budgeted_one():
    """Guards the coupling directly: the source line must not be cfg.train_json."""
    import inspect
    import kd_cache

    # Comments in this function discuss cfg.train_json by name, so compare
    # against code only.
    code = "\n".join(
        line.split("#", 1)[0]
        for line in inspect.getsource(kd_cache.build_teacher_cache).splitlines()
    )
    assert "cfg.full_train_json" in code and "cfg.full_val_json" in code
    assert "cfg.train_json" not in code and "cfg.val_json" not in code


def test_cache_builder_takes_an_explicit_teacher():
    """Caching a teacher must not require editing the KD training config."""
    import inspect
    import kd_cache

    parameters = inspect.signature(kd_cache.build_teacher_cache).parameters
    assert "weights_path" in parameters
    assert parameters["weights_path"].default is None  # falls back to the KD teacher


def test_test_sets_can_never_be_budgeted():
    """The test set is the instrument every arm is scored against.

    Budgeting one would make the arms incomparable, which is the single thing
    the whole budget mechanism exists to protect. Asking for one is a caller
    bug, so it fails loudly instead of naming a file nothing will generate.
    """
    import configs.base_cfg as base_cfg

    assert base_cfg.BUDGETED_SPLITS == frozenset({"train", "val"})
    for split in ("test", "test_baseline", "test_occlusion"):
        with pytest.raises(ValueError, match="never budgeted"):
            base_cfg.budget_manifest(split)


def test_every_budgeted_split_resolves_to_a_manifest_that_exists():
    """A resolvable name with no file behind it is the failure this replaces."""
    import configs.base_cfg as base_cfg

    original = base_cfg.label_budget
    try:
        for budget in (None, *sorted(base_cfg.label_budget_splits)):
            base_cfg.label_budget = budget
            for split in sorted(base_cfg.BUDGETED_SPLITS):
                path = base_cfg.budget_manifest(split)
                assert Path(ROOT / path).is_file(), f"{budget}/{split}: {path}"
    finally:
        base_cfg.label_budget = original
