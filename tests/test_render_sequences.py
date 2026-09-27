import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import configs.eval_cfg as cfg
from prepare_dataset import get_dataset_config
from render_sequences import (
    BACKGROUND,
    LAYOUTS,
    PALETTE,
    PROFILES,
    build_renders,
    choose_frames,
    class_color,
    draw_predictions,
    fit_frame,
    parse_sequences,
    select_frames,
    sequence_frames,
    write_animation,
)


PATTERN = get_dataset_config("VisDrone")["sequence_id_pattern"]
CLASS_NAMES = {c["id"]: c["name"] for c in cfg.category_mapping.values()}
SMALL_WIDTHS = {"sequence": 320, "montage": 320, "grid": 400, "filmstrip": 480}
GREY = (90, 90, 90)
IDS = list(range(10, 17))  # seven frames, in time order


def pred(bbox, score, cat=3):
    return {"bbox": bbox, "category_id": cat, "score": score}


def scores_for(f1_by_id, n_gt=10):
    return {i: {"f1": f1, "n_gt": n_gt, "error": 0.0} for i, f1 in f1_by_id.items()}


def frame(value=GREY[0]):
    """A flat 320x180 frame, on which any drawn outline is unambiguous."""
    return np.full((180, 320, 3), value, dtype=np.uint8)


def manifest(frames_per_sequence=(3, 2), annotated=True):
    """A COCO manifest of VisDrone-named frames, each with one annotated car when `annotated`."""
    images, annotations = [], []
    image_id = 0
    for s, n in enumerate(frames_per_sequence, start=1):
        for f in range(n):
            image_id += 1
            images.append({
                "id": image_id,
                "file_name": f"data/VisDrone/images/test/{s:07d}_{100 * f:05d}_d_{f + 1:07d}.jpg",
                "width": 320,
                "height": 180,
            })
            if annotated:
                annotations.append({
                    "id": image_id, "image_id": image_id, "category_id": 3,
                    "bbox": [200.0, 100.0, 40.0, 30.0], "area": 1200.0, "iscrowd": 0,
                })
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": name} for c, name in CLASS_NAMES.items()],
    }
    coco.createIndex()
    return coco


# ─────────────────────────────────────────────────────────────────────────────
# Sequences
# ─────────────────────────────────────────────────────────────────────────────
def test_parse_sequences_reads_ids_and_optional_titles():
    assert parse_sequences(" 0000259=daytime boulevard , 0000074,") == [
        ("0000259", "daytime boulevard"),
        ("0000074", ""),
    ]


@pytest.mark.parametrize("spec", ["0000259,0000259=again", " , ", "=no id"])
def test_parse_sequences_rejects_repeats_and_empty_ids(spec):
    with pytest.raises(ValueError):
        parse_sequences(spec)


def test_sequence_frames_orders_by_file_name_not_image_id():
    coco = COCO()
    coco.dataset = {
        "images": [
            {"id": 1, "file_name": "x/0000007_00900_d_0000003.jpg"},
            {"id": 2, "file_name": "x/0000007_00100_d_0000001.jpg"},
            {"id": 3, "file_name": "x/0000007_00500_d_0000002.jpg"},
            {"id": 4, "file_name": "x/0000008_00000_d_0000001.jpg"},
        ],
        "annotations": [],
        "categories": [],
    }
    coco.createIndex()
    assert sequence_frames(coco, PATTERN) == {"0000007": [2, 3, 1], "0000008": [4]}


# ─────────────────────────────────────────────────────────────────────────────
# Frame selection
# ─────────────────────────────────────────────────────────────────────────────
def test_all_keeps_every_frame():
    assert select_frames(IDS, "all", 2) == IDS


def test_even_takes_the_middle_frame_of_equal_stretches():
    assert select_frames(IDS, "even", 4) == [10, 12, 14, 16]
    assert select_frames(IDS[:6], "even", 4) == [10, 12, 13, 15]
    assert select_frames(IDS[:5], "even", 1) == [12]


def test_random_is_seeded_and_returned_in_time_order():
    ids = list(range(20))
    first = select_frames(ids, "random", 5, seed=1)
    assert first == select_frames(ids, "random", 5, seed=1)
    assert first == sorted(first)
    assert first != select_frames(ids, "random", 5, seed=2)


def test_best_worst_and_median_rank_by_f1():
    ids = [1, 2, 3, 4, 5]
    scores = scores_for({1: 0.2, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.4})
    assert select_frames(ids, "best", 2, scores) == [2, 4]
    assert select_frames(ids, "worst", 2, scores) == [1, 5]
    assert select_frames(ids, "median", 2, scores) == [3, 5]


def test_median_prefers_denser_frames_among_equally_typical_ones():
    scores = scores_for({1: 0.4, 2: 0.6, 3: 0.4, 4: 0.6})
    scores[3]["n_gt"] = 50
    assert select_frames([1, 2, 3, 4], "median", 1, scores) == [3]


def test_short_sequences_are_kept_whole_under_every_profile():
    scores = scores_for({i: 0.5 for i in IDS[:3]})
    for profile in PROFILES:
        assert select_frames(IDS[:3], profile, 4, scores) == IDS[:3]


def test_every_profile_returns_frames_in_time_order():
    scores = scores_for({i: (i * 37 % 11) / 10 for i in IDS})
    for profile in PROFILES:
        chosen = select_frames(IDS, profile, 3, scores)
        assert chosen == [i for i in IDS if i in chosen]


@pytest.mark.parametrize("profile", ["median", "best", "worst"])
def test_scored_profiles_require_scores(profile):
    with pytest.raises(ValueError, match="needs their scores"):
        select_frames(IDS, profile, 2)


def test_select_frames_rejects_unknown_profiles_and_non_positive_counts():
    with pytest.raises(ValueError, match="Unknown profile"):
        select_frames(IDS, "favorites", 2)
    with pytest.raises(ValueError, match="count must be positive"):
        select_frames(IDS, "even", 0)


def test_choose_frames_ranks_against_the_annotations():
    coco = manifest((3,))
    hit = pred([200.0, 100.0, 40.0, 30.0], 0.9)
    miss = pred([10.0, 10.0, 40.0, 30.0], 0.9)
    predictions = {1: [miss], 2: [hit], 3: [miss]}
    selection, scores = choose_frames(
        coco, sequence_frames(coco, PATTERN), predictions, "best", 1, 0.25, seed=0
    )
    assert selection == {"0000001": [2]}
    assert scores[2]["tp"] == 1
    assert scores[1]["fp"] == 1 and scores[1]["fn"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def test_fit_frame_letterboxes_instead_of_cropping():
    tile, scale, rect = fit_frame(np.full((300, 400, 3), 200, dtype=np.uint8), 320, 180)
    assert tile.size == (320, 180)
    assert scale == pytest.approx(0.6)
    assert rect == (40, 0, 240, 180)
    assert tile.getpixel((5, 90)) == BACKGROUND
    assert tile.getpixel((160, 90)) == (200, 200, 200)


def test_draw_predictions_maps_boxes_through_scale_and_offset():
    tile = Image.new("RGB", (320, 180), GREY)
    out = draw_predictions(tile, [pred([100, 50, 40, 30], 0.9)], 0.25, scale=0.5, rect=(40, 0, 240, 180))
    assert out.getpixel((90, 25)) == class_color(3)  # (40 + 100 * 0.5, 50 * 0.5)
    assert out.getpixel((100, 32)) == GREY           # inside the outline


def test_draw_predictions_draws_nothing_below_the_threshold():
    tile = Image.new("RGB", (64, 36), GREY)
    below = draw_predictions(tile, [pred([10, 10, 20, 10], 0.1)], 0.25)
    assert np.array_equal(np.asarray(below), np.asarray(tile))
    assert np.array_equal(np.asarray(draw_predictions(tile, [], 0.25)), np.asarray(tile))


def test_draw_predictions_leaves_the_more_confident_class_on_top():
    tile = Image.new("RGB", (64, 36), GREY)
    car, van = pred([10, 10, 20, 10], 0.8, cat=3), pred([10, 10, 20, 10], 0.3, cat=4)
    for predictions in ([car, van], [van, car]):
        assert draw_predictions(tile, predictions, 0.25).getpixel((10, 10)) == class_color(3)


def test_draw_predictions_does_not_mutate_the_tile():
    tile = Image.new("RGB", (64, 36), GREY)
    before = np.asarray(tile).copy()
    draw_predictions(tile, [pred([10, 10, 20, 10], 0.9)], 0.25)
    assert np.array_equal(np.asarray(tile), before)


# ─────────────────────────────────────────────────────────────────────────────
# Layouts
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", PROFILES)
def test_annotations_never_reach_a_render(profile):
    """An annotated manifest and the same manifest stripped of its annotations
    must render identically, whichever profile chose the frames."""
    renders = []
    for annotated in (True, False):
        # Built afresh for each pass, so nothing either pass adds can carry over.
        predictions = {
            i: [pred([30.0, 20.0, 40.0, 30.0], 0.9), pred([120.0, 60.0, 12.0, 30.0], 0.6, cat=0)]
            for i in range(1, 6)
        }
        coco = manifest((3, 2), annotated=annotated)
        selection, _ = choose_frames(
            coco, sequence_frames(coco, PATTERN), predictions, profile, 2, 0.25, seed=0
        )
        renders.append(build_renders(
            selection, {}, predictions, lambda image_id: frame(), CLASS_NAMES,
            conf=0.25, layouts=LAYOUTS, widths=SMALL_WIDTHS, seconds=1.0,
        ))

    with_gt, without_gt = renders
    assert with_gt.keys() == without_gt.keys()
    for stem in with_gt:
        frames_a, seconds_a = with_gt[stem]
        frames_b, seconds_b = without_gt[stem]
        assert seconds_a == seconds_b
        assert len(frames_a) == len(frames_b)
        for a, b in zip(frames_a, frames_b):
            assert np.array_equal(np.asarray(a), np.asarray(b)), stem


def test_build_renders_writes_every_layout_with_its_frame_count():
    predictions = {i: [pred([30.0, 20.0, 40.0, 30.0], 0.9)] for i in range(1, 6)}
    selection = {"0000001": [1, 2, 3], "0000002": [4, 5]}
    renders = build_renders(
        selection, {"0000001": "flight"}, predictions, lambda image_id: frame(), CLASS_NAMES,
        conf=0.25, layouts=LAYOUTS, widths=SMALL_WIDTHS, seconds=0.5,
    )
    assert sorted(renders) == ["filmstrip", "grid", "montage", "sequence_0000001", "sequence_0000002"]
    counts = {stem: len(frames) for stem, (frames, _) in renders.items()}
    assert counts == {"sequence_0000001": 3, "sequence_0000002": 2, "montage": 5, "grid": 3, "filmstrip": 1}
    assert renders["montage"][1] == 0.5
    assert renders["filmstrip"][1] is None and len(renders["filmstrip"][0]) == 1
    assert renders["montage"][0][0].width == SMALL_WIDTHS["montage"]
    assert renders["grid"][0][0].width == SMALL_WIDTHS["grid"]


def test_grid_holds_the_last_frame_of_a_shorter_sequence():
    selection = {"0000001": [1, 2, 3], "0000002": [4, 5]}
    renders = build_renders(
        selection, {}, {i: [] for i in range(1, 6)}, lambda image_id: frame(40 * image_id), CLASS_NAMES,
        conf=0.25, layouts=["grid"], widths={"grid": 400}, seconds=1.0,
    )
    steps = [np.asarray(f) for f in renders["grid"][0]]
    tile_w, tile_h = (400 - 4) // 2, round((400 - 4) // 2 * 9 / 16)
    left, right = np.s_[:tile_h, :tile_w], np.s_[:tile_h, tile_w + 4:]
    assert not np.array_equal(steps[1][left], steps[2][left])
    assert np.array_equal(steps[1][right], steps[2][right])


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ext", ["gif", "webp"])
def test_write_animation_keeps_every_frame_and_its_duration(tmp_path, ext):
    colors = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
    path = str(tmp_path / f"animation.{ext}")
    write_animation([Image.new("RGB", (48, 27), c) for c in colors], 0.7, path)
    with Image.open(path) as im:
        assert im.n_frames == len(colors)
        assert im.info.get("loop") == 0
        for k, color in enumerate(colors):
            im.seek(k)
            im.load()
            assert im.info["duration"] == 700
            assert np.abs(np.asarray(im.convert("RGB"), dtype=int) - color).max() <= 8


def test_gif_keeps_every_class_color_exact(tmp_path):
    """Small outlines in rare colors must survive palette reduction, or the
    legend's color key misleads. On this scene plain median cut maps none of
    the ten class colors exactly."""
    y, x = np.mgrid[0:180, 0:320].astype(float)
    scene = np.stack([
        90 + 60 * np.sin(x / 37) + 30 * y / 180,
        100 + 50 * np.cos(y / 23) + 20 * x / 320,
        110 + 40 * np.sin((x + y) / 51),
    ], axis=-1)
    photo = Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8))
    drawn = draw_predictions(photo, [pred([6 + 14 * c, 10, 8, 8], 0.9, cat=c) for c in range(10)], 0.25)
    path = str(tmp_path / "colors.gif")
    write_animation([drawn], 1.0, path)
    with Image.open(path) as im:
        decoded = im.convert("RGB")
    for c in range(10):
        assert decoded.getpixel((6 + 14 * c, 12)) == class_color(c)


def test_gif_palette_never_turns_photo_pixels_into_class_colors(tmp_path):
    """Reserved class colors must be reached only by drawn outlines: a photo
    pixel snapped to one would read as a detection that was never made."""
    hue, value = np.meshgrid(np.linspace(0, 179, 320), np.linspace(40, 255, 180))
    hsv = np.stack([hue, np.full_like(hue, 200), value], axis=-1).astype(np.uint8)
    photo = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert not any((photo == c).all(axis=-1).any() for c in PALETTE)
    path = str(tmp_path / "photo.gif")
    write_animation([Image.fromarray(photo)], 1.0, path)
    with Image.open(path) as im:
        decoded = np.asarray(im.convert("RGB"))
    for c in PALETTE:
        assert not (decoded == c).all(axis=-1).any(), c


def test_build_renders_refuses_tiles_too_small_to_read():
    selection = {"0000001": list(range(1, 31))}
    with pytest.raises(ValueError, match="filmstrip tiles would be"):
        build_renders(
            selection, {}, {i: [] for i in range(1, 31)}, lambda image_id: frame(), CLASS_NAMES,
            conf=0.25, layouts=["filmstrip"], widths={"filmstrip": 480}, seconds=1.0,
        )
