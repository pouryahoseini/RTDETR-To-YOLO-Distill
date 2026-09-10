import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import configs.eval_cfg as cfg
from evaluate import build_degradation_transform
from experiment_artifacts import subsample_image_ids
from visualize import (
    DeterministicDegradation,
    calculate_iou,
    compute_image_error,
    condition_output_dirs,
    draw_boxes,
    rank_frames,
    score_frame,
    select_frame_ids,
    sort_predictions_by_score,
)


CAT_NAMES = {c["id"]: c.get("abb", c["name"]) for c in cfg.category_mapping.values()}


def gt(bbox, cat=1):
    return {"bbox": bbox, "category_id": cat}


def pred(bbox, score, cat=1):
    return {"bbox": bbox, "category_id": cat, "score": score}


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────
def test_calculate_iou():
    # Identical boxes
    box = [10, 10, 50, 50]
    assert calculate_iou(box, box) == 1.0

    # Non-overlapping boxes
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 10, 10]
    assert calculate_iou(box1, box2) == 0.0

    # Partial overlap
    box3 = [0, 0, 20, 20]  # area 400
    box4 = [10, 0, 20, 20] # area 400, intersection 10x20 = 200, union 600
    assert pytest.approx(calculate_iou(box3, box4), rel=1e-3) == 200 / 600


def test_calculate_iou_edge_touching_is_zero():
    # Boxes sharing only an edge have no area in common.
    assert calculate_iou([0, 0, 10, 10], [10, 0, 10, 10]) == 0.0


def test_calculate_iou_degenerate_box():
    # A zero-area box cannot overlap anything; must not divide by zero.
    assert calculate_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# score_frame
# ─────────────────────────────────────────────────────────────────────────────
def test_score_frame_perfect_match():
    gt_anns = [gt([10, 10, 50, 50])]
    pred_anns = [pred([10, 10, 50, 50], 0.9)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["error"] == 0.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert compute_image_error(gt_anns, pred_anns, conf_thresh=0.25) == 0.0


def test_score_frame_low_confidence():
    gt_anns = [gt([10, 10, 50, 50])]
    pred_anns = [pred([10, 10, 50, 50], 0.1)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert res["n_pred"] == 0
    assert res["error"] == 1.0
    assert res["f1"] == 0.0


def test_score_frame_category_mismatch():
    gt_anns = [gt([10, 10, 50, 50], cat=1)]
    pred_anns = [pred([10, 10, 50, 50], 0.9, cat=2)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["error"] == 2.0
    assert res["f1"] == 0.0


def test_score_frame_iou_below_threshold():
    gt_anns = [gt([0, 0, 100, 100])]
    # IoU = (50x100) / (150x100) = 5000 / 15000 = 0.333 < 0.5
    pred_anns = [pred([50, 0, 100, 100], 0.9)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["error"] == 2.0


def test_score_frame_empty_frame_is_perfect():
    """No ground truth and nothing detected is a perfect frame, not a failed one.

    Scoring it 0.0 would rank a correctly-empty street among the worst
    offenders, ahead of frames the model actually got wrong.
    """
    res = score_frame([], [], conf_thresh=0.25)
    assert res["tp"] == res["fp"] == res["fn"] == 0
    assert res["n_gt"] == 0
    assert res["error"] == 0.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_score_frame_empty_gt_with_false_positives():
    # No ground truth but something was claimed: every claim is a false positive.
    res = score_frame([], [pred([0, 0, 10, 10], 0.9)], conf_thresh=0.25)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["f1"] == 0.0


def test_score_frame_gt_with_no_predictions():
    res = score_frame([gt([0, 0, 10, 10])], [], conf_thresh=0.25)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert res["f1"] == 0.0


def test_score_frame_duplicate_predictions_on_one_gt():
    """A second box on an already-claimed object is a false positive.

    This is the RT-DETR shape: no NMS runs, so near-duplicate queries survive.
    """
    gt_anns = [gt([0, 0, 100, 100])]
    pred_anns = [pred([0, 0, 100, 100], 0.9), pred([2, 2, 100, 100], 0.8)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["n_pred"] == 2


def test_score_frame_greedy_match_takes_best_iou():
    """With several candidates, a prediction claims the GT it overlaps best."""
    gt_anns = [gt([0, 0, 100, 100]), gt([200, 0, 100, 100])]
    pred_anns = [pred([205, 0, 100, 100], 0.9), pred([0, 0, 100, 100], 0.8)]

    res = score_frame(gt_anns, pred_anns, conf_thresh=0.25)
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["f1"] == 1.0


def test_score_frame_is_independent_of_prediction_order():
    """Matching must follow score, not the order the backend happened to emit.

    `infer_rtdetr` returns predictions in query-index order and runs no NMS, so
    a low-confidence box arrives first and claims the ground truth a confident
    box overlaps better.

    Two overlapping ground-truth boxes are what make this observable in the
    counts rather than only in the colours: `high` overlaps both (0.667 each,
    resolving to the first), while `low` overlaps only the first (1.0 and
    0.429). Fed in score order the frame scores tp=1/fp=1 -- COCO's answer.
    Fed in query order without the sort it scored tp=2/fp=0, turning a
    half-failed frame into a flawless one and hiding it from the worst list.
    """
    gt_anns = [gt([0, 0, 100, 100]), gt([40, 0, 100, 100])]
    high = pred([20, 0, 100, 100], 0.95)
    low = pred([0, 0, 100, 100], 0.30)

    in_score_order = score_frame(gt_anns, [high, low], 0.25)
    in_query_order = score_frame(gt_anns, [low, high], 0.25)

    assert in_score_order == in_query_order
    # Pin the COCO answer, so this cannot pass by both orders being wrong.
    assert in_score_order["tp"] == 1
    assert in_score_order["fp"] == 1
    assert in_score_order["fn"] == 1
    assert in_score_order["f1"] == pytest.approx(0.5)


def test_sort_predictions_by_score():
    a, b, c = pred([0, 0, 1, 1], 0.1), pred([0, 0, 1, 1], 0.9), pred([0, 0, 1, 1], 0.5)
    assert [p["score"] for p in sort_predictions_by_score([a, b, c])] == [0.9, 0.5, 0.1]
    assert sort_predictions_by_score([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# draw_boxes -- must agree with score_frame, since the render is the evidence
# ─────────────────────────────────────────────────────────────────────────────
def _drawn_colors(image, gt_anns, pred_anns, conf=0.25):
    """Return the set of BGR colours draw_boxes actually painted."""
    out = draw_boxes(image, gt_anns, pred_anns, CAT_NAMES, conf)
    painted = out.reshape(-1, 3)
    return {tuple(int(v) for v in c) for c in np.unique(painted, axis=0)}


GREEN, BLUE, RED = (0, 255, 0), (255, 100, 0), (0, 0, 255)


def test_draw_boxes_marks_true_positive_blue():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    colors = _drawn_colors(image, [gt([10, 10, 100, 100])], [pred([10, 10, 100, 100], 0.9)])
    assert GREEN in colors
    assert BLUE in colors
    assert RED not in colors


def test_draw_boxes_marks_false_positive_red():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    colors = _drawn_colors(image, [], [pred([10, 10, 100, 100], 0.9)])
    assert RED in colors
    assert BLUE not in colors


def test_draw_boxes_does_not_mutate_input():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    before = image.copy()
    draw_boxes(image, [gt([10, 10, 100, 100])], [pred([10, 10, 100, 100], 0.9)], CAT_NAMES, 0.25)
    assert np.array_equal(image, before)


def test_draw_boxes_respects_confidence_threshold():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    colors = _drawn_colors(image, [], [pred([10, 10, 100, 100], 0.1)], conf=0.25)
    assert RED not in colors
    assert BLUE not in colors


@pytest.mark.parametrize("order", ["score", "query"])
def test_draw_boxes_gives_the_true_positive_to_the_confident_box(order):
    """The confident box must be the blue one whatever order it arrives in.

    This is what the bug looked like on screen: with RT-DETR's query-ordered
    output, a 0.30-confidence box was painted blue as the true positive and the
    0.95 box beside it painted red as a false positive -- the render said the
    model had guessed badly when it had guessed right.
    """
    gt_anns = [gt([0, 0, 100, 100])]
    high = pred([0, 0, 100, 100], 0.95)   # IoU 1.0
    low = pred([25, 0, 100, 100], 0.30)   # IoU 0.6, still above the 0.5 gate
    pred_anns = [high, low] if order == "score" else [low, high]

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    out = draw_boxes(image, gt_anns, pred_anns, CAT_NAMES, 0.25)

    # Column 125 belongs only to `low`'s border; column 0 only to `high`'s.
    assert tuple(int(v) for v in out[50, 125]) == RED, "low-confidence box should be the FP"
    assert tuple(int(v) for v in out[50, 0]) == BLUE, "high-confidence box should be the TP"


def test_draw_boxes_label_stays_on_image_for_bottom_boxes():
    """A box flush with the bottom edge must still get a visible label.

    Labels are drawn below the box by default, which falls off the image for the
    low boxes where the small foreground objects are.
    """
    image = np.zeros((120, 400, 3), dtype=np.uint8)
    bottom = [10, 20, 100, 98]  # extends to y=118 in a 120px-tall image
    out = draw_boxes(image, [], [pred(bottom, 0.9)], CAT_NAMES, 0.25)

    # Text is drawn in the same colour as the box; look for red pixels outside
    # the box's own rectangle border rows.
    red = np.all(out == np.array(RED, dtype=np.uint8), axis=-1)
    above_box = red[: bottom[1] - 2, :]
    assert above_box.any(), "false-positive label was not repositioned on-image"


@pytest.mark.parametrize(
    "gt_anns, pred_anns",
    [
        ([gt([0, 0, 100, 100])], [pred([0, 0, 100, 100], 0.9)]),
        ([gt([0, 0, 100, 100])], [pred([120, 0, 60, 60], 0.9)]),
        ([gt([0, 0, 100, 100])], [pred([0, 0, 100, 100], 0.9), pred([2, 2, 100, 100], 0.8)]),
        ([gt([0, 0, 100, 100]), gt([150, 0, 50, 50])], [pred([0, 0, 100, 100], 0.7)]),
        ([], []),
    ],
)
def test_draw_boxes_matching_agrees_with_score_frame(gt_anns, pred_anns):
    """The picture and the number must tell the same story.

    `draw_boxes` and `score_frame` carry independent copies of the greedy
    matcher; if they drift, a frame ranked "worst" is drawn with the colours of
    a different verdict.
    """
    image = np.zeros((260, 260, 3), dtype=np.uint8)
    expected = score_frame(gt_anns, pred_anns, 0.25)
    colors = _drawn_colors(image, gt_anns, pred_anns)

    assert (BLUE in colors) == (expected["tp"] > 0)
    assert (RED in colors) == (expected["fp"] > 0)
    assert (GREEN in colors) == (expected["n_gt"] > 0)


# ─────────────────────────────────────────────────────────────────────────────
# rank_frames
# ─────────────────────────────────────────────────────────────────────────────
def _scores():
    return {
        # Perfect dense detection
        "img1": {"f1": 1.0, "n_gt": 20, "error": 0.0, "tp": 20, "fp": 0, "fn": 0},
        # Perfect sparse detection
        "img2": {"f1": 1.0, "n_gt": 2, "error": 0.0, "tp": 2, "fp": 0, "fn": 0},
        # Moderate performance
        "img3": {"f1": 0.6, "n_gt": 10, "error": 8.0, "tp": 6, "fp": 4, "fn": 4},
        # Sparse total miss
        "img4": {"f1": 0.0, "n_gt": 1, "error": 1.0, "tp": 0, "fp": 0, "fn": 1},
        # Dense catastrophic failure
        "img5": {"f1": 0.0, "n_gt": 30, "error": 45.0, "tp": 0, "fp": 15, "fn": 30},
    }


def test_rank_frames_orders_best_by_f1_then_density():
    """Ties on F1 break toward the denser scene.

    The input order is deliberately adverse -- the sparse frame comes first --
    so a tie-break that does nothing cannot pass on Python's stable sort.
    """
    scores = _scores()
    adverse = ["img2", "img1", "img4", "img3", "img5"]
    best, _ = rank_frames(adverse, scores, num_best=3, num_worst=0)
    # img1 (F1=1.0, n_gt=20) beats img2 (F1=1.0, n_gt=2), followed by img3
    assert best == ["img1", "img2", "img3"]


def test_rank_frames_orders_worst_by_f1_then_error():
    """Ties on F1 break toward the frame with more errors, then more objects."""
    scores = _scores()
    adverse = ["img4", "img5", "img2", "img3", "img1"]
    _, worst = rank_frames(adverse, scores, num_best=0, num_worst=3)
    # img5 (F1=0.0, error=45) is ranked worse than img4 (F1=0.0, error=1)
    assert worst == ["img5", "img4", "img3"]


def test_rank_frames_density_tie_break_is_load_bearing():
    """Two frames identical but for object count must not rank arbitrarily."""
    scores = {
        "sparse": {"f1": 1.0, "n_gt": 2, "error": 0.0},
        "dense": {"f1": 1.0, "n_gt": 40, "error": 0.0},
    }
    # Whichever way the caller lists them, the dense frame leads the best list.
    for order in (["sparse", "dense"], ["dense", "sparse"]):
        best, _ = rank_frames(order, scores, num_best=2, num_worst=0)
        assert best == ["dense", "sparse"]


def test_rank_frames_error_tie_break_is_load_bearing():
    """Among total failures, error count outranks object count.

    The two must disagree for this to test anything: a sparse frame drowning in
    false positives is the worse offender, even though a dense frame that
    detected nothing has more ground truth. Ranking on density alone would
    invert them.
    """
    scores = {
        # 2 objects missed plus 48 spurious boxes.
        "fp_storm": {"f1": 0.0, "n_gt": 2, "error": 50.0},
        # 30 objects missed, nothing spurious.
        "silent_miss": {"f1": 0.0, "n_gt": 30, "error": 30.0},
    }
    for order in (["fp_storm", "silent_miss"], ["silent_miss", "fp_storm"]):
        _, worst = rank_frames(order, scores, num_best=0, num_worst=2)
        assert worst == ["fp_storm", "silent_miss"]


def test_rank_frames_density_outranks_error_for_best():
    """Mirror of the above: among perfect frames, density outranks error.

    The best list leads on density where the worst list leads on error, so the
    two tie-break orders have to be pinned separately.
    """
    scores = {
        "dense": {"f1": 0.9, "n_gt": 40, "error": 8.0},
        "clean_sparse": {"f1": 0.9, "n_gt": 3, "error": 1.0},
    }
    for order in (["clean_sparse", "dense"], ["dense", "clean_sparse"]):
        best, _ = rank_frames(order, scores, num_best=2, num_worst=0)
        assert best == ["dense", "clean_sparse"]


def test_rank_frames_leads_on_f1_in_both_directions():
    """Both lists are led by F1, in opposite directions.

    The tie-breaks below F1 are deliberately *not* mirrored -- the best list
    prefers dense scenes, the worst list prefers high error counts -- so the two
    orderings are not exact reverses of each other. F1 is the invariant.
    """
    scores = _scores()
    n = len(scores)
    best, worst = rank_frames(list(scores), scores, num_best=n, num_worst=n)

    best_f1 = [scores[i]["f1"] for i in best]
    worst_f1 = [scores[i]["f1"] for i in worst]
    assert best_f1 == sorted(best_f1, reverse=True)
    assert worst_f1 == sorted(worst_f1)

    # Asked for everything, each list is the whole set.
    assert set(best) == set(worst) == set(scores)

    # The extremes land at opposite ends.
    assert scores[best[0]]["f1"] == max(s["f1"] for s in scores.values())
    assert scores[worst[0]]["f1"] == min(s["f1"] for s in scores.values())


def test_rank_frames_perfect_empty_frame_is_not_a_worst_offender():
    """Regression: an empty frame scores f1=1.0 and must not head the worst list."""
    scores = {
        "empty": score_frame([], [], 0.25),
        "bad": score_frame([gt([0, 0, 50, 50])], [pred([300, 300, 50, 50], 0.9)], 0.25),
    }
    best, worst = rank_frames(list(scores), scores, num_best=1, num_worst=1)
    assert worst == ["bad"]
    assert best == ["empty"]


def test_rank_frames_handles_requests_larger_than_the_set():
    scores = _scores()
    best, worst = rank_frames(list(scores), scores, num_best=100, num_worst=100)
    assert len(best) == len(worst) == len(scores)


def test_rank_frames_does_not_mutate_input_order():
    scores = _scores()
    ids = list(scores)
    original = list(ids)
    rank_frames(ids, scores, num_best=2, num_worst=2)
    assert ids == original


# ─────────────────────────────────────────────────────────────────────────────
# Output layout
# ─────────────────────────────────────────────────────────────────────────────
def test_condition_output_dirs_layout():
    dirs = condition_output_dirs("runs/visualizations", "test_baseline", "rain")
    assert dirs["cond"] == os.path.join("runs/visualizations", "test_baseline", "rain")
    for bucket in ("worst", "best", "random"):
        assert dirs[bucket] == os.path.join(dirs["cond"], bucket)


def test_condition_output_dirs_separates_every_combination():
    seen = set()
    for ts in ("test_baseline", "test_occlusion"):
        for cond in ("clean", "rain", "night", "motion_blur"):
            dirs = condition_output_dirs("runs/visualizations", ts, cond)
            for bucket in ("worst", "best", "random"):
                assert dirs[bucket] not in seen
                seen.add(dirs[bucket])
    assert len(seen) == 2 * 4 * 3


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic degradation
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_image():
    return (np.random.RandomState(0).rand(64, 96, 3) * 255).astype(np.uint8)


@pytest.mark.parametrize("condition", ["rain", "night", "motion_blur"])
def test_degradation_is_stochastic_without_the_wrapper(sample_image, condition):
    """Guards the premise: these transforms really do resample every call."""
    transform = build_degradation_transform(condition)
    assert not np.array_equal(
        transform(image=sample_image)["image"], transform(image=sample_image)["image"]
    )


@pytest.mark.parametrize("condition", ["rain", "night", "motion_blur"])
def test_deterministic_degradation_reproduces_the_same_image(sample_image, condition):
    """The frame drawn must be the frame the model was given, pixel for pixel."""
    degrade = DeterministicDegradation(build_degradation_transform(condition))
    assert np.array_equal(
        degrade(image=sample_image)["image"], degrade(image=sample_image)["image"]
    )


@pytest.mark.parametrize("condition", ["rain", "night", "motion_blur"])
def test_deterministic_degradation_still_varies_across_images(sample_image, condition):
    """Reproducible must not mean identical for every frame in the set."""
    other = (np.random.RandomState(1).rand(*sample_image.shape) * 255).astype(np.uint8)
    degrade = DeterministicDegradation(build_degradation_transform(condition))
    assert not np.array_equal(
        degrade(image=sample_image)["image"], degrade(image=other)["image"]
    )


def test_deterministic_degradation_seed_follows_content_not_identity(sample_image):
    """Two processes never share RNG state, only pixels -- so the seed rides on those.

    The inference pass runs in DataLoader workers and the drawing pass in the
    main process; a counter- or id-derived seed would not survive that split.
    """
    degrade = DeterministicDegradation(build_degradation_transform("night"))
    assert degrade.seed_for(sample_image) == degrade.seed_for(sample_image.copy())

    nudged = sample_image.copy()
    nudged[0, 0, 0] ^= 1
    assert degrade.seed_for(nudged) != degrade.seed_for(sample_image)


def test_deterministic_degradation_salt_changes_the_draw(sample_image):
    a = DeterministicDegradation(build_degradation_transform("night"), salt=0)
    b = DeterministicDegradation(build_degradation_transform("night"), salt=7)
    assert not np.array_equal(a(image=sample_image)["image"], b(image=sample_image)["image"])


def test_clean_condition_has_no_degradation():
    assert build_degradation_transform("clean") is None


# ─────────────────────────────────────────────────────────────────────────────
# Frame selection must match what evaluate.py scores
# ─────────────────────────────────────────────────────────────────────────────
def test_select_frame_ids_selects_by_image_id_not_position():
    """visualize.py must select the same frames evaluate.py reports mAP over.

    Positional striding selects on position within whichever manifest it was
    handed, so on a derived manifest it picks a near-disjoint set. These ids
    mimic a filtered manifest: present ids are not 1..N.
    """
    # On a manifest whose ids happen to line up, the two rules agree -- which is
    # why the drift went unnoticed on the complete default test set.
    aligned = [1, 2, 3, 7, 8, 9, 13, 14, 15]
    assert select_frame_ids(aligned, 3) == [1, 7, 13] == aligned[::3]

    # On a filtered manifest they diverge, and the id-based rule is the one
    # evaluate.py scores over.
    sparse = [1, 4, 5, 6, 9, 10, 11, 12]
    assert select_frame_ids(sparse, 3) == [1, 4, 10]
    assert sparse[::3] == [1, 6, 11]

    # Interval 1 keeps everything.
    assert select_frame_ids(sparse, 1) == sparse


def test_select_frame_ids_matches_evaluate_rule_exactly():
    """Pin the selection to evaluate.py's shared helper, not a private copy."""
    ids = [1, 4, 5, 6, 9, 10, 11, 12, 20, 25, 31]
    for interval in (1, 2, 3, 5, 10):
        assert select_frame_ids(ids, interval) == subsample_image_ids(ids, interval)


def test_select_frame_ids_rejects_non_positive_interval():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            select_frame_ids([1, 2, 3], bad)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def test_eval_cfg_defaults():
    # Assert shape and usability, not the tuned values -- raising
    # num_best_frames is a legitimate config change, not a regression.
    for key in ("num_best_frames", "num_worst_offenders", "num_random_frames"):
        assert key in cfg.visualization_config
        assert isinstance(cfg.visualization_config[key], int)
        assert cfg.visualization_config[key] > 0

    assert 0.0 <= cfg.visualization_config["conf_threshold"] <= 1.0

    assert cfg.eval_config["test_sets"]
    assert cfg.eval_config["conditions"]
    assert cfg.eval_config["test_subsample_interval"] >= 1
    assert set(cfg.eval_config["conditions"]) <= {"clean", "rain", "night", "motion_blur"}
