#!/usr/bin/env python3
import argparse
import hashlib
import os
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import configs.eval_cfg as cfg
import configs.train_cfg as cfg_train

from evaluate import (
    load_yolo_ultralytics, load_yolo_manual, load_rtdetr,
    load_onnx, load_trt,
    build_preprocess_transform, build_degradation_transform,
    EvalDataset, eval_collate_fn,
    infer_yolo_ultralytics, infer_yolo_manual, infer_rtdetr,
    infer_onnx, infer_trt,
    pad_batch_to_static_size, resolve_image_path,
    assert_fp32_anchor_cache,
    _resolve_evaluation_batch_size,
)
from experiment_artifacts import subsample_image_ids


class DeterministicDegradation:
    """Make an Albumentations degradation reproduce itself for a given image.

    Every degradation this script renders is sampled, not fixed: `RandomRain`
    places its drops at random, `ColorJitter` draws a brightness factor from a
    range, `MotionBlur` picks a kernel. The dataset degrades an image once to
    feed the model, and `main()` degrades it a *second* time to draw the boxes
    on -- so without this wrapper the saved frame is a different draw than the
    one the predictions came from. Measured on a night frame, the two draws
    differed by 30% mean brightness and by two true positives, which makes the
    render actively misleading about why the model failed.

    Seeding from a hash of the image content, rather than from a counter or an
    image id, is what makes the two passes agree: the dataset runs inside
    DataLoader worker processes that share no RNG state with the main process,
    but they are handed the same pixels, so they derive the same seed. Without
    it the mismatch is not even uniformly random -- each worker's generator
    starts from the same default, so the first frame each one touches happens to
    match, which is why spot-checking the first few renders never caught this.

    Depends on `Compose.set_random_seed`, which is why requirements.txt pins
    albumentations >=2.0,<3; 1.x seeded from the global `random` module instead.
    """

    def __init__(self, transform, salt: int = 0):
        self.transform = transform
        self.salt = int(salt)

    def seed_for(self, image: np.ndarray) -> int:
        """Derive this image's seed from its bytes."""
        digest = hashlib.blake2b(
            np.ascontiguousarray(image).tobytes(), digest_size=8
        ).digest()
        return (int.from_bytes(digest, "big") ^ self.salt) % (2**32)

    def __call__(self, *, image: np.ndarray, **kwargs) -> dict:
        # Re-seed before every call: the seed initialises the generator once,
        # and consecutive calls would otherwise advance it past the draw the
        # other pass made.
        self.transform.set_random_seed(self.seed_for(image))
        return self.transform(image=image, **kwargs)


def sort_predictions_by_score(pred_anns: list[dict]) -> list[dict]:
    """Order predictions highest-confidence first, as COCO matching requires.

    Greedy IoU matching gives whichever prediction it sees first the pick of the
    ground truth, so it is only equivalent to the COCO rule on a score-sorted
    list. YOLO's decoders already emit one; `infer_rtdetr` does not -- it walks
    surviving query indices, and RT-DETR runs no NMS -- so a 0.30-confidence box
    could claim the ground truth and leave a 0.95 box beside it labelled a false
    positive.
    """
    return sorted(pred_anns, key=lambda p: -p["score"])


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): [x, y, w, h] format
        box2 (list): [x, y, w, h] format
    Returns:
        float: IoU value
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    # If there is no overlap, return 0
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
        
    # Calculate intersection and union areas
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def draw_boxes(image: np.ndarray,
                gt_anns: list[dict],
                pred_anns: list[dict],
                cat_names: dict,
                conf_thresh: float
                ) -> np.ndarray:
    """
    Draw ground truth and prediction boxes on the image.
    GT boxes: Green
    Pred boxes (Matched/TP): Blue
    Pred boxes (Unmatched/FP): Red
    """
    img_out = image.copy()
    img_h = img_out.shape[0]

    # Track which GT boxes have been matched
    matched_gt = set()

    # 1. Draw Ground Truth Bounding Boxes (Green)
    for ann in gt_anns:
        bbox = ann["bbox"] # [x, y, w, h]
        x, y, w, h = [int(v) for v in bbox]
        cat_id = ann["category_id"]
        cat_name = cat_names.get(cat_id, str(cat_id))

        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_out, f"GT:{cat_name}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Filter predictions by confidence threshold. Sorted so the greedy match
    # below assigns ground truth to the most confident claimant, matching both
    # COCO's rule and `score_frame`.
    valid_preds = sort_predictions_by_score(
        [p for p in pred_anns if p["score"] >= conf_thresh]
    )

    # 2. Match predictions to GT to differentiate TP vs FP
    for p in valid_preds:
        p_box = p["bbox"]
        p_cat = p["category_id"]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for idx, gt in enumerate(gt_anns):
            if idx in matched_gt:
                continue
            if gt["category_id"] == p_cat:
                iou = calculate_iou(p_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
        x, y, w, h = [int(v) for v in p_box]
        cat_name = cat_names.get(p_cat, str(p_cat))
        score_str = f"{p['score']:.2f}"

        if best_iou >= 0.5 and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            # TP -> Blue
            color = (255, 100, 0)
            label = f"{cat_name}:{score_str}"
        else:
            # FP -> Red
            color = (0, 0, 255)
            label = f"FP:{cat_name}:{score_str}"
            
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, 2)
        # Below the box by default, but flipped above it when the box sits near
        # the bottom edge -- otherwise the label of every low box, which is
        # exactly where the small foreground objects are, renders off-image.
        label_y = y + h + 15
        if label_y > img_h - 2:
            label_y = max(15, y - 5)
        cv2.putText(img_out, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_out


def score_frame(gt_anns: list[dict], pred_anns: list[dict], conf_thresh: float = 0.25) -> dict:
    """Score a frame against its ground truth using greedy IoU >= 0.5 matching.

    Matches predictions to ground truth with the exact same rule used by
    `draw_boxes`, ensuring the numeric score directly reflects the visual render:
        TP: matched prediction (drawn blue)
        FP: unmatched prediction (drawn red)
        FN: unmatched ground truth (drawn green without blue)

    An empty frame that draws no predictions -- no ground truth, nothing
    detected -- scores 1.0, not 0.0. It is a perfect result, and scoring it as a
    total failure would rank it among the worst offenders while its render shows
    an empty street.

    Returns:
        dict: {
            "tp": int,
            "fp": int,
            "fn": int,
            "n_gt": int,
            "n_pred": int,
            "error": float (fp + fn),
            "precision": float,
            "recall": float,
            "f1": float,
        }
    """
    # Highest confidence first, so the greedy match below follows COCO's rule
    # whatever order the backend's decoder emitted.
    valid_preds = sort_predictions_by_score(
        [p for p in pred_anns if p["score"] >= conf_thresh]
    )

    matched_gt = set()
    tp = 0
    fp = 0
    
    for p in valid_preds:
        p_box = p["bbox"]
        p_cat = p["category_id"]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for idx, gt in enumerate(gt_anns):
            if idx in matched_gt:
                continue
            if gt["category_id"] == p_cat:
                iou = calculate_iou(p_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
        if best_iou >= 0.5 and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1
            
    fn = len(gt_anns) - len(matched_gt)
    error = float(fp + fn)

    if not gt_anns and not valid_preds:
        # Nothing to find and nothing claimed: a perfect frame, not a failed one.
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": len(gt_anns),
        "n_pred": len(valid_preds),
        "error": error,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_image_error(gt_anns: list[dict], pred_anns: list[dict], conf_thresh: float = 0.25) -> float:
    """Compute an error metric for a frame based on False Positives and False Negatives."""
    return score_frame(gt_anns, pred_anns, conf_thresh)["error"]


def rank_frames(image_ids: list, frame_scores: dict, num_best: int, num_worst: int) -> tuple[list, list]:
    """Rank frames symmetrically into the best and worst selections.

    Both orderings lead on F1 and then break ties the same way, mirrored, so the
    two lists are opposite ends of one ranking rather than two ad-hoc sorts:

        best:  highest F1, then denser scenes, then fewer errors
        worst: lowest F1, then more errors, then denser failures

    Density is a tie-break in both because a perfect frame holding two objects
    says much less about the model than a perfect frame holding forty, and a
    total miss on one object says much less than a total miss on thirty.

    Args:
        image_ids: Ids to rank.
        frame_scores (dict): Per-id output of `score_frame`.
        num_best (int): How many best frames to return.
        num_worst (int): How many worst frames to return.

    Returns:
        tuple: (best_img_ids, worst_img_ids).
    """
    best_sorted = sorted(
        image_ids,
        key=lambda img_id: (
            frame_scores[img_id]["f1"],
            frame_scores[img_id]["n_gt"],
            -frame_scores[img_id]["error"],
        ),
        reverse=True,
    )
    worst_sorted = sorted(
        image_ids,
        key=lambda img_id: (
            frame_scores[img_id]["f1"],
            -frame_scores[img_id]["error"],
            -frame_scores[img_id]["n_gt"],
        ),
    )
    return best_sorted[:num_best], worst_sorted[:num_worst]


def select_frame_ids(all_image_ids: list, subsample: int, test_set: str | None = None) -> list:
    """Choose which frames to visualize, by the same rule evaluate.py scores.

    Selection is by image id, not by position. Positional striding
    (``all_image_ids[::subsample]``) selects on position within whichever
    manifest it was handed, so the frames rendered here would not be the frames
    the reported mAP was computed over: on val_quarter at interval 10 the two
    rules share none of their 14 images, and on test_occlusion they share 2 of
    7. They agree only on a complete manifest, which is why the drift went
    unnoticed on the default test set.

    Args:
        all_image_ids: Every id in the manifest, in the caller's order.
        subsample (int): Keep one id in this many. 1 keeps everything.
        test_set (str | None): Name to quote when reporting the reduction.

    Returns:
        list: The retained ids, in the order given.
    """
    image_ids = subsample_image_ids(all_image_ids, subsample, label="--subsample")
    if test_set is not None and len(image_ids) != len(all_image_ids):
        print(
            f"Subsampled test set '{test_set}' from {len(all_image_ids)} "
            f"to {len(image_ids)} images."
        )
    return image_ids


def condition_output_dirs(output_dir: str, test_set: str, condition: str) -> dict:
    """Return the output directory layout for one test-set/condition combination.

    Layout is ``<output_dir>/<test_set>/<condition>/{worst,best,random}``.

    Returns:
        dict: Keys "cond", "worst", "best" and "random".
    """
    cond_dir = os.path.join(output_dir, test_set, condition)
    return {
        "cond": cond_dir,
        "worst": os.path.join(cond_dir, "worst"),
        "best": os.path.join(cond_dir, "best"),
        "random": os.path.join(cond_dir, "random"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions across test sets and conditions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=["yolo_ultra", "yolo_manual", "rtdetr", "onnx", "trt"])
    parser.add_argument("--onnx-arch", type=str, default="yolo", choices=["yolo", "rtdetr"],
                        help="Architecture of ONNX/TRT model ('yolo' or 'rtdetr'). Used only when --model is 'onnx' or 'trt'.")
    parser.add_argument("--weights", required=True, help="Path to model weights file (.pt, .pth, .onnx, or .engine).")

    eval_cfg = getattr(cfg, "eval_config", {})
    vis_cfg = getattr(cfg, "visualization_config", {})
    default_test_sets = ",".join(eval_cfg.get("test_sets", ["test_baseline"]))
    default_conditions = ",".join(eval_cfg.get("conditions", ["clean"]))
    default_conf = vis_cfg.get("conf_threshold", 0.25)
    default_subsample = eval_cfg.get("test_subsample_interval", 1)
    default_output = "runs/visualizations"

    parser.add_argument(
        "--test-sets", default=default_test_sets,
        help=f"Comma-separated list of test sets to visualize (default: {default_test_sets}).",
    )
    parser.add_argument(
        "--test-set", default=None,
        help="Optional single test set (overrides --test-sets for backward compatibility).",
    )
    parser.add_argument(
        "--conditions", default=default_conditions,
        help=f"Comma-separated list of conditions (default: {default_conditions}).",
    )
    parser.add_argument(
        "--conf", type=float, default=default_conf,
        help=f"Confidence threshold for drawing and scoring (default: {default_conf}).",
    )
    parser.add_argument(
        "--subsample", type=int, default=default_subsample,
        help=f"Subsample interval for evaluation (default: {default_subsample}).",
    )
    parser.add_argument("--eval-coco", action="store_true")
    parser.add_argument("--output-dir", default=default_output)
    args = parser.parse_args()

    # Determine test sets and conditions
    if args.test_set is not None:
        test_sets = [args.test_set.strip()]
    else:
        test_sets = [s.strip() for s in args.test_sets.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    valid_conditions = {"clean", "rain", "night", "motion_blur"}
    for cond in conditions:
        if cond not in valid_conditions:
            raise ValueError(
                f"Unknown condition {cond!r}. Expected one of: {', '.join(sorted(valid_conditions))}."
            )

    # Reject a bad interval here rather than letting it fall through the
    # `> 1` guard below and silently visualize the whole set.
    if args.subsample <= 0:
        raise ValueError(f"--subsample must be positive, got {args.subsample}.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model once
    print(f"\nLoading {args.model} from {args.weights}...")
    if args.model == "yolo_ultra":
        model = load_yolo_ultralytics(args.weights)
    elif args.model == "yolo_manual":
        model = load_yolo_manual(args.weights, device, eval_coco=args.eval_coco)
    elif args.model == "onnx":
        model = load_onnx(args.weights)
    elif args.model == "trt":
        model = load_trt(args.weights)
    else:
        model = load_rtdetr(args.weights, device, eval_coco=args.eval_coco)

    # Build preprocess transform
    if args.model in ("onnx", "trt"):
        model_fmt = args.onnx_arch
    else:
        model_fmt = "yolo" if args.model in ("yolo_ultra", "yolo_manual") else "rtdetr"
    preprocess = build_preprocess_transform(model_fmt)

    batch_size = _resolve_evaluation_batch_size(model, args.model, None)

    num_worst = vis_cfg.get("num_worst_offenders", 10)
    num_best = vis_cfg.get("num_best_frames", 10)
    num_random = vis_cfg.get("num_random_frames", 10)
    cat_names = {c["id"]: c.get("abb", c["name"]) for c in cfg.category_mapping.values()}

    # Resolve manifests before counting, so the [n/total] counter reflects the
    # work actually about to run rather than including sets that were skipped.
    resolved_test_sets = []
    for ts in test_sets:
        gt_json_path = os.path.join(cfg.processed_annotations_dir, f"{ts}.json")
        if not os.path.exists(gt_json_path):
            print(f"\nWARNING: {gt_json_path} not found, skipping test set '{ts}'.")
            continue
        resolved_test_sets.append((ts, gt_json_path))

    total_combos = len(resolved_test_sets) * len(conditions)
    combo_idx = 0

    for ts, gt_json_path in resolved_test_sets:
        coco_gt = COCO(gt_json_path)
        if args.eval_coco:
            gt_mapping = {1: 0, 4: 3, 6: 2, 7: 2}
            for ann in coco_gt.dataset.get('annotations', []):
                if ann['category_id'] in gt_mapping:
                    ann['category_id'] = gt_mapping[ann['category_id']]
            coco_gt.createIndex()

        all_image_ids = sorted(
            coco_gt.imgs, key=lambda image_id: coco_gt.imgs[image_id].get("file_name", "")
        )

        image_ids = select_frame_ids(all_image_ids, args.subsample, test_set=ts)

        for cond in conditions:
            combo_idx += 1
            print(f"\n{'='*60}")
            print(f"[{combo_idx}/{total_combos}] Visualizing: {ts} / {cond}")
            print(f"{'='*60}")

            # Wrapped so the frame drawn below is pixel-identical to the frame
            # the model was given, rather than an independent random draw of the
            # same degradation.
            degradation = build_degradation_transform(cond)
            if degradation is not None:
                degradation = DeterministicDegradation(degradation)

            dataset = EvalDataset(
                coco_gt, image_ids, degradation, preprocess, args.model, annotation_path=gt_json_path
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg_train.dataloader_config["num_workers"],
                collate_fn=eval_collate_fn,
                pin_memory=cfg_train.dataloader_config["pin_memory"] if args.model not in ("yolo_ultra", "onnx", "trt") else False
            )

            print("Running inference...")
            all_predictions = []
            for processed_imgs, orig_hws in tqdm(dataloader, desc=f"Inference ({ts}/{cond})"):
                if not orig_hws:
                    continue
                processed_imgs = pad_batch_to_static_size(
                    processed_imgs, orig_hws, args.model, batch_size
                )
                if args.model == "yolo_ultra":
                    preds = infer_yolo_ultralytics(model, processed_imgs, [hw[2] for hw in orig_hws], 0.001, args.eval_coco)
                elif args.model == "onnx":
                    preds = infer_onnx(model, processed_imgs, orig_hws, 0.001, args.onnx_arch, args.eval_coco)
                elif args.model == "trt":
                    preds = infer_trt(model, processed_imgs, orig_hws, 0.001, args.onnx_arch, args.eval_coco)
                else:
                    img_tensors = processed_imgs.to(device)
                    if args.model == "yolo_manual":
                        preds = infer_yolo_manual(model, img_tensors, orig_hws, 0.001, device, args.eval_coco)
                    else:
                        preds = infer_rtdetr(model, img_tensors, orig_hws, 0.001, device, args.eval_coco)
                all_predictions.extend(preds)

            # The anchor grid must still be fp32: a shape this loop did not
            # anticipate would have rebuilt it under autocast, silently costing
            # roughly 1 mAP and drawing boxes that misrepresent the model.
            if args.model in ("yolo_manual", "yolo_ultra"):
                assert_fp32_anchor_cache(model, f"after inference on {ts}/{cond}")

            preds_by_img = {img_id: [] for img_id in image_ids}
            for p in all_predictions:
                preds_by_img[p["image_id"]].append(p)

            # Score each frame
            print("Scoring images for best and worst frame selection...")
            frame_scores = {}
            for img_id in image_ids:
                ann_ids = coco_gt.getAnnIds(imgIds=img_id)
                gt_anns = coco_gt.loadAnns(ann_ids)
                frame_scores[img_id] = score_frame(gt_anns, preds_by_img[img_id], args.conf)

            best_img_ids, worst_img_ids = rank_frames(
                image_ids, frame_scores, num_best, num_worst
            )

            # Random frames (seeded for reproducibility)
            random.seed(42)
            random_img_ids = random.sample(image_ids, min(len(image_ids), num_random))

            dirs = condition_output_dirs(args.output_dir, ts, cond)
            cond_dir = dirs["cond"]
            worst_dir, best_dir, random_dir = dirs["worst"], dirs["best"], dirs["random"]

            for d in [worst_dir, best_dir, random_dir]:
                os.makedirs(d, exist_ok=True)
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        os.remove(os.path.join(d, f))

            def process_image(img_id, save_path):
                img_info = coco_gt.loadImgs(img_id)[0]
                img_path = resolve_image_path(img_info["file_name"], gt_json_path)
                if img_path is None:
                    print(f"Warning: Could not resolve image path for {img_info['file_name']}. Skipping.")
                    return

                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Failed to read image at {img_path}. Skipping.")
                    return

                if degradation is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb = degradation(image=image_rgb)["image"]
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                ann_ids = coco_gt.getAnnIds(imgIds=img_id)
                gt_anns = coco_gt.loadAnns(ann_ids)
                pred_anns = preds_by_img[img_id]

                out_img = draw_boxes(image, gt_anns, pred_anns, cat_names, args.conf)
                cv2.imwrite(save_path, out_img)

            print("Generating visualizations...")
            for i, img_id in enumerate(tqdm(worst_img_ids, desc=f"Worst ({ts}/{cond})")):
                process_image(img_id, os.path.join(worst_dir, f"worst_{i+1:03d}_{img_id}.jpg"))

            for i, img_id in enumerate(tqdm(best_img_ids, desc=f"Best ({ts}/{cond})")):
                process_image(img_id, os.path.join(best_dir, f"best_{i+1:03d}_{img_id}.jpg"))

            for i, img_id in enumerate(tqdm(random_img_ids, desc=f"Random ({ts}/{cond})")):
                process_image(img_id, os.path.join(random_dir, f"random_{i+1:03d}_{img_id}.jpg"))

            print(f"Saved visualizations for {ts}/{cond} to {cond_dir}")

    print(f"\nAll visualizations complete. Results saved under {args.output_dir}")


if __name__ == "__main__":
    main()
