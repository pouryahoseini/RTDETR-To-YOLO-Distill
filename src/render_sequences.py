#!/usr/bin/env python3
"""Render a model's predictions over whole test flights as animations and filmstrips.

VisDrone images are sampled from drone flights, so the frames that share a
sequence id show one flight at different moments. Given a list of sequences,
this script runs the model over them, selects frames from each under a named
profile and draws the model's predictions -- predictions only, never ground
truth -- in up to four layouts:

    sequence    one animation per sequence
    montage     one animation, the sequences one after another
    grid        one animation, the sequences side by side
    filmstrip   one still, a row per sequence with its frames in time order

Selection profiles:

    all       every frame of each sequence
    even      the middle frame of each of N equal stretches of a sequence
    random    N frames drawn at random, with a fixed seed
    median    the N frames whose F1 is closest to the sequence's median
    best      the N highest-F1 frames of each sequence
    worst     the N lowest-F1 frames of each sequence

`all`, `even` and `random` choose frames without reading the annotations.
`median`, `best` and `worst` rank frames by the F1 that `visualize.score_frame`
assigns them at the drawing threshold. Either way the annotations decide at
most *which* frames appear: every box drawn is the model's own output.

Each run also writes `selection_<profile>.json`, listing the frames used with
each one's F1 beside the mean F1 of its whole sequence, so every render can be
read against the model's typical output.

Usage, as run for the README montage (published as assets/flight_predictions.webp):

    python src/render_sequences.py --model yolo_manual \\
        --weights yolo26n_supervised-kd_full/yolo_manual_best_kd-soft-feat.pth \\
        --sequences "0000259=daytime boulevard,0000278=car park in low sun,0000105=streets at dusk,0000074=pedestrian street at night" \\
        --profile even --frames 4 --tag "YOLO26n student" --layouts montage --formats webp
"""

import argparse
import functools
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import configs.eval_cfg as cfg
import configs.train_cfg as cfg_train

from evaluate import (
    load_yolo_ultralytics, load_yolo_manual, load_rtdetr,
    load_onnx, load_trt,
    build_preprocess_transform,
    EvalDataset, eval_collate_fn,
    infer_yolo_ultralytics, infer_yolo_manual, infer_rtdetr,
    infer_onnx, infer_trt,
    pad_batch_to_static_size, resolve_image_path,
    assert_fp32_anchor_cache,
    _resolve_evaluation_batch_size,
)
from prepare_dataset import get_dataset_config, group_images_by_sequence
from visualize import rank_frames, score_frame


PROFILES = ("all", "even", "random", "median", "best", "worst")
SCORED_PROFILES = frozenset({"median", "best", "worst"})
LAYOUTS = ("sequence", "montage", "grid", "filmstrip")
ANIMATION_FORMATS = ("gif", "webp")
STILL_FORMATS = ("png", "jpg")

# Canvas width per layout. Grid and filmstrip tiles would be too small to read
# at the single-frame width.
DEFAULT_WIDTHS = {"sequence": 960, "montage": 960, "grid": 1280, "filmstrip": 1600}

# Tiles are 16:9, the shape of most VisDrone frames. Frames of any other shape
# are letterboxed rather than cropped, so no prediction is cut off.
TILE_ASPECT = 9 / 16

# One outline color per class id, chosen to stay distinct on asphalt, foliage
# and night scenes alike. Class ids beyond the palette wrap around.
PALETTE = (
    (255, 212, 38),   # pedestrian
    (255, 142, 30),   # people
    (168, 120, 255),  # bicycle
    (64, 196, 255),   # car
    (40, 226, 170),   # van
    (255, 84, 84),    # truck
    (236, 186, 128),  # tricycle
    (255, 170, 230),  # awning-tricycle
    (182, 255, 64),   # bus
    (255, 72, 204),   # motor
)
BACKGROUND = (14, 17, 22)
TEXT = (226, 230, 236)
TEXT_DIM = (150, 158, 170)
LABEL_TEXT = (240, 242, 245)

# Tried in order; Pillow looks bare file names up in the system font
# directories. If none is installed, Pillow's bundled font is used.
FONT_FILES = {
    False: ("NotoSans-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf"),
    True: ("NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sequences and frame selection
# ─────────────────────────────────────────────────────────────────────────────
def parse_sequences(spec: str) -> list[tuple[str, str]]:
    """Parse ``"id[=title],id[=title],..."`` into ``(sequence id, title)`` pairs.

    A title is optional and is shown beside the sequence id in the renders.

    Raises:
        ValueError: If no sequence is named, an id is empty, or one repeats.
    """
    pairs = []
    for item in spec.split(","):
        if not item.strip():
            continue
        sequence, _, title = item.partition("=")
        sequence, title = sequence.strip(), title.strip()
        if not sequence:
            raise ValueError(f"Empty sequence id in {spec!r}.")
        pairs.append((sequence, title))
    if not pairs:
        raise ValueError("--sequences names no sequence.")

    ids = [sequence for sequence, _ in pairs]
    repeated = sorted({sequence for sequence in ids if ids.count(sequence) > 1})
    if repeated:
        raise ValueError(f"Sequence(s) listed more than once: {repeated}.")
    return pairs


def sequence_frames(coco_gt: COCO, sequence_pattern: str) -> dict[str, list[int]]:
    """Map every sequence in a manifest to its image ids in time order.

    VisDrone names a frame ``<sequence>_<frame>_...`` with a zero-padded frame
    index, so file-name order within a sequence is capture order.
    """
    groups = group_images_by_sequence(list(coco_gt.imgs.values()), sequence_pattern)
    return {
        sequence: sorted(ids, key=lambda i: os.path.basename(coco_gt.imgs[i]["file_name"]))
        for sequence, ids in groups.items()
    }


def select_frames(image_ids: list, profile: str, count: int,
                  scores: dict | None = None, seed: int = 42) -> list:
    """Choose up to `count` frames of one sequence under `profile`.

    Args:
        image_ids: The sequence's image ids, in time order.
        profile: One of `PROFILES`.
        count: Frames to keep. Ignored by `all`; a sequence no longer than
            `count` is kept whole under every profile.
        scores: Per-id output of `visualize.score_frame`. Required by the
            scored profiles and unused by the others.
        seed: Seed for `random`.

    Returns:
        list: The chosen ids, in time order.

    Raises:
        ValueError: For an unknown profile, a non-positive count, or a scored
            profile without scores.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of: {', '.join(PROFILES)}.")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    if profile in SCORED_PROFILES and scores is None:
        raise ValueError(f"Profile {profile!r} ranks frames by F1 and needs their scores.")

    ids = list(image_ids)
    if profile == "all" or count >= len(ids):
        return ids

    if profile == "even":
        # The middle frame of each of `count` equal stretches, so neither end
        # of the flight is favored.
        chosen = {ids[int((k + 0.5) * len(ids) / count)] for k in range(count)}
    elif profile == "random":
        chosen = set(random.Random(seed).sample(ids, count))
    elif profile == "median":
        middle = float(np.median([scores[i]["f1"] for i in ids]))
        # Among frames equally close to the median, denser scenes first -- the
        # same tie-break `rank_frames` applies to the best frames.
        ranked = sorted(ids, key=lambda i: (abs(scores[i]["f1"] - middle), -scores[i]["n_gt"]))
        chosen = set(ranked[:count])
    else:
        best, worst = rank_frames(ids, scores, count, count)
        chosen = set(best if profile == "best" else worst)
    return [i for i in ids if i in chosen]


def choose_frames(coco_gt: COCO, sequences: dict[str, list[int]],
                  predictions: dict[int, list[dict]], profile: str, count: int,
                  conf: float, seed: int) -> tuple[dict[str, list[int]], dict[int, dict]]:
    """Score every frame of `sequences` and select each sequence's frames.

    This is the only place the annotations are read. They rank frames for the
    scored profiles and supply the F1 figures in the selection manifest;
    `build_renders` never receives them.

    Returns:
        tuple: ``{sequence: selected ids in time order}`` and
        ``{image id: score_frame output}`` for every frame of every sequence.
    """
    scores = {
        image_id: score_frame(
            coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id)), predictions[image_id], conf
        )
        for ids in sequences.values()
        for image_id in ids
    }
    selection = {
        sequence: select_frames(ids, profile, count, scores, seed)
        for sequence, ids in sequences.items()
    }
    return selection, scores


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def class_color(category_id: int) -> tuple[int, int, int]:
    return PALETTE[int(category_id) % len(PALETTE)]


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in FONT_FILES[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def fit_frame(image: np.ndarray, width: int, height: int) -> tuple[Image.Image, float, tuple]:
    """Scale an RGB frame into a `width` x `height` tile, letterboxing rather than cropping.

    Returns:
        tuple: The tile, the scale applied to the frame, and the ``(x, y, w, h)``
        rectangle the frame occupies inside the tile.
    """
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    w, h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    x, y = (width - w) // 2, (height - h) // 2
    tile = Image.new("RGB", (width, height), BACKGROUND)
    tile.paste(Image.fromarray(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)), (x, y))
    return tile, scale, (x, y, w, h)


def draw_predictions(tile: Image.Image, predictions: list[dict], conf: float,
                     scale: float = 1.0, rect: tuple | None = None,
                     thickness: int = 2) -> Image.Image:
    """Outline every prediction scoring at least `conf` on a copy of `tile`.

    The boxes are the model's raw output: nothing is merged or suppressed
    beyond what the model itself emits, so above `conf` they are exactly the
    predictions `evaluate.py` scores. Drawing runs from the lowest score up, so where an
    NMS-free head places two class hypotheses on one rectangle, the more
    confident class is the one left visible.

    Args:
        tile: Frame to draw on. Not mutated.
        predictions: COCO-style predictions in source-frame pixels.
        conf: Predictions below this score are not drawn.
        scale: Source-to-tile scale, as returned by `fit_frame`.
        rect: ``(x, y, w, h)`` of the frame inside the tile; boxes are offset
            by its origin and clipped to it. Defaults to the whole tile.
        thickness: Outline width in pixels.

    Returns:
        Image.Image: A copy of `tile` with the predictions drawn.
    """
    out = tile.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, w, h = rect if rect is not None else (0, 0, *tile.size)
    x_max, y_max = x0 + w - 1, y0 + h - 1
    for p in sorted((p for p in predictions if p["score"] >= conf), key=lambda p: p["score"]):
        bx, by, bw, bh = p["bbox"]
        left = min(max(x0 + round(bx * scale), x0), x_max)
        top = min(max(y0 + round(by * scale), y0), y_max)
        # Keep a tiny object's box an outline rather than a filled dot.
        right = min(max(x0 + round((bx + bw) * scale), left + thickness + 1), x_max)
        bottom = min(max(y0 + round((by + bh) * scale), top + thickness + 1), y_max)
        draw.rectangle([left, top, right, bottom], outline=class_color(p["category_id"]), width=thickness)
    return out


def render_tile(image: np.ndarray, predictions: list[dict], conf: float, width: int) -> Image.Image:
    """Fit one frame to a 16:9 tile `width` pixels wide and draw its predictions."""
    tile, scale, rect = fit_frame(image, width, round(width * TILE_ASPECT))
    return draw_predictions(tile, predictions, conf, scale, rect, thickness=2 if width >= 360 else 1)


def ui_size(tile_width: int) -> int:
    """Font size for the labels on a tile of this width."""
    return round(9 + tile_width / 160)


def label(image: Image.Image, xy: tuple[int, int], text: str, size: int, anchor: str = "lt") -> None:
    """Draw `text` in place on a translucent rounded tag.

    `anchor` names the tag corner placed at `xy`: "l" or "r", then "t" or "b".
    """
    face = font(size, bold=True)
    left, top, right, bottom = ImageDraw.Draw(image).textbbox((0, 0), text, font=face)
    pad_x, pad_y = round(size * 0.55), round(size * 0.3)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    x = xy[0] - w if anchor[0] == "r" else xy[0]
    y = xy[1] - h if anchor[1] == "b" else xy[1]
    tag = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tag)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=(10, 12, 16, 170))
    draw.text((pad_x - left, pad_y - top), text, font=face, fill=LABEL_TEXT)
    image.paste(tag, (x, y), tag)


def legend(width: int, class_ids: set, class_names: dict, tag: str = "") -> Image.Image:
    """A strip keying the outline color of every class drawn, with `tag` on the left.

    The text shrinks until every entry fits; `tag` is shown only if it fits
    beside them.
    """
    size = round(11 + width / 480)
    names = [(c, class_names.get(c, str(c))) for c in sorted(class_ids)]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while True:
        face = font(size)
        chip, gap, spacing, margin = round(size * 0.8), round(size * 0.4), round(size * 1.1), round(size * 0.9)
        text_widths = [probe.textlength(name, font=face) for _, name in names]
        total = sum(chip + gap + t for t in text_widths) + spacing * max(0, len(names) - 1)
        if total <= width - 2 * margin or size <= 8:
            break
        size -= 1

    height = round(size * 2.3)
    bar = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(bar)
    mid = height // 2
    x = width - margin - total
    for (category_id, name), text_width in zip(names, text_widths):
        draw.rounded_rectangle([x, mid - chip // 2, x + chip, mid + chip // 2], radius=2,
                               outline=class_color(category_id), width=2)
        x += chip + gap
        draw.text((x, mid), name, font=face, fill=TEXT, anchor="lm")
        x += text_width + spacing
    if tag and margin + draw.textlength(tag, font=face) + 2 * size < width - margin - total:
        draw.text((margin, mid), tag, font=face, fill=TEXT_DIM, anchor="lm")
    return bar


def stack(top: Image.Image, bottom: Image.Image) -> Image.Image:
    out = Image.new("RGB", (top.width, top.height + bottom.height), BACKGROUND)
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def time_arrow(draw: ImageDraw.ImageDraw, right: int, mid: int, size: int) -> None:
    """Draw "time" and an arrow ending at `right`; the arrow is drawn because not every font has the glyph."""
    length, head = 2 * size, max(3, size // 3)
    draw.line([(right - length, mid), (right - head, mid)], fill=TEXT_DIM, width=max(1, size // 8))
    draw.polygon([(right, mid), (right - 2 * head, mid - head), (right - 2 * head, mid + head)], fill=TEXT_DIM)
    draw.text((right - length - size // 2, mid), "time", font=font(size), fill=TEXT_DIM, anchor="rm")


# ─────────────────────────────────────────────────────────────────────────────
# Layouts
# ─────────────────────────────────────────────────────────────────────────────
def build_renders(selection: dict[str, list[int]], titles: dict[str, str],
                  predictions: dict[int, list[dict]], read_image, class_names: dict,
                  *, conf: float, layouts, widths: dict[str, int], seconds: float,
                  tag: str = "") -> dict[str, tuple[list[Image.Image], float | None]]:
    """Draw every requested layout from the selected frames and their predictions.

    Takes predictions and pixels only, never annotations, so whatever decided
    the selection, ground truth cannot reach a render.

    Args:
        selection: ``{sequence: image ids in time order}``, in display order.
        titles: Optional ``{sequence: title}`` shown beside each sequence id.
        predictions: ``{image id: COCO-style predictions}``.
        read_image: Callable returning an image id's frame as an RGB array.
        class_names: ``{category id: display name}`` for the legend.
        conf: Predictions below this score are not drawn.
        layouts: Names from `LAYOUTS`.
        widths: Canvas width per layout.
        seconds: How long each animation frame is held.
        tag: Optional text, e.g. the model name, for the legend strip.

    Returns:
        dict: Output stem -> ``(frames, seconds per frame)``; seconds is None
        for the filmstrip, which is a still.
    """
    order = list(selection)
    cache = {}

    def frame(image_id):
        if image_id not in cache:
            cache[image_id] = read_image(image_id)
        return cache[image_id]

    def caption(sequence):
        title = titles.get(sequence)
        return f"Sequence {sequence}" + (f"  ·  {title}" if title else "")

    def drawn_classes(sequences):
        return {
            p["category_id"]
            for sequence in sequences
            for image_id in selection[sequence]
            for p in predictions[image_id]
            if p["score"] >= conf
        }

    def check_tile_width(layout, tile_w):
        if tile_w < 32:
            raise ValueError(
                f"{layout} tiles would be {tile_w} px wide at a {widths[layout]} px canvas; "
                "raise --width or select fewer frames."
            )

    renders = {}
    for layout in layouts:
        width = widths[layout]

        if layout in ("sequence", "montage"):
            groups = [[s] for s in order] if layout == "sequence" else [order]
            check_tile_width(layout, width)
            size, margin = ui_size(width), max(4, round(width / 80))
            for group in groups:
                bar = legend(width, drawn_classes(group), class_names, tag)
                frames = []
                for sequence in group:
                    ids = selection[sequence]
                    for k, image_id in enumerate(ids):
                        tile = render_tile(frame(image_id), predictions[image_id], conf, width)
                        label(tile, (margin, margin), caption(sequence), size)
                        label(tile, (width - margin, margin), f"{k + 1} / {len(ids)}", size, anchor="rt")
                        frames.append(stack(tile, bar))
                stem = f"sequence_{group[0]}" if layout == "sequence" else "montage"
                renders[stem] = (frames, seconds)

        elif layout == "grid":
            gap = 4
            cols = math.ceil(math.sqrt(len(order)))
            rows = math.ceil(len(order) / cols)
            tile_w = (width - gap * (cols - 1)) // cols
            check_tile_width(layout, tile_w)
            tile_h = round(tile_w * TILE_ASPECT)
            canvas_w = cols * tile_w + gap * (cols - 1)
            size, margin = ui_size(tile_w), max(4, round(tile_w / 80))
            bar = legend(canvas_w, drawn_classes(order), class_names, tag)
            frames = []
            for step in range(max(len(selection[s]) for s in order)):
                canvas = Image.new("RGB", (canvas_w, rows * tile_h + gap * (rows - 1)), BACKGROUND)
                for n, sequence in enumerate(order):
                    ids = selection[sequence]
                    # A shorter sequence holds its last frame rather than
                    # looping back, so time only runs forward in every tile.
                    image_id = ids[min(step, len(ids) - 1)]
                    tile = render_tile(frame(image_id), predictions[image_id], conf, tile_w)
                    label(tile, (margin, margin), caption(sequence), size)
                    canvas.paste(tile, ((n % cols) * (tile_w + gap), (n // cols) * (tile_h + gap)))
                frames.append(stack(canvas, bar))
            renders["grid"] = (frames, seconds)

        elif layout == "filmstrip":
            margin, gap = 14, 6
            cols = max(len(selection[s]) for s in order)
            tile_w = (width - 2 * margin - gap * (cols - 1)) // cols
            check_tile_width(layout, tile_w)
            tile_h = round(tile_w * TILE_ASPECT)
            canvas_w = 2 * margin + cols * tile_w + gap * (cols - 1)
            size = ui_size(tile_w)
            header = round(size * 2.8)
            # Sequences with fewer frames leave their trailing cells empty.
            body = Image.new("RGB", (canvas_w, header + len(order) * (tile_h + gap) + margin // 2), BACKGROUND)
            draw = ImageDraw.Draw(body)
            if tag:
                draw.text((margin, header // 2), tag, font=font(size + 2, bold=True), fill=TEXT, anchor="lm")
            time_arrow(draw, canvas_w - margin, header // 2, size + 1)
            for r, sequence in enumerate(order):
                for c, image_id in enumerate(selection[sequence]):
                    tile = render_tile(frame(image_id), predictions[image_id], conf, tile_w)
                    if c == 0:
                        label(tile, (6, 6), caption(sequence), size)
                    body.paste(tile, (margin + c * (tile_w + gap), header + r * (tile_h + gap)))
            renders["filmstrip"] = ([stack(body, legend(canvas_w, drawn_classes(order), class_names))], None)

        else:
            raise ValueError(f"Unknown layout {layout!r}; expected one of: {', '.join(LAYOUTS)}.")
    return renders


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
def quantize_frame(frame: Image.Image) -> Image.Image:
    """Reduce an RGB frame to a 256-color palette that keeps the box colors exact.

    Every frame gets its own palette, since one shared by a daytime and a
    night-time flight would posterize both. Median cut alone spends its
    entries on the photograph and folds the thin class-colored outlines into
    nearby photo colors, which breaks the legend's color key. So the class and
    interface colors hold reserved entries, the photograph is quantized with
    the rest, and only pixels exactly matching a reserved color are assigned
    one. Mapping every pixel to its nearest entry instead would turn hundreds
    of photo pixels per frame into class-colored specks that read as
    detections.

    One k-means pass refines the median-cut palette. Without it, sparse bright
    regions such as a white car roof take on the tint of whatever they were
    binned with.
    """
    reserved = [*PALETTE, BACKGROUND, TEXT, TEXT_DIM, LABEL_TEXT]
    adaptive = frame.quantize(colors=256 - len(reserved), method=Image.Quantize.MEDIANCUT, kmeans=1)
    indices = np.asarray(adaptive, dtype=np.uint8) + len(reserved)
    rgb = np.asarray(frame.convert("RGB"))
    for k, color in enumerate(reserved):
        indices[(rgb == color).all(axis=-1)] = k
    out = Image.frombytes("P", frame.size, indices.tobytes())
    out.putpalette([v for color in reserved for v in color]
                   + adaptive.getpalette()[:3 * (256 - len(reserved))])
    return out


def write_animation(frames: list[Image.Image], seconds: float, path: str) -> None:
    """Write a looping GIF or WebP, by the extension of `path`."""
    duration = round(seconds * 1000)
    if path.endswith(".gif"):
        frames = [quantize_frame(f) for f in frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                       loop=0, disposal=1, optimize=False)
    else:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                       loop=0, quality=82, method=6)


def write_still(image: Image.Image, path: str) -> None:
    if path.endswith(".png"):
        image.save(path, optimize=True)
    else:
        image.save(path, quality=90, optimize=True, progressive=True)


def selection_report(coco_gt: COCO, sequences: dict[str, list[int]], titles: dict[str, str],
                     selection: dict[str, list[int]], scores: dict[int, dict]) -> list[dict]:
    """Per-sequence record of the frames used and how they score against the whole sequence."""
    report = []
    for sequence, ids in sequences.items():
        chosen = selection[sequence]
        report.append({
            "sequence": sequence,
            "title": titles.get(sequence, ""),
            "frames_in_sequence": len(ids),
            "mean_f1_sequence": round(float(np.mean([scores[i]["f1"] for i in ids])), 4),
            "mean_f1_selected": round(float(np.mean([scores[i]["f1"] for i in chosen])), 4),
            "frames": [
                {
                    "file_name": os.path.basename(coco_gt.imgs[i]["file_name"]),
                    "f1": round(scores[i]["f1"], 4),
                    **{key: scores[i][key] for key in ("n_gt", "n_pred", "tp", "fp", "fn")},
                }
                for i in chosen
            ],
        })
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_type: str, weights: str, device: torch.device):
    if model_type == "yolo_ultra":
        return load_yolo_ultralytics(weights)
    if model_type == "yolo_manual":
        return load_yolo_manual(weights, device)
    if model_type == "onnx":
        return load_onnx(weights)
    if model_type == "trt":
        return load_trt(weights)
    return load_rtdetr(weights, device)


def predict(model, model_type: str, onnx_arch: str, coco_gt: COCO, image_ids: list,
            annotation_path: str, conf: float, device: torch.device) -> dict[int, list[dict]]:
    """Run `model` over `image_ids`, returning its predictions grouped by image id."""
    if model_type in ("onnx", "trt"):
        model_format = onnx_arch
    else:
        model_format = "yolo" if model_type in ("yolo_ultra", "yolo_manual") else "rtdetr"
    batch_size = _resolve_evaluation_batch_size(model, model_type, None)

    dataset = EvalDataset(
        coco_gt, image_ids, None, build_preprocess_transform(model_format), model_type,
        annotation_path=annotation_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg_train.dataloader_config["num_workers"],
        collate_fn=eval_collate_fn,
        pin_memory=(cfg_train.dataloader_config["pin_memory"]
                    if model_type not in ("yolo_ultra", "onnx", "trt") else False),
    )

    predictions = {image_id: [] for image_id in image_ids}
    for images, orig_hws in tqdm(loader, desc="Inference"):
        if not orig_hws:
            continue
        images = pad_batch_to_static_size(images, orig_hws, model_type, batch_size)
        if model_type == "yolo_ultra":
            batch = infer_yolo_ultralytics(model, images, [hw[2] for hw in orig_hws], conf)
        elif model_type == "onnx":
            batch = infer_onnx(model, images, orig_hws, conf, onnx_arch)
        elif model_type == "trt":
            batch = infer_trt(model, images, orig_hws, conf, onnx_arch)
        elif model_type == "yolo_manual":
            batch = infer_yolo_manual(model, images.to(device), orig_hws, conf, device)
        else:
            batch = infer_rtdetr(model, images.to(device), orig_hws, conf, device)
        for p in batch:
            predictions[p["image_id"]].append(p)

    # As in visualize.py: an anchor grid rebuilt under autocast would shift the
    # boxes and misrepresent the model.
    if model_type in ("yolo_manual", "yolo_ultra"):
        assert_fp32_anchor_cache(model, "after inference in render_sequences.py")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def split_choices(value: str, choices, flag: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in choices]
    if unknown or not items:
        raise ValueError(f"{flag} expects a comma-separated list from: {', '.join(choices)}; got {value!r}.")
    return list(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    vis_cfg = getattr(cfg, "visualization_config", {})
    eval_cfg = getattr(cfg, "eval_config", {})
    default_test_set = eval_cfg.get("test_sets", ["test_baseline"])[0]
    default_conf = vis_cfg.get("conf_threshold", 0.25)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=["yolo_ultra", "yolo_manual", "rtdetr", "onnx", "trt"])
    parser.add_argument("--onnx-arch", default="yolo", choices=["yolo", "rtdetr"],
                        help="Architecture of an ONNX/TRT model. Used only when --model is 'onnx' or 'trt'.")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt, .pth, .onnx, or .engine).")
    parser.add_argument("--sequences", required=True,
                        help="Comma-separated sequence ids, each optionally titled: "
                             '"0000259=daytime boulevard,0000074".')
    parser.add_argument("--profile", default="even", choices=PROFILES,
                        help="How frames are chosen from each sequence, as listed above (default: even).")
    parser.add_argument("--frames", type=int, default=4,
                        help="Frames per sequence (default: 4). Ignored by --profile all.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --profile random (default: 42).")
    parser.add_argument("--test-set", default=default_test_set,
                        help=f"Manifest under {cfg.processed_annotations_dir}/ to draw sequences from "
                             f"(default: {default_test_set}).")
    parser.add_argument("--conf", type=float, default=default_conf,
                        help=f"Confidence threshold for drawing and scoring (default: {default_conf}).")
    parser.add_argument("--layouts", default=",".join(LAYOUTS),
                        help=f"Comma-separated layouts from: {', '.join(LAYOUTS)} (default: all of them).")
    parser.add_argument("--formats", default="gif,png",
                        help="Comma-separated output formats: gif and webp for the animations, "
                             "png and jpg for the filmstrip (default: gif,png).")
    parser.add_argument("--width", type=int, default=None,
                        help="Canvas width in pixels for every layout (default: "
                             + ", ".join(f"{k} {v}" for k, v in DEFAULT_WIDTHS.items()) + ").")
    parser.add_argument("--seconds", type=float, default=1.2,
                        help="Seconds each animation frame is held (default: 1.2).")
    parser.add_argument("--tag", default="", help='Optional text for the legend strip, e.g. "YOLO26n student".')
    parser.add_argument("--output-dir", default="runs/visualizations/sequences",
                        help="Where the renders and the selection manifest are written "
                             "(default: runs/visualizations/sequences).")
    return parser.parse_args()


def main():
    args = parse_args()
    requested = parse_sequences(args.sequences)
    layouts = split_choices(args.layouts, LAYOUTS, "--layouts")
    formats = split_choices(args.formats, ANIMATION_FORMATS + STILL_FORMATS, "--formats")
    if args.frames <= 0:
        raise ValueError(f"--frames must be positive, got {args.frames}.")
    if args.seconds <= 0:
        raise ValueError(f"--seconds must be positive, got {args.seconds}.")
    if args.width is not None and args.width <= 0:
        raise ValueError(f"--width must be positive, got {args.width}.")
    if any(layout != "filmstrip" for layout in layouts) and not set(formats) & set(ANIMATION_FORMATS):
        raise ValueError("--formats has no animation format (gif or webp) for the animated layouts.")
    if "filmstrip" in layouts and not set(formats) & set(STILL_FORMATS):
        raise ValueError("--formats has no still format (png or jpg) for the filmstrip.")

    gt_path = os.path.join(cfg.processed_annotations_dir, f"{args.test_set}.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"{gt_path} not found. Run src/prepare_dataset.py first.")
    coco_gt = COCO(gt_path)
    available = sequence_frames(coco_gt, get_dataset_config(cfg.dataset_name)["sequence_id_pattern"])
    missing = [sequence for sequence, _ in requested if sequence not in available]
    if missing:
        raise ValueError(
            f"Sequence(s) {missing} not found in {args.test_set}, which holds "
            f"{len(available)} sequences: {', '.join(sorted(available))}."
        )
    sequences = {sequence: available[sequence] for sequence, _ in requested}
    titles = {sequence: title for sequence, title in requested if title}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} from {args.weights}...")
    model = load_model(args.model, args.weights, device)
    image_ids = [image_id for ids in sequences.values() for image_id in ids]
    predictions = predict(model, args.model, args.onnx_arch, coco_gt, image_ids, gt_path, args.conf, device)

    selection, scores = choose_frames(
        coco_gt, sequences, predictions, args.profile, args.frames, args.conf, args.seed
    )

    def read_image(image_id):
        file_name = coco_gt.imgs[image_id]["file_name"]
        path = resolve_image_path(file_name, gt_path)
        image = cv2.imread(path) if path else None
        if image is None:
            raise FileNotFoundError(f"Could not read image {image_id}: {file_name}.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    renders = build_renders(
        selection, titles, predictions, read_image,
        {c["id"]: c["name"] for c in cfg.category_mapping.values()},
        conf=args.conf,
        layouts=layouts,
        widths={layout: args.width or DEFAULT_WIDTHS[layout] for layout in LAYOUTS},
        seconds=args.seconds,
        tag=args.tag,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nWriting to {args.output_dir}:")
    for stem, (frames, seconds) in renders.items():
        kinds = STILL_FORMATS if seconds is None else ANIMATION_FORMATS
        for ext in (f for f in formats if f in kinds):
            path = os.path.join(args.output_dir, f"{stem}_{args.profile}.{ext}")
            if seconds is None:
                write_still(frames[0], path)
            else:
                write_animation(frames, seconds, path)
            print(f"  {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''}, "
                  f"{os.path.getsize(path) / 1e6:.2f} MB)")

    report = selection_report(coco_gt, sequences, titles, selection, scores)
    manifest_path = os.path.join(args.output_dir, f"selection_{args.profile}.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "model": args.model,
            "weights": args.weights,
            "test_set": args.test_set,
            "conf_threshold": args.conf,
            "profile": args.profile,
            "frames_per_sequence": None if args.profile == "all" else args.frames,
            "seed": args.seed if args.profile == "random" else None,
            "sequences": report,
        }, f, indent=2)
    print(f"  {os.path.basename(manifest_path)}")

    print(f"\nSelection ({args.profile}, F1 at conf >= {args.conf}):")
    for entry in report:
        print(f"  {entry['sequence']}: {len(entry['frames'])} of {entry['frames_in_sequence']} frames, "
              f"mean F1 {entry['mean_f1_selected']:.3f} (whole sequence {entry['mean_f1_sequence']:.3f})")


if __name__ == "__main__":
    main()
