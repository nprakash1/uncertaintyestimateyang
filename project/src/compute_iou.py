"""Step 3: Compute reader-reader bounding-box overlap (mask IoU).

We rasterize each reader's set of boxes into a binary mask on a fixed
[MASK_GRID_SIZE, MASK_GRID_SIZE] grid and compute the union-mask IoU.

PadChest-GR boxes are normalized in [0,1], so the grid size is arbitrary
for the purpose of IoU (it just controls rasterization precision).

This module is also used by the MedSAM 3 inference notebook
(`project/notebooks/medsam3_padchest_gr_colab.ipynb`) for predicted-mask
vs. ground-truth-box IoU; the helpers below the original `boxes_to_mask`
/ `mask_iou` block (intersection mask, outer-bbox union mask, pairwise
inter-annotator IoUs, predicted-prob → grid-mask resizer) were added for
that use case.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import MASK_GRID_SIZE, SAMPLES_TWO_READERS_CSV


# ---------------------------------------------------------------------------
# Core rasterization + IoU (unchanged behavior, used by the original Step 3
# pipeline that compares reader1 vs reader2 union masks).
# ---------------------------------------------------------------------------


def boxes_to_mask(
    boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
    normalized: bool = True,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> np.ndarray:
    """Rasterize a list of boxes into a binary union mask (pixel-OR).

    Each box paints a rectangle of 1's onto a zeroed [grid_size, grid_size]
    canvas; the result is the per-pixel OR over all boxes. Empty input →
    all-zero mask.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not boxes:
        return mask
    for box in boxes:
        x1, y1, x2, y2 = box
        if normalized:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        else:
            nx1 = x1 / image_width
            ny1 = y1 / image_height
            nx2 = x2 / image_width
            ny2 = y2 / image_height
        # Map normalized coords to integer pixel ranges in [0, grid_size]
        px1 = int(np.floor(nx1 * grid_size))
        py1 = int(np.floor(ny1 * grid_size))
        px2 = int(np.ceil(nx2 * grid_size))
        py2 = int(np.ceil(ny2 * grid_size))
        px1 = max(0, min(grid_size, px1))
        py1 = max(0, min(grid_size, py1))
        px2 = max(0, min(grid_size, px2))
        py2 = max(0, min(grid_size, py2))
        if px2 <= px1 or py2 <= py1:
            continue
        mask[py1:py2, px1:px2] = 1
    return mask


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, int, int, int, int]:
    """Return (iou, area1, area2, intersection, union)."""
    inter = int(np.logical_and(mask1, mask2).sum())
    union = int(np.logical_or(mask1, mask2).sum())
    a1 = int(mask1.sum())
    a2 = int(mask2.sum())
    iou = inter / union if union > 0 else float("nan")
    return iou, a1, a2, inter, union


def compute_iou_for_sample(
    reader1_boxes: List[List[float]],
    reader2_boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
) -> dict:
    m1 = boxes_to_mask(reader1_boxes, grid_size=grid_size, normalized=True)
    m2 = boxes_to_mask(reader2_boxes, grid_size=grid_size, normalized=True)
    iou, a1, a2, inter, union = mask_iou(m1, m2)
    return {
        "reader_iou": iou,
        "reader_disagreement": (1.0 - iou) if union > 0 else float("nan"),
        "reader1_area": a1,
        "reader2_area": a2,
        "intersection_area": inter,
        "union_area": union,
        "num_reader1_boxes": len(reader1_boxes),
        "num_reader2_boxes": len(reader2_boxes),
    }


def add_iou_columns(df: pd.DataFrame, grid_size: int = MASK_GRID_SIZE) -> pd.DataFrame:
    """Given a samples DataFrame with reader1_boxes / reader2_boxes JSON
    strings, append IoU metrics columns."""
    results = []
    for _, row in df.iterrows():
        r1 = json.loads(row["reader1_boxes"]) if isinstance(row["reader1_boxes"], str) else row["reader1_boxes"]
        r2 = json.loads(row["reader2_boxes"]) if isinstance(row["reader2_boxes"], str) else row["reader2_boxes"]
        results.append(compute_iou_for_sample(r1, r2, grid_size=grid_size))
    iou_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), iou_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Extended helpers for predicted-mask IoU (MedSAM 3 notebook).
# ---------------------------------------------------------------------------


def boxes_to_intersection_mask(
    boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
) -> np.ndarray:
    """Per-pixel AND over a list of boxes.

    Region covered by *all* boxes simultaneously. Empty / single-box inputs
    return the boxes_to_mask result (since the intersection over one set is
    that set, and over zero sets is the empty set).
    """
    if not boxes:
        return np.zeros((grid_size, grid_size), dtype=np.uint8)
    masks = [boxes_to_mask([b], grid_size=grid_size, normalized=True) for b in boxes]
    inter = masks[0].copy()
    for m in masks[1:]:
        inter = np.logical_and(inter, m).astype(np.uint8)
    return inter


def boxes_to_outer_bbox_mask(
    boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
) -> np.ndarray:
    """Single tightest rectangle containing every input box.

    A geometric union variant: instead of pixel-OR (which can leave a
    disjoint, holey mask when boxes don't touch), we take the
    [min(x1), min(y1), max(x2), max(y2)] outer box and rasterize that.
    Useful as a "smart union" baseline alongside `boxes_to_mask` (pixel-OR).
    """
    if not boxes:
        return np.zeros((grid_size, grid_size), dtype=np.uint8)
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    outer = [min(xs1), min(ys1), max(xs2), max(ys2)]
    return boxes_to_mask([outer], grid_size=grid_size, normalized=True)


def pairwise_box_ious(
    boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
) -> List[float]:
    """IoU(box_i, box_j) for every i<j. Empty/length-1 input → []."""
    if not boxes or len(boxes) < 2:
        return []
    masks = [boxes_to_mask([b], grid_size=grid_size, normalized=True) for b in boxes]
    out: List[float] = []
    for i, j in combinations(range(len(masks)), 2):
        iou, *_ = mask_iou(masks[i], masks[j])
        out.append(float(iou))
    return out


def pred_probs_to_grid_mask(
    probs: np.ndarray,
    grid_size: int = MASK_GRID_SIZE,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert a MedSAM 3 sigmoid prob output to a binary mask on the
    fixed [grid_size, grid_size] IoU grid.

    Accepts shapes:
        (H, W)         - single mask
        (N, H, W)      - N candidate masks  → max-over-N then threshold
        (1, N, H, W)   - batched variant    → squeezed, then as (N, H, W)

    Returns a uint8 mask of shape (grid_size, grid_size).
    """
    arr = np.asarray(probs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        # collapse multiple candidate masks by taking the per-pixel max prob
        arr = arr.max(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) / (N,H,W) / (1,N,H,W); got {probs.shape}.")

    # Lazy import so this module stays usable without scikit-image installed.
    try:
        from skimage.transform import resize  # type: ignore
        resized = resize(
            arr.astype(np.float32),
            (grid_size, grid_size),
            order=1,           # bilinear
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        )
    except Exception:
        # Pure-numpy fallback: nearest-neighbor.
        h, w = arr.shape
        ys = (np.linspace(0, h - 1, grid_size)).astype(np.int64)
        xs = (np.linspace(0, w - 1, grid_size)).astype(np.int64)
        resized = arr[ys[:, None], xs[None, :]]
    return (resized > threshold).astype(np.uint8)


def compute_pred_vs_gt_iou_bundle(
    pred_probs: np.ndarray,
    reader1_boxes: List[List[float]],
    reader2_boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
    threshold: float = 0.5,
) -> dict:
    """All MedSAM-3-vs-PadChest-GR IoUs we want per sample.

    Returns
    -------
    dict with keys:
        per_box_ious                 : list[float], one per GT box (both readers)
        max_per_box_iou              : float, max of per_box_ious (NaN if no boxes)
        mean_per_box_iou             : float, mean of per_box_ious
        iou_with_pixel_or_union      : float, IoU(pred, pixel-OR of all GT boxes)
        iou_with_outer_bbox_union    : float, IoU(pred, single outer bbox of all GT boxes)
        iou_with_intersection        : float, IoU(pred, pixel-AND of all GT boxes)
        iou_with_reader1_union       : float, IoU(pred, pixel-OR over reader1 boxes)
        iou_with_reader2_union       : float, IoU(pred, pixel-OR over reader2 boxes)
        reader_iou_union             : float, IoU(reader1_union, reader2_union)
        pairwise_box_ious            : list[float], IoU(box_i, box_j) for i<j across both readers
        mean_pairwise_iou            : float, mean of pairwise_box_ious
        pred_area                    : int, predicted mask pixel count on the grid
        num_boxes_total, num_reader1_boxes, num_reader2_boxes : int
    """
    pred_mask = pred_probs_to_grid_mask(
        pred_probs, grid_size=grid_size, threshold=threshold
    )
    all_boxes = (reader1_boxes or []) + (reader2_boxes or [])

    # Per-box IoUs
    per_box: List[float] = []
    for b in all_boxes:
        bm = boxes_to_mask([b], grid_size=grid_size, normalized=True)
        iou, *_ = mask_iou(pred_mask, bm)
        per_box.append(float(iou))

    # Pixel-OR / outer-bbox / intersection over all boxes
    union_mask_pix  = boxes_to_mask(all_boxes, grid_size=grid_size)
    union_mask_obb  = boxes_to_outer_bbox_mask(all_boxes, grid_size=grid_size)
    inter_mask_all  = boxes_to_intersection_mask(all_boxes, grid_size=grid_size)
    iou_or, *_      = mask_iou(pred_mask, union_mask_pix)
    iou_obb, *_     = mask_iou(pred_mask, union_mask_obb)
    iou_inter, *_   = mask_iou(pred_mask, inter_mask_all)

    # Per-reader unions and reader-vs-reader IoU
    r1_union = boxes_to_mask(reader1_boxes or [], grid_size=grid_size)
    r2_union = boxes_to_mask(reader2_boxes or [], grid_size=grid_size)
    iou_r1, *_ = mask_iou(pred_mask, r1_union)
    iou_r2, *_ = mask_iou(pred_mask, r2_union)
    iou_rr, *_ = mask_iou(r1_union, r2_union)

    # Pairwise inter-annotator IoUs across all GT boxes (both readers pooled)
    pw = pairwise_box_ious(all_boxes, grid_size=grid_size)

    return {
        "per_box_ious": per_box,
        "max_per_box_iou":  float(np.max(per_box))  if per_box else float("nan"),
        "mean_per_box_iou": float(np.mean(per_box)) if per_box else float("nan"),
        "iou_with_pixel_or_union":   float(iou_or),
        "iou_with_outer_bbox_union": float(iou_obb),
        "iou_with_intersection":     float(iou_inter),
        "iou_with_reader1_union":    float(iou_r1),
        "iou_with_reader2_union":    float(iou_r2),
        "reader_iou_union":          float(iou_rr),
        "pairwise_box_ious":         pw,
        "mean_pairwise_iou":         float(np.mean(pw)) if pw else float("nan"),
        "pred_area":                 int(pred_mask.sum()),
        "num_boxes_total":           len(all_boxes),
        "num_reader1_boxes":         len(reader1_boxes or []),
        "num_reader2_boxes":         len(reader2_boxes or []),
    }


# ---------------------------------------------------------------------------
# CLI for the original reader1 vs reader2 IoU pipeline (Step 3).
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SAMPLES_TWO_READERS_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--grid_size", type=int, default=MASK_GRID_SIZE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_df = add_iou_columns(df, grid_size=args.grid_size)
    out_path = args.output or args.input.with_name(args.input.stem + "_with_iou.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df[["reader_iou"]].describe())


if __name__ == "__main__":
    main()
