"""Step 3: Compute reader-reader bounding-box overlap (mask IoU).

We rasterize each reader's set of boxes into a binary mask on a fixed
[MASK_GRID_SIZE, MASK_GRID_SIZE] grid and compute the union-mask IoU.

PadChest-GR boxes are normalized in [0,1], so the grid size is arbitrary
for the purpose of IoU (it just controls rasterization precision).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import MASK_GRID_SIZE, SAMPLES_TWO_READERS_CSV


def boxes_to_mask(
    boxes: List[List[float]],
    grid_size: int = MASK_GRID_SIZE,
    normalized: bool = True,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> np.ndarray:
    """Rasterize a list of boxes into a binary union mask."""
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
