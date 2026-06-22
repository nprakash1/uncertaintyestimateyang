"""Step 7: Visualizations.

Generates:
    figures/iou_by_uncertainty_group.png    (boxplot + strip)
    figures/iou_histogram_by_group.png      (overlaid histograms)
    figures/example_grid.png                (example sentences with boxes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from config import (
    FIG_EXAMPLE_GRID,
    FIG_IOU_BY_GROUP,
    FIG_IOU_HISTOGRAM,
    SAMPLES_WITH_IOU_CSV,
)


def plot_iou_by_group(df: pd.DataFrame, out_path: Path = FIG_IOU_BY_GROUP) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ["certain", "uncertain"]
    data = [df.loc[df["uncertainty_label"] == g, "reader_iou"].dropna().to_numpy() for g in groups]
    tick_labels = [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)]
    try:
        # matplotlib >= 3.9
        bp = ax.boxplot(data, tick_labels=tick_labels,
                        showfliers=False, patch_artist=True)
    except TypeError:
        # matplotlib < 3.9
        bp = ax.boxplot(data, labels=tick_labels,
                        showfliers=False, patch_artist=True)
    colors = ["#4C9F70", "#E07B39"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # Overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        x = rng.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.15, s=6, color=colors[i])
    ax.set_ylabel("Reader-reader IoU")
    ax.set_title("Reader bounding-box overlap by report uncertainty")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_histogram(df: pd.DataFrame, out_path: Path = FIG_IOU_HISTOGRAM) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 31)
    for grp, color in [("certain", "#4C9F70"), ("uncertain", "#E07B39")]:
        vals = df.loc[df["uncertainty_label"] == grp, "reader_iou"].dropna().to_numpy()
        ax.hist(vals, bins=bins, alpha=0.55, label=f"{grp} (n={len(vals)})",
                color=color, edgecolor="black", linewidth=0.4, density=True)
    ax.set_xlabel("Reader-reader IoU")
    ax.set_ylabel("Density")
    ax.set_title("IoU distribution by report-language uncertainty")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _draw_boxes_on_axis(ax, boxes, color, label):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=2,
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)


def plot_example_grid(df: pd.DataFrame, out_path: Path = FIG_EXAMPLE_GRID) -> None:
    """Plot 4 example panels (no real image; boxes only on a unit canvas)."""
    quadrants = {
        "uncertain + low IoU": df[df["uncertainty_label"] == "uncertain"]
            .sort_values("reader_iou").head(1),
        "certain + high IoU": df[df["uncertainty_label"] == "certain"]
            .sort_values("reader_iou", ascending=False).head(1),
        "uncertain + high IoU": df[df["uncertainty_label"] == "uncertain"]
            .sort_values("reader_iou", ascending=False).head(1),
        "certain + low IoU": df[df["uncertainty_label"] == "certain"]
            .sort_values("reader_iou").head(1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (title, sub) in zip(axes.flat, quadrants.items()):
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)  # image-style y axis
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f8f8")
        if sub.empty:
            ax.set_title(f"{title}\n(no example)")
            continue
        row = sub.iloc[0]
        r1 = json.loads(row["reader1_boxes"]) if isinstance(row["reader1_boxes"], str) else row["reader1_boxes"]
        r2 = json.loads(row["reader2_boxes"]) if isinstance(row["reader2_boxes"], str) else row["reader2_boxes"]
        _draw_boxes_on_axis(ax, r1, "tab:blue", "Reader 1")
        _draw_boxes_on_axis(ax, r2, "tab:red", "Reader 2")
        ax.legend(loc="upper right", fontsize=8)
        sentence = row["sentence"]
        if len(sentence) > 90:
            sentence = sentence[:87] + "..."
        ax.set_title(
            f"{title}\n"
            f"IoU={row['reader_iou']:.2f}  label={row['uncertainty_label']}\n"
            f'"{sentence}"',
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Example reader boxes by sentence-uncertainty category", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SAMPLES_WITH_IOU_CSV)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    plot_iou_by_group(df)
    plot_histogram(df)
    plot_example_grid(df)


if __name__ == "__main__":
    main()
