"""Step 1: Load PadChest-GR grounded reports and filter to positive finding
sentences that have bounding boxes from BOTH readers.

The PadChest-GR `grounded_reports` JSON has the structure:

    [
      {
        "StudyID": "...",
        "ImageID": "..._xxx.png",
        "findings": [
          {
            "sentence_en": "...",
            "sentence_es": "...",
            "abnormal": true|false,
            "boxes":       [[x1,y1,x2,y2], ...],   # reader 1 (normalized 0-1)
            "extra_boxes": [[x1,y1,x2,y2], ...],   # reader 2 (normalized 0-1)
            "labels": ["..."],
            "locations": ["..."],
            "progression": null
          },
          ...
        ]
      },
      ...
    ]

We treat `boxes` as reader 1 and `extra_boxes` as reader 2.

Output: data/processed/samples_with_two_readers.csv  with columns
    sample_id, image_id, study_id, sentence, finding_label,
    image_width, image_height, reader1_boxes, reader2_boxes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import RAW_GROUNDED_REPORTS, SAMPLES_TWO_READERS_CSV


def _clean_boxes(raw_boxes) -> List[List[float]]:
    """Return a list of [x1,y1,x2,y2] boxes, dropping malformed entries.

    PadChest-GR boxes are normalized in [0,1].  We coerce to floats, clamp to
    [0,1], require strictly positive width/height, and ensure x1<x2, y1<y2.
    """
    cleaned: List[List[float]] = []
    if not raw_boxes:
        return cleaned
    for box in raw_boxes:
        if box is None or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        # Order corners
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # Clamp to [0,1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cleaned.append([x1, y1, x2, y2])
    return cleaned


def load_samples(
    json_path: Path = RAW_GROUNDED_REPORTS,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        studies = json.load(f)

    rows = []
    for study in studies:
        study_id = study.get("StudyID")
        image_id = study.get("ImageID")
        findings = study.get("findings") or []
        for f_idx, finding in enumerate(findings):
            if not finding.get("abnormal"):
                continue  # only positive findings
            sentence = (finding.get("sentence_en") or "").strip()
            if not sentence:
                continue
            r1 = _clean_boxes(finding.get("boxes"))
            r2 = _clean_boxes(finding.get("extra_boxes"))
            if not r1 or not r2:
                continue  # require both readers

            labels = finding.get("labels") or []
            finding_label = labels[0] if labels else "unknown"

            rows.append(
                {
                    "sample_id": f"{image_id}__f{f_idx}",
                    "image_id": image_id,
                    "study_id": study_id,
                    "sentence": sentence,
                    "finding_label": finding_label,
                    # PadChest-GR boxes are normalized; image_width/height are
                    # not strictly needed for mask IoU on a fixed grid, but we
                    # store 1.0 here so the box format remains normalized.
                    "image_width": 1.0,
                    "image_height": 1.0,
                    "reader1_boxes": json.dumps(r1),
                    "reader2_boxes": json.dumps(r2),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--input", type=Path, default=RAW_GROUNDED_REPORTS,
        help="Path to PadChest-GR grounded_reports JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=SAMPLES_TWO_READERS_CSV,
    )
    args = parser.parse_args()

    df = load_samples(args.input, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} samples to {args.output}")
    print(df["finding_label"].value_counts().head(15))


if __name__ == "__main__":
    main()
