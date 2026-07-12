"""Emit the FULL validation-split manifest from the MIMIC-ILS master JSON.

Mirrors the existing test manifest (same columns / image-list format) but for the
official `val` split only. Streams the 785 MB JSON with ijson so memory stays low.

Usage:
    python build_val_manifest.py \
        --dataset "/Users/nealprakash/Downloads/mimic-cxr-ext-ils" \
        --outdir  project/data/mimic_ils
"""
import argparse
import csv
import os
from pathlib import Path

import ijson

SPLIT = "val"
FIELDNAMES = [
    "split", "study_id", "subject_id", "dicom_id", "image_path",
    "section_name", "pair_id", "polarity", "type", "target", "location",
    "instruction", "answer", "seg", "seg_mask_path", "section_content",
]


def iter_pairs(study_id, study):
    subject_id = study.get("subject_id")
    dicom_id = study.get("dicom_id")
    image_path = study.get("image_path")
    section_name = study.get("section_name")
    section_content = study.get("section_content")
    iap = study.get("instruction_answer_pairs") or {}
    for polarity in ("positive_pairs", "negative_pairs"):
        for pair in (iap.get(polarity) or []):
            loc = pair.get("location")
            if isinstance(loc, list):
                loc = "; ".join(str(x) for x in loc)
            yield {
                "split": SPLIT,
                "study_id": study_id,
                "subject_id": subject_id,
                "dicom_id": dicom_id,
                "image_path": image_path,
                "section_name": section_name,
                "pair_id": pair.get("pair_id"),
                "polarity": "positive" if polarity == "positive_pairs" else "negative",
                "type": pair.get("type"),
                "target": pair.get("target"),
                "location": loc,
                "instruction": pair.get("instruction"),
                "answer": pair.get("answer"),
                "seg": pair.get("seg"),
                "seg_mask_path": pair.get("seg_mask_path"),
                "section_content": section_content,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Users/nealprakash/Downloads/mimic-cxr-ext-ils")
    ap.add_argument("--outdir", default="project/data/mimic_ils")
    args = ap.parse_args()

    json_path = os.path.join(args.dataset, "mimic_ils_instruction_answer.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(json_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "mimic_ils_val_manifest.csv"
    out_list = outdir / "mimic_ils_val_image_list.txt"

    n_rows = 0
    n_studies = 0
    n_pos = 0
    images = set()
    with open(json_path, "rb") as f, open(out_csv, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=FIELDNAMES)
        w.writeheader()
        for study_id, study in ijson.kvitems(f, SPLIT):
            n_studies += 1
            for row in iter_pairs(study_id, study):
                w.writerow(row)
                n_rows += 1
                if row["polarity"] == "positive":
                    n_pos += 1
                if row["image_path"]:
                    images.add(row["image_path"])
            if n_studies % 20000 == 0:
                print(f"  ... {n_studies:,} val studies, {n_rows:,} pairs")

    out_list.write_text("\n".join(sorted(images)) + "\n")

    print("\n=== DONE (val split) ===")
    print(f"val studies         : {n_studies:,}")
    print(f"val pairs (rows)    : {n_rows:,}  ({n_pos:,} positive)  -> {out_csv}")
    print(f"unique val images   : {len(images):,}  -> {out_list}")
    print(f"csv size            : {out_csv.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
