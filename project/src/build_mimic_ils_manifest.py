"""Build image/annotation manifests from the MIMIC-ILS dataset.

MIMIC-ILS ships annotations only (masks + instruction/answer pairs + report
text); the actual chest X-rays live in credentialed MIMIC-CXR-JPG and are NOT
included. This script streams the (785 MB) master JSON and emits:

  1. mimic_ils_full_manifest.csv   -- EVERY instruction-answer pair across all
     splits (train/val/test). One row per pair, with the image linkage columns
     (subject_id / study_id / dicom_id / image_path) so it doubles as the file
     list to hand to whoever has MIMIC-CXR-JPG on the cluster. De-dupe on
     `image_path` to get the unique set of images to copy.

  2. mimic_ils_subset_manifest.csv -- a reasonable-size MIXED sample that spans
     train + val + test. We sample whole STUDIES (so every pair + mask for a
     sampled study stays together and maps cleanly to one image), then emit all
     their pairs.

Also writes two convenience image lists (unique image_path per manifest):
  mimic_ils_full_image_list.txt, mimic_ils_subset_image_list.txt

Usage:
    python project/src/build_mimic_ils_manifest.py \
        --dataset "/Users/nealprakash/Downloads/mimic-cxr-ext-ils" \
        --outdir  project/data/mimic_ils \
        --subset-train 400 --subset-val 150 --subset-test 150 --seed 42
"""
import argparse
import csv
import os
import random
from pathlib import Path

import ijson

SPLITS = ["train", "val", "test"]

FIELDNAMES = [
    "split",
    "study_id",
    "subject_id",
    "dicom_id",
    "image_path",          # relative MIMIC-CXR-JPG path -> the file to fetch
    "section_name",
    "pair_id",
    "polarity",            # positive | negative
    "type",
    "target",              # one of the 7 lesion types
    "location",            # ; -joined
    "instruction",
    "answer",
    "seg",                 # bool: does this pair have a segmentation mask
    "seg_mask_path",       # relative path under lesion_mask/
    "section_content",     # full parsed report section text
]


def iter_pairs(study_id, study, split):
    """Yield one flat row dict per instruction-answer pair in a study."""
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
                "split": split,
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


def build(dataset_dir, outdir, subset_counts, seed):
    json_path = os.path.join(dataset_dir, "mimic_ils_instruction_answer.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(json_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_csv = outdir / "mimic_ils_full_manifest.csv"
    subset_csv = outdir / "mimic_ils_subset_manifest.csv"

    # ---- PASS 1: collect study_ids per split (cheap: keys + tiny fields) -----
    # We only need the study_id keys to choose the subset sample; we grab them
    # by streaming map_keys under each split without materializing values.
    print("Pass 1/2: enumerating study ids per split ...")
    studies_by_split = {s: [] for s in SPLITS}
    with open(json_path, "rb") as f:
        # kvitems over the root would load whole study objects; instead walk the
        # parse events and record keys exactly two levels deep (split -> study).
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if event == "map_key" and prefix in SPLITS:
                studies_by_split[prefix].append(value)
    for s in SPLITS:
        print(f"  {s:5s}: {len(studies_by_split[s]):>7,} studies")

    rng = random.Random(seed)
    subset_ids = {}
    for s in SPLITS:
        ids = studies_by_split[s]
        k = min(subset_counts[s], len(ids))
        subset_ids[s] = set(rng.sample(ids, k)) if k > 0 else set()
        print(f"  subset {s:5s}: sampling {len(subset_ids[s])} studies")

    # ---- PASS 2: stream every study, write full + subset rows ---------------
    print("Pass 2/2: streaming studies and writing CSVs ...")
    n_full_rows = 0
    n_subset_rows = 0
    n_studies = 0
    full_images = set()
    subset_images = set()

    with open(json_path, "rb") as f, \
         open(full_csv, "w", newline="") as ff, \
         open(subset_csv, "w", newline="") as sf:
        fw = csv.DictWriter(ff, fieldnames=FIELDNAMES)
        sw = csv.DictWriter(sf, fieldnames=FIELDNAMES)
        fw.writeheader()
        sw.writeheader()
        for split in SPLITS:
            # kvitems yields (study_id, study_obj) one study at a time -> the
            # only thing held in memory is a single study's dict.
            f.seek(0)
            for study_id, study in ijson.kvitems(f, split):
                n_studies += 1
                in_subset = study_id in subset_ids[split]
                for row in iter_pairs(study_id, study, split):
                    fw.writerow(row)
                    n_full_rows += 1
                    if row["image_path"]:
                        full_images.add(row["image_path"])
                    if in_subset:
                        sw.writerow(row)
                        n_subset_rows += 1
                        if row["image_path"]:
                            subset_images.add(row["image_path"])
                if n_studies % 20000 == 0:
                    print(f"    ... {n_studies:,} studies, {n_full_rows:,} pairs")

    # ---- unique image lists (the actual files to request) -------------------
    full_list = outdir / "mimic_ils_full_image_list.txt"
    subset_list = outdir / "mimic_ils_subset_image_list.txt"
    full_list.write_text("\n".join(sorted(full_images)) + "\n")
    subset_list.write_text("\n".join(sorted(subset_images)) + "\n")

    print("\n=== DONE ===")
    print(f"Studies processed      : {n_studies:,}")
    print(f"Full manifest rows     : {n_full_rows:,}   -> {full_csv}")
    print(f"  unique images        : {len(full_images):,}   -> {full_list}")
    print(f"Subset manifest rows   : {n_subset_rows:,}   -> {subset_csv}")
    print(f"  unique images        : {len(subset_images):,}   -> {subset_list}")
    for p in (full_csv, subset_csv):
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Users/nealprakash/Downloads/mimic-cxr-ext-ils")
    ap.add_argument("--outdir", default="project/data/mimic_ils")
    ap.add_argument("--subset-train", type=int, default=400)
    ap.add_argument("--subset-val", type=int, default=150)
    ap.add_argument("--subset-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(
        args.dataset,
        args.outdir,
        {"train": args.subset_train, "val": args.subset_val, "test": args.subset_test},
        args.seed,
    )


if __name__ == "__main__":
    main()
