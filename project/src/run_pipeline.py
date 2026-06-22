"""End-to-end driver.

Order of operations follows the spec:

    1. Load + filter PadChest-GR  (load_padchest_gr.py)
    2. Score uncertainty           (medgemma_uncertainty.py, rule or medgemma)
    3. Compute reader IoU + merge  (compute_iou.py + analyze_results.py)
    4-6,9. Group stats + tests + per-finding control + baseline (analyze)
    7. Plots                       (plot_results.py)

Usage:
    python run_pipeline.py --scorer rule          # baseline, no GPU needed
    python run_pipeline.py --scorer medgemma      # MedGemma classifier
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import analyze_results
import plot_results
from config import (
    MEDGEMMA_SCORES_JSONL,
    RAW_GROUNDED_REPORTS,
    RULE_SCORES_JSONL,
    SAMPLES_TWO_READERS_CSV,
)
from load_padchest_gr import load_samples
from medgemma_uncertainty import (
    MedGemmaUncertaintyScorer,
    score_dataframe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["rule", "medgemma", "both"], default="medgemma")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only first N samples (smoke test).")
    parser.add_argument("--skip_load", action="store_true",
                        help="Skip Step 1 if samples CSV already exists.")
    parser.add_argument("--input", type=Path, default=RAW_GROUNDED_REPORTS)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for MedGemma generation (>1 on GPU).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cuda', 'mps', or 'cpu'.")
    args = parser.parse_args()

    # ---- Step 1: Load + filter ---------------------------------------------
    if args.skip_load and SAMPLES_TWO_READERS_CSV.exists():
        print(f"[1] Using existing {SAMPLES_TWO_READERS_CSV}")
        df = pd.read_csv(SAMPLES_TWO_READERS_CSV)
    else:
        print("[1] Loading and filtering PadChest-GR...")
        df = load_samples(args.input, limit=args.limit)
        df.to_csv(SAMPLES_TWO_READERS_CSV, index=False)
        print(f"[1] Wrote {len(df)} samples to {SAMPLES_TWO_READERS_CSV}")

    if args.limit is not None:
        df = df.head(args.limit)

    # ---- Step 2: Uncertainty scoring ---------------------------------------
    if args.scorer == "rule":
        print("[2] Rule-based uncertainty scoring...")
        score_dataframe(df, RULE_SCORES_JSONL, scorer="rule")
    elif args.scorer == "medgemma":
        print("[2] MedGemma uncertainty scoring (exclusive)...")
        print("    Loading MedGemma model (this can take several minutes)...")
        mg = MedGemmaUncertaintyScorer(device=args.device)
        score_dataframe(
            df, MEDGEMMA_SCORES_JSONL, scorer="medgemma",
            medgemma=mg, batch_size=args.batch_size,
        )
    elif args.scorer == "both":
        print("[2a] Rule-based uncertainty scoring...")
        score_dataframe(df, RULE_SCORES_JSONL, scorer="rule")
        print("[2b] MedGemma uncertainty scoring...")
        print("     Loading MedGemma model (this can take several minutes)...")
        mg = MedGemmaUncertaintyScorer(device=args.device)
        score_dataframe(
            df, MEDGEMMA_SCORES_JSONL, scorer="medgemma",
            medgemma=mg, batch_size=args.batch_size,
        )

    label_source = "medgemma" if args.scorer in ("medgemma", "both") else "rule"

    # ---- Steps 3-6 + 9: analysis -------------------------------------------
    print(f"[3-6,9] Running analysis with label_source={label_source}")
    merged = analyze_results.run(label_source=label_source)

    # ---- Step 7: plots -----------------------------------------------------
    print("[7] Generating plots...")
    plot_results.plot_iou_by_group(merged)
    plot_results.plot_histogram(merged)
    plot_results.plot_example_grid(merged)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
