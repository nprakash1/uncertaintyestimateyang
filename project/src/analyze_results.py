"""Steps 4-6 + 9: Statistical analysis comparing reader IoU between
certain and uncertain finding sentences.

Produces:
    outputs/group_statistics.json
    outputs/statistical_tests.json
    outputs/per_finding_analysis.csv
    outputs/regression_results.json
    outputs/baseline_comparison.json
    outputs/examples_uncertain_low_iou.csv
    outputs/examples_certain_high_iou.csv
    data/processed/samples_with_uncertainty_and_iou.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from compute_iou import add_iou_columns
from config import (
    BASELINE_COMPARISON_JSON,
    BOOTSTRAP_N,
    EXAMPLES_CERTAIN_HIGH_IOU,
    EXAMPLES_UNCERTAIN_LOW_IOU,
    GROUP_STATS_JSON,
    MEDGEMMA_SCORES_JSONL,
    MIN_PER_FINDING_GROUP,
    PERMUTATION_N,
    PER_FINDING_ANALYSIS_CSV,
    RANDOM_SEED,
    REGRESSION_RESULTS_JSON,
    RULE_SCORES_JSONL,
    SAMPLES_TWO_READERS_CSV,
    SAMPLES_WITH_IOU_CSV,
    STAT_TESTS_JSON,
)
from medgemma_uncertainty import load_scores


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_analysis_df(
    samples_csv: Path = SAMPLES_TWO_READERS_CSV,
    scores_jsonl: Path = MEDGEMMA_SCORES_JSONL,
    label_source: str = "medgemma",
) -> pd.DataFrame:
    """Merge samples with IoU + uncertainty labels and save a single CSV."""
    samples = pd.read_csv(samples_csv)
    samples = add_iou_columns(samples)

    scores = load_scores(scores_jsonl)
    if scores.empty:
        raise RuntimeError(
            f"No uncertainty scores found in {scores_jsonl}. Run "
            "medgemma_uncertainty.py first."
        )

    # Keep only the columns we need from the score table
    score_cols = ["sample_id", "uncertainty_label", "confidence", "uncertainty_triggers", "reason"]
    score_cols = [c for c in score_cols if c in scores.columns]
    scores = scores[score_cols].rename(columns={"confidence": "medgemma_confidence"})

    merged = samples.merge(scores, on="sample_id", how="inner")
    merged["label_source"] = label_source
    # Drop samples MedGemma failed to classify and rows with no overlap to score
    merged = merged[merged["uncertainty_label"].isin(["certain", "uncertain"])]
    merged = merged.dropna(subset=["reader_iou"])
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: group statistics + Step 5: stat tests
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(values: np.ndarray, n: int = BOOTSTRAP_N,
                      seed: int = RANDOM_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def _bootstrap_delta_ci(certain: np.ndarray, uncertain: np.ndarray,
                       n: int = BOOTSTRAP_N, seed: int = RANDOM_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    if len(certain) == 0 or len(uncertain) == 0:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    nc, nu = len(certain), len(uncertain)
    ic = rng.integers(0, nc, size=(n, nc))
    iu = rng.integers(0, nu, size=(n, nu))
    boots = certain[ic].mean(axis=1) - uncertain[iu].mean(axis=1)
    return {
        "delta_mean": float(certain.mean() - uncertain.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def _permutation_pvalue(certain: np.ndarray, uncertain: np.ndarray,
                       n: int = PERMUTATION_N, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    observed = certain.mean() - uncertain.mean()
    pooled = np.concatenate([certain, uncertain])
    nc = len(certain)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        d = pooled[:nc].mean() - pooled[nc:].mean()
        if abs(d) >= abs(observed):
            count += 1
    return (count + 1) / (n + 1)


def group_statistics(df: pd.DataFrame) -> Dict:
    out: Dict[str, Dict] = {}
    for grp in ("certain", "uncertain"):
        vals = df.loc[df["uncertainty_label"] == grp, "reader_iou"].to_numpy(dtype=float)
        ci = _bootstrap_mean_ci(vals)
        out[grp] = {
            "n": int(len(vals)),
            "mean_iou": float(vals.mean()) if len(vals) else float("nan"),
            "median_iou": float(np.median(vals)) if len(vals) else float("nan"),
            "std_iou": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "mean_iou_ci_low": ci["ci_low"],
            "mean_iou_ci_high": ci["ci_high"],
        }
    out["delta_mean_iou_certain_minus_uncertain"] = (
        out["certain"]["mean_iou"] - out["uncertain"]["mean_iou"]
    )
    return out


def statistical_tests(df: pd.DataFrame) -> Dict:
    from scipy.stats import mannwhitneyu

    certain = df.loc[df["uncertainty_label"] == "certain", "reader_iou"].to_numpy(dtype=float)
    uncertain = df.loc[df["uncertainty_label"] == "uncertain", "reader_iou"].to_numpy(dtype=float)
    results: Dict = {
        "n_certain": int(len(certain)),
        "n_uncertain": int(len(uncertain)),
    }
    if len(certain) > 0 and len(uncertain) > 0:
        mw = mannwhitneyu(certain, uncertain, alternative="greater")
        results["mannwhitneyu"] = {
            "U": float(mw.statistic),
            "p_value_certain_gt_uncertain": float(mw.pvalue),
        }
        mw2 = mannwhitneyu(certain, uncertain, alternative="two-sided")
        results["mannwhitneyu"]["p_value_two_sided"] = float(mw2.pvalue)
        results["bootstrap_delta"] = _bootstrap_delta_ci(certain, uncertain)
        results["permutation_p_value"] = _permutation_pvalue(certain, uncertain)
    return results


# ---------------------------------------------------------------------------
# Step 6: per-finding control + simple regression
# ---------------------------------------------------------------------------


def per_finding_analysis(df: pd.DataFrame, min_n: int = MIN_PER_FINDING_GROUP) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("finding_label"):
        c = sub.loc[sub["uncertainty_label"] == "certain", "reader_iou"].to_numpy(dtype=float)
        u = sub.loc[sub["uncertainty_label"] == "uncertain", "reader_iou"].to_numpy(dtype=float)
        if len(c) < min_n or len(u) < min_n:
            continue
        rows.append({
            "finding_label": label,
            "n_certain": int(len(c)),
            "n_uncertain": int(len(u)),
            "mean_iou_certain": float(c.mean()),
            "mean_iou_uncertain": float(u.mean()),
            "delta_mean_iou": float(c.mean() - u.mean()),
        })
    return pd.DataFrame(rows).sort_values("delta_mean_iou", ascending=False)


def simple_regression(df: pd.DataFrame) -> Dict:
    """reader_iou ~ uncertainty_label + finding_label + log(union_area).

    Implemented with a manual OLS over a dense one-hot design matrix so we
    do not require statsmodels.  Returns the coefficient on
    `is_uncertain` and a t-statistic.
    """
    work = df.copy()
    work = work[work["uncertainty_label"].isin(["certain", "uncertain"])]
    work = work.dropna(subset=["reader_iou"])
    if work.empty:
        return {"error": "no data"}

    y = work["reader_iou"].to_numpy(dtype=float)
    n = len(y)
    # Features
    is_uncertain = (work["uncertainty_label"].values == "uncertain").astype(float)
    log_area = np.log1p(work["union_area"].to_numpy(dtype=float))
    # One-hot for finding label, drop first
    fl = work["finding_label"].astype("category")
    fl_codes = fl.cat.codes.to_numpy()
    n_cats = len(fl.cat.categories)
    onehot = np.zeros((n, max(n_cats - 1, 0)))
    for i, code in enumerate(fl_codes):
        if code > 0:
            onehot[i, code - 1] = 1.0
    X = np.column_stack([np.ones(n), is_uncertain, log_area, onehot])
    # OLS with pseudoinverse for stability
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(n - X.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    # Index 1 = is_uncertain
    coef = float(beta[1])
    coef_se = float(se[1])
    t = coef / coef_se if coef_se > 0 else float("nan")
    # Two-sided p-value via normal approx (dof typically large)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else float("nan")
    return {
        "n": int(n),
        "coef_is_uncertain": coef,
        "se_is_uncertain": coef_se,
        "t_is_uncertain": float(t),
        "p_value_is_uncertain": float(p),
        "coef_log_union_area": float(beta[2]),
        "intercept": float(beta[0]),
        "n_finding_categories": int(n_cats),
        "interpretation": (
            "Negative coef_is_uncertain means uncertain sentences have lower "
            "reader IoU after controlling for finding type and box area."
        ),
    }


# ---------------------------------------------------------------------------
# Step 9: baseline (rule-based) vs MedGemma comparison
# ---------------------------------------------------------------------------


def baseline_comparison(
    samples_csv: Path = SAMPLES_TWO_READERS_CSV,
    medgemma_jsonl: Path = MEDGEMMA_SCORES_JSONL,
    rule_jsonl: Path = RULE_SCORES_JSONL,
) -> Dict:
    if not rule_jsonl.exists():
        return {"warning": f"No rule scores at {rule_jsonl}."}

    rule_df = build_analysis_df(samples_csv, rule_jsonl, label_source="rule")
    rule_stats = group_statistics(rule_df)
    out: Dict = {"rule": rule_stats}

    if medgemma_jsonl.exists():
        mg_df = build_analysis_df(samples_csv, medgemma_jsonl, label_source="medgemma")
        mg_stats = group_statistics(mg_df)
        out["medgemma"] = mg_stats

        # Agreement / kappa on overlapping sample_ids
        merged = mg_df[["sample_id", "uncertainty_label"]].merge(
            rule_df[["sample_id", "uncertainty_label"]],
            on="sample_id",
            suffixes=("_medgemma", "_rule"),
        )
        if not merged.empty:
            same = (merged["uncertainty_label_medgemma"] == merged["uncertainty_label_rule"]).mean()
            from sklearn.metrics import cohen_kappa_score  # type: ignore
            try:
                kappa = float(
                    cohen_kappa_score(
                        merged["uncertainty_label_medgemma"],
                        merged["uncertainty_label_rule"],
                    )
                )
            except Exception:
                kappa = float("nan")
            out["agreement"] = {
                "n": int(len(merged)),
                "agreement_rate": float(same),
                "cohen_kappa": kappa,
            }
    return out


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def save_examples(df: pd.DataFrame, k: int = 25) -> None:
    cols = [
        "sample_id", "image_id", "study_id", "sentence", "finding_label",
        "uncertainty_label", "medgemma_confidence", "reader_iou",
        "reader_disagreement", "num_reader1_boxes", "num_reader2_boxes",
    ]
    cols = [c for c in cols if c in df.columns]
    unc = df[df["uncertainty_label"] == "uncertain"].sort_values("reader_iou").head(k)
    cer = df[df["uncertainty_label"] == "certain"].sort_values("reader_iou", ascending=False).head(k)
    unc[cols].to_csv(EXAMPLES_UNCERTAIN_LOW_IOU, index=False)
    cer[cols].to_csv(EXAMPLES_CERTAIN_HIGH_IOU, index=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(label_source: str = "medgemma", scores_jsonl: Optional[Path] = None) -> pd.DataFrame:
    if scores_jsonl is None:
        scores_jsonl = MEDGEMMA_SCORES_JSONL if label_source == "medgemma" else RULE_SCORES_JSONL

    df = build_analysis_df(
        samples_csv=SAMPLES_TWO_READERS_CSV,
        scores_jsonl=scores_jsonl,
        label_source=label_source,
    )
    df.to_csv(SAMPLES_WITH_IOU_CSV, index=False)
    print(f"Wrote merged analysis table: {SAMPLES_WITH_IOU_CSV} ({len(df)} rows)")

    # Step 4 + 5
    gs = group_statistics(df)
    with open(GROUP_STATS_JSON, "w") as f:
        json.dump(gs, f, indent=2)
    print(f"Wrote group statistics: {GROUP_STATS_JSON}")

    st = statistical_tests(df)
    with open(STAT_TESTS_JSON, "w") as f:
        json.dump(st, f, indent=2)
    print(f"Wrote statistical tests: {STAT_TESTS_JSON}")

    # Step 6
    pf = per_finding_analysis(df)
    pf.to_csv(PER_FINDING_ANALYSIS_CSV, index=False)
    print(f"Wrote per-finding analysis: {PER_FINDING_ANALYSIS_CSV}")

    reg = simple_regression(df)
    with open(REGRESSION_RESULTS_JSON, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"Wrote regression results: {REGRESSION_RESULTS_JSON}")

    # Step 9
    bc = baseline_comparison()
    with open(BASELINE_COMPARISON_JSON, "w") as f:
        json.dump(bc, f, indent=2)
    print(f"Wrote baseline comparison: {BASELINE_COMPARISON_JSON}")

    save_examples(df)
    print(f"Wrote examples: {EXAMPLES_UNCERTAIN_LOW_IOU}, {EXAMPLES_CERTAIN_HIGH_IOU}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_source", choices=["medgemma", "rule"], default="medgemma")
    parser.add_argument("--scores_jsonl", type=Path, default=None)
    args = parser.parse_args()
    run(label_source=args.label_source, scores_jsonl=args.scores_jsonl)


if __name__ == "__main__":
    main()
