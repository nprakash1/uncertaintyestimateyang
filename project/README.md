# Report-Language Uncertainty Predicts Radiologist Spatial Disagreement in Chest X-ray Grounding

This project tests one hypothesis:

> Do radiology report sentences classified as **uncertain** have lower
> radiologist bounding-box agreement than sentences classified as
> **certain**?

It uses the **PadChest-GR** grounded-report dataset, in which each
positive finding sentence is annotated with bounding boxes by two
independent readers.  We classify each sentence's language as `certain`
or `uncertain` (with **MedGemma**, plus a keyword-based baseline), compute
the mask-IoU between the two readers' bounding boxes, and ask whether
uncertain sentences have systematically lower reader-reader IoU.

## Dataset

Input file: `project/data/raw/grounded_reports_20240819.json` (the
PadChest-GR `grounded_reports` JSON).  Each study contains a list of
findings; for each *positive* finding (`abnormal: true`) PadChest-GR
provides:

| JSON field   | Meaning                              |
|--------------|--------------------------------------|
| `sentence_en`| English finding sentence             |
| `boxes`      | Reader 1 boxes (normalized `[x1,y1,x2,y2]`) |
| `extra_boxes`| Reader 2 boxes (normalized `[x1,y1,x2,y2]`) |
| `labels`     | Finding label(s)                     |

We keep only positive findings where *both* readers contributed at
least one valid box (≈5,242 sentences in this release).

## Pipeline

```
1.  load_padchest_gr.py        →  data/processed/samples_with_two_readers.csv
2.  medgemma_uncertainty.py    →  data/processed/medgemma_uncertainty_scores.jsonl
                                  data/processed/rule_uncertainty_scores.jsonl
3.  compute_iou.py             →  reader_iou per sample (mask IoU)
4-6,9.  analyze_results.py     →  outputs/group_statistics.json
                                  outputs/statistical_tests.json
                                  outputs/per_finding_analysis.csv
                                  outputs/regression_results.json
                                  outputs/baseline_comparison.json
                                  outputs/examples_*.csv
                                  data/processed/samples_with_uncertainty_and_iou.csv
7.  plot_results.py            →  figures/iou_by_uncertainty_group.png
                                  figures/iou_histogram_by_group.png
                                  figures/example_grid.png
```

End-to-end runner: `src/run_pipeline.py`.

### IoU computation

For each sample we rasterize reader 1's and reader 2's boxes into a
binary mask at `MASK_GRID_SIZE × MASK_GRID_SIZE` and compute
`|m1 ∩ m2| / |m1 ∪ m2|`.  Union-mask IoU is used because either reader
may draw multiple boxes for the same finding.

### MedGemma uncertainty scorer

`src/medgemma_uncertainty.py` loads `google/medgemma-4b-it` from Hugging
Face and calls it with a strict-JSON prompt asking for the schema:

```json
{
  "uncertainty_label": "certain" | "uncertain",
  "confidence": float,
  "uncertainty_triggers": [string],
  "reason": string
}
```

Generation is deterministic (`temperature=0`).  Output is parsed; on
failure we retry once, then mark the sample `parse_failed` and skip it.
Every result is written to a JSONL cache so the run resumes after
interruption.

A rule-based baseline (`rule_based_classify`) uses a hedge-word
dictionary (`possible`, `subtle`, `cannot exclude`, …) and is always
run.  This gives us:

* Step 9's MedGemma-vs-baseline comparison (agreement rate, Cohen's κ).
* A working pipeline result even when MedGemma is not installed.

## Quick start

### Option A — Google Colab (recommended for MedGemma)

Open `project/notebooks/medgemma_pipeline_colab.ipynb` in Colab
(`Runtime → Change runtime type → A100 GPU`).  The notebook:

1. Clones this repo from GitHub.
2. Installs `transformers>=4.50`, etc.
3. Asks for your Hugging Face token (you must have already accepted the
   MedGemma license at https://huggingface.co/google/medgemma-4b-it).
4. Runs `python src/run_pipeline.py --scorer medgemma --batch_size 16`
   end-to-end (≈10–20 min on an A100).
5. Renders all stats, tables, and figures inline in the notebook.

### Option B — Local

```bash
cd project
pip install -r requirements.txt

# MedGemma is the default scorer
python src/run_pipeline.py --scorer medgemma --batch_size 16

# (optional) rule-based smoke test
python src/run_pipeline.py --scorer rule
```

`--limit N` runs only the first N samples (useful for debugging).
`--skip_load` reuses `samples_with_two_readers.csv` if it exists.
`--device cuda|mps|cpu` forces a particular device for MedGemma.
`--batch_size N` controls the MedGemma generation batch (16 fits on A100 40GB).

## Outputs

After a successful run you will have:

* `outputs/group_statistics.json` — n, mean, median, std, 95% bootstrap
  CI for each group plus `delta = mean_iou(certain) − mean_iou(uncertain)`.
* `outputs/statistical_tests.json` — Mann-Whitney U (one- and two-sided),
  bootstrap CI for the delta, and a permutation p-value.
* `outputs/per_finding_analysis.csv` — control analysis per finding label.
* `outputs/regression_results.json` — OLS
  `reader_iou ~ is_uncertain + log(union_area) + finding_label`,
  reporting the coefficient on `is_uncertain` and its t-stat / p-value.
* `outputs/baseline_comparison.json` — group stats for the rule-based
  labels, plus agreement / Cohen's κ vs MedGemma.
* `outputs/examples_uncertain_low_iou.csv`,
  `outputs/examples_certain_high_iou.csv` — top-k illustrative cases.
* `figures/iou_by_uncertainty_group.png` — boxplot of reader IoU.
* `figures/iou_histogram_by_group.png` — overlaid histograms.
* `figures/example_grid.png` — 2×2 panel of example boxes.

## Example result (rule-based labels, 5,242 samples)

| group     | n     | mean IoU | median IoU |
|-----------|-------|----------|------------|
| certain   | 4,922 | 0.502    | 0.531      |
| uncertain |   320 | 0.445    | 0.468      |

* delta = 0.057, 95% bootstrap CI [0.034, 0.081]
* Mann-Whitney U one-sided (certain > uncertain) p ≈ 1e-6
* Permutation p ≈ 2e-4

The direction matches the hypothesis: uncertain sentences correspond to
lower reader-reader spatial agreement.  Per-finding subgroup analysis
(`per_finding_analysis.csv`) shows the effect is not uniform across
labels.  Once MedGemma labels are in place, rerunning the analysis with
`--scorer medgemma` produces a parallel set of statistics that can be
compared via `outputs/baseline_comparison.json`.

## File layout

```
project/
  data/
    raw/grounded_reports_20240819.json
    processed/
      samples_with_two_readers.csv
      rule_uncertainty_scores.jsonl
      medgemma_uncertainty_scores.jsonl
      samples_with_uncertainty_and_iou.csv
  outputs/
    group_statistics.json
    statistical_tests.json
    per_finding_analysis.csv
    regression_results.json
    baseline_comparison.json
    examples_uncertain_low_iou.csv
    examples_certain_high_iou.csv
  figures/
    iou_by_uncertainty_group.png
    iou_histogram_by_group.png
    example_grid.png
  src/
    config.py
    load_padchest_gr.py
    medgemma_uncertainty.py
    compute_iou.py
    analyze_results.py
    plot_results.py
    run_pipeline.py
  requirements.txt
  README.md
```

## Notes on scope

* This first experiment does **not** require BioViL-T or MedSAM and does
  not generate any model masks.  It establishes the foundational signal:
  whether **report-language** uncertainty corresponds to **human
  spatial** disagreement.
* If this signal is positive (it is, in the baseline run above), the
  next stage is to plug in BioViL-T + MedSAM to produce model
  segmentations and ask whether model uncertainty tracks the same
  human-disagreement signal.
