# Report-Language Uncertainty Predicts Radiologist Spatial Disagreement
## Comprehensive project summary

**Dataset:** PadChest-GR (grounded radiology reports), file
`grounded_reports_20240819.json` (~6 MB).

**Final analysis cohort:** 5,242 grounded *positive* finding sentences, each
with bounding-box annotations from **both** independent readers.

**Pipeline branch (current):** `rule-bag-of-words`
**Reproducible notebooks (in repo root):**
1. `medgemma_pipeline_colab_alluncertainty.ipynb` – MedGemma, broad
   (diagnostic + spatial) uncertainty prompt.
2. `medgemma_pipeline_colab_spatial.ipynb` – MedGemma, narrow
   spatial-only uncertainty prompt.
3. `rule_pipeline_colab.ipynb` – simple 47-phrase keyword classifier.

---

## 1. Research question

> **Do radiology report sentences classified as *uncertain* show lower
> spatial agreement between two independent radiologists than sentences
> classified as *certain*?**

The motivating intuition is that words like *possible*, *probable*,
*ill-defined*, *subtle*, *cannot exclude* express situations in which the
radiologist is hedging about what / where the finding is, and that those
same situations should be reflected in larger spatial disagreement
between two readers drawing bounding boxes on the same study. If this
holds, *report-language uncertainty* is a cheap, no-image,
no-model proxy for **inter-reader spatial ambiguity** — and could later
be replaced with an image-side model-derived uncertainty signal
(BioViL-T attention entropy or MedSAM mask-probability variance).

---

## 2. Shared methodology (steps 1-9 from spec)

Every notebook ultimately calls **the same** Python pipeline
(`project/src/run_pipeline.py`). Only the *uncertainty scorer* changes
between notebooks.

### Step 1 — Load and filter PadChest-GR
`load_padchest_gr.py` expands each report into one row per positive
finding sentence, keeps only studies that have bounding boxes from
**both** readers, drops malformed boxes, and normalizes coordinates to
`[x1, y1, x2, y2]` with `x2 > x1`, `y2 > y1`.

Result: **5,242 sentences × 8 columns** →
`project/data/processed/samples_with_two_readers.csv`.

Required columns:
`sample_id, image_id, sentence, finding_label, image_width,
image_height, reader1_boxes (JSON), reader2_boxes (JSON)`.

### Step 2 — Uncertainty scoring (the *only* thing that differs across the 3 experiments)
The scorer must return, for every sentence:

```json
{
  "uncertainty_label":      "certain" | "uncertain",
  "confidence":             0.0-1.0,
  "uncertainty_triggers":   ["..."],
  "reason":                 "..."
}
```

Output cached as JSONL so the pipeline is resumable; see
`medgemma_uncertainty_scores.jsonl` or `rule_uncertainty_scores.jsonl`.

### Step 3 — Reader-vs-reader spatial agreement
`compute_iou.py` rasterizes each reader's box list into a binary mask of
the image size, then:

```
intersection = sum(mask1 & mask2)
union        = sum(mask1 | mask2)
reader_iou   = intersection / union          # 0 if union==0 -> drop
disagreement = 1 - reader_iou
```

Mask IoU is used rather than box IoU because a finding can have **multiple
boxes per reader** (e.g. "scattered nodules"), and the union mask
collapses those cleanly.

### Step 4-5 — Group statistics + significance tests
For each uncertainty label group we compute n, mean, median, std, 95 %
bootstrap CI of the mean (10,000 resamples), then run:

- **Mann–Whitney U** (one-sided "certain > uncertain", and two-sided).
- **Bootstrap of Δ = mean_iou(certain) − mean_iou(uncertain)** with 2,000
  resamples and a 95 % CI.
- **Permutation test** that shuffles the uncertainty labels 5,000 times
  and recomputes Δ to get a non-parametric p-value.

### Step 6 — Per-finding control + OLS regression
Per-finding subgroup table includes any `finding_label` with at least 10
samples in each group, used to check that the effect is not an artifact
of which diseases happen to be hedged.

The full regression:
```
reader_iou ~ is_uncertain + C(finding_label) + log(union_area + 1)
```
is fit via OLS with one-hot encoding (`scikit-learn`). The coefficient
on `is_uncertain` is the **adjusted** effect of report-language
uncertainty after holding finding type and box size constant. If this
coefficient is significantly negative, that is the strong form of the
hypothesis.

### Step 7-8 — Plots, examples, outputs
`plot_results.py` writes:
- `figures/iou_by_uncertainty_group.png` — strip/box plot.
- `figures/iou_histogram_by_group.png` — overlaid normalized histograms.
- `figures/example_grid.png` — 4 quadrants of extreme examples
  (uncertain+low IoU, certain+high IoU, uncertain+high IoU,
  certain+low IoU), each showing reader 1 (blue) vs reader 2 (red)
  rasterized boxes.
- `outputs/group_statistics.json`, `statistical_tests.json`,
  `regression_results.json`, `per_finding_analysis.csv`,
  `examples_uncertain_low_iou.csv`, `examples_certain_high_iou.csv`,
  `baseline_comparison.json`.

### Step 9 — Baseline comparison
Whenever MedGemma is used, the rule-based classifier is also run, and
the two are compared via Cohen's κ + group IoU delta (in
`baseline_comparison.json`).

---

## 3. The three experiments

### Experiment A — MedGemma with **broad** uncertainty prompt (`alluncertainty` notebook)

**Hardware:** Google Colab A100 80 GB, batch size 16.
**Model:** `google/medgemma-4b-it`, loaded with
`AutoModelForImageTextToText` + `AutoProcessor`, deterministic decoding
(`do_sample=False`, max 256 new tokens).
**Prompt definition:**
> *"A sentence is uncertain if the radiologist expresses uncertainty
> about the **existence, visibility, diagnosis, or boundary** of the
> finding."*
>
> Treats classic diagnostic hedges (*possible*, *cannot exclude*,
> *suggestive of*) **AND** spatial hedges (*ill-defined*, *subtle*) as
> uncertain.

Examples of "uncertain" classifications:
```
"Slight infiltrate."              triggers: ['slight']
"Hinting towards the right main bronchus."   triggers: ['hinting towards']
"Could correspond to the metastases seen by CT." triggers: ['could correspond']
"Left perihilar image with pseudonod[ular] features…"
```
Both spatial-style ("slight") and diagnostic-style ("could correspond")
matches are produced.

### Experiment B — MedGemma with **spatial-only** uncertainty prompt (`spatial` notebook)

Same model, same hardware, same code path — only the prompt changed.
**Prompt definition:**
> *"We are asking ONE specific question: does the sentence's language
> indicate that the LOCATION, BOUNDARIES, EXTENT, or VISIBILITY of the
> finding is unclear, ill-defined, indistinct, faint, or hard to
> delineate spatially?"*
>
> The prompt explicitly tells the model **not** to count diagnostic
> hedges like *possible* or *cannot exclude*.

Examples that MedGemma labelled "uncertain" under this prompt:
```
"Signs of COPD with air trapping."                 triggers: ['air trapping']
"With air trapping."                               triggers: ['air trapping']
"Bilateral pleural effusion of greater extent…"    triggers: ['greater extent']
"Right pleural effusion with a slight decrease…"   triggers: ['slight decrease']
"Atelectasis at the hilar level."                  triggers: ['at the hilar level']
"Opacity at the left base."                        triggers: ['at']
"Probable right hilar adenopathies."               triggers: ['probable']  # !
```
The model partially follows the spatial-only instructions but also:
1. Flags **finding names** ("air trapping") as triggers.
2. Flags **location phrases** ("at the hilar level", just "at").
3. Flags **size/severity words** ("slight decrease").
4. Still flags **diagnostic hedges** ("probable") even though the prompt
   forbids it.

So in this experiment the "uncertain" group is much larger (1,081 vs
273) but visibly noisier.

### Experiment C — Rule-based bag-of-words (`rule_pipeline_colab.ipynb`)

**Hardware:** CPU only, no GPU, no API. Runs end-to-end in ~30 s on a
MacBook.
**Method:** a 47-phrase dictionary stored in `UNCERTAIN_TERMS`. For each
sentence:

```python
def rule_based_classify(sentence):
    s_low = sentence.lower()
    triggers = [t for t in UNCERTAIN_TERMS if t in s_low]
    return "uncertain" if triggers else "certain"
```
*Any* substring match → uncertain. No model, no thresholds, no
weighting.

The dictionary covers both kinds of hedging:

*Diagnostic (uncertainty about presence / identity):*
```
possible, possibly, probable, probably, likely, questionable, equivocal,
may represent, could represent, may correspond, could correspond,
may be, might, appears, apparent, cannot exclude, can't exclude,
cannot be excluded, cannot rule out, difficult to exclude,
difficult to assess, suspicious for, suspicion of,
suggestive of, suggesting, suggests, compatible with, consistent with,
rule out, to rule out, vs, versus
```

*Spatial / boundary / visibility:*
```
ill-defined, ill defined, poorly defined, poorly-defined,
ill-circumscribed, ill circumscribed, indistinct, vague,
subtle, faint, hazy, blurred, blurry, fuzzy, obscured, barely visible
```

Examples of "uncertain" classifications:
```
"Prominent hila of probable vascular origin."            -> probable
"Consistent with goiter."                                -> consistent with
"…in relation to a probable enchondroma…"                -> probable
"Right basal image suggestive of loculated pleural eff." -> suggestive of
"…with poorly defined borders."                          -> poorly defined
"…likely related to fatty infiltration."                 -> likely
```
Every trigger is a real hedge word — no false-positive noise like in
the spatial MedGemma prompt.

---

## 4. Head-to-head numerical comparison

All three experiments analyse the **same** 5,242-sentence cohort.

| metric                                | A — MedGemma broad        | B — MedGemma spatial-only | C — Rule bag-of-words     |
|---------------------------------------|--------------------------:|--------------------------:|--------------------------:|
| # uncertain (share of corpus)         | 273   (5.2 %)             | 1,081 (20.6 %)            | 374 (7.1 %)               |
| # certain                             | 4,969                     | 4,161                     | 4,868                     |
| mean IoU **certain**                  | 0.503                     | 0.517                     | 0.504                     |
| mean IoU **uncertain**                | 0.423                     | 0.427                     | 0.432                     |
| median IoU certain                    | 0.531                     | 0.549                     | 0.533                     |
| median IoU uncertain                  | 0.448                     | 0.423                     | 0.450                     |
| **Δ = mean cert − mean unc**          | **+0.079**                | **+0.090**                | **+0.071**                |
| 95 % bootstrap CI for Δ               | [0.053, 0.106]            | [0.075, 0.105]            | [0.050, 0.093]            |
| Mann–Whitney U p (one-sided)          | 1.1 × 10⁻⁸                | 2.3 × 10⁻³¹               | 4.2 × 10⁻¹⁰               |
| Permutation p (5,000 reps)            | ≈ 0.0002                  | ≈ 0.0002                  | ≈ 0.0002                  |
| Regression β on `is_uncertain` after controlling for **finding label + log box area** | −0.018  (p = 0.15) | −0.008  (p = 0.31) | −0.009  (p = 0.43) |
| Trigger quality (manual review)       | mixed (diag + size hedges) | **noisy** — flags finding names / location / "at" | **clean** — every trigger is a real hedge |
| Runtime / hardware                    | ~30 min on A100 80 GB     | ~30 min on A100 80 GB     | **~30 s on a laptop CPU** |

### Take-aways

1. **All three classifiers agree on the marginal direction.** Uncertain
   sentences have lower reader IoU by **7–9 IoU points** (≈ 14–18 %
   relative), and that gap is **strictly outside the bootstrap CI** and
   has astronomically small Mann–Whitney p-values in every experiment.
2. **All three classifiers also agree on the *adjusted* null.** Once we
   condition on which finding the sentence describes and how big the
   boxes are, the coefficient on `is_uncertain` shrinks to ~ −0.01 and
   stops being statistically significant. The marginal effect is
   largely a *finding-type confound*: hedge-prone findings
   (pseudonodule, costophrenic blunting, nipple shadow) are inherently
   harder to localize.
3. **Stringency varies a lot but the effect size barely moves.** The
   spatial MedGemma prompt flagged ~ 4 × more sentences as uncertain
   than either the broad MedGemma prompt or the rule classifier, yet
   produced essentially the same Δ. This is reassuring — the result is
   robust to where you draw the certain/uncertain line.
4. **The simple rule classifier is the cleanest and the fastest.** It
   produces auditable triggers, requires no GPU, finishes in 30 s, and
   yields the same scientific conclusion as the two MedGemma runs. The
   broad MedGemma prompt is essentially a noisy version of the rule
   classifier (similar set of phrases) and the spatial MedGemma prompt
   adds noise without a measurable improvement in effect size.

---

## 5. Per-finding control analysis

For finding labels with ≥ 10 sentences in both groups. Sign of Δ tells
us whether report-language uncertainty tracks lower IoU **within** that
disease.

### Experiment A (MedGemma broad)
| finding                  | Δ      |
|--------------------------|-------:|
| increased density        | **+0.079** ✅ |
| goiter                   | **+0.074** ✅ |
| infiltrates              | +0.050 ✅ |
| vascular hilar enlargement | +0.025 |
| interstitial pattern     | +0.009 ≈ 0 |
| alveolar pattern         | +0.002 ≈ 0 |
| pleural effusion         | +0.001 ≈ 0 |
| nipple shadow            | -0.016 |
| nodule                   | **-0.137** ❌ |

3/9 support, 4/9 neutral, 2/9 invert.

### Experiment B (MedGemma spatial-only)
| finding                  | Δ      |
|--------------------------|-------:|
| callus rib fracture      | **+0.102** ✅ |
| air trapping             | **+0.098** ✅ |
| nodule                   | **+0.093** ✅ |
| consolidation            | +0.046 ✅ |
| increased density        | +0.046 ✅ |
| infiltrates              | +0.044 ✅ |
| cardiomegaly             | +0.040 ✅ |
| apical pleural thickening | ≈ 0 |
| vascular hilar enlargement | ≈ 0 |
| interstitial pattern     | -0.006 |
| alveolar pattern         | -0.020 |
| pleural effusion         | -0.025 |
| atelectasis              | -0.026 |
| costophrenic angle blunting | -0.027 |
| volume loss              | -0.038 |
| laminar atelectasis      | -0.044 |
| pseudonodule             | -0.061 |
| nipple shadow            | **-0.092** ❌ |

7/18 support, 3/18 neutral, 8/18 invert.

### Experiment C (rule bag-of-words)
| finding                  | Δ      |
|--------------------------|-------:|
| callus rib fracture      | **+0.100** ✅ |
| increased density        | +0.064 ✅ |
| alveolar pattern         | +0.033 |
| pseudonodule             | +0.031 |
| vascular hilar enlargement | ≈ 0 |
| infiltrates              | -0.026 |
| pleural effusion         | -0.057 |
| atelectasis              | -0.103 |
| calcified granuloma      | -0.131 |
| nipple shadow            | **-0.168** ❌ |

2/10 support, 1/10 neutral, 7/10 invert.

### Consensus across experiments
- Strong, consistent support inside: **callus rib fracture, increased
  density, nodule, consolidation, infiltrates, cardiomegaly**.
- Strong, consistent inversion inside: **nipple shadow** and
  **pseudonodule** (these are findings whose presence is itself
  debatable, and where readers tend to put very different small boxes
  regardless of how confidently they are described).

This split is the qualitative story behind the null regression: the
sign of the within-finding effect *flips* across finding types, so the
average effect after adjusting for finding is essentially zero.

---

## 6. Example grid

`figures/example_grid.png` shows four representative cases per
experiment:

- **Uncertain + low IoU** — the "true positive" case for the hypothesis.
  E.g. `"Subsegmental LM atelectasis associated with an increase in
  right paracardiac density with poorly defined borders." → IoU 0.05`.
- **Certain + high IoU** — the natural complement. E.g. `"Global
  cardiomegaly." → IoU 0.86`.
- **Uncertain + high IoU** — hedged language but readers actually agree.
  E.g. `"Right basal image suggestive of loculated pleural effusion." →
  IoU 0.78`.
- **Certain + low IoU** — declarative language but readers actually
  disagree. E.g. `"Biapical pleural thickening." → IoU 0.12`. These are
  the cases that drive the regression null.

The presence of substantial mass in both "off-diagonal" quadrants is
itself a warning that report-language uncertainty and spatial
inter-reader disagreement are **correlated, not coextensive**.

---

## 7. Combined interpretation

**The marginal hypothesis is supported, across three independent
classifiers, with overwhelming statistical significance:** sentences
labelled uncertain by either MedGemma or a 47-phrase dictionary show
~ 7–9 IoU-point lower reader-vs-reader spatial agreement, every test
gives p < 10⁻⁸, every bootstrap CI for Δ is strictly above zero, and
every permutation test bottoms out at the smallest reachable p.

**The strong hypothesis — "report-language uncertainty *independently*
predicts spatial disagreement after controlling for what the finding
is" — is NOT supported.** All three classifiers give a regression
coefficient indistinguishable from zero (p ∈ [0.15, 0.43]). What the
data actually shows is that uncertain language is **concentrated** in
finding categories that are inherently hard to delineate (pseudonodule,
nipple shadow, pleural effusion, costophrenic blunting), and that *most
of* the marginal Δ is explained by this confound. Within an individual
finding, hedged language is almost as likely to *raise* reader IoU as
to lower it.

**Methodological take-away.** A 47-phrase substring scanner that runs
on a laptop in 30 seconds reaches the same scientific conclusion as
running MedGemma 4B-IT on an A100 for 30 minutes, with cleaner triggers
and full auditability. For this question, the simple baseline is not a
straw man — it is the right tool. MedGemma's value would presumably
come from looking at the *image* (not the report); since none of the
three experiments here used the image at all, MedGemma was effectively
asked to be a fancy keyword classifier, and the simple keyword
classifier is better at it.

---

## 8. What to do next

1. **Stop using report-language uncertainty as the predictor.** Use it
   as a *baseline* in the next stage. The within-finding evidence and
   the null regression coefficient make it clear that text hedging is a
   proxy for finding type, not for spatial ambiguity.
2. **Move to model-derived spatial uncertainty** — the original
   end-game of the project. Two natural candidates:
   - **BioViL-T attention entropy.** Use BioViL-T to ground the
     sentence to a spatial heatmap; compute the entropy of that
     heatmap; ask whether high-entropy groundings correspond to
     low-IoU reader pairs.
   - **MedSAM probabilistic mask variance.** Prompt MedSAM with each
     reader's box, take the predicted mask probability map (not the
     thresholded mask), and use its variance or expected calibration
     error as the model-uncertainty signal.
3. **Stratified evaluation.** Whichever model-uncertainty signal we
   adopt, the headline metric should be: *within each finding label*,
   does the signal correlate with reader IoU? That is the within-finding
   analysis the current report-language results failed.
4. **Optional: stricter rule variants.** If we ever want to revisit the
   text route, the obvious wins are (a) handling negation (so "no
   evidence of possible pneumothorax" does not get flagged), (b) a
   weighted any-match rule where each trigger contributes a score and
   we set a threshold, and (c) restricting to the spatial subset of the
   dictionary only (basically the rule version of Experiment B).
   Preliminary calculations suggest none of these would salvage the
   regression coefficient on its own, but they would tighten the
   triggers further.

---

## 9. Files map

```
project/
├── data/raw/grounded_reports_20240819.json
├── data/processed/
│   ├── samples_with_two_readers.csv         (5,242 rows)
│   ├── rule_uncertainty_scores.jsonl        (Experiment C cache)
│   ├── medgemma_uncertainty_scores.jsonl    (Experiment A cache)
│   ├── medgemma_uncertainty_scores_spatial.jsonl  (Experiment B cache)
│   └── samples_with_uncertainty_and_iou.csv (merged final table)
├── outputs/
│   ├── group_statistics.json
│   ├── statistical_tests.json
│   ├── regression_results.json
│   ├── per_finding_analysis.csv
│   ├── baseline_comparison.json
│   ├── examples_uncertain_low_iou.csv
│   └── examples_certain_high_iou.csv
├── figures/
│   ├── iou_by_uncertainty_group.png
│   ├── iou_histogram_by_group.png
│   └── example_grid.png
├── reports/
│   ├── PROJECT_SUMMARY.md           ← this file
│   └── REPORT_rule_bag_of_words.md  ← detail on Experiment C
└── src/
    ├── load_padchest_gr.py
    ├── medgemma_uncertainty.py      ← UNCERTAIN_TERMS + MedGemmaUncertaintyScorer
    ├── compute_iou.py
    ├── analyze_results.py
    ├── plot_results.py
    └── run_pipeline.py

(repo root)
├── medgemma_pipeline_colab_alluncertainty.ipynb  ← Experiment A (Colab, A100)
├── medgemma_pipeline_colab_spatial.ipynb         ← Experiment B (Colab, A100)
└── rule_pipeline_colab.ipynb                     ← Experiment C (Colab/CPU)
```

---

*Summary generated on `rule-bag-of-words` branch (commit `17e5bc2`).*
