"""Configuration for the report-language uncertainty pipeline."""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Input file: PadChest-GR grounded reports JSON
RAW_GROUNDED_REPORTS = RAW_DIR / "grounded_reports_20240819.json"

# Processed outputs
SAMPLES_TWO_READERS_CSV = PROCESSED_DIR / "samples_with_two_readers.csv"
# Spatial-uncertainty prompt has its own cache so prior diagnostic-uncertainty
# runs are not overwritten or mixed.  See PROMPT_TEMPLATE in
# medgemma_uncertainty.py.
MEDGEMMA_SCORES_JSONL = PROCESSED_DIR / "medgemma_uncertainty_scores_spatial.jsonl"
RULE_SCORES_JSONL = PROCESSED_DIR / "rule_uncertainty_scores.jsonl"
SAMPLES_WITH_IOU_CSV = PROCESSED_DIR / "samples_with_uncertainty_and_iou.csv"

# Output artifacts
GROUP_STATS_JSON = OUTPUTS_DIR / "group_statistics.json"
STAT_TESTS_JSON = OUTPUTS_DIR / "statistical_tests.json"
EXAMPLES_UNCERTAIN_LOW_IOU = OUTPUTS_DIR / "examples_uncertain_low_iou.csv"
EXAMPLES_CERTAIN_HIGH_IOU = OUTPUTS_DIR / "examples_certain_high_iou.csv"
PER_FINDING_ANALYSIS_CSV = OUTPUTS_DIR / "per_finding_analysis.csv"
REGRESSION_RESULTS_JSON = OUTPUTS_DIR / "regression_results.json"
BASELINE_COMPARISON_JSON = OUTPUTS_DIR / "baseline_comparison.json"

# Figures
FIG_IOU_BY_GROUP = FIGURES_DIR / "iou_by_uncertainty_group.png"
FIG_IOU_HISTOGRAM = FIGURES_DIR / "iou_histogram_by_group.png"
FIG_EXAMPLE_GRID = FIGURES_DIR / "example_grid.png"

# ---- Mask / IoU settings ---------------------------------------------------
# PadChest-GR boxes are stored as normalized [0,1] coordinates [x1,y1,x2,y2].
# We rasterize them onto a fixed resolution grid for mask IoU. The grid is
# only used to compute IoU; it does not need to match the original image size.
MASK_GRID_SIZE = 1024  # pixels per side

# ---- MedGemma settings -----------------------------------------------------
MEDGEMMA_MODEL_NAME = "google/medgemma-4b-it"
MEDGEMMA_TEMPERATURE = 0.0
MEDGEMMA_MAX_NEW_TOKENS = 256

# ---- Analysis settings -----------------------------------------------------
BOOTSTRAP_N = 2000
PERMUTATION_N = 5000
RANDOM_SEED = 42
MIN_PER_FINDING_GROUP = 10  # min samples per (finding_label, group) for control analysis
