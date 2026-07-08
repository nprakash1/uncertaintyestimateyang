"""Build project/notebooks/rosalia_padchest_gr_colab.ipynb (two-phase).

Phase A (MedGemma): classify every UNIQUE PadChest-GR finding label into one
of ROSALIA's 7 lesion concepts, or 'none' (drop). MedGemma-4b-it is a Gemma-3
image-text-to-text model that requires transformers>=4.50.

Phase B (ROSALIA): reinstall transformers==4.31.0 (required by LISA/ROSALIA),
restart the runtime, load the cached mapping, and run instruction-guided
segmentation + the uncertainty-stratified IoU eval.

The two phases MUST be separated by a runtime restart because their
transformers version requirements are mutually exclusive. The finding->concept
map is persisted to /content AND GCS so it survives the restart.
"""
from pathlib import Path
import json
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = []

# =============================================================================
# TITLE
# =============================================================================
cells.append(md(r"""# ROSALIA on PadChest-GR — uncertainty-stratified IoU eval (two-phase)

For every (image, grounded finding) pair in PadChest-GR whose finding maps to
one of ROSALIA's **7 supported lesion concepts**
(`cardiomegaly, pneumonia, atelectasis, opacity, consolidation, edema, effusion`),
we run ROSALIA (LISA-7B + SAM-H fine-tuned on 1.1M MIMIC-ILS pairs) to get a
predicted lesion mask, then compute a bundle of IoUs vs the two annotators'
ground-truth boxes, aggregated by the rule-based spatial-uncertainty label.

## ⚠️ This notebook has TWO phases separated by a runtime restart

**Why:** MedGemma-4b-it is a **Gemma-3** model that needs `transformers>=4.50`,
but ROSALIA/LISA needs **exactly** `transformers==4.31.0`. The two cannot live
in the same runtime. So:

* **Phase A — MedGemma classifier.** Uses MedGemma to map every *unique*
  PadChest-GR `finding_label` to one of the 7 ROSALIA concepts (or `none`,
  which is dropped). Runs once, saves `finding_concept_map.json` to `/content`
  and GCS.
* **↻ Restart runtime + reinstall.**
* **Phase B — ROSALIA.** Loads the cached map, runs segmentation + IoU eval.

**Run order:** Phase A cells → Cell B1 (installs ROSALIA deps) → **Restart
runtime** → Phase B cells (B2 onward).

**Runtime:** GPU with ≥16 GB VRAM (L4/A100 recommended; ROSALIA-7B bf16 is
~15 GB, borderline OOM on a T4). MedGemma-4b needs only ~9 GB, so Phase A is
fine on any GPU.
"""))

# =============================================================================
# Cell 0 — GCS extraction (shared, idempotent)
# =============================================================================
cells.append(md(r"""## Cell 0 — one-time GCS extraction of the split-zip PadChest-GR archive

Idempotent: if `gs://${BUCKET}/images/` is already populated, it no-ops. You
can run this in either phase (it doesn't touch transformers).
"""))

cells.append(code(r"""BUCKET      = "yang-padchest-gr"
GCS_PROJECT = "yang-uncertainty-eval"
SRC_PREFIX  = "Padchest_GR_files"
DST_PREFIX  = "images"

from google.colab import auth
auth.authenticate_user()
get_ipython().system(f"gcloud config set project {GCS_PROJECT} -q")
print("Active identity:")
get_ipython().system("gcloud auth list --filter=status:ACTIVE --format='value(account)'")

import subprocess
r = subprocess.run(
    f"gcloud storage ls gs://{BUCKET}/ 2>&1 | head -10",
    capture_output=True, text=True, shell=True,
)
print(f"\nPreflight `gcloud storage ls gs://{BUCKET}/`:")
print(r.stdout)
if "ERROR" in r.stdout or "403" in r.stdout or "404" in r.stdout or "401" in r.stdout:
    raise RuntimeError(
        f"Cannot read gs://{BUCKET}/ as the active account. Check bucket name, "
        f"signed-in account, and project '{GCS_PROJECT}'."
    )

from google.cloud import storage
_client = storage.Client(project=GCS_PROJECT)
_bucket = _client.bucket(BUCKET)

def gcs_has_images():
    return len(list(_bucket.list_blobs(prefix=f"{DST_PREFIX}/", max_results=1))) > 0

if gcs_has_images():
    print(f"\ngs://{BUCKET}/{DST_PREFIX}/ already populated — skipping extraction.")
else:
    print("\nExtracting PadChest-GR from split-zip parts. One-time slow step.")
    !mkdir -p /content/zips /content/images /content/extracted
    get_ipython().system(
        f"gcloud storage cp 'gs://{BUCKET}/{SRC_PREFIX}/PadChest_GR.zip.*' /content/zips/")
    !ls /content/zips/PadChest_GR.zip.* | sort > /content/zip_parts.txt
    !cat $(cat /content/zip_parts.txt | tr '\n' ' ') > /content/full.zip
    !rm -f /content/zips/PadChest_GR.zip.*
    !unzip -q -o /content/full.zip -d /content/extracted/
    !rm -f /content/full.zip
    !find /content/extracted -type f -name '*.png' -print0 | xargs -0 -I{} cp {} /content/images/
    get_ipython().system(
        f"gcloud storage rsync /content/images/ gs://{BUCKET}/{DST_PREFIX}/ --recursive")
    !rm -rf /content/extracted /content/images
    print("Extraction + upload complete.")
"""))

# =============================================================================
# PHASE A — MedGemma classifier
# =============================================================================
cells.append(md(r"""# ══════════════════ PHASE A — MedGemma finding classifier ══════════════════

Run Cells A1 → A3 in a **fresh runtime with modern transformers** (Colab's
default). This produces `finding_concept_map.json`. When A3 finishes, go to
Cell B1.
"""))

cells.append(md(r"""## Cell A1 — install MedGemma deps (modern transformers)

MedGemma-4b-it is Gemma-3 (`image-text-to-text`) and needs `transformers>=4.50`
and a recent `accelerate`. Colab usually has these, but we pin to be safe.
"""))

cells.append(code(r"""!pip -q install --upgrade 'transformers>=4.50' 'accelerate>=0.30' gcsfs google-cloud-storage huggingface_hub pandas

import transformers, torch
print("transformers", transformers.__version__)
print("torch       ", torch.__version__)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 50), \
    "Need transformers>=4.50 for MedGemma (Gemma-3). Restart runtime if it just upgraded."
"""))

cells.append(md(r"""## Cell A2 — authenticate (Google + Hugging Face)

* Google auth → save the map to GCS.
* HF login → `google/medgemma-4b-it` is a **gated** model; you must have
  accepted its license at https://huggingface.co/google/medgemma-4b-it and log
  in with a token that has access.
"""))

cells.append(code(r"""from google.colab import auth
auth.authenticate_user()

from huggingface_hub import notebook_login
notebook_login()
"""))

cells.append(md(r"""## Cell A3 — classify each unique finding label with MedGemma

1. Download the samples CSV from GitHub.
2. Collect the **unique** `finding_label`s + one representative sentence each
   (so MedGemma sees context, e.g. "opacity consistent with pneumonia").
3. Ask MedGemma to pick **exactly one** of
   `{cardiomegaly, pneumonia, atelectasis, opacity, consolidation, edema,
   effusion, none}`. `none` = "not one of ROSALIA's 7 → will be dropped".
   Few-shot examples anchor the behavior; parsing is strict (invalid →
   `none`, the safe default).
4. Save the mapping to `/content/finding_concept_map.json` and
   `gs://${BUCKET}/outputs/rosalia_v1/finding_concept_map.json` so it survives
   the Phase-B runtime restart.
5. Print a review table so you can eyeball MedGemma's choices.

This runs once over ~100-200 unique labels (not per sample), so it's fast.
"""))

cells.append(code(r"""import os, json, re, gc
import pandas as pd
import torch

BUCKET       = "yang-padchest-gr"
GCS_PROJECT  = "yang-uncertainty-eval"
MAP_GCS_PATH = f"{BUCKET}/outputs/rosalia_v1/finding_concept_map.json"
MAP_LOCAL    = "/content/finding_concept_map.json"

REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"
SAMPLES_CSV  = "/content/samples_with_uncertainty_and_iou.csv"
if not os.path.exists(SAMPLES_CSV):
    get_ipython().system(
        f"curl -fsSL '{REPO_RAW_URL}/project/data/processed/samples_with_uncertainty_and_iou.csv' -o '{SAMPLES_CSV}'")

samples = pd.read_csv(SAMPLES_CSV)
print(f"Loaded {len(samples)} rows; {samples['finding_label'].nunique()} unique finding labels")

# One representative sentence per unique label (first non-null occurrence)
rep = (samples.dropna(subset=["finding_label"])
              .groupby("finding_label")["sentence"]
              .first().to_dict())
unique_labels = sorted(rep.keys())

ROSALIA_LESIONS = ["cardiomegaly","pneumonia","atelectasis","opacity",
                   "consolidation","edema","effusion"]
ALLOWED = set(ROSALIA_LESIONS) | {"none"}

FEWSHOT = (
    "Examples:\n"
    "Finding: 'pleural effusion' | Sentence: 'blunting of the costophrenic angle' -> effusion\n"
    "Finding: 'cardiomegaly' | Sentence: 'the cardiac silhouette is enlarged' -> cardiomegaly\n"
    "Finding: 'alveolar pattern' | Sentence: 'alveolar opacities in the right base' -> opacity\n"
    "Finding: 'laminar atelectasis' | Sentence: 'linear atelectasis at left base' -> atelectasis\n"
    "Finding: 'pulmonary edema' | Sentence: 'signs of interstitial edema' -> edema\n"
    "Finding: 'consolidation' | Sentence: 'dense consolidation in right lower lobe' -> consolidation\n"
    "Finding: 'pulmonary nodule' | Sentence: 'a solitary nodule is seen' -> none\n"
    "Finding: 'pneumothorax' | Sentence: 'apical pneumothorax' -> none\n"
    "Finding: 'aortic elongation' | Sentence: 'unfolded thoracic aorta' -> none\n"
)

def build_prompt(label, sentence):
    return (
        "You are a radiology label normalizer. A downstream segmentation model "
        "can ONLY segment these 7 chest X-ray lesion types:\n"
        f"{', '.join(ROSALIA_LESIONS)}.\n\n"
        "Given a PadChest-GR finding label and an example sentence, respond with "
        "EXACTLY ONE word: the single best-matching lesion type from that list, "
        "or 'none' if the finding is not one of those 7 (e.g. nodule, mass, "
        "pneumothorax, fracture, aortic changes, hardware/devices, normal "
        "anatomy). Output only one lowercase word, no punctuation.\n\n"
        f"{FEWSHOT}\n"
        f"Finding: '{label}' | Sentence: '{str(sentence)[:200]}' ->"
    )

# --- load MedGemma (Gemma-3 image-text-to-text; needs transformers>=4.50) ---
from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL_NAME = "google/medgemma-4b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device} ...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tok = getattr(processor, "tokenizer", None) or processor
tok.padding_side = "left"
if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
    tok.pad_token = tok.eos_token
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

def classify(label, sentence):
    prompt = build_prompt(label, sentence)
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=6, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    raw = tok.batch_decode(gen, skip_special_tokens=True)[0]
    word = re.sub(r"[^a-z]", "", raw.strip().lower().split()[0]) if raw.strip() else "none"
    return word if word in ALLOWED else "none", raw.strip()

from tqdm.auto import tqdm
finding_concept_map = {}
raw_outputs = {}
for lbl in tqdm(unique_labels, desc="MedGemma classify"):
    concept, raw = classify(lbl, rep[lbl])
    finding_concept_map[lbl] = concept
    raw_outputs[lbl] = raw

# --- review table ---
review = (pd.DataFrame({
            "finding_label": list(finding_concept_map),
            "concept": list(finding_concept_map.values()),
            "medgemma_raw": [raw_outputs[k] for k in finding_concept_map],
            "n_samples": [int((samples["finding_label"]==k).sum()) for k in finding_concept_map],
         })
         .sort_values(["concept","n_samples"], ascending=[True, False]))
print("\n=== MedGemma finding -> ROSALIA concept mapping ===")
import pandas as _pd; _pd.set_option("display.max_rows", 300)
print(review.to_string(index=False))

n_kept = int((review["concept"]!="none").sum())
kept_samples = int(samples["finding_label"].map(finding_concept_map).isin(ROSALIA_LESIONS).sum())
print(f"\nUnique labels: {len(unique_labels)} | mapped to a concept: {n_kept} | 'none': {len(unique_labels)-n_kept}")
print(f"Samples kept (in-vocab): {kept_samples} / {len(samples)} ({100*kept_samples/len(samples):.1f}%)")
print("\nConcept distribution (by sample count):")
print(samples["finding_label"].map(finding_concept_map).value_counts().to_string())

# --- persist to /content + GCS (survives the Phase-B restart) ---
with open(MAP_LOCAL, "w") as f:
    json.dump(finding_concept_map, f, indent=2)
print(f"\nSaved {MAP_LOCAL}")
try:
    import gcsfs
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    with fs.open(MAP_GCS_PATH, "w") as f:
        f.write(json.dumps(finding_concept_map, indent=2))
    print(f"Saved gs://{MAP_GCS_PATH}")
except Exception as e:
    print(f"[warn] GCS save failed ({e}); the /content copy will still work if you don't delete the runtime.")

# --- free MedGemma VRAM before Phase B ---
del model, processor
gc.collect(); torch.cuda.empty_cache()
print("\nMedGemma unloaded. Proceed to Cell B1.")
"""))

# =============================================================================
# PHASE B — ROSALIA
# =============================================================================
cells.append(md(r"""# ══════════════════ PHASE B — ROSALIA segmentation ══════════════════

Cell B1 reinstalls the ROSALIA dependency stack (which **downgrades**
transformers to 4.31.0 and is incompatible with MedGemma). After B1 you MUST
**restart the runtime**, then run B2 onward.
"""))

cells.append(md(r"""## Cell B1 — install ROSALIA deps + clone repo, then RESTART

`transformers==4.31.0` is required by LISA's custom `LISAForCausalLM`. This
will uninstall the MedGemma-compatible transformers — that's expected. The
`finding_concept_map.json` from Phase A is already saved to disk + GCS, so it
survives the restart.
"""))

cells.append(code(r"""import os, sys, subprocess

# LISA/ROSALIA was originally pinned to transformers==4.31.0 + tokenizers 0.13.x.
# But tokenizers 0.13.x has NO prebuilt wheel for Python 3.12 (Colab's current
# default), and its Rust source no longer compiles cleanly with modern rustc.
#
# Solution: use tokenizers 0.14.1 (has prebuilt cp310/cp311/cp312 wheels) with
# transformers 4.34.1 (accepts tokenizers>=0.14,<0.15 and is known to run
# LISA-family models). --only-binary=:all: on tokenizers guarantees no Rust
# compile step is attempted.
py = sys.version_info
print(f"Python {py.major}.{py.minor}.{py.micro}")
TRANSFORMERS_VER = "4.34.1"
TOKENIZERS_VER   = "0.14.1"

# DO NOT force-reinstall torch. Colab's preinstalled torch is built and tested
# against this runtime's exact CUDA driver. Previously we did:
#     pip install --force-reinstall --no-deps torch torchvision torchaudio \
#         --index-url .../cu126
# which pulled the LATEST torch (2.7.x, cu126) with --no-deps -> mismatched
# torchvision/torchaudio and a CUDA build newer than the driver. LISA's
# model.evaluate() then SEGFAULTED the kernel mid-forward-pass on an A100
# ("AsyncIOLoopKernelRestarter: restarting kernel") with no Python traceback.
# transformers 4.34.1 + LISA are happy on Colab's stock torch 2.x, so we
# leave it untouched.
import torch as _t
print(f"Using Colab's preinstalled torch {_t.__version__} "
      f"(CUDA {_t.version.cuda}, available={_t.cuda.is_available()})")


# Robust clone: nuke any partial/stale directory and re-clone. The Python
# module resolver in Cell B6 does 'from model.LISA import LISAForCausalLM'
# via sys.path.insert(0, '/content/rosalia_repo'), so /content/rosalia_repo/
# MUST contain model/LISA.py, model/llava/, model/segment_anything/, utils/.
ROSALIA_REPO_DIR = "/content/rosalia_repo"
if (not os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py")
    or not os.path.isfile(f"{ROSALIA_REPO_DIR}/utils/utils.py")):
    if os.path.isdir(ROSALIA_REPO_DIR):
        print(f"Removing incomplete {ROSALIA_REPO_DIR} ...")
        subprocess.run(["rm","-rf",ROSALIA_REPO_DIR], check=True)
    print("Cloning ROSALIA...")
    _r = subprocess.run(
        ["git","clone","--depth","1",
         "https://github.com/checkoneee/ROSALIA.git", ROSALIA_REPO_DIR],
        capture_output=True, text=True)
    print(_r.stdout); print(_r.stderr)
    if _r.returncode != 0:
        raise RuntimeError(f"git clone ROSALIA failed: {_r.stderr}")
assert os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py"), \
    f"post-clone check failed — {ROSALIA_REPO_DIR}/model/LISA.py missing"
assert os.path.isfile(f"{ROSALIA_REPO_DIR}/utils/utils.py"), \
    f"post-clone check failed — {ROSALIA_REPO_DIR}/utils/utils.py missing"
print(f"ROSALIA repo OK at {ROSALIA_REPO_DIR}")


# Wheel-only install of tokenizers -> no Rust compile is even attempted.
r = subprocess.run(
    ["pip","install","-q","--only-binary=:all:", f"tokenizers=={TOKENIZERS_VER}"],
    capture_output=True, text=True,
)
print(r.stdout[-800:]); print(r.stderr[-800:])
if r.returncode != 0:
    raise RuntimeError(
        f"No prebuilt tokenizers=={TOKENIZERS_VER} wheel for this Python. "
        "Use Runtime -> Change runtime type -> Fallback runtime (Python 3.11) "
        "and retry.")

# Rest of the LISA/ROSALIA stack. NOTE: we do NOT pin opencv-python==4.8.0.74
# any more -- that old wheel was compiled against NumPy 1.x and dies on
# Colab's default NumPy 2.x with 'numpy.core.multiarray failed to import'.
# Downgrading numpy in turn breaks pandas (built against NumPy 2), so we
# instead upgrade opencv-python to the current line (>=4.10) whose wheels
# are ABI-compatible with NumPy 2. Same for pycocotools / scikit-image.
!pip -q install "transformers=={TRANSFORMERS_VER}" 'peft==0.4.0' 'einops==0.4.1' \
    'sentencepiece' 'opencv-python>=4.10' 'pycocotools' \
    'scikit-image' 'bitsandbytes'

# gcsfs/google-cloud-storage for GCS streaming. Do NOT --upgrade huggingface_hub
# blindly: transformers 4.34.1 requires huggingface_hub<1.0, but the current
# default is 1.x -> "huggingface-hub>=0.16.4,<1.0 is required ... found 1.21.0".
!pip -q install --upgrade gcsfs google-cloud-storage
!pip -q install 'huggingface_hub>=0.16.4,<1.0'

subprocess.run(["python","-c",
                "import numpy,transformers,tokenizers,peft,einops,torch,cv2;"
                "print('numpy       ',numpy.__version__);"
                "print('transformers',transformers.__version__);"
                "print('tokenizers  ',tokenizers.__version__);"
                "print('peft        ',peft.__version__);"
                "print('torch       ',torch.__version__);"
                "print('cv2         ',cv2.__version__);"])
print("\n>>> RESTART THE RUNTIME NOW, then run Cell B2 onward. <<<")


"""))



cells.append(md(r"""## Cell B2 — authenticate (Google + Hugging Face)

`checkone/ROSALIA-7B-v1` and `xinlai/LISA-7B-v1` are public/ungated; auth just
raises rate limits and enables GCS access.
"""))

cells.append(code(r"""from google.colab import auth
auth.authenticate_user()
from huggingface_hub import notebook_login
notebook_login()
"""))

cells.append(md(r"""## Cell B3 — config"""))

cells.append(code(r"""GCS_PROJECT = "yang-uncertainty-eval"
BUCKET      = "yang-padchest-gr"
IMG_PREFIX  = "images"
OUT_PREFIX  = "outputs/rosalia_v1"
MASK_PREFIX = f"{OUT_PREFIX}/masks"
MAP_GCS_PATH = f"{BUCKET}/{OUT_PREFIX}/finding_concept_map.json"
MAP_LOCAL    = "/content/finding_concept_map.json"

DRIVE_REPO_ROOT = "/content/drive/MyDrive/uncertaintyestimate"

ROSALIA_REPO      = "checkone/ROSALIA-7B-v1"
LISA_TOKENIZER    = "xinlai/LISA-7B-v1"
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14"

# Instruction template.
#   "bounded" = "Segment the {target} in the {location}."  <-- ROSALIA's OWN
#               inference_rosalia_example.py uses exactly this form
#               ('Segment the opacity in the right lung.'). ROSALIA/LISA was
#               trained almost entirely on LOCATION-CONDITIONED instructions,
#               so a location-free prompt is out-of-distribution and the model
#               tends to answer "There is no <finding>." with an EMPTY mask.
#   "global"  = "Segment the {target}."  (kept for ablation; expect many
#               empty masks / "There is no ..." responses.)
# The location is parsed from the radiology SENTENCE in Cell B4 (this is the
# report text the radiologist wrote -- NOT ground-truth-box leakage).
INSTRUCTION_MODE       = "bounded"
INSTRUCTION_TEMPLATE_G = "Segment the {target}."
INSTRUCTION_TEMPLATE_B = "Segment the {target} in the {location}."
INSTRUCTION_DEFAULT_LOCATION = "lung"   # fallback when no location is parsed


ROSALIA_LESIONS = {"cardiomegaly","pneumonia","atelectasis","opacity",
                   "consolidation","edema","effusion"}

MASK_GRID   = 1024
MASK_THRESHOLD = 0.5
SMOKE_N = 50
SMOKE_SEED = 42

print("Config loaded. Instruction mode:", INSTRUCTION_MODE)
"""))

cells.append(md(r"""## Cell B4 — load samples + rule labels, apply MedGemma map, filter to vocab

Loads the cached `finding_concept_map.json` (from Phase A) — first from
`/content`, else from GCS. Applies it, drops `none`, and builds the ROSALIA
instruction per row. If the map is missing, re-run Phase A.
"""))

cells.append(code(r"""import os, json, pandas as pd
import gcsfs
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)

REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"
SAMPLES_CSV  = "/content/samples_with_uncertainty_and_iou.csv"
RULE_JSONL   = "/content/rule_uncertainty_scores.jsonl"
for fn, url in [
    (SAMPLES_CSV, f"{REPO_RAW_URL}/project/data/processed/samples_with_uncertainty_and_iou.csv"),
    (RULE_JSONL,  f"{REPO_RAW_URL}/project/data/processed/rule_uncertainty_scores.jsonl"),
]:
    if not os.path.exists(fn):
        get_ipython().system(f"curl -fsSL '{url}' -o '{fn}'")

# --- load the MedGemma-produced map ---
finding_concept_map = None
if os.path.exists(MAP_LOCAL):
    finding_concept_map = json.load(open(MAP_LOCAL))
    print(f"Loaded map from {MAP_LOCAL} ({len(finding_concept_map)} labels)")
else:
    try:
        with fs.open(MAP_GCS_PATH, "r") as f:
            finding_concept_map = json.loads(f.read())
        print(f"Loaded map from gs://{MAP_GCS_PATH} ({len(finding_concept_map)} labels)")
    except Exception as e:
        raise RuntimeError(
            f"finding_concept_map.json not found in /content or GCS ({e}). "
            "Re-run Phase A (Cells A1-A3).")

# Drive (best-effort, for Cell B12)
try:
    from google.colab import drive
    drive.mount('/content/drive'); DRIVE_MOUNTED = True
except Exception as e:
    print(f"[warn] Drive not mounted ({e})."); DRIVE_MOUNTED = False

samples = pd.read_csv(SAMPLES_CSV)
rule_rows = []
with open(RULE_JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            try: rule_rows.append(json.loads(line))
            except Exception: pass
rule_df = pd.DataFrame(rule_rows)[["sample_id","uncertainty_label"]].rename(
    columns={"uncertainty_label":"uncertainty_label_rule"})
samples = samples.rename(columns={"uncertainty_label":"uncertainty_label_medgemma"})
samples = samples.merge(rule_df, on="sample_id", how="left")

# Apply the MedGemma map
samples["rosalia_target"] = samples["finding_label"].map(finding_concept_map)
samples.loc[~samples["rosalia_target"].isin(ROSALIA_LESIONS), "rosalia_target"] = None

n_total = len(samples); n_mapped = samples["rosalia_target"].notna().sum()
print(f"\nMapped to ROSALIA vocab: {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%)")
print("Dropped finding_labels (MedGemma said 'none'), top 15:")
print(samples[samples["rosalia_target"].isna()].finding_label.value_counts().head(15).to_string())

samples = samples[samples["rosalia_target"].notna()].reset_index(drop=True)
print(f"\nWorking set: {len(samples)} rows")
print("\nROSALIA target x rule uncertainty:")
print(pd.crosstab(samples["rosalia_target"], samples["uncertainty_label_rule"], margins=True))

# --- parse an anatomical location from the radiology sentence -----------------
# ROSALIA's own example prompt is 'Segment the opacity in the right lung.', i.e.
# LOCATION-CONDITIONED. We extract a location phrase from the report SENTENCE
# (what the radiologist wrote) so the prompt matches ROSALIA's training
# distribution. This is report text, NOT the ground-truth boxes, so it does not
# leak the annotation we are evaluating against.
import re as _re

# Ordered so more specific phrases win; first match is used.
_LOC_PATTERNS = [
    (r"\bright\s+(?:upper|lower|middle)\s+(?:lobe|zone|field)\b", None),
    (r"\bleft\s+(?:upper|lower|middle)\s+(?:lobe|zone|field)\b",  None),
    (r"\bright\s+base\b|\bright\s+basal\b|\bright\s+costophrenic\b", "right lower lung"),
    (r"\bleft\s+base\b|\bleft\s+basal\b|\bleft\s+costophrenic\b",    "left lower lung"),
    (r"\bright\s+ap(?:ex|ical)\b", "right upper lung"),
    (r"\bleft\s+ap(?:ex|ical)\b",  "left upper lung"),
    (r"\b(?:right|left)\s+(?:peri)?hil(?:um|ar)\b", None),
    (r"\bretrocardiac\b", "left lower lung"),
    (r"\bbibasal\b|\bbibasilar\b|\bbilateral\s+bases?\b", "bilateral lower lung"),
    (r"\bbilateral\b", "bilateral lung"),
    (r"\bright\s+lung\b|\bright\s+hemithorax\b|\bright\b", "right lung"),
    (r"\bleft\s+lung\b|\bleft\s+hemithorax\b|\bleft\b",    "left lung"),
    (r"\bbases?\b|\bbasal\b", "lower lung"),
    (r"\bap(?:ex|ical|ices)\b", "upper lung"),
]

def parse_location(sentence, default=INSTRUCTION_DEFAULT_LOCATION):
    if not isinstance(sentence, str) or not sentence.strip():
        return default
    s = sentence.lower()
    for pat, canon in _LOC_PATTERNS:
        m = _re.search(pat, s)
        if m:
            return canon if canon else m.group(0).strip()
    return default

samples["rosalia_location"] = samples["sentence"].apply(parse_location)

def _build_instruction(target, mode=INSTRUCTION_MODE, location=None):
    if mode == "global":
        return INSTRUCTION_TEMPLATE_G.format(target=target)
    loc = location or INSTRUCTION_DEFAULT_LOCATION
    return INSTRUCTION_TEMPLATE_B.format(target=target, location=loc)

samples["rosalia_instruction"] = samples.apply(
    lambda r: _build_instruction(r["rosalia_target"], location=r["rosalia_location"]),
    axis=1)

print("\nParsed-location coverage (share NOT falling back to default "
      f"'{INSTRUCTION_DEFAULT_LOCATION}'):")
_cov = (samples["rosalia_location"] != INSTRUCTION_DEFAULT_LOCATION).mean()
print(f"  {_cov*100:.1f}% of rows got a specific location from the sentence")
print("\nExample instructions:")
print(samples["rosalia_instruction"].value_counts().head(12).to_string())

samples.head(5)[["sample_id","image_id","finding_label","rosalia_target",
                 "rosalia_location","rosalia_instruction","uncertainty_label_rule"]]

"""))

cells.append(md(r"""## Cell B5 — GCS streaming image loader

Streams one PadChest-GR PNG, percentile-windows 16-bit → 8-bit RGB, returns a
PIL image. Cell B6 converts PIL → RGB uint8 numpy for ROSALIA.
"""))

cells.append(code(r"""import io
import numpy as np
from PIL import Image

def _window_to_uint8(arr, low_pct=1.0, high_pct=99.0):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi - lo < 1.0: hi = lo + 1.0
    return (np.clip((arr - lo)/(hi - lo), 0, 1) * 255).astype(np.uint8)

def load_image_gcs(image_id):
    with fs.open(f"{BUCKET}/{IMG_PREFIX}/{image_id}", "rb") as f:
        data = f.read()
    arr = np.array(Image.open(io.BytesIO(data)))
    if arr.ndim == 3: arr = arr[..., 0]
    return Image.fromarray(_window_to_uint8(arr)).convert("RGB")

_img = load_image_gcs(samples.iloc[0]["image_id"]); _a = np.array(_img)
print(f"OK — {samples.iloc[0]['image_id']}: size={_img.size}, "
      f"min={_a.min()}, max={_a.max()}, mean={_a.mean():.1f}")
"""))

cells.append(md(r"""## Cell B6 — load ROSALIA (LISA-7B + SAM-H)

Imports the custom `LISAForCausalLM` from the cloned repo, loads
`checkone/ROSALIA-7B-v1` (public, ungated, ~15 GB bf16) with the LISA-7B
tokenizer + CLIP-ViT-L/14 vision tower, and defines
`segment_image_rosalia(...) -> (binary_mask, text_output)`.
"""))

cells.append(code(r"""import sys, os, subprocess, faulthandler
faulthandler.enable()   # dump a C-level traceback if a native lib segfaults
ROSALIA_REPO_DIR = "/content/rosalia_repo"


# Verify (and if necessary auto-repair) the ROSALIA repo clone. Cell B1 does
# this too, but if the user restarted the runtime and jumped back to B6, or
# if /content was wiped since B1, we recover here so 'from model.LISA import
# LISAForCausalLM' can actually resolve.
def _rosalia_repo_ok():
    return (os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py")
            and os.path.isfile(f"{ROSALIA_REPO_DIR}/utils/utils.py"))

if not _rosalia_repo_ok():
    print(f"[B6] {ROSALIA_REPO_DIR} is missing / incomplete — re-cloning.")
    if os.path.isdir(ROSALIA_REPO_DIR):
        subprocess.run(["rm","-rf",ROSALIA_REPO_DIR], check=True)
    _r = subprocess.run(
        ["git","clone","--depth","1",
         "https://github.com/checkoneee/ROSALIA.git", ROSALIA_REPO_DIR],
        capture_output=True, text=True)
    print(_r.stdout); print(_r.stderr)

if not _rosalia_repo_ok():
    print("Diagnostic: contents of /content/ and /content/rosalia_repo/ ->")
    subprocess.run(["ls","-la","/content/"])
    subprocess.run(["ls","-la",ROSALIA_REPO_DIR])
    raise RuntimeError(
        f"ROSALIA repo still not usable at {ROSALIA_REPO_DIR}. Check that "
        "https://github.com/checkoneee/ROSALIA is reachable from this runtime.")

sys.path.insert(0, ROSALIA_REPO_DIR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# Self-healing numpy guard (inverted from earlier attempts): we now WANT
# NumPy 2.x — pandas, gcsfs, and the CUDA-12.6 PyTorch wheels on Colab are all
# built against it. We upgraded opencv-python>=4.10 in Cell B1 precisely so
# it's ABI-compatible with NumPy 2. If someone earlier accidentally pinned
# numpy<2, pandas dies at Cell B4 with:
#   ValueError: numpy.dtype size changed, may indicate binary incompatibility.
#   Expected 96 from C header, got 88 from PyObject
# Recover by force-reinstalling NumPy 2 and asking for a restart.
import numpy as _np_check
if _np_check.__version__.startswith("1."):
    import subprocess as _sp, sys as _sys
    print(f"numpy {_np_check.__version__} detected — stack needs numpy 2.x. "
          "Upgrading now...")
    _sp.run([_sys.executable, "-m", "pip", "install", "-q",
             "--force-reinstall", "--no-deps", "numpy>=2"], check=True)
    raise RuntimeError(
        "numpy has been upgraded to >=2. NOW DO THIS: Runtime -> Restart "
        "runtime, then re-run cells B2, B3, B4, B5, B6 in order. (No need to "
        "rerun B1 or Phase A.)")

import cv2, numpy as np, torch

# ---------------------------------------------------------------------------
# CRITICAL COMPAT SHIM for LISA/ROSALIA under modern transformers.
# The vendored LLaVA code inside LISA (model/llava/model/language_model/*.py)
# calls:
#     AutoConfig.register("llava", LlavaConfig)
#     AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
# But transformers>=~4.34 already ships native LLaVA, so those calls raise:
#     ValueError: 'llava' is already used by a Transformers config, pick another name.
# We can't drop transformers to 4.31 (tokenizers 0.13 has no cp312 wheel, and
# rustc won't compile it on modern Colab).
# Fix: monkey-patch the underlying registry so re-registration is a no-op
# (exist_ok=True). This is safe for *inference*: LISA's LlavaLlamaForCausalLM
# needs to be reachable via `from model.LISA import ...` which is what we do,
# not via AutoModel.from_pretrained("llava"), so overriding the registry
# entry doesn't matter for us.
# ---------------------------------------------------------------------------
import transformers  # noqa: E402
from transformers.models.auto.configuration_auto import _LazyConfigMapping
_orig_cfg_register = _LazyConfigMapping.register
def _safe_cfg_register(self, key, value, exist_ok=False):
    return _orig_cfg_register(self, key, value, exist_ok=True)
_LazyConfigMapping.register = _safe_cfg_register

try:
    from transformers.models.auto.auto_factory import _LazyAutoMapping
    _orig_am_register = _LazyAutoMapping.register
    def _safe_am_register(self, key, value, exist_ok=False):
        try:
            return _orig_am_register(self, key, value, exist_ok=True)
        except TypeError:
            # older signature: no exist_ok kw
            try: return _orig_am_register(self, key, value)
            except ValueError: pass
    _LazyAutoMapping.register = _safe_am_register
except Exception as _e:
    print(f"[warn] _LazyAutoMapping patch skipped: {_e}")

# Some LISA versions also register a custom tokenizer.
try:
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
    _orig_tok_register = TOKENIZER_MAPPING.register
    def _safe_tok_register(key, value, exist_ok=False):
        try:
            return _orig_tok_register(key, value, exist_ok=True)
        except TypeError:
            try: return _orig_tok_register(key, value)
            except ValueError: pass
    TOKENIZER_MAPPING.register = _safe_tok_register
except Exception as _e:
    print(f"[warn] TOKENIZER_MAPPING patch skipped: {_e}")

print(f"[B6] transformers {transformers.__version__} — register shim installed.")

import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.LISA import LISAForCausalLM

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("ROSALIA requires a CUDA GPU (A100/L4).")
IMAGE_SIZE = 1024

def _sam_preprocess(x, img_size=IMAGE_SIZE):
    pixel_mean = torch.tensor([123.675,116.28,103.53]).view(-1,1,1)
    pixel_std  = torch.tensor([58.395,57.12,57.375]).view(-1,1,1)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))

def load_rosalia():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        LISA_TOKENIZER, cache_dir=None, model_max_length=512,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    dtype = torch.bfloat16
    model = LISAForCausalLM.from_pretrained(
        ROSALIA_REPO, low_cpu_mem_usage=True, vision_tower=CLIP_VISION_TOWER,
        seg_token_idx=seg_token_idx, torch_dtype=dtype)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype, device=0)
    model = model.bfloat16().cuda().eval()
    clip_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(IMAGE_SIZE)
    if torch.cuda.is_available():
        print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer, clip_processor, transform

model, tokenizer, clip_processor, transform = load_rosalia()

def segment_image_rosalia(model, tokenizer, clip_processor, transform, pil_image, instruction):
    image_np = np.array(pil_image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    original_size_list = [image_np.shape[:2]]
    conv = conversation_lib.conv_templates["llava_v1"].copy(); conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + instruction
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN,
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], prompt); conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    image_clip = (clip_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
                  .unsqueeze(0).cuda().bfloat16())
    image_resized = transform.apply_image(image_np)
    resize_list = [image_resized.shape[:2]]
    image = (_sam_preprocess(torch.from_numpy(image_resized).permute(2,0,1).contiguous())
             .unsqueeze(0).cuda().bfloat16())
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(
            image_clip, image, input_ids, resize_list, original_size_list,
            max_new_tokens=256, tokenizer=tokenizer)
    out_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_out = tokenizer.decode(out_ids, skip_special_tokens=False)
    text_out = text_out.replace("\n","").replace("  "," ").split("ASSISTANT:")[-1].split("</s>")[0].strip()
    if len(pred_masks) == 0 or pred_masks[0].numel() == 0:
        mask = np.zeros(original_size_list[0], dtype=np.uint8)
    else:
        mask = (pred_masks[0] > 0).detach().cpu().numpy().astype(np.uint8)
        if mask.ndim == 3: mask = mask[0]
    return mask, text_out

_row = samples.iloc[0]; _img = load_image_gcs(_row.image_id)
_m, _t = segment_image_rosalia(model, tokenizer, clip_processor, transform, _img, _row.rosalia_instruction)
print(f"Smoke: instr={_row.rosalia_instruction!r} mask_sum={int(_m.sum())} text={_t!r}")
"""))

cells.append(md(r"""## Cell B6b — DIAGNOSTIC: why are masks empty? (preprocessing × prompt sweep)

The model runs but keeps replying *"There is no &lt;finding&gt;..."* with an empty
mask. Before trusting the full run, isolate the cause. ROSALIA's official
example feeds a **raw `cv2.imread` 8-bit RGB** image with **no windowing**,
whereas our loader applies 1-99 percentile windowing — a prime domain-shift
suspect. This cell runs a few samples through 4 combinations and prints the
text output + mask pixel count for each, so we can see what actually makes
ROSALIA emit a `[SEG]` mask.
"""))

cells.append(code(r"""import io, numpy as np
from PIL import Image

# variant loaders -------------------------------------------------------------
def _load_raw_rgb(image_id):
    "No windowing: mimic ROSALIA's cv2.imread 8-bit path as closely as possible."
    with fs.open(f"{BUCKET}/{IMG_PREFIX}/{image_id}", "rb") as f:
        arr = np.array(Image.open(io.BytesIO(f.read())))
    if arr.ndim == 3:
        arr = arr[..., 0]
    # If 16-bit, scale by max (NOT percentile) -> preserves global contrast.
    if arr.dtype != np.uint8:
        m = float(arr.max()) or 1.0
        arr = (arr.astype(np.float32) / m * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def _global_instr(target):  return f"Segment the {target}."
def _bounded_instr(target, loc): return f"Segment the {target} in the {loc}."

# pick a few rows that HAVE a specific parsed location (more likely truly present)
_diag = samples.copy()
_diag = _diag[_diag["rosalia_location"] != INSTRUCTION_DEFAULT_LOCATION].head(4)
if len(_diag) < 4:
    _diag = samples.head(4)

print("Legend: windowed=our 1-99% loader, raw=max-scaled 8-bit (ROSALIA-style)\n")
for row in _diag.itertuples():
    print("="*90)
    print(f"{row.sample_id} | finding={row.finding_label} -> target={row.rosalia_target} "
          f"| parsed_loc={row.rosalia_location}")
    print(f"  sentence: {str(row.sentence)[:120]}")
    img_win = load_image_gcs(row.image_id)     # windowed
    img_raw = _load_raw_rgb(row.image_id)       # raw/max-scaled
    variants = [
        ("windowed + bounded", img_win, _bounded_instr(row.rosalia_target, row.rosalia_location)),
        ("raw      + bounded", img_raw, _bounded_instr(row.rosalia_target, row.rosalia_location)),
        ("windowed + global ", img_win, _global_instr(row.rosalia_target)),
        ("raw      + global ", img_raw, _global_instr(row.rosalia_target)),
    ]
    for name, img, instr in variants:
        m, t = segment_image_rosalia(model, tokenizer, clip_processor, transform, img, instr)
        print(f"  [{name}] mask_px={int(m.sum()):>7d}  instr={instr!r}")
        print(f"       -> {t!r}")
print("="*90)
print("\nInterpretation:")
print(" * If 'raw' variants fire (mask_px>0) but 'windowed' don't -> our percentile")
print("   windowing is the culprit; switch Cell B5 loader to the raw/max-scaled path.")
print(" * If 'global' fires but 'bounded' doesn't (or vice-versa) -> prompt phrasing.")
print(" * If NOTHING fires on findings the readers boxed -> genuine MIMIC->PadChest")
print("   recall gap; that is itself a valid result to report, not a bug.")
"""))

cells.append(md(r"""## Cell B6c — INSPECT raw mask logits / probabilities (pre-threshold)

The binary mask used everywhere else comes from ONE hard cutoff inside
`segment_image_rosalia`:

```python
mask = (pred_masks[0] > 0)      #  logit > 0   ==   sigmoid(logit) > 0.5
```

`pred_masks[0]` is SAM-H's **raw per-pixel logit map** (a float tensor at the
original image resolution). Every pixel gets a continuous score; the mask is
just the pixels whose score clears the threshold. This cell reruns a few
examples but keeps those *pre-threshold* scores so we can see them: the sigmoid
probability heatmap, min/max/mean logit, how many pixels beat the 0.5 cutoff,
and the probability histogram with the threshold line. Handy for telling apart
"almost fired" (peak prob just under 0.5) from "confidently absent" (all probs
~0)."""))

cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

# segment_image_rosalia() thresholds pred_masks[0] > 0 to make the binary mask.
# pred_masks[0] is SAM-H's RAW per-pixel LOGIT map. This variant returns that
# float logit map (at original image size) instead of the binary mask so we can
# inspect the pre-threshold scores. (logit > 0) == (sigmoid(logit) > 0.5).
def segment_image_rosalia_logits(model, tokenizer, clip_processor, transform, pil_image, instruction):
    image_np = np.array(pil_image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    original_size_list = [image_np.shape[:2]]
    conv = conversation_lib.conv_templates["llava_v1"].copy(); conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + instruction
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN,
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], prompt); conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    image_clip = (clip_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
                  .unsqueeze(0).cuda().bfloat16())
    image_resized = transform.apply_image(image_np)
    resize_list = [image_resized.shape[:2]]
    image = (_sam_preprocess(torch.from_numpy(image_resized).permute(2,0,1).contiguous())
             .unsqueeze(0).cuda().bfloat16())
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(
            image_clip, image, input_ids, resize_list, original_size_list,
            max_new_tokens=256, tokenizer=tokenizer)
    out_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_out = tokenizer.decode(out_ids, skip_special_tokens=False)
    text_out = text_out.replace("\n","").replace("  "," ").split("ASSISTANT:")[-1].split("</s>")[0].strip()
    if len(pred_masks) == 0 or pred_masks[0].numel() == 0:
        logits = None
    else:
        lm = pred_masks[0]
        if lm.ndim == 3: lm = lm[0]
        logits = lm.float().detach().cpu().numpy()
    return logits, text_out

# pick a few rows with a specific parsed location (more likely to actually fire)
N_EXAMPLES = 9
_ex = samples[samples["rosalia_location"] != INSTRUCTION_DEFAULT_LOCATION].head(N_EXAMPLES)
if len(_ex) < N_EXAMPLES:
    _ex = samples.head(N_EXAMPLES)


for row in _ex.itertuples():
    img = load_image_gcs(row.image_id)
    logits, text_out = segment_image_rosalia_logits(
        model, tokenizer, clip_processor, transform, img, row.rosalia_instruction)
    print("="*84)
    print(f"{row.sample_id} | target={row.rosalia_target} | instr={row.rosalia_instruction!r}")
    print(f"  text_output: {text_out!r}")
    if logits is None:
        print("  [no SEG mask returned - model emitted no [SEG] token, so there is no logit map]")
        continue
    probs   = 1.0 / (1.0 + np.exp(-logits))     # per-pixel P(lesion)
    binmask = (logits > 0).astype(np.uint8)     # EXACT rule used elsewhere
    print(f"  logit : min={logits.min():.2f}  max={logits.max():.2f}  mean={logits.mean():.3f}")
    print(f"  prob  : min={probs.min():.3f}  max={probs.max():.3f}  mean={probs.mean():.3f}")
    print(f"  pixels above threshold (logit>0 == prob>0.5): {int(binmask.sum())} "
          f"({100*binmask.mean():.3f}% of image)")
    print("  peak prob {:.3f} -> {}".format(
        probs.max(),
        "FIRES (a pixel clears 0.5)" if probs.max() > 0.5 else "does NOT fire (peak below 0.5)"))

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("image"); ax[0].axis("off")
    im1 = ax[1].imshow(probs, cmap="jet", vmin=0, vmax=1)
    ax[1].set_title("P(lesion) = sigmoid(logit)"); ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(img, cmap="gray")
    ax[2].imshow(np.ma.masked_where(probs < 0.05, probs), cmap="jet", vmin=0, vmax=1, alpha=0.55)
    ax[2].set_title("prob heatmap overlay"); ax[2].axis("off")
    ax[3].imshow(binmask, cmap="gray")
    ax[3].set_title(f"binary mask @0.5 ({int(binmask.sum())} px)"); ax[3].axis("off")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,3))
    plt.hist(probs.ravel(), bins=60, color="tab:purple", alpha=0.85)
    plt.axvline(0.5, color="red", ls="--", label="threshold 0.5")
    plt.yscale("log"); plt.xlabel("per-pixel P(lesion)"); plt.ylabel("pixel count (log)")
    plt.title(f"{row.sample_id}: probability distribution"); plt.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## Cell B7 — IoU helpers (mirror project/src/compute_iou.py)"""))



cells.append(code(r"""import numpy as np
from itertools import combinations

def boxes_to_mask(boxes, grid_size=MASK_GRID):
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not boxes: return mask
    for x1,y1,x2,y2 in boxes:
        px1=max(0,min(grid_size,int(np.floor(x1*grid_size)))); py1=max(0,min(grid_size,int(np.floor(y1*grid_size))))
        px2=max(0,min(grid_size,int(np.ceil(x2*grid_size))));  py2=max(0,min(grid_size,int(np.ceil(y2*grid_size))))
        if px2>px1 and py2>py1: mask[py1:py2, px1:px2]=1
    return mask

def mask_iou(m1, m2):
    inter=int(np.logical_and(m1,m2).sum()); union=int(np.logical_or(m1,m2).sum())
    return inter/union if union>0 else float("nan")

def boxes_to_intersection_mask(boxes, grid_size=MASK_GRID):
    if not boxes: return np.zeros((grid_size,grid_size),np.uint8)
    masks=[boxes_to_mask([b],grid_size) for b in boxes]; out=masks[0].copy()
    for m in masks[1:]: out=np.logical_and(out,m).astype(np.uint8)
    return out

def boxes_to_outer_bbox_mask(boxes, grid_size=MASK_GRID):
    if not boxes: return np.zeros((grid_size,grid_size),np.uint8)
    xs1=[b[0] for b in boxes]; ys1=[b[1] for b in boxes]; xs2=[b[2] for b in boxes]; ys2=[b[3] for b in boxes]
    return boxes_to_mask([[min(xs1),min(ys1),max(xs2),max(ys2)]], grid_size)

def pairwise_box_ious(boxes, grid_size=MASK_GRID):
    if not boxes or len(boxes)<2: return []
    masks=[boxes_to_mask([b],grid_size) for b in boxes]
    return [mask_iou(masks[i],masks[j]) for i,j in combinations(range(len(masks)),2)]

def pred_probs_to_grid_mask(probs, grid_size=MASK_GRID, threshold=MASK_THRESHOLD):
    arr=np.asarray(probs)
    if arr.ndim==4 and arr.shape[0]==1: arr=arr[0]
    if arr.ndim==3: arr=arr.max(axis=0)
    if arr.ndim!=2: raise ValueError(f"Bad shape {probs.shape}")
    try:
        from skimage.transform import resize
        resized=resize(arr.astype(np.float32),(grid_size,grid_size),order=1,mode="edge",
                       anti_aliasing=False,preserve_range=True)
    except Exception:
        h,w=arr.shape
        ys=np.linspace(0,h-1,grid_size).astype(np.int64); xs=np.linspace(0,w-1,grid_size).astype(np.int64)
        resized=arr[ys[:,None],xs[None,:]]
    return (resized>threshold).astype(np.uint8)

def compute_iou_bundle(pred_probs, r1_boxes, r2_boxes, grid_size=MASK_GRID, threshold=MASK_THRESHOLD):
    pm=pred_probs_to_grid_mask(pred_probs,grid_size,threshold)
    all_boxes=(r1_boxes or [])+(r2_boxes or [])
    per_box=[mask_iou(pm,boxes_to_mask([b],grid_size)) for b in all_boxes]
    return {
        "per_box_ious":per_box,
        "max_per_box_iou":float(np.max(per_box)) if per_box else float("nan"),
        "mean_per_box_iou":float(np.mean(per_box)) if per_box else float("nan"),
        "iou_with_pixel_or_union":mask_iou(pm,boxes_to_mask(all_boxes,grid_size)),
        "iou_with_outer_bbox_union":mask_iou(pm,boxes_to_outer_bbox_mask(all_boxes,grid_size)),
        "iou_with_intersection":mask_iou(pm,boxes_to_intersection_mask(all_boxes,grid_size)),
        "iou_with_reader1_union":mask_iou(pm,boxes_to_mask(r1_boxes or [],grid_size)),
        "iou_with_reader2_union":mask_iou(pm,boxes_to_mask(r2_boxes or [],grid_size)),
        "reader_iou_union":mask_iou(boxes_to_mask(r1_boxes or [],grid_size),boxes_to_mask(r2_boxes or [],grid_size)),
        "pairwise_box_ious":pairwise_box_ious(all_boxes,grid_size),
        "mean_pairwise_iou":float(np.mean(pairwise_box_ious(all_boxes,grid_size))) if len(all_boxes)>=2 else float("nan"),
        "pred_area":int(pm.sum()),
        "num_boxes_total":len(all_boxes),
        "num_reader1_boxes":len(r1_boxes or []),
        "num_reader2_boxes":len(r2_boxes or []),
    }, pm
"""))

cells.append(md(r"""## Cell B8 — SMOKE TEST (50 samples)"""))

cells.append(code(r"""import random, json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt, matplotlib.patches as patches
random.seed(SMOKE_SEED)

def stratified_sample(df, n_per_class):
    out=[]
    for cls in ["certain","uncertain"]:
        sub=df[df["uncertainty_label_rule"]==cls]
        out.append(sub.sample(n=min(n_per_class,len(sub)), random_state=SMOKE_SEED))
    return pd.concat(out).sample(frac=1, random_state=SMOKE_SEED).reset_index(drop=True)

smoke = stratified_sample(samples, SMOKE_N//2)
print(f"Smoke set: {len(smoke)}")
smoke_rows=[]; sample_masks={}
for row in tqdm(smoke.itertuples(), total=len(smoke), desc="smoke"):
    img=load_image_gcs(row.image_id)
    mask,text_out=segment_image_rosalia(model,tokenizer,clip_processor,transform,img,row.rosalia_instruction)
    r1=json.loads(row.reader1_boxes) if isinstance(row.reader1_boxes,str) else row.reader1_boxes
    r2=json.loads(row.reader2_boxes) if isinstance(row.reader2_boxes,str) else row.reader2_boxes
    bundle,pm=compute_iou_bundle(mask,r1,r2)
    smoke_rows.append({"sample_id":row.sample_id,"image_id":row.image_id,
        "finding_label":row.finding_label,"rosalia_target":row.rosalia_target,
        "rosalia_instruction":row.rosalia_instruction,"rosalia_text_output":text_out,
        "uncertainty_label_rule":row.uncertainty_label_rule,
        "uncertainty_label_medgemma":row.uncertainty_label_medgemma,
        "sentence_raw":row.sentence, **bundle})
    sample_masks[row.sample_id]=pm

smoke_df=pd.DataFrame(smoke_rows)

# --- summary: how often did ROSALIA actually emit a mask? -------------------
n_nonempty = int((smoke_df["pred_area"] > 0).sum())
print(f"\nNon-empty masks: {n_nonempty}/{len(smoke_df)} "
      f"({100*n_nonempty/len(smoke_df):.0f}%). "
      "ROSALIA abstains (empty mask) on findings it judges subtle/absent.")
if n_nonempty:
    nz = smoke_df[smoke_df["pred_area"] > 0]
    print("Among NON-EMPTY masks:  mean IoU(pixel-OR union)="
          f"{nz['iou_with_pixel_or_union'].mean():.3f}  "
          f"median={nz['iou_with_pixel_or_union'].median():.3f}")
    print("Non-empty rate by uncertainty group:")
    print((smoke_df.assign(fired=smoke_df['pred_area']>0)
                   .groupby('uncertainty_label_rule')['fired']
                   .mean().round(3).to_string()))

display(smoke_df[["sample_id","uncertainty_label_rule","rosalia_target",
    "pred_area","iou_with_pixel_or_union","iou_with_outer_bbox_union",
    "mean_per_box_iou","reader_iou_union","num_boxes_total"]]
    .sort_values("pred_area", ascending=False).head(8))

# Visualize the samples that ACTUALLY produced a mask (fall back to head if none)
vis = smoke_df.sort_values("pred_area", ascending=False)
vis = vis[vis["pred_area"] > 0].head(5)
if len(vis) == 0:
    vis = smoke_df.head(5)
n_vis = len(vis)
fig,axes=plt.subplots(n_vis,4,figsize=(16,4*n_vis))
if n_vis == 1: axes = axes.reshape(1, -1)
for i,row in enumerate(vis.itertuples()):

    img=load_image_gcs(row.image_id); iw,ih=img.size
    r1=json.loads(samples.loc[samples.sample_id==row.sample_id,"reader1_boxes"].iloc[0])
    r2=json.loads(samples.loc[samples.sample_id==row.sample_id,"reader2_boxes"].iloc[0])
    pm=sample_masks[row.sample_id]
    axes[i,0].imshow(img,cmap="gray"); axes[i,0].set_title("image"); axes[i,0].axis("off")
    axes[i,1].imshow(img,cmap="gray")
    for b,c in [(r1,"red"),(r2,"blue")]:
        for x1,y1,x2,y2 in b:
            axes[i,1].add_patch(patches.Rectangle((x1*iw,y1*ih),(x2-x1)*iw,(y2-y1)*ih,lw=2,edgecolor=c,facecolor="none"))
    axes[i,1].set_title("GT (R1=red,R2=blue)"); axes[i,1].axis("off")
    axes[i,2].imshow(pm,cmap="gray"); axes[i,2].set_title(f"pred IoU(or)={row.iou_with_pixel_or_union:.2f}"); axes[i,2].axis("off")
    axes[i,3].imshow(img,cmap="gray")
    pm_disp=np.array(Image.fromarray((pm*255).astype(np.uint8)).resize((iw,ih),Image.NEAREST))
    axes[i,3].imshow(np.ma.masked_where(pm_disp==0,pm_disp),cmap="Reds",alpha=0.5)
    axes[i,3].set_title(f"[{row.uncertainty_label_rule}] {row.rosalia_target}\n{row.rosalia_text_output[:60]}"); axes[i,3].axis("off")
plt.tight_layout(); plt.savefig("/content/rosalia_smoke_grid.png",dpi=110); plt.show()
"""))

cells.append(md(r"""## Cell B9 — FULL RUN (resumable)"""))

cells.append(code(r"""import io, json
import numpy as np
from tqdm.auto import tqdm

def exists_in_bucket(path):
    try: return fs.exists(path)
    except Exception: return False
def upload_json(path,obj):
    with fs.open(path,"w") as f: f.write(json.dumps(obj))
def upload_npz(path,mask):
    buf=io.BytesIO(); np.savez_compressed(buf,mask=mask); buf.seek(0)
    with fs.open(path,"wb") as f: f.write(buf.read())

all_records=[]; n_skipped=0; n_new=0
for row in tqdm(samples.itertuples(), total=len(samples), desc="full"):
    out_json=f"{BUCKET}/{OUT_PREFIX}/{row.sample_id}.json"
    out_npz =f"{BUCKET}/{MASK_PREFIX}/{row.sample_id}.npz"
    if exists_in_bucket(out_json):
        with fs.open(out_json,"r") as f: all_records.append(json.loads(f.read()))
        n_skipped+=1; continue
    try:
        img=load_image_gcs(row.image_id)
        mask,text_out=segment_image_rosalia(model,tokenizer,clip_processor,transform,img,row.rosalia_instruction)
        r1=json.loads(row.reader1_boxes) if isinstance(row.reader1_boxes,str) else row.reader1_boxes
        r2=json.loads(row.reader2_boxes) if isinstance(row.reader2_boxes,str) else row.reader2_boxes
        bundle,pm=compute_iou_bundle(mask,r1,r2)
        rec={"sample_id":row.sample_id,"image_id":row.image_id,"finding_label":row.finding_label,
             "rosalia_target":row.rosalia_target,"rosalia_instruction":row.rosalia_instruction,
             "rosalia_text_output":text_out,"uncertainty_label_rule":row.uncertainty_label_rule,
             "uncertainty_label_medgemma":row.uncertainty_label_medgemma,"sentence_raw":row.sentence,**bundle}
        upload_json(out_json,rec); upload_npz(out_npz,pm); all_records.append(rec); n_new+=1
    except Exception as e:
        print(f"[skip] {row.sample_id}: {type(e).__name__}: {e}"); continue

results_df=pd.DataFrame(all_records)
print(f"\nDone. skipped={n_skipped} new={n_new} total={len(results_df)}")
results_df.to_csv("/content/rosalia_per_sample_ious.csv", index=False)
"""))

cells.append(md(r"""## Cell B10 — aggregates stratified by rule-based uncertainty"""))

cells.append(code(r"""import json, numpy as np
RNG=np.random.default_rng(42); BOOT_N=2000
IOU_COLS=["iou_with_pixel_or_union","iou_with_outer_bbox_union","iou_with_intersection",
          "iou_with_reader1_union","iou_with_reader2_union","reader_iou_union",
          "mean_per_box_iou","max_per_box_iou","mean_pairwise_iou"]
for c in IOU_COLS: results_df[c]=pd.to_numeric(results_df[c],errors="coerce")

print("=== By uncertainty group ===")
print(results_df.groupby("uncertainty_label_rule")[IOU_COLS].agg(["count","mean","median","std"]))
print("\n=== By ROSALIA target x uncertainty ===")
pt=results_df.groupby(["rosalia_target","uncertainty_label_rule"])[IOU_COLS].agg(["count","mean","median"])
print(pt); pt.to_csv("/content/rosalia_per_target_ious.csv")
results_df.groupby(["uncertainty_label_rule","finding_label"])[IOU_COLS].agg(
    ["count","mean","median"]).to_csv("/content/rosalia_per_finding_ious.csv")

def boot_mean_ci(x,n=BOOT_N,alpha=0.05):
    x=np.asarray(x,dtype=float); x=x[~np.isnan(x)]
    if len(x)==0: return (float("nan"),)*3
    idx=RNG.integers(0,len(x),size=(n,len(x))); means=x[idx].mean(axis=1)
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2]); return float(means.mean()),float(lo),float(hi)

agg_json={}
for cls,sub in results_df.groupby("uncertainty_label_rule"):
    mean,lo,hi=boot_mean_ci(sub["iou_with_pixel_or_union"])
    agg_json[str(cls)]={"n":int(len(sub)),"iou_with_pixel_or_union_mean":mean,
        "iou_with_pixel_or_union_ci95":[lo,hi],
        "iou_with_outer_bbox_union_mean":float(sub["iou_with_outer_bbox_union"].mean()),
        "iou_with_intersection_mean":float(sub["iou_with_intersection"].mean()),
        "reader_iou_union_mean":float(sub["reader_iou_union"].mean()),
        "mean_per_box_iou_mean":float(sub["mean_per_box_iou"].mean())}
json.dump(agg_json, open("/content/rosalia_aggregate_ious.json","w"), indent=2)
agg_json
"""))

cells.append(md(r"""## Cell B11 — plots"""))

cells.append(code(r"""import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
for cls,color in [("certain","tab:blue"),("uncertain","tab:orange")]:
    sub=results_df[results_df["uncertainty_label_rule"]==cls]["iou_with_pixel_or_union"].dropna()
    plt.hist(sub,bins=40,alpha=0.55,label=f"{cls} (n={len(sub)})",color=color)
plt.xlabel("IoU(pred, pixel-OR union of GT boxes)"); plt.ylabel("Count")
plt.title("ROSALIA IoU by rule-based spatial uncertainty"); plt.legend(); plt.tight_layout()
plt.savefig("/content/rosalia_iou_histogram_by_group.png",dpi=120); plt.show()

plot_cols=["iou_with_pixel_or_union","iou_with_outer_bbox_union","iou_with_intersection",
           "mean_per_box_iou","max_per_box_iou","reader_iou_union"]
means=results_df.groupby("uncertainty_label_rule")[plot_cols].mean()
fig,ax=plt.subplots(figsize=(10,4)); means.T.plot(kind="bar",ax=ax)
ax.set_ylabel("Mean IoU"); ax.set_title("ROSALIA mean IoU per metric, by uncertainty group")
ax.set_xticklabels(plot_cols,rotation=30,ha="right"); plt.tight_layout()
plt.savefig("/content/rosalia_iou_by_uncertainty_group.png",dpi=120); plt.show()

plt.figure(figsize=(6,5))
for cls,color in [("certain","tab:blue"),("uncertain","tab:orange")]:
    sub=results_df[results_df["uncertainty_label_rule"]==cls]
    plt.scatter(sub["reader_iou_union"],sub["iou_with_pixel_or_union"],s=10,alpha=0.5,label=cls,color=color)
plt.xlabel("Reader-Reader IoU (agreement)"); plt.ylabel("ROSALIA IoU")
plt.title("ROSALIA quality vs annotator agreement"); plt.legend(); plt.tight_layout()
plt.savefig("/content/rosalia_agreement_vs_iou_scatter.png",dpi=120); plt.show()
"""))

cells.append(md(r"""## Cell B12 — push aggregated results back to the repo via Drive"""))

cells.append(code(r"""import shutil, os
if globals().get("DRIVE_MOUNTED",False) and os.path.isdir("/content/drive/MyDrive"):
    out_dir=f"{DRIVE_REPO_ROOT}/project/outputs"; fig_dir=f"{DRIVE_REPO_ROOT}/project/figures"
else:
    out_dir="/content/outputs"; fig_dir="/content/figures"
    print("Drive not mounted — writing to /content/.")
os.makedirs(out_dir,exist_ok=True); os.makedirs(fig_dir,exist_ok=True)
for fn in ("rosalia_per_sample_ious.csv","rosalia_per_finding_ious.csv",
           "rosalia_per_target_ious.csv","rosalia_aggregate_ious.json",
           "finding_concept_map.json"):
    if os.path.exists(f"/content/{fn}"): shutil.copy(f"/content/{fn}", f"{out_dir}/{fn}")
for fn in ("rosalia_iou_histogram_by_group.png","rosalia_iou_by_uncertainty_group.png",
           "rosalia_agreement_vs_iou_scatter.png","rosalia_smoke_grid.png"):
    if os.path.exists(f"/content/{fn}"): shutil.copy(f"/content/{fn}", f"{fig_dir}/{fn}")
print(f"Done. outputs -> {out_dir}, figures -> {fig_dir}")
"""))

# =============================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}
# Write relative to THIS script's location so the builder is portable across
# machines/checkouts (previously this was a hardcoded absolute path that only
# worked on one laptop). Override with env var NOTEBOOK_OUT if desired.
import os
_default_out = Path(__file__).resolve().parent / "project" / "notebooks" / "rosalia_padchest_gr_colab.ipynb"
out_path = Path(os.environ.get("NOTEBOOK_OUT", _default_out))
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")

