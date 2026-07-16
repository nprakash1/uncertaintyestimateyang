"""Build project/notebooks/mimic_ils_rosalia_reproduction_colab.ipynb

Goal: reproduce the ROSALIA paper's instruction-guided lesion-segmentation
results on the *in-distribution* MIMIC-ILS data (the dataset ROSALIA was
actually trained on), on a 700-study mixed-split subset, AND label each finding
with MedGemma report-language uncertainty so we can ask the project's core
question in-distribution: does ROSALIA's segmentation quality drop on findings
whose report language is "uncertain"?

Pipeline (two phases, separated by a runtime restart because MedGemma needs
transformers>=4.50 and ROSALIA needs the old 4.3x line):

  PHASE A  (MedGemma, modern transformers)
    - Load the 700-study subset manifest (mimic_ils_subset_manifest.csv).
    - For every POSITIVE (image, finding) pair, ask MedGemma whether the
      report language about that finding is certain/uncertain (same
      existence/visibility/diagnosis/boundary definition as the PadChest work).
    - Save uncertainty labels to Drive (survives the restart).

  ↻ restart runtime + reinstall

  PHASE B  (ROSALIA = LISA-7B + SAM-H)
    - Reload the manifest + MedGemma uncertainty labels.
    - Read the local MIMIC-CXR-JPG images + the MIMIC-ILS silver masks.
    - Feed ROSALIA the dataset's ROSALIA-style instruction
      ("Segment the {target} in the {location}."), get a predicted mask.
    - Compare pred vs SILVER mask with the paper-style metrics: per-sample IoU
      (-> gIoU), cumulative IoU (cIoU), Dice; overall and per lesion type.
    - Check abstention on NEGATIVE pairs (should emit empty mask / "There is no
      ...").
    - Stratify all of the above by MedGemma uncertainty label.

Data expected on disk (put on Google Drive, or adjust the paths in Cell S0):
  <MIMIC_SUBSET_DIR>/pXX/pXXXXXXXX/sSTUDY/<dicom_id>.jpg   (700 images)
  <LESION_MASK_DIR>/sSTUDY/sSTUDY_{lesion}_{idx}.png       (silver masks)
  mimic_ils_subset_manifest.csv                            (from this repo)
"""
from pathlib import Path
import json
import os
import nbformat as nbf


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


cells = []

# =============================================================================
cells.append(md(r"""# Reproducing ROSALIA on MIMIC-ILS (in-distribution) + MedGemma uncertainty

This notebook evaluates **ROSALIA** (LISA-7B + SAM-H, instruction-guided CXR
lesion segmentation) on a **700-study mixed-split subset of MIMIC-ILS** — the
dataset ROSALIA was *trained* on — and compares its predicted masks to the
**silver-standard masks** shipped with MIMIC-ILS, using the paper's metrics
(**gIoU**, **cIoU**, **Dice**). It also runs **MedGemma** to label each
finding's report language as *certain/uncertain*, so we can ask: *does
segmentation quality track report-language uncertainty, in-distribution?*

## ⚠️ TWO phases separated by a runtime restart
* **Phase A — MedGemma** (`transformers>=4.50`): report-language uncertainty
  labels for every positive (image, finding) pair.
* **↻ Restart runtime + reinstall.**
* **Phase B — ROSALIA** (`transformers==4.34.1`): segmentation + metrics vs the
  silver masks, stratified by the Phase-A uncertainty labels.

**Data:** 700 MIMIC-CXR-JPG images (`.jpg`) + MIMIC-ILS silver masks (`.png`) +
`mimic_ils_subset_manifest.csv` (507 positive pairs across all 7 lesions, plus
negatives). Put the images/masks on Google Drive and set the paths in Cell S0.

**Runtime:** GPU with ≥16 GB VRAM (L4/A100). ROSALIA-7B bf16 is ~15 GB.
"""))

# =============================================================================
# Cell S0 — shared config + data location
# =============================================================================
cells.append(md(r"""## Cell S0 — mount Drive + fetch the manifest

Your data lives in a **Shared-with-me** Drive folder `MIMIC-CXR-Ext-ILS/` as two
zips:
* `mimic_subset.zip` (~1.14 GB) — the 700 MIMIC-CXR-JPG images (`pXX/...`).
* `mimic-cxr-ext-ils.zip` (~231 MB) — the annotations incl. `lesion_mask/`.

⚠️ **Shared-with-me folders don't auto-mount.** In Drive, right-click the
`MIMIC-CXR-Ext-ILS` folder → **Organise → Add shortcut to Drive → My Drive**.
After that it appears at `/content/drive/MyDrive/MIMIC-CXR-Ext-ILS/`.

Phase A (MedGemma) only needs the manifest text, so we don't unzip here — the
images/masks are extracted in **Phase B (Cell B3b)**. This cell just mounts
Drive (for saving Phase-A labels) and downloads the manifest.
"""))

cells.append(code(r"""import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = True
except Exception as e:
    print(f"[warn] Drive not mounted ({e}); using local paths.")
    DRIVE = False

# Shared-with-me folder (after you add a shortcut to My Drive). Image zips:
DRIVE_DATA_DIR = "/content/drive/MyDrive/MIMIC-CXR-Ext-ILS"
# NOTE: "mimic_subset (1).zip" (3.96 GB) = full TEST-split images;
#       "mimic_val_subset.zip"           = full VAL-split images;
#       "mimic_subset.zip"  (1.14 GB)    = old smoke subset.
EXT_ZIP    = os.path.join(DRIVE_DATA_DIR, "mimic-cxr-ext-ils.zip")


# Persist Phase-A outputs so they survive the restart
WORK_DIR = "/content/drive/MyDrive/mimic_ils_rosalia" if DRIVE else "/content/mimic_ils_rosalia"
os.makedirs(WORK_DIR, exist_ok=True)
UNC_LABELS_PATH = os.path.join(WORK_DIR, "medgemma_uncertainty_subset.json")

# Manifest: pulled from the repo (small, 2.6 MB). Fallback: local copy.
REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"

# ---- which manifest/split to run -------------------------------------------
# "subset"=smoke subset | "test"=held-out TEST (2,445 imgs) |
# "val"=held-out VAL (1,508 imgs) | "valtest"=VAL+TEST pooled
MANIFEST_CHOICE = "test"          # "subset" | "test" | "val" | "valtest"
_MANIFEST_FILES = {
    "subset":  ["mimic_ils_subset_manifest.csv"],
    "test":    ["mimic_ils_test_manifest.csv"],
    "val":     ["mimic_ils_val_manifest.csv"],
    "valtest": ["mimic_ils_test_manifest.csv", "mimic_ils_val_manifest.csv"],
}[MANIFEST_CHOICE]
_IMG_ZIP_NAMES = {
    "subset":  ["mimic_subset.zip"],
    "test":    ["mimic_subset (1).zip"],
    "val":     ["mimic_val_subset.zip"],
    "valtest": ["mimic_subset (1).zip", "mimic_val_subset.zip"],
}[MANIFEST_CHOICE]
IMG_ZIPS   = [os.path.join(DRIVE_DATA_DIR, z) for z in _IMG_ZIP_NAMES]
SUBSET_ZIP = IMG_ZIPS[0]          # back-compat alias

import pandas as pd
_dfs = []
for _f in _MANIFEST_FILES:
    _p = f"/content/{_f}"
    if not os.path.exists(_p):
        os.system(f"curl -fsSL '{REPO_RAW_URL}/project/data/mimic_ils/{_f}' -o '{_p}'")
    _dfs.append(pd.read_csv(_p))
SUBSET_MANIFEST = f"/content/mimic_ils_{MANIFEST_CHOICE}_manifest.csv"
pd.concat(_dfs, ignore_index=True).to_csv(SUBSET_MANIFEST, index=False)
print(f"[config] MANIFEST_CHOICE={MANIFEST_CHOICE} -> {SUBSET_MANIFEST}")

_m = pd.read_csv(SUBSET_MANIFEST)
print("manifest rows:", len(_m), "| unique studies:", _m.study_id.nunique())
print("positive pairs:", (_m.polarity=="positive").sum(),
      "| negative pairs:", (_m.polarity=="negative").sum())
print("splits:", _m.split.value_counts().to_dict())
print("image zips present?", {os.path.basename(z): os.path.exists(z) for z in IMG_ZIPS},
      "| ext:", os.path.exists(EXT_ZIP))
if not all(os.path.exists(z) for z in IMG_ZIPS):
    print("[hint] add a shortcut to the shared 'MIMIC-CXR-Ext-ILS' folder into My Drive,",
          "or edit DRIVE_DATA_DIR above.")
"""))



# =============================================================================
# PHASE A — MedGemma uncertainty labelling
# =============================================================================
cells.append(md(r"""# ═════════════════ PHASE A — MedGemma uncertainty labelling ═════════════════

Runs in a fresh runtime with modern transformers. Produces one
certain/uncertain label per POSITIVE (study, finding) pair, saved to Drive.
"""))

cells.append(md(r"""## Cell A1 — install MedGemma deps (transformers>=4.50)"""))

cells.append(code(r"""!pip -q install --upgrade 'transformers>=4.50' 'accelerate>=0.30' huggingface_hub pandas
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 50), \
    "Need transformers>=4.50 for MedGemma; restart runtime if it just upgraded."
"""))

cells.append(md(r"""## Cell A2 — Hugging Face login (MedGemma is gated)

Accept the license at https://huggingface.co/google/medgemma-4b-it first.
"""))

cells.append(code(r"""from huggingface_hub import notebook_login
notebook_login()
"""))

cells.append(md(r"""## Cell A3 — label report-language uncertainty for each positive finding

For each positive (study, finding) pair we give MedGemma the parsed report
section text (`section_content`) and the finding (`target`), and ask for a
single word `certain`/`uncertain` under the same definition used in the
PadChest study: uncertain = the report hedges about the finding's *existence,
visibility, diagnosis, or boundary*. Output is cached to Drive so Phase B can
read it after the restart.
"""))

cells.append(code(r"""import os, re, json
os.environ["HF_HUB_DISABLE_XET"] = "1"   # Xet backend can hang on Colab; use plain HTTPS
import pandas as pd
import torch
from tqdm.auto import tqdm

manifest = pd.read_csv(SUBSET_MANIFEST)

pos = manifest[manifest.polarity == "positive"].copy()
# one MedGemma call per (study_id, target); report text is per study
pos["key"] = pos["study_id"] + "|" + pos["target"].astype(str)
uniq = pos.drop_duplicates("key")[["study_id","target","section_name","section_content"]]
print(f"positive pairs: {len(pos)} | unique (study,target) to label: {len(uniq)}")

FEWSHOT = (
    "Examples:\n"
    "Finding: 'pneumonia' | Report: 'Findings suggestive of pneumonia in the right base.' -> uncertain\n"
    "Finding: 'effusion' | Report: 'Moderate right pleural effusion.' -> certain\n"
    "Finding: 'opacity' | Report: 'Ill-defined opacity, possibly atelectasis versus consolidation.' -> uncertain\n"
    "Finding: 'cardiomegaly' | Report: 'The cardiac silhouette is enlarged.' -> certain\n"
    "Finding: 'edema' | Report: 'Mild pulmonary edema.' -> certain\n"
    "Finding: 'consolidation' | Report: 'Cannot exclude early consolidation at the left base.' -> uncertain\n"
)

def build_prompt(target, report):
    return (
        "You label radiology report language as certain or uncertain about a "
        "specific finding. A finding is 'uncertain' if the report hedges about "
        "its EXISTENCE, VISIBILITY, DIAGNOSIS, or BOUNDARY (e.g. possible, "
        "probable, cannot exclude, suggestive of, ill-defined, subtle, vs). "
        "Otherwise it is 'certain'. Answer with EXACTLY ONE lowercase word: "
        "certain or uncertain.\n\n"
        f"{FEWSHOT}\n"
        f"Finding: '{target}' | Report: '{str(report)[:600]}' ->"
    )

from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL_NAME = "google/medgemma-4b-it"
print(f"Loading {MODEL_NAME} ...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tok = getattr(processor, "tokenizer", None) or processor
tok.padding_side = "left"
if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
    tok.pad_token = tok.eos_token
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()

ALLOWED = {"certain", "uncertain"}
def classify(target, report):
    prompt = build_prompt(target, report)
    msgs = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    w = re.sub(r"[^a-z]", "", raw.strip().lower().split()[0]) if raw.strip() else "certain"
    return (w if w in ALLOWED else "certain"), raw.strip()

labels = {}
for row in tqdm(uniq.itertuples(), total=len(uniq), desc="MedGemma uncertainty"):
    lab, raw = classify(row.target, row.section_content)
    labels[f"{row.study_id}|{row.target}"] = {"uncertainty_label": lab, "raw": raw,
                                               "study_id": row.study_id, "target": row.target}

with open(UNC_LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)

lab_series = pd.Series([v["uncertainty_label"] for v in labels.values()])
print("\nSaved", UNC_LABELS_PATH)
print("label distribution (per study,target):")
print(lab_series.value_counts().to_string())

# free VRAM
import gc
del model, processor
gc.collect(); torch.cuda.empty_cache()
print("\nMedGemma unloaded. Proceed to Cell B1, then RESTART the runtime.")
"""))

# =============================================================================
# PHASE B — ROSALIA
# =============================================================================
cells.append(md(r"""# ═════════════════════ PHASE B — ROSALIA segmentation ═════════════════════

Cell B1 installs the ROSALIA stack (downgrades transformers -> incompatible
with MedGemma). After B1, **restart the runtime**, then run B2 onward. The
uncertainty labels from Phase A are on Drive and survive the restart.
"""))

cells.append(md(r"""## Cell B1 — install ROSALIA deps + clone repo, then RESTART"""))

cells.append(code(r"""import os, sys, subprocess

TRANSFORMERS_VER = "4.34.1"
TOKENIZERS_VER   = "0.14.1"
import torch as _t
print(f"Using preinstalled torch {_t.__version__} (CUDA {_t.version.cuda})")

ROSALIA_REPO_DIR = "/content/rosalia_repo"
if (not os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py")
    or not os.path.isfile(f"{ROSALIA_REPO_DIR}/utils/utils.py")):
    if os.path.isdir(ROSALIA_REPO_DIR):
        subprocess.run(["rm","-rf",ROSALIA_REPO_DIR], check=True)
    _r = subprocess.run(["git","clone","--depth","1",
        "https://github.com/checkoneee/ROSALIA.git", ROSALIA_REPO_DIR],
        capture_output=True, text=True)
    print(_r.stdout[-500:]); print(_r.stderr[-500:])
assert os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py"), "clone failed"
print("ROSALIA repo OK")

r = subprocess.run(["pip","install","-q","--only-binary=:all:",
                    f"tokenizers=={TOKENIZERS_VER}"], capture_output=True, text=True)
print(r.stdout[-400:]); print(r.stderr[-400:])
!pip -q install "transformers=={TRANSFORMERS_VER}" 'peft==0.4.0' 'einops==0.4.1' \
    'sentencepiece' 'opencv-python>=4.10' 'pycocotools' 'scikit-image' 'bitsandbytes'
!pip -q install 'huggingface_hub>=0.16.4,<1.0'
subprocess.run(["python","-c",
    "import numpy,transformers,tokenizers,torch,cv2;"
    "print('numpy',numpy.__version__,'transformers',transformers.__version__,"
    "'tokenizers',tokenizers.__version__)"])
print("\n>>> RESTART THE RUNTIME NOW, then run Cell B2 onward. <<<")
"""))

cells.append(md(r"""## Cell B2 — Hugging Face login (ROSALIA/LISA weights are public)"""))

cells.append(code(r"""from huggingface_hub import notebook_login
notebook_login()
"""))

cells.append(md(r"""## Cell B3 — config + re-declare paths (post-restart)"""))

cells.append(code(r"""import os
# Re-declared because the restart cleared state (must match Cell S0).
try:
    from google.colab import drive; drive.mount('/content/drive'); DRIVE=True
except Exception as e:
    print("[warn]", e); DRIVE=False

DRIVE_DATA_DIR = "/content/drive/MyDrive/MIMIC-CXR-Ext-ILS"
EXT_ZIP    = os.path.join(DRIVE_DATA_DIR, "mimic-cxr-ext-ils.zip")

WORK_DIR = "/content/drive/MyDrive/mimic_ils_rosalia" if DRIVE else "/content/mimic_ils_rosalia"
os.makedirs(WORK_DIR, exist_ok=True)
UNC_LABELS_PATH = os.path.join(WORK_DIR, "medgemma_uncertainty_subset.json")
REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"

# ---- which manifest/split to run (MUST match Cell S0) ----------------------
# "subset"=smoke subset | "test"=held-out TEST | "val"=held-out VAL | "valtest"=VAL+TEST
MANIFEST_CHOICE = "test"          # "subset" | "test" | "val" | "valtest"
# Skip negatives -> ~halve inference. The IoU regression only needs positives;
# set False if you also want the abstention / negative analysis.
POSITIVES_ONLY  = True
_MANIFEST_FILES = {
    "subset":  ["mimic_ils_subset_manifest.csv"],
    "test":    ["mimic_ils_test_manifest.csv"],
    "val":     ["mimic_ils_val_manifest.csv"],
    "valtest": ["mimic_ils_test_manifest.csv", "mimic_ils_val_manifest.csv"],
}[MANIFEST_CHOICE]
_IMG_ZIP_NAMES = {
    "subset":  ["mimic_subset.zip"],
    "test":    ["mimic_subset (1).zip"],
    "val":     ["mimic_val_subset.zip"],
    "valtest": ["mimic_subset (1).zip", "mimic_val_subset.zip"],
}[MANIFEST_CHOICE]
IMG_ZIPS   = [os.path.join(DRIVE_DATA_DIR, z) for z in _IMG_ZIP_NAMES]
SUBSET_ZIP = IMG_ZIPS[0]          # back-compat alias

import pandas as _pd
_dfs = []
for _f in _MANIFEST_FILES:
    _p = f"/content/{_f}"
    if not os.path.exists(_p):
        os.system(f"curl -fsSL '{REPO_RAW_URL}/project/data/mimic_ils/{_f}' -o '{_p}'")
    _dfs.append(_pd.read_csv(_p))
SUBSET_MANIFEST = f"/content/mimic_ils_{MANIFEST_CHOICE}_manifest.csv"
_pd.concat(_dfs, ignore_index=True).to_csv(SUBSET_MANIFEST, index=False)
print(f"[config] MANIFEST_CHOICE={MANIFEST_CHOICE} -> {SUBSET_MANIFEST} | POSITIVES_ONLY={POSITIVES_ONLY}")


ROSALIA_REPO      = "checkone/ROSALIA-7B-v1"
LISA_TOKENIZER    = "xinlai/LISA-7B-v1"
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14"

GRID = 512            # common grid for pred-vs-silver mask comparison
MASK_BIN_THRESH = 127 # silver PNG is 0/255; >127 -> foreground
LESIONS = ["cardiomegaly","pneumonia","atelectasis","opacity",
           "consolidation","edema","effusion"]
print("Config loaded. zips present? subset:", os.path.exists(SUBSET_ZIP),
      "| ext:", os.path.exists(EXT_ZIP))
"""))

cells.append(md(r"""## Cell B3b — extract the zips to fast local disk + auto-detect roots

Unzips `mimic_subset.zip` and `mimic-cxr-ext-ils.zip` to `/content` (fast local
SSD — far quicker than reading 700 files over Drive during the run), then
auto-detects `MIMIC_SUBSET_DIR` (the folder that contains `p10/ p11/ ...`) and
`LESION_MASK_DIR` (the `lesion_mask/` folder) regardless of how the zips nest
their contents. Re-running is cheap (skips extraction if already unzipped).
"""))

cells.append(code(r"""import os, re, zipfile, glob

EXTRACT_ROOT = "/content/mimic_data"
IMG_EXTRACT  = os.path.join(EXTRACT_ROOT, "images")
EXT_EXTRACT  = os.path.join(EXTRACT_ROOT, "ext")

def _unzip(zip_path, dest):
    # per-zip sentinel so multiple image zips (e.g. val+test) all extract into
    # the same dest instead of the 2nd being skipped by a shared marker.
    os.makedirs(dest, exist_ok=True)
    sentinel = os.path.join(dest, "._extracted_" + re.sub(r"\W+", "_", os.path.basename(zip_path)))
    if os.path.exists(sentinel):
        print(f"[skip] already extracted {os.path.basename(zip_path)} -> {dest}")
        return
    assert os.path.exists(zip_path), f"missing zip: {zip_path} (add the shared folder shortcut to My Drive)"
    print(f"unzipping {os.path.basename(zip_path)} -> {dest} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    open(sentinel, "w").close()
    print("  done.")

for _z in IMG_ZIPS:                 # one or more image zips (val+test both extract)
    _unzip(_z, IMG_EXTRACT)
_unzip(EXT_ZIP, EXT_EXTRACT)


def find_image_roots(base):
    # ALL roots that directly contain pXX/ dirs (multiple when val+test zips
    # extract into different top folders, e.g. mimic_subset/ and mimic_val_subset/).
    roots = []
    for root, dirs, _ in os.walk(base):
        if any(re.fullmatch(r"p1\d", d) for d in dirs):
            roots.append(root)
            dirs[:] = [d for d in dirs if not re.fullmatch(r"p1\d", d)]  # don't descend into pXX
    if not roots:
        raise RuntimeError(f"could not find p1x/ image folders under {base}")
    return roots

def find_mask_root(base):
    for root, dirs, _ in os.walk(base):
        if os.path.basename(root) == "lesion_mask":
            return root
        if "lesion_mask" in dirs:
            return os.path.join(root, "lesion_mask")
    raise RuntimeError(f"could not find lesion_mask/ under {base}")

MIMIC_SUBSET_DIRS = find_image_roots(IMG_EXTRACT)
MIMIC_SUBSET_DIR  = MIMIC_SUBSET_DIRS[0]   # back-compat alias
LESION_MASK_DIR   = find_mask_root(EXT_EXTRACT)
print("MIMIC_SUBSET_DIRS =", MIMIC_SUBSET_DIRS)
print("LESION_MASK_DIR   =", LESION_MASK_DIR)
for _rt in MIMIC_SUBSET_DIRS:
    print(f"  {_rt}: #patient dirs = {len([d for d in os.listdir(_rt) if re.fullmatch(r'p1\d', d)])}")
print("  #mask studies :", len(os.listdir(LESION_MASK_DIR)))
"""))


cells.append(md(r"""## Cell B4 — load manifest + MedGemma labels; local image & mask loaders


The MIMIC-CXR-JPG images are already 8-bit, so (unlike the PadChest notebook)
we do **no percentile windowing** — we load the JPG straight to RGB. Silver
masks are 512×512 binary PNGs; a positive pair can reference one mask file.
"""))

cells.append(code(r"""import os, json
import numpy as np
import pandas as pd
from PIL import Image

manifest = pd.read_csv(SUBSET_MANIFEST)

# merge MedGemma uncertainty labels (per study_id|target) onto positive pairs
unc = {}
if os.path.exists(UNC_LABELS_PATH):
    unc = {k: v["uncertainty_label"] for k, v in json.load(open(UNC_LABELS_PATH)).items()}
    print(f"loaded {len(unc)} MedGemma uncertainty labels")
else:
    print("[warn] no MedGemma labels found; uncertainty stratification will be 'unknown'")
manifest["uncertainty_label"] = manifest.apply(
    lambda r: unc.get(f"{r.study_id}|{r.target}", "unknown"), axis=1)

pos = manifest[manifest.polarity == "positive"].reset_index(drop=True)
neg = manifest[manifest.polarity == "negative"].reset_index(drop=True)
print(f"positive pairs: {len(pos)} | negative pairs: {len(neg)}")
print("positive per-lesion:", pos.target.value_counts().to_dict())

# images may live under several roots (val+test zips extract to different top
# folders); resolve each image_path across all roots, with a small cache.
_IMG_PATH_CACHE = {}
def resolve_image(image_path):
    if not isinstance(image_path, str):   # NaN/float from unmatched merges -> no image
        return None
    if image_path in _IMG_PATH_CACHE:
        return _IMG_PATH_CACHE[image_path]

    for _rt in MIMIC_SUBSET_DIRS:
        fp = os.path.join(_rt, image_path)
        if os.path.exists(fp):
            _IMG_PATH_CACHE[image_path] = fp
            return fp
    _IMG_PATH_CACHE[image_path] = None
    return None

def load_image(image_path):
    fp = resolve_image(image_path)
    if fp is None:
        raise FileNotFoundError(f"{image_path} not found under any of {MIMIC_SUBSET_DIRS}")
    return Image.open(fp).convert("RGB")


def load_silver_mask(seg_mask_path):
    fp = os.path.join(LESION_MASK_DIR, seg_mask_path)
    m = np.array(Image.open(fp).convert("L"))
    return (m > MASK_BIN_THRESH).astype(np.uint8)

# ---- coverage check: how many positive images are actually on local disk? ----
_present = pos["image_path"].map(lambda p: resolve_image(p) is not None)
print(f"positive images present locally: {int(_present.sum())}/{len(pos)}")
if _present.sum() == 0:
    print("[ERROR] none of the manifest's images are in the extracted zip(s).")
    for _rt in MIMIC_SUBSET_DIRS:
        print(f"        {_rt}: {sorted(os.listdir(_rt))[:20]}")

    print("        -> your image zip is incomplete/wrong for this split. It must contain")
    print("           every path in the split's *_image_list.txt (pXX/pXXXXX/sSTUDY/<dicom>.jpg).")
    raise FileNotFoundError("no manifest images found on disk; rebuild the image zip for this split.")
elif _present.sum() < len(pos):
    print(f"[warn] {int((~_present).sum())} positive images are missing from the zip and will be skipped in B8.")

# smoke check on the first positive pair whose image is present
_r = pos[_present].iloc[0]
_img = load_image(_r.image_path); _gt = load_silver_mask(_r.seg_mask_path)
print(f"OK - {_r.study_id} {_r.target}: image={_img.size}, "
      f"silver_mask={_gt.shape}, fg_px={int(_gt.sum())}, instr={_r.instruction!r}")
"""))


cells.append(md(r"""## Cell B5 — load ROSALIA (LISA-7B + SAM-H)

Same loader/compat-shim as the PadChest notebook. Defines
`segment(pil_image, instruction) -> (binary_mask_at_image_size, logits, text)`.
"""))

cells.append(code(r"""import sys, os, subprocess, faulthandler
faulthandler.enable()
os.environ["HF_HUB_DISABLE_XET"] = "1"   # Xet backend can hang on Colab; use plain HTTPS
ROSALIA_REPO_DIR = "/content/rosalia_repo"

if not (os.path.isfile(f"{ROSALIA_REPO_DIR}/model/LISA.py")
        and os.path.isfile(f"{ROSALIA_REPO_DIR}/utils/utils.py")):
    if os.path.isdir(ROSALIA_REPO_DIR): subprocess.run(["rm","-rf",ROSALIA_REPO_DIR])
    subprocess.run(["git","clone","--depth","1",
        "https://github.com/checkoneee/ROSALIA.git", ROSALIA_REPO_DIR])
sys.path.insert(0, ROSALIA_REPO_DIR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as _np
if _np.__version__.startswith("1."):
    subprocess.run([sys.executable,"-m","pip","install","-q","--force-reinstall",
                    "--no-deps","numpy>=2"], check=True)
    raise RuntimeError("numpy upgraded to 2.x -> Runtime > Restart, then re-run B2-B5.")

import cv2, numpy as np, torch, torch.nn.functional as F
import transformers
from transformers.models.auto.configuration_auto import _LazyConfigMapping
_orig = _LazyConfigMapping.register
_LazyConfigMapping.register = lambda self,k,v,exist_ok=False: _orig(self,k,v,exist_ok=True)
try:
    from transformers.models.auto.auto_factory import _LazyAutoMapping
    _o2 = _LazyAutoMapping.register
    def _s2(self,k,v,exist_ok=False):
        try: return _o2(self,k,v,exist_ok=True)
        except TypeError:
            try: return _o2(self,k,v)
            except ValueError: pass
    _LazyAutoMapping.register = _s2
except Exception as e: print("[warn]", e)

from transformers import CLIPImageProcessor
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

IMAGE_SIZE = 1024
def _sam_preprocess(x, img_size=IMAGE_SIZE):
    mean = torch.tensor([123.675,116.28,103.53]).view(-1,1,1)
    std  = torch.tensor([58.395,57.12,57.375]).view(-1,1,1)
    x = (x-mean)/std
    h,w = x.shape[-2:]
    return F.pad(x, (0, img_size-w, 0, img_size-h))

def load_rosalia():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        LISA_TOKENIZER, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = LISAForCausalLM.from_pretrained(
        ROSALIA_REPO, low_cpu_mem_usage=True, vision_tower=CLIP_VISION_TOWER,
        seg_token_idx=seg_idx, torch_dtype=torch.bfloat16)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch.bfloat16, device=0)
    model = model.bfloat16().cuda().eval()
    clip = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    tfm = ResizeLongestSide(IMAGE_SIZE)
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer, clip, tfm

# robust, resumable weight download (avoids ChunkedEncodingError / IncompleteRead
# when Colab's connection to HuggingFace drops mid-download of the ~10GB shard)
from huggingface_hub import snapshot_download
for _a in range(8):
    try:
        snapshot_download(ROSALIA_REPO, resume_download=True, max_workers=2)
        print("[weights] ROSALIA snapshot fully downloaded"); break
    except Exception as _e:
        print(f"[weights][retry {_a+1}/8] {type(_e).__name__}: {_e}")
else:
    raise RuntimeError("ROSALIA weight download kept failing; rerun this cell (it resumes).")

model, tokenizer, clip_processor, transform = load_rosalia()


def segment(pil_image, instruction):
    image_np = np.array(pil_image)
    if image_np.ndim == 2: image_np = np.stack([image_np]*3, -1)
    elif image_np.shape[-1] == 4: image_np = image_np[...,:3]
    orig = [image_np.shape[:2]]
    conv = conversation_lib.conv_templates["llava_v1"].copy(); conv.messages=[]
    p = DEFAULT_IMAGE_TOKEN + "\n" + instruction
    p = p.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], p); conv.append_message(conv.roles[1], "")
    p = conv.get_prompt()
    image_clip = clip_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().bfloat16()
    image_resized = transform.apply_image(image_np)
    resize_list = [image_resized.shape[:2]]
    image = _sam_preprocess(torch.from_numpy(image_resized).permute(2,0,1).contiguous()).unsqueeze(0).cuda().bfloat16()
    input_ids = tokenizer_image_token(p, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(image_clip, image, input_ids,
            resize_list, orig, max_new_tokens=256, tokenizer=tokenizer)
    ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    txt = tokenizer.decode(ids, skip_special_tokens=False)
    txt = txt.replace("\n","").replace("  "," ").split("ASSISTANT:")[-1].split("</s>")[0].strip()
    if len(pred_masks)==0 or pred_masks[0].numel()==0:
        return np.zeros(orig[0],np.uint8), None, txt
    lm = pred_masks[0]
    if lm.ndim==3: lm = lm[0]
    logits = lm.float().detach().cpu().numpy()
    return (logits>0).astype(np.uint8), logits, txt

_m,_l,_t = segment(_img, _r.instruction)
print(f"smoke: instr={_r.instruction!r} mask_px={int(_m.sum())} text={_t!r}")
"""))

cells.append(md(r"""## Cell B6 — metrics (mirror the ROSALIA/LISA paper)

* **IoU** and **Dice** per positive sample, computed on a common `GRID×GRID`
  raster (pred resized nearest to the silver-mask grid).
* **gIoU** = mean of per-sample IoU (LISA's "generalized IoU").
* **cIoU** = cumulative IoU = Σintersectionⱼ / Σunionⱼ over all samples.
* **Negatives:** abstention = pred mask empty. We report the empty-mask rate on
  negative pairs (higher is better) and on positives (lower is better).
"""))

cells.append(code(r"""import numpy as np
from skimage.transform import resize as _skresize

def to_grid(mask, grid=GRID):
    m = np.asarray(mask).astype(np.float32)
    if m.shape != (grid, grid):
        m = _skresize(m, (grid, grid), order=0, mode="edge",
                      anti_aliasing=False, preserve_range=True)
    return (m > 0.5).astype(np.uint8)

def iou_dice(pred, gt, grid=GRID):
    p = to_grid(pred, grid); g = to_grid(gt, grid)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    psum, gsum = int(p.sum()), int(g.sum())
    iou = inter/union if union > 0 else float("nan")
    dice = (2*inter)/(psum+gsum) if (psum+gsum) > 0 else float("nan")
    return iou, dice, inter, union

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x, dtype=np.float64)))

def logit_stats(logits, gt=None):
    # Per-example confidence summaries from ROSALIA's raw mask logits:
    #   max_prob        = peak sigmoid prob anywhere (did the model 'fire' hard?)
    #   mean_prob       = mean sigmoid prob over the whole map
    #   mean_prob_in_gt = mean sigmoid prob inside the silver-mask region
    if logits is None:

        return dict(max_logit=np.nan, max_prob=np.nan, mean_prob=np.nan, mean_prob_in_gt=np.nan)
    p = _sigmoid(logits)
    out = dict(max_logit=float(np.max(logits)), max_prob=float(p.max()),
               mean_prob=float(p.mean()), mean_prob_in_gt=float("nan"))
    if gt is not None and np.asarray(gt).sum() > 0:
        g = _skresize(np.asarray(gt).astype(np.float32), p.shape, order=0,
                      mode="edge", anti_aliasing=False, preserve_range=True) > 0.5
        if g.sum() > 0:
            out["mean_prob_in_gt"] = float(p[g].mean())
    return out

print("metrics ready. gIoU=mean per-sample IoU; cIoU=sum(inter)/sum(union).")
print("logit_stats -> max_prob, mean_prob, mean_prob_in_gt per example.")
"""))


cells.append(md(r"""## Cell B7 — SMOKE test: 6 positives with silver-mask overlay"""))

cells.append(code(r"""import matplotlib.pyplot as plt
from tqdm.auto import tqdm

smoke = pos.groupby("target").head(1).head(6)   # up to 1 per lesion
n = len(smoke)
fig, ax = plt.subplots(n, 4, figsize=(16, 4*n))
if n == 1: ax = ax.reshape(1,-1)
for i, row in enumerate(smoke.itertuples()):
    img = load_image(row.image_path); gt = load_silver_mask(row.seg_mask_path)
    pred, logits, txt = segment(img, row.instruction)
    iou, dice, *_ = iou_dice(pred, gt)
    iw, ih = img.size
    ax[i,0].imshow(img, cmap="gray"); ax[i,0].set_title(f"{row.target}"); ax[i,0].axis("off")
    ax[i,1].imshow(gt, cmap="Greens"); ax[i,1].set_title("silver mask"); ax[i,1].axis("off")
    ax[i,2].imshow(pred, cmap="Reds"); ax[i,2].set_title(f"pred (IoU={iou:.2f} Dice={dice:.2f})"); ax[i,2].axis("off")
    ax[i,3].imshow(img, cmap="gray")
    gt_disp = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((iw,ih), Image.NEAREST))
    pr_disp = np.array(Image.fromarray((pred*255).astype(np.uint8)).resize((iw,ih), Image.NEAREST))
    ax[i,3].imshow(np.ma.masked_where(gt_disp==0, gt_disp), cmap="Greens", alpha=0.45)
    ax[i,3].imshow(np.ma.masked_where(pr_disp==0, pr_disp), cmap="Reds", alpha=0.45)
    ax[i,3].set_title(f"[{row.uncertainty_label}] {txt[:40]}"); ax[i,3].axis("off")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"mimic_ils_smoke.png"), dpi=110); plt.show()
"""))

cells.append(md(r"""## Cell B8 — FULL RUN over positives (+ negatives for abstention)

Resumable: per-pair results are cached to a CSV on Drive; re-running skips
already-scored `pair_id`s.
"""))

cells.append(code(r"""import os, json
import pandas as pd
from tqdm.auto import tqdm

RESULTS_CSV = os.path.join(WORK_DIR, "rosalia_mimic_ils_per_pair.csv")
done = set()
if os.path.exists(RESULTS_CSV):
    done = set(pd.read_csv(RESULTS_CSV)["pair_id"].tolist())
    print(f"resuming; {len(done)} pairs already scored")

def append_row(rec):
    df = pd.DataFrame([rec])
    df.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

# ---- image-availability check (test manifest may reference images not in the
#      extracted subset zip; missing ones are reported here, then skipped below) ----
_avail = pos["image_path"].map(lambda p: resolve_image(p) is not None)
print(f"[check] positive images present locally: {int(_avail.sum())}/{len(pos)}")

if _avail.sum() < len(pos):
    print(f"[warn] {int((~_avail).sum())} positive images are NOT on local disk and will be skipped.")
    print("       For the full TEST split you need those images extracted (see the 500GB/GCS note).")

# ---- positives: IoU/Dice vs silver mask ----
for row in tqdm(pos.itertuples(), total=len(pos), desc="positives"):
    if row.pair_id in done: continue
    try:
        img = load_image(row.image_path); gt = load_silver_mask(row.seg_mask_path)
        pred, logits, txt = segment(img, row.instruction)
        iou, dice, inter, union = iou_dice(pred, gt)
        append_row({"pair_id":row.pair_id,"study_id":row.study_id,"split":row.split,
            "polarity":"positive","target":row.target,"location":row.location,
            "instruction":row.instruction,"uncertainty_label":row.uncertainty_label,
            "text_output":txt,"pred_px":int(pred.sum()),"gt_px":int(gt.sum()),
            "inter":inter,"union":union,"iou":iou,"dice":dice,"fired":int(pred.sum()>0),
            **logit_stats(logits, gt)})

    except Exception as e:
        print(f"[skip] {row.pair_id}: {type(e).__name__}: {e}")

# ---- negatives: should abstain (empty mask). Skipped when POSITIVES_ONLY. ----
if not POSITIVES_ONLY:
    neg_eval = neg.sample(n=min(len(neg), len(pos)), random_state=42)
    for row in tqdm(neg_eval.itertuples(), total=len(neg_eval), desc="negatives"):
        if row.pair_id in done: continue
        try:
            img = load_image(row.image_path)
            pred, logits, txt = segment(img, row.instruction)
            append_row({"pair_id":row.pair_id,"study_id":row.study_id,"split":row.split,
                "polarity":"negative","target":row.target,"location":row.location,
                "instruction":row.instruction,"uncertainty_label":row.uncertainty_label,
                "text_output":txt,"pred_px":int(pred.sum()),"gt_px":0,
                "inter":0,"union":int(pred.sum()),"iou":float("nan"),"dice":float("nan"),
                "fired":int(pred.sum()>0)})
        except Exception as e:
            print(f"[skip] {row.pair_id}: {type(e).__name__}: {e}")
else:
    print("[info] POSITIVES_ONLY=True -> skipping negatives (no abstention analysis).")

# de-duplicate rows from any earlier re-runs (keep the latest per pair_id) so
# downstream counts/regressions aren't inflated by duplicates.
results = pd.read_csv(RESULTS_CSV)
_before = len(results)
results = results.drop_duplicates("pair_id", keep="last")
if len(results) < _before:
    results.to_csv(RESULTS_CSV, index=False)
    print(f"[dedup] removed {_before - len(results)} duplicate pair_id rows")
print(f"\nDone. total scored rows: {len(results)}")
"""))


cells.append(md(r"""## Cell B9 — aggregate metrics (overall, per lesion, by uncertainty)

Reproduces the paper-style table (gIoU / cIoU / Dice) and adds the project's
uncertainty stratification.
"""))

cells.append(code(r"""import numpy as np, pandas as pd
results = pd.read_csv(RESULTS_CSV)
P = results[results.polarity=="positive"].copy()
N = results[results.polarity=="negative"].copy()

def block(df):
    d = df.dropna(subset=["iou"])
    giou = d["iou"].mean()
    ciou = d["inter"].sum()/d["union"].sum() if d["union"].sum()>0 else float("nan")
    dice = d["dice"].mean()
    return pd.Series({"n":len(df),"fire_rate":df["fired"].mean(),
                      "gIoU":giou,"cIoU":ciou,"mDice":dice})

print("="*60,"\nPOSITIVES — overall (paper metrics):")
print(block(P).round(4).to_string())

print("\nPOSITIVES — per lesion type:")
print(P.groupby("target").apply(block).round(4).to_string())

print("\nPOSITIVES — by MedGemma uncertainty label:")
print(P.groupby("uncertainty_label").apply(block).round(4).to_string())

print("\nPOSITIVES — lesion × uncertainty (gIoU):")
piv = P.dropna(subset=["iou"]).pivot_table(index="target",
        columns="uncertainty_label", values="iou", aggfunc="mean")
print(piv.round(3).to_string())

print("\n" + "="*60, "\nNEGATIVES — abstention (empty-mask rate; higher=better):")
print(f"  abstention rate = {(1-N['fired']).mean():.3f}  (n={len(N)})")
print(f"  positives fire rate = {P['fired'].mean():.3f}  (n={len(P)})")

# persist a compact JSON summary
summary = {
    "positives_overall": block(P).round(4).to_dict(),
    "per_lesion": {k: block(v).round(4).to_dict() for k,v in P.groupby("target")},
    "by_uncertainty": {k: block(v).round(4).to_dict() for k,v in P.groupby("uncertainty_label")},
    "negative_abstention_rate": float((1-N["fired"]).mean()),
}
import json, os
json.dump(summary, open(os.path.join(WORK_DIR,"rosalia_mimic_ils_summary.json"),"w"), indent=2)
print("\nsaved summary ->", os.path.join(WORK_DIR,"rosalia_mimic_ils_summary.json"))
"""))

cells.append(md(r"""## Cell B10 — plots"""))

cells.append(code(r"""import matplotlib.pyplot as plt, numpy as np, os
P = results[results.polarity=="positive"].dropna(subset=["iou"])

# per-lesion gIoU bar
fig, ax = plt.subplots(figsize=(9,4))
g = P.groupby("target")["iou"].mean().sort_values(ascending=False)
g.plot(kind="bar", ax=ax, color="teal")

ax.set_ylabel("gIoU (mean IoU vs silver)"); ax.set_title("ROSALIA on MIMIC-ILS subset: gIoU by lesion")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"giou_by_lesion.png"), dpi=120); plt.show()

# IoU distribution by uncertainty
fig, ax = plt.subplots(figsize=(7,4))
for lab, c in [("certain","tab:blue"),("uncertain","tab:orange"),("unknown","gray")]:
    s = P[P.uncertainty_label==lab]["iou"]
    if len(s): ax.hist(s, bins=30, alpha=0.55, label=f"{lab} (n={len(s)})", color=c)
ax.set_xlabel("IoU vs silver mask"); ax.set_ylabel("count")
ax.set_title("ROSALIA IoU by MedGemma report-language uncertainty"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"iou_by_uncertainty.png"), dpi=120); plt.show()

# certain vs uncertain COUNT per lesion (grouped bar)
Pall = results[results.polarity=="positive"].copy()
ct = (Pall.groupby(["target","uncertainty_label"]).size()
          .unstack(fill_value=0).reindex(LESIONS, fill_value=0))
for lab in ["certain","uncertain"]:
    if lab not in ct.columns: ct[lab] = 0
labels = list(ct.index); x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(11,4.5))
b1 = ax.bar(x - w/2, ct["certain"],   w, label="certain",   color="tab:blue")
b2 = ax.bar(x + w/2, ct["uncertain"], w, label="uncertain", color="tab:orange")
ax.bar_label(b1, fontsize=8); ax.bar_label(b2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("# positive findings"); ax.legend()
ax.set_title("MedGemma report-language uncertainty per lesion (certain vs uncertain counts)")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"uncertainty_counts_by_lesion.png"), dpi=120); plt.show()
print(ct[["certain","uncertain"]].to_string())
"""))


cells.append(md(r"""## Cell B10b — probability heat-map overlays: 10 uncertain + 10 certain

Re-runs ROSALIA on 10 uncertain and 10 certain positive findings and overlays
the **per-pixel sigmoid probability** map (jet) on the image, with the silver
mask outline (green). This is the same logit/probability visualization as
before, now split by MedGemma report-language uncertainty. If the hypothesis
holds, uncertain findings should show cooler / more diffuse probability maps.
"""))

cells.append(code(r"""import matplotlib.pyplot as plt, numpy as np, os
from matplotlib import cm

def _contour(ax, mask, color):
    try:
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
    except Exception:
        pass

def plot_heatmaps(df_sel, title, fname):
    n = len(df_sel)
    if n == 0:
        print("no examples for", title); return
    cols = 5; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_1d(axes).ravel()
    for k, row in enumerate(df_sel.itertuples()):
        ax = axes[k]
        img = load_image(row.image_path); gt = load_silver_mask(row.seg_mask_path)
        pred, logits, txt = segment(img, row.instruction)
        ax.imshow(img, cmap="gray")
        if logits is not None:
            prob = _sigmoid(logits)
            prob_r = _skresize(prob, np.array(img).shape[:2], order=1,
                               mode="edge", anti_aliasing=True, preserve_range=True)
            im = ax.imshow(prob_r, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            mp = float(prob.max())
        else:
            mp = float("nan")
        gt_r = _skresize(gt.astype(float), np.array(img).shape[:2], order=0,
                         mode="edge", anti_aliasing=False, preserve_range=True)
        _contour(ax, gt_r, "lime")
        iou, dice, *_ = iou_dice(pred, gt)
        ax.set_title(f"{row.target} | IoU={iou:.2f}\nmax_p={mp:.2f}", fontsize=9)
        ax.axis("off")
    for k in range(n, len(axes)): axes[k].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(right=0.9); cbar_ax = fig.add_axes([0.92,0.15,0.015,0.7])
    fig.colorbar(cm.ScalarMappable(cmap="jet"), cax=cbar_ax, label="sigmoid prob")
    plt.savefig(os.path.join(WORK_DIR, fname), dpi=110, bbox_inches="tight"); plt.show()

# prefer examples where the model actually produced a mask, for legible heatmaps
_pos_fire = pos.copy()
if os.path.exists(RESULTS_CSV):
    _fired = pd.read_csv(RESULTS_CSV)
    _fired = set(_fired[(_fired.polarity=="positive") & (_fired.fired==1)]["pair_id"])
    _pref = pos[pos.pair_id.isin(_fired)]
    if len(_pref) >= 10: _pos_fire = _pref

sel_unc = _pos_fire[_pos_fire.uncertainty_label=="uncertain"].head(10)
sel_cer = _pos_fire[_pos_fire.uncertainty_label=="certain"].head(10)
plot_heatmaps(sel_unc, "UNCERTAIN findings — ROSALIA probability heat-map", "heatmaps_uncertain.png")
plot_heatmaps(sel_cer, "CERTAIN findings — ROSALIA probability heat-map",   "heatmaps_certain.png")
"""))

cells.append(md(r"""## Cell B10c — subset-wide probability distributions + significance test

Over **all** scored positives, plot the distribution of ROSALIA's confidence
(`max_prob` = peak sigmoid probability, and `mean_prob_in_gt` = mean probability
inside the silver mask) separately for **certain** vs **uncertain** findings,
then run a one-sided **Mann–Whitney U** test of the hypothesis
*uncertain < certain* (non-parametric; also report Cohen's d and medians).
"""))

cells.append(code(r"""import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

results = pd.read_csv(RESULTS_CSV)
P = results[results.polarity=="positive"].copy()

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float("nan")
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2)/(na+nb-2))
    return (a.mean()-b.mean())/sp if sp > 0 else float("nan")

METRICS = [("max_prob", "peak sigmoid probability"),
           ("mean_prob_in_gt", "mean prob inside silver mask")]

fig, axes = plt.subplots(1, len(METRICS), figsize=(7*len(METRICS), 4.5))
axes = np.atleast_1d(axes)
for ax, (metric, nice) in zip(axes, METRICS):
    cer = P[P.uncertainty_label=="certain"][metric].dropna().values
    unc = P[P.uncertainty_label=="uncertain"][metric].dropna().values
    bins = np.linspace(0, 1, 31)
    ax.hist(cer, bins=bins, alpha=0.55, density=True, color="tab:blue",
            label=f"certain (n={len(cer)}, med={np.median(cer):.2f})")
    ax.hist(unc, bins=bins, alpha=0.55, density=True, color="tab:orange",
            label=f"uncertain (n={len(unc)}, med={np.median(unc):.2f})")
    ax.axvline(np.median(cer), color="tab:blue", ls="--"); ax.axvline(np.median(unc), color="tab:orange", ls="--")
    if len(cer) >= 3 and len(unc) >= 3:
        U, p = mannwhitneyu(unc, cer, alternative="less")   # H1: uncertain < certain
        d = cohens_d(unc, cer)
        ax.set_title(f"{nice}\nMann-Whitney U (uncertain<certain): p={p:.3g}, d={d:.2f}")
        print(f"[{metric}] certain med={np.median(cer):.3f} mean={cer.mean():.3f} | "
              f"uncertain med={np.median(unc):.3f} mean={unc.mean():.3f} | "
              f"U={U:.0f} p(one-sided uncertain<certain)={p:.3g} Cohen_d={d:.3f}")
    else:
        ax.set_title(f"{nice}\n(insufficient n for test)")
    ax.set_xlabel(metric); ax.set_ylabel("density"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"prob_dist_by_uncertainty.png"), dpi=120); plt.show()

# box/strip summary for the primary metric
fig, ax = plt.subplots(figsize=(5,4))
data = [P[P.uncertainty_label=="certain"]["max_prob"].dropna(),
        P[P.uncertainty_label=="uncertain"]["max_prob"].dropna()]
ax.boxplot(data, labels=["certain","uncertain"], showmeans=True)
ax.set_ylabel("max_prob (peak sigmoid)"); ax.set_title("ROSALIA peak confidence by uncertainty")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"maxprob_box.png"), dpi=120); plt.show()
"""))

cells.append(md(r"""## Cell B10d — example gallery per uncertainty TYPE

For each uncertainty **type** (certain / presence / spatial / diagnostic /
borderline, from the regression notebook's cached `unc_type` labels on Drive),
show `N_PER_TYPE` examples with: the **image**, the **target silver mask**, and
**ROSALIA's predicted mask** overlaid (red=pred, green=silver), annotated with
the **disease** and **IoU**. Falls back to the binary certain/uncertain label if
the 5-way type cache isn't present yet.

⚠️ Predicted masks aren't cached, so this cell re-runs ROSALIA `segment()` on the
shown examples (needs Cell B5 loaded). At `N_PER_TYPE=15` that's up to 75
inferences — a few minutes on a GPU.
"""))

cells.append(code(r"""import os, textwrap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image


N_PER_TYPE = 15         # examples shown per uncertainty type
# de-emphasise cardiomegaly (it dominates & its "mask" is the whole heart):
EXCLUDE_TARGETS = []               # e.g. ["cardiomegaly"] to drop it entirely
MAX_PER_DISEASE = {"cardiomegaly": 2}   # cap how many of a disease appear per type

LABELS5 = ["certain", "presence", "spatial", "diagnostic", "borderline"]

# defined here too so this cell runs standalone (B8 may have been skipped)
RESULTS_CSV = os.path.join(WORK_DIR, "rosalia_mimic_ils_per_pair.csv")
TYPES_CSV = os.path.join(WORK_DIR, "uncertainty_types_regression.csv")
assert os.path.exists(RESULTS_CSV), f"per-pair CSV not found: {RESULTS_CSV} (run B8 first)"

res = pd.read_csv(RESULTS_CSV)

resP = res[res.polarity == "positive"].copy()
# attach image/mask paths + report text from the manifest (CSV doesn't store them)
_cols = ["pair_id", "image_path", "seg_mask_path"] + \
        [c for c in ["section_content", "section_name"] if c in pos.columns]
resP = resP.merge(pos[_cols], on="pair_id", how="left")


def _key(df):
    return df["study_id"].astype(str) + "|" + df["target"].astype(str)

if os.path.exists(TYPES_CSV):
    _t = pd.read_csv(TYPES_CSV)
    _map = dict(zip(_t["study_id"].astype(str) + "|" + _t["target"].astype(str), _t["unc_type"]))
    resP["unc_type"] = _key(resP).map(_map)
    TYPES = [t for t in LABELS5 if (resP["unc_type"] == t).any()]
    print(f"using 5-way unc_type from {TYPES_CSV}")
else:
    resP["unc_type"] = resP["uncertainty_label"]
    TYPES = [t for t in ["certain", "uncertain", "unknown"] if (resP["unc_type"] == t).any()]
    print("[warn] no 5-way type cache found; falling back to binary uncertainty_label")

print("examples available per type:",
      {t: int((resP["unc_type"] == t).sum()) for t in TYPES})

def select_for_type(unc_type):
    # fired-first ordering, drop excluded diseases + those without images on disk,
    # then cap how many of any one disease appear (e.g. cardiomegaly) per type.
    cand = pd.concat([_fired[_fired["unc_type"] == unc_type],
                      resP[resP["unc_type"] == unc_type]])
    cand = cand[~cand.index.duplicated(keep="first")]
    cand = cand[~cand["target"].isin(EXCLUDE_TARGETS)]
    cand = cand[cand["image_path"].notna()]      # val/train rows unmatched by the test manifest
    cand = cand[cand["image_path"].map(lambda p: resolve_image(p) is not None)]

    picked, per = [], {}
    for row in cand.itertuples():
        cap = MAX_PER_DISEASE.get(row.target)
        if cap is not None and per.get(row.target, 0) >= cap:
            continue
        per[row.target] = per.get(row.target, 0) + 1
        picked.append(row.Index)
        if len(picked) >= N_PER_TYPE:
            break
    return cand.loc[picked]

def _color_overlay(ax, mask_bool, rgb, alpha=0.5):
    # draw a SOLID-colour transparent overlay (avoids colormap-normalisation
    # turning a constant mask value into near-white).
    h, w = mask_bool.shape
    ov = np.zeros((h, w, 4), dtype=float)
    ov[..., 0], ov[..., 1], ov[..., 2] = rgb
    ov[..., 3] = np.where(mask_bool, alpha, 0.0)
    ax.imshow(ov)

def show_examples(df_sel, unc_type):
    df_sel = df_sel.head(N_PER_TYPE)
    if len(df_sel) == 0:

        print(f"[{unc_type}] no examples with images on disk"); return
    n = len(df_sel)
    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1: ax = ax.reshape(1, -1)
    for i, row in enumerate(df_sel.itertuples()):
        img = load_image(row.image_path); gt = load_silver_mask(row.seg_mask_path)
        pred, logits, txt = segment(img, row.instruction)
        iou, dice, *_ = iou_dice(pred, gt)
        iw, ih = img.size
        gt_b = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize((iw, ih), Image.NEAREST)) > 0
        pr_b = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize((iw, ih), Image.NEAREST)) > 0
        ax[i, 0].imshow(img, cmap="gray"); ax[i, 0].axis("off")
        ax[i, 0].set_title(f"{row.target}  [{unc_type}]")
        ax[i, 1].imshow(img, cmap="gray")
        _color_overlay(ax[i, 1], gt_b, (0, 1, 0), alpha=0.5)          # silver = green
        ax[i, 1].set_title("target (silver) mask"); ax[i, 1].axis("off")
        ax[i, 2].imshow(img, cmap="gray")
        _color_overlay(ax[i, 2], gt_b, (0, 1, 0), alpha=0.40)         # silver = green
        _color_overlay(ax[i, 2], pr_b, (1, 0, 0), alpha=0.45)         # pred   = red
        ax[i, 2].set_title(f"pred (red) vs silver (green)\nIoU={iou:.2f}  Dice={dice:.2f}")
        ax[i, 2].axis("off")
        # report finding text under the row (wrapped) + printed to stdout
        report = getattr(row, "section_content", None)
        if isinstance(report, str) and report.strip():
            snippet = " ".join(report.split())
            wrapped = textwrap.fill(f"Report: {snippet}", width=110)
            ax[i, 0].text(0.0, -0.04, wrapped, transform=ax[i, 0].transAxes,
                          ha="left", va="top", fontsize=8, color="black", wrap=True)
            print(f"[{unc_type}] {row.target} (IoU={iou:.2f}): {snippet[:300]}")

    fig.suptitle(f"Uncertainty type: {unc_type}", fontsize=14)
    plt.tight_layout()

    _pp = os.path.join(WORK_DIR, f"examples_{unc_type}.png")
    plt.savefig(_pp, dpi=110, bbox_inches="tight"); plt.show()
    print("saved ->", _pp)

# prefer examples where ROSALIA actually produced a mask (legible overlays);
# select_for_type() tops up with non-firing rows and applies EXCLUDE/cap.
_fired = resP[resP.get("fired", 1) == 1] if "fired" in resP.columns else resP
for _ty in TYPES:
    sel = select_for_type(_ty)
    print(f"[{_ty}] showing {len(sel)} (disease mix: {sel['target'].value_counts().to_dict()})")
    show_examples(sel, _ty)
"""))


cells.append(md(r"""## Cell B11 — notes on reproduction



* **gIoU / cIoU** are the LISA-family metrics; **Dice** is included for
  comparability with segmentation papers. Compare the per-lesion gIoU here to
  the ROSALIA paper's test-set table (this subset mixes train/val/test, so
  numbers on *train* studies will be optimistic — filter `split=='test'` in
  Cell B9 for a clean held-out comparison).
* **Silver masks** are automatically generated pseudo-labels, so IoU is against
  a noisy target — treat absolute numbers accordingly; the *relative* pattern
  (per-lesion ordering, certain vs uncertain) is the informative part.
* To restrict to held-out test studies only, add `P = P[P.split=='test']` at the
  top of Cell B9.
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
_default_out = Path(__file__).resolve().parent / "project" / "notebooks" / "mimic_ils_rosalia_reproduction_colab.ipynb"
out_path = Path(os.environ.get("NOTEBOOK_OUT", _default_out))
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")
