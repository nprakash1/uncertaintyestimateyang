"""Build project/notebooks/text_conditioning_ablation_colab.ipynb

Question answered: does the TEXT instruction actually change RoSALIA's
segmentation, or does it segment from image/anatomy priors and largely ignore
the prompt?

Design (discussed with the user):
  * For every POSITIVE study x finding pair (has a silver lesion mask), run
    RoSALIA with 4 prompt "flavors" on the SAME image:
        correct        - the manifest instruction (right disease + right place)
        wrong_disease  - swap the finding to a different one (in-distribution)
        wrong_location - swap the location to a spatially DISJOINT one
        wrong_both     - both wrong
  * Corruptions are "distribution-aware" (Option A): the swapped disease/location
    is rendered with a template that disease actually uses in the manifest, so the
    prompt stays in-distribution and only the MEANING is wrong (no
    "cardiomegaly in the left lung" out-of-distribution strings).
  * OPTIONAL Option B: MedGemma proposes the corruptions; every field is VALIDATED
    against the same allowed vocab + grammar rules and FALLS BACK to the
    deterministic Option-A value on any violation (nothing is dropped).
  * Measure two things per pair:
        (a) accuracy vs GT   : IoU / Dice / p_in_gt for each flavor -> paired
            deltas (Wilcoxon + paired t) => "does wrong text hurt accuracy?"
        (b) mask movement    : correct-vs-wrong mask IoU, symmetric change,
            area ratio, centroid shift, mean|dprob|, prob-map correlation
            => "does the text move the mask at all?"
    Together these disentangle "text ignored" from "text used but robust".
  * A qualitative gallery prints >=30 examples: image | GT | correct pred |
    wrong_disease | wrong_location | wrong_both, with the prompts + stats.

Reuses the exact environment/model cells from the balanced-kfold notebook so
nothing drifts. Run Cell 1, restart runtime, then run the rest top-to-bottom.
"""
import json
from pathlib import Path

import nbformat as nbf


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


cells = []

# =============================================================================
cells.append(md(r"""# Text-conditioning ablation for RoSALIA (Option B: MedGemma corruptions)

**Does the instruction text actually change the segmentation?** We run RoSALIA on
every positive pair with the correct prompt and three corrupted prompts
(*wrong disease*, *wrong location*, *wrong both*) and measure both **accuracy vs
the silver mask** and **how much the predicted mask itself moves**.

Run order: **Cell 1 (install) → Restart runtime → Cells 2..end**. The inference
cell is resumable (safe to re-run after a disconnect); a full run over all
positives × 4 flavors can take several hours.
"""))

# ---- Cell 1: install (verbatim from balanced-kfold) ------------------------
cells.append(md(r"""## Cell 1 — install pinned deps, then **RESTART RUNTIME**"""))
cells.append(code(r"""import subprocess, sys
# LISA/ROSALIA was originally pinned to transformers==4.31.0 + tokenizers 0.13.x,
# but tokenizers 0.13.x has NO prebuilt wheel for Python 3.12 (Colab's current
# default) and its Rust source no longer compiles cleanly -> build fails.
# Known-good stack (same as the other RoSALIA notebooks): transformers 4.34.1 +
# tokenizers 0.14.1, which ships cp310/cp311/cp312 wheels and runs LISA-family
# models. --only-binary guarantees no Rust compile is even attempted.
TOKENIZERS_VER   = "0.14.1"
TRANSFORMERS_VER = "4.34.1"

py = sys.version_info
print(f"Python {py.major}.{py.minor}.{py.micro}")
r = subprocess.run(["pip","install","-q","--only-binary=:all:",
                    f"tokenizers=={TOKENIZERS_VER}"], capture_output=True, text=True)
print(r.stdout[-400:]); print(r.stderr[-400:])
if r.returncode != 0:
    raise RuntimeError(
        f"No prebuilt tokenizers=={TOKENIZERS_VER} wheel for Python "
        f"{py.major}.{py.minor}. Use Runtime -> Change runtime type -> "
        "Fallback runtime version (Python 3.11) and re-run this cell.")
!pip -q install "transformers=={TRANSFORMERS_VER}" 'peft==0.4.0' 'einops==0.4.1' \
    'sentencepiece' 'opencv-python>=4.10' 'pycocotools' 'scikit-image' 'bitsandbytes'
!pip -q install 'huggingface_hub>=0.16.4,<1.0'
subprocess.run(["python","-c",
    "import numpy,transformers,tokenizers,torch,cv2;"
    "print('numpy',numpy.__version__,'transformers',transformers.__version__,"
    "'tokenizers',tokenizers.__version__)"])
print("\n>>> RESTART THE RUNTIME NOW, then run Cell 2 onward. <<<")
"""))


# ---- Cell 2: HF login ------------------------------------------------------
cells.append(md(r"""## Cell 2 — Hugging Face login (ROSALIA/LISA public; MedGemma needs access)"""))
cells.append(code(r"""from huggingface_hub import notebook_login
notebook_login()
"""))

# ---- Cell 3: config --------------------------------------------------------
cells.append(md(r"""## Cell 3 — config + paths (post-restart)

`MANIFEST_CHOICE="valtest"` pools VAL+TEST positives (matches the other
notebooks). `USE_MEDGEMMA_CORRUPTION` toggles Option B; when False the
deterministic Option-A swap is used everywhere.
"""))
cells.append(code(r"""import os
try:
    from google.colab import drive; drive.mount('/content/drive'); DRIVE=True
except Exception as e:
    print("[warn]", e); DRIVE=False

DRIVE_DATA_DIR = "/content/drive/MyDrive/MIMIC-CXR-Ext-ILS"
EXT_ZIP    = os.path.join(DRIVE_DATA_DIR, "mimic-cxr-ext-ils.zip")

WORK_DIR = "/content/drive/MyDrive/mimic_ils_rosalia" if DRIVE else "/content/mimic_ils_rosalia"
os.makedirs(WORK_DIR, exist_ok=True)
REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"

MANIFEST_CHOICE = "valtest"       # "subset" | "test" | "val" | "valtest"
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

import pandas as _pd
_dfs = []
for _f in _MANIFEST_FILES:
    _p = f"/content/{_f}"
    if not os.path.exists(_p):
        os.system(f"curl -fsSL '{REPO_RAW_URL}/project/data/mimic_ils/{_f}' -o '{_p}'")
    _dfs.append(_pd.read_csv(_p))
SUBSET_MANIFEST = f"/content/mimic_ils_{MANIFEST_CHOICE}_manifest.csv"
_pd.concat(_dfs, ignore_index=True).to_csv(SUBSET_MANIFEST, index=False)

ROSALIA_REPO      = "checkone/ROSALIA-7B-v1"
LISA_TOKENIZER    = "xinlai/LISA-7B-v1"
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14"
GRID = 512
MASK_BIN_THRESH = 127
LESIONS = ["cardiomegaly","pneumonia","atelectasis","opacity",
           "consolidation","edema","effusion"]

# ---- ablation knobs ---------------------------------------------------------
SEED                    = 42        # global seed; per-pair RNG is derived from pair_id
LOCATION_STRICT         = True      # True: wrong_location must be spatially DISJOINT
                                    # False: allow any different attested location (relaxed)
# Option B (MedGemma) needs transformers>=4.50, but RoSALIA (Cell 5) needs 4.34.x
# and the two CANNOT coexist in one runtime. So MedGemma is a SEPARATE Phase A:
# run Cell 8 in a FRESH runtime (before Cell 5) to cache the prompt CSV, then
# restart and run RoSALIA which just reads the cache. Default is deterministic
# Option A (no transformers change, already validated: 0 grammar violations).
USE_MEDGEMMA_CORRUPTION = False     # Option B: MedGemma proposes corruptions (validated)
MEDGEMMA_ON_FAIL        = "fallback"  # "fallback" (use Option A) | "drop"

N_GALLERY               = 36        # >=30 qualitative examples to print
ABL_CSV = os.path.join(WORK_DIR, f"text_cond_ablation_{MANIFEST_CHOICE}.csv")
CORRUPT_CSV = os.path.join(WORK_DIR, f"text_cond_prompts_{MANIFEST_CHOICE}.csv")
print(f"[config] {MANIFEST_CHOICE} | strict_loc={LOCATION_STRICT} | "
      f"medgemma={USE_MEDGEMMA_CORRUPTION} | seed={SEED}")
print("zips present?", {os.path.basename(z): os.path.exists(z) for z in IMG_ZIPS},
      "| ext:", os.path.exists(EXT_ZIP))
"""))

# ---- Cell 3b: extract zips (verbatim) --------------------------------------
cells.append(md(r"""## Cell 3b — extract zips to local disk + auto-detect roots"""))
cells.append(code(r"""import os, re, zipfile

EXTRACT_ROOT = "/content/mimic_data"
IMG_EXTRACT  = os.path.join(EXTRACT_ROOT, "images")
EXT_EXTRACT  = os.path.join(EXTRACT_ROOT, "ext")

def _unzip(zip_path, dest):
    os.makedirs(dest, exist_ok=True)
    sentinel = os.path.join(dest, "._extracted_" + re.sub(r"\W+", "_", os.path.basename(zip_path)))
    if os.path.exists(sentinel):
        print(f"[skip] already extracted {os.path.basename(zip_path)}"); return
    assert os.path.exists(zip_path), f"missing zip: {zip_path}"
    print(f"unzipping {os.path.basename(zip_path)} -> {dest} ...")
    with zipfile.ZipFile(zip_path) as z: z.extractall(dest)
    open(sentinel, "w").close(); print("  done.")

for _z in IMG_ZIPS: _unzip(_z, IMG_EXTRACT)
_unzip(EXT_ZIP, EXT_EXTRACT)

def find_image_roots(base):
    roots = []
    for root, dirs, _ in os.walk(base):
        if any(re.fullmatch(r"p1\d", d) for d in dirs):
            roots.append(root); dirs[:] = [d for d in dirs if not re.fullmatch(r"p1\d", d)]
    if not roots: raise RuntimeError(f"no p1x/ folders under {base}")
    return roots

def find_mask_root(base):
    for root, dirs, _ in os.walk(base):
        if os.path.basename(root) == "lesion_mask": return root
        if "lesion_mask" in dirs: return os.path.join(root, "lesion_mask")
    raise RuntimeError(f"no lesion_mask/ under {base}")

MIMIC_SUBSET_DIRS = find_image_roots(IMG_EXTRACT)
LESION_MASK_DIR   = find_mask_root(EXT_EXTRACT)
print("MIMIC_SUBSET_DIRS =", MIMIC_SUBSET_DIRS)
print("LESION_MASK_DIR   =", LESION_MASK_DIR)
"""))

# ---- Cell 4: manifest + loaders (ALL positives, no uncertainty filter) -----
cells.append(md(r"""## Cell 4 — load manifest + image/mask loaders (ALL positives)

Unlike the uncertainty notebooks, we keep **every** positive pair that has an
image + silver mask locally (no certain/uncertain filtering) — the ablation is
about text conditioning, not uncertainty.
"""))
cells.append(code(r"""import os
import numpy as np, pandas as pd
from PIL import Image

manifest = pd.read_csv(SUBSET_MANIFEST)
pos = manifest[manifest.polarity == "positive"].reset_index(drop=True)
print(f"positive pairs in manifest: {len(pos)}")

_IMG_PATH_CACHE = {}
def resolve_image(image_path):
    if not isinstance(image_path, str): return None
    if image_path in _IMG_PATH_CACHE: return _IMG_PATH_CACHE[image_path]
    for _rt in MIMIC_SUBSET_DIRS:
        fp = os.path.join(_rt, image_path)
        if os.path.exists(fp): _IMG_PATH_CACHE[image_path] = fp; return fp
    _IMG_PATH_CACHE[image_path] = None; return None

def load_image(image_path):
    fp = resolve_image(image_path)
    if fp is None: raise FileNotFoundError(image_path)
    return Image.open(fp).convert("RGB")

def load_silver_mask(seg_mask_path):
    m = np.array(Image.open(os.path.join(LESION_MASK_DIR, seg_mask_path)).convert("L"))
    return (m > MASK_BIN_THRESH).astype(np.uint8)

_present = pos["image_path"].map(lambda p: resolve_image(p) is not None)
_hasmask = pos["seg_mask_path"].map(lambda p: isinstance(p, str) and len(p) > 0)
pos = pos[_present & _hasmask].reset_index(drop=True)
print(f"positives with image AND silver mask present locally: {len(pos)}")
print("by disease:\n", pos.target.value_counts().to_string())
"""))

# ---- Cell 5: load ROSALIA + segment() (verbatim) ---------------------------
cells.append(md(r"""## Cell 5 — load ROSALIA (LISA-7B + SAM-H)

`segment(pil, instruction) -> (binary_mask, logits, text)`. `sigmoid(logits)` is
the probability map (mask == logits>0 == prob>0.5).
"""))
cells.append(code(r"""import sys, os, subprocess, faulthandler
faulthandler.enable()
os.environ["HF_HUB_DISABLE_XET"] = "1"
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
    raise RuntimeError("numpy upgraded to 2.x -> Runtime > Restart, then re-run Cell 2 onward.")

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
    x = (x-mean)/std; h,w = x.shape[-2:]
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

from huggingface_hub import snapshot_download
for _a in range(8):
    try:
        snapshot_download(ROSALIA_REPO, resume_download=True, max_workers=2)
        print("[weights] ROSALIA snapshot downloaded"); break
    except Exception as _e:
        print(f"[weights][retry {_a+1}/8] {type(_e).__name__}: {_e}")
else:
    raise RuntimeError("weight download kept failing; rerun this cell (it resumes).")

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

print("segment() ready.")
"""))

# ---- Cell 6: metrics (accuracy + mask-change) ------------------------------
cells.append(md(r"""## Cell 6 — metrics: accuracy vs GT **and** mask-movement between prompts"""))
cells.append(code(r"""import numpy as np
from skimage.transform import resize as _skresize

def _grid(mask, grid=GRID):
    m = np.asarray(mask).astype(np.float32)
    if m.shape != (grid, grid):
        m = _skresize(m, (grid, grid), order=0, mode="edge",
                      anti_aliasing=False, preserve_range=True)
    return (m > 0.5).astype(np.uint8)

def _prob_grid(logits, grid=GRID):
    if logits is None: return None
    p = 1.0/(1.0+np.exp(-np.asarray(logits, dtype=np.float64)))
    if p.shape != (grid, grid):
        p = _skresize(p, (grid, grid), order=1, mode="edge",
                      anti_aliasing=True, preserve_range=True)
    return p.astype(np.float32)

def iou_dice(pred, gt, grid=GRID):
    p, g = _grid(pred, grid), _grid(gt, grid)
    inter = int(np.logical_and(p, g).sum()); union = int(np.logical_or(p, g).sum())
    iou = inter/union if union>0 else float("nan")
    dice = (2*inter)/(p.sum()+g.sum()) if (p.sum()+g.sum())>0 else float("nan")
    return iou, dice

def acc_metrics(pred, logits, gt):
    iou, dice = iou_dice(pred, gt)
    prob = _prob_grid(logits)
    if prob is None:
        return dict(iou=iou, dice=dice, p_in_gt=np.nan, pred_px=int(_grid(pred).sum()), fired=0)
    g = _grid(gt)
    p_in_gt = float(prob[g>0].mean()) if g.sum()>0 else np.nan
    predb = _grid(pred)
    return dict(iou=iou, dice=dice, p_in_gt=p_in_gt,
                pred_px=int(predb.sum()), fired=int(predb.sum()>0))

def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs)==0: return None
    return (xs.mean()/mask.shape[1], ys.mean()/mask.shape[0])

def mask_change(pred_a, prob_a, pred_b, prob_b, grid=GRID):
    # a = correct, b = wrong. How different is b from a?
    A, B = _grid(pred_a, grid), _grid(pred_b, grid)
    inter = int(np.logical_and(A, B).sum()); union = int(np.logical_or(A, B).sum())
    m_iou = inter/union if union>0 else (1.0 if (A.sum()+B.sum())==0 else 0.0)
    sym = (np.logical_xor(A, B).sum()/union) if union>0 else 0.0   # frac of region that moved
    area_ratio = (B.sum()/A.sum()) if A.sum()>0 else (np.nan if B.sum()==0 else np.inf)
    ca, cb = _centroid(A), _centroid(B)
    if ca is None or cb is None:
        cshift = np.nan
    else:
        cshift = float(np.hypot(ca[0]-cb[0], ca[1]-cb[1]))
    if prob_a is None or prob_b is None:
        mad = rms = corr = np.nan
    else:
        d = prob_a - prob_b
        mad = float(np.abs(d).mean()); rms = float(np.sqrt((d**2).mean()))
        fa, fb = prob_a.ravel(), prob_b.ravel()
        if fa.std()>1e-6 and fb.std()>1e-6:
            corr = float(np.corrcoef(fa, fb)[0,1])
        else:
            corr = np.nan
    return dict(mask_iou_vs_correct=m_iou, sym_change=sym, area_ratio=area_ratio,
                centroid_shift=cshift, mean_abs_dprob=mad, rms_dprob=rms, prob_corr=corr)

print("metrics ready: acc_metrics(vs GT) + mask_change(correct vs wrong).")
"""))

# ---- Cell 7: Option A corruption engine ------------------------------------
cells.append(md(r"""## Cell 7 — corruption engine (Option A: distribution-aware, deterministic)

Parse each positive instruction into `(disease, location, has_type_suffix)`, build
per-disease **template banks** from the manifest, then generate seeded wrong
prompts that stay in-distribution. `wrong_location` is set to **N/A** when no
spatially-disjoint location exists (location-free findings like cardiomegaly, or
bilateral "both lungs" findings) — unless `LOCATION_STRICT=False`.

> **Where does "location" come from?** We do **not** use the manifest's
> `location` column — it's empty for ~67% of positives. Instead we regex-parse
> location out of the `instruction` string itself (e.g. `"Segment the edema in
> the left lung."` → `left lung`), which is exactly the text RoSALIA is
> conditioned on. Instructions with no location clause (`"Segment the
> cardiomegaly."`) parse to `location=None` and become `wrong_location = N/A`.
"""))

cells.append(code(r"""import re, hashlib
import numpy as np, pandas as pd

DISEASES = ["cardiomegaly","edema","atelectasis","effusion","opacity","pneumonia","consolidation"]
_INSTR_RE = re.compile(r"^Segment the (.+?)(?: in the (.+?))?(?: and predict its type)?\.$")

def parse_instruction(instr):
    m = _INSTR_RE.match(str(instr).strip())
    if not m: return (None, None, False)
    dis = m.group(1); loc = m.group(2)
    suf = "predict its type" in str(instr)
    return (dis, loc, suf)

def loc_tokens(loc):
    if not isinstance(loc, str): return set()
    t = set()
    for s in ["left","right"]:
        if s in loc: t.add(s)
    for z in ["base","mid zone","upper zone"]:
        if z in loc: t.add(z)
    return t

# per-disease template banks from ALL positives in this manifest
# NOTE: column names avoid leading underscore because pd.itertuples() drops those.
_parsed = pos["instruction"].map(parse_instruction)
pos = pos.assign(pdis=[p[0] for p in _parsed],
                 ploc=[p[1] for p in _parsed],
                 psuf=[p[2] for p in _parsed])
LOC_BANK = {d: sorted(set(g.ploc.dropna().unique())) for d, g in pos.groupby("pdis")}
SUFFIX_DISEASES = set(pos[pos.psuf].pdis.unique())   # which findings ever use the suffix

print("location bank sizes:", {d: len(v) for d, v in LOC_BANK.items()})
print("suffix-using findings:", sorted(SUFFIX_DISEASES))

def _rng_for(pair_id):
    h = int(hashlib.md5(f"{pair_id}|{SEED}".encode()).hexdigest(), 16) & 0xFFFFFFFF
    return np.random.default_rng(h)

def render(disease, location, suffix):
    s = f"Segment the {disease}"
    if isinstance(location, str) and location:
        s += f" in the {location}"
    if suffix:
        s += " and predict its type"
    return s + "."

def _pick_loc_for(disease, rng, avoid_tokens=None, strict=True):
    bank = LOC_BANK.get(disease, [])
    if not bank: return None                     # location-free finding (cardiomegaly)
    if avoid_tokens is None:
        return bank[rng.integers(len(bank))]
    disjoint = [L for L in bank if loc_tokens(L) and loc_tokens(L).isdisjoint(avoid_tokens)]
    if disjoint:
        return disjoint[rng.integers(len(disjoint))]
    if strict:
        return None                              # no disjoint option (bilateral) -> N/A
    diff = [L for L in bank if L not in (None,)]
    return diff[rng.integers(len(diff))] if diff else None

def corrupt_option_a(pair_id, disease, location, suffix, strict=True):
    rng = _rng_for(pair_id)
    out = {}
    # wrong_disease: pick a different finding, render in ITS grammar
    wd = disease
    others = [d for d in DISEASES if d != disease]
    wd = others[rng.integers(len(others))]
    wd_loc = _pick_loc_for(wd, rng)              # any attested loc for wd (or None)
    wd_suf = wd in SUFFIX_DISEASES and bool(rng.integers(2)) if wd == "opacity" else False
    out["wrong_disease"] = dict(text=render(wd, wd_loc, wd_suf), finding=wd, location=wd_loc)
    # wrong_location: same finding, disjoint location
    if location is None:
        # originally location-free: injecting a specific location is the "wrong" signal,
        # but only if this finding actually uses locations (bank non-empty)
        wl_loc = _pick_loc_for(disease, rng)
    else:
        wl_loc = _pick_loc_for(disease, rng, avoid_tokens=loc_tokens(location), strict=strict)
    if wl_loc is None or (location is None and not LOC_BANK.get(disease)):
        out["wrong_location"] = dict(text=None, finding=disease, location=None)
    else:
        out["wrong_location"] = dict(text=render(disease, wl_loc, suffix), finding=disease, location=wl_loc)
    # wrong_both: wrong disease + a wrong location valid for it
    wb_loc = _pick_loc_for(wd, rng)
    out["wrong_both"] = dict(text=render(wd, wb_loc, wd_suf), finding=wd, location=wb_loc)
    return out

# quick demo
for _pid, _d, _l, _s in [("demo1","opacity","right lung",True),
                          ("demo2","cardiomegaly",None,False),
                          ("demo3","edema","right lung and left lung",False)]:
    print(f"\n[{_d} / {_l}]")
    print("  correct       :", render(_d,_l,_s))
    for k,v in corrupt_option_a(_pid,_d,_l,_s,LOCATION_STRICT).items():
        print(f"  {k:14s}:", v["text"])
"""))

# ---- Cell 8: Option B MedGemma corruption (validated, with fallback) -------
cells.append(md(r"""## Cell 8 — Option B: MedGemma proposes corruptions (validated → fallback to A)

Runs only if `USE_MEDGEMMA_CORRUPTION=True`. MedGemma sees the controlled vocab +
grammar rules and returns **structured JSON**. Every field is validated (allowed
finding/location, `wrong_disease != true`, disjointness, null semantics for
location-free/bilateral, suffix only on opacity). Any violation → we substitute
the deterministic **Option-A** value for that field (`MEDGEMMA_ON_FAIL`). We label
each `(instruction)` once (many pairs share the same instruction) and cache it.
"""))
cells.append(code(r"""import json, os, re, sys, hashlib, subprocess
import numpy as np, pandas as pd

# --- Self-contained bootstrap so this cell ALSO works in a FRESH MedGemma-only
# --- runtime where Cells 4 & 7 were NOT run (needed because MedGemma can't share
# --- a kernel with RoSALIA). If Cell 7 already ran, we transparently reuse it.
DISEASES = ["cardiomegaly","edema","atelectasis","effusion","opacity","pneumonia","consolidation"]
if "parse_instruction" not in globals():
    _INSTR_RE = re.compile(r"^Segment the (.+?)(?: in the (.+?))?(?: and predict its type)?\.$")
    def parse_instruction(instr):
        m = _INSTR_RE.match(str(instr).strip())
        return (m.group(1), m.group(2), "predict its type" in str(instr)) if m else (None, None, False)
if "loc_tokens" not in globals():
    def loc_tokens(loc):
        if not isinstance(loc, str): return set()
        t = set()
        for s in ["left","right"]:
            if s in loc: t.add(s)
        for z in ["base","mid zone","upper zone"]:
            if z in loc: t.add(z)
        return t
if "render" not in globals():
    def render(d, l, s):
        x = f"Segment the {d}"
        if isinstance(l, str) and l: x += f" in the {l}"
        if s: x += " and predict its type"
        return x + "."
try:
    pos; assert {"pdis","ploc","psuf"}.issubset(pos.columns)
except Exception:
    _man = pd.read_csv(SUBSET_MANIFEST)
    pos = _man[_man.polarity == "positive"].reset_index(drop=True).copy()
    _pp = pos["instruction"].map(parse_instruction)
    pos = pos.assign(pdis=[a[0] for a in _pp], ploc=[a[1] for a in _pp], psuf=[a[2] for a in _pp])
    print(f"[bootstrap] rebuilt {len(pos)} positives from manifest (image-independent).")
if "LOC_BANK" not in globals():
    LOC_BANK = {d: sorted(set(g.ploc.dropna().unique())) for d, g in pos.groupby("pdis")}
if "SUFFIX_DISEASES" not in globals():
    SUFFIX_DISEASES = set(pos[pos.psuf].pdis.unique())
if "corrupt_option_a" not in globals():
    def _rng_for(pid):
        h = int(hashlib.md5(f"{pid}|{SEED}".encode()).hexdigest(), 16) & 0xFFFFFFFF
        return np.random.default_rng(h)
    def _pick_loc_for(disease, rng, avoid_tokens=None, strict=True):
        bank = LOC_BANK.get(disease, [])
        if not bank: return None
        if avoid_tokens is None: return bank[rng.integers(len(bank))]
        dj = [L for L in bank if loc_tokens(L) and loc_tokens(L).isdisjoint(avoid_tokens)]
        if dj: return dj[rng.integers(len(dj))]
        if strict: return None
        return bank[rng.integers(len(bank))]
    def corrupt_option_a(pair_id, disease, location, suffix, strict=True):
        rng = _rng_for(pair_id); out = {}
        others = [d for d in DISEASES if d != disease]; wd = others[rng.integers(len(others))]
        wd_loc = _pick_loc_for(wd, rng)
        wd_suf = wd in SUFFIX_DISEASES and bool(rng.integers(2)) if wd == "opacity" else False
        out["wrong_disease"] = dict(text=render(wd, wd_loc, wd_suf), finding=wd, location=wd_loc)
        wl = _pick_loc_for(disease, rng) if location is None else _pick_loc_for(disease, rng, loc_tokens(location), strict)
        if wl is None or (location is None and not LOC_BANK.get(disease)):
            out["wrong_location"] = dict(text=None, finding=disease, location=None)
        else:
            out["wrong_location"] = dict(text=render(disease, wl, suffix), finding=disease, location=wl)
        wb = _pick_loc_for(wd, rng)
        out["wrong_both"] = dict(text=render(wd, wb, wd_suf), finding=wd, location=wb)
        return out

ALLOWED_LOCS = set(sum(LOC_BANK.values(), [])) | {None}


def _valid_field(finding, location, *, require_finding=None, forbid_finding=None,
                 avoid_tokens=None, allow_null_loc=True, suffix=None, text=None):
    if finding not in DISEASES: return False
    if require_finding is not None and finding != require_finding: return False
    if forbid_finding is not None and finding == forbid_finding: return False
    if location is None:
        if not allow_null_loc: return False
    else:
        if location not in ALLOWED_LOCS: return False
        if location not in LOC_BANK.get(finding, []): return False   # in-distribution for that finding
        if avoid_tokens is not None and not loc_tokens(location).isdisjoint(avoid_tokens):
            return False
    # suffix only allowed on opacity
    if text is not None and ("predict its type" in text) and finding != "opacity":
        return False
    return True

MEDGEMMA_SYS = (
    "You corrupt radiology segmentation instructions for a controlled ablation.\n"
    "Given ONE correct instruction 'Segment the <FINDING>[ in the <LOCATION>][ and predict its type].',\n"
    "produce WRONG-but-well-formed variants that stay in-distribution.\n\n"
    "ALLOWED FINDINGS: cardiomegaly, edema, atelectasis, effusion, opacity, pneumonia, consolidation\n"
    "ALLOWED LOCATIONS: right lung, left lung, right lung and left lung, right lung base, "
    "left lung base, right upper zone lung, left upper zone lung, right mid zone lung, left mid zone lung\n\n"
    "GRAMMAR RULES:\n"
    "- cardiomegaly is ALWAYS location-free ('Segment the cardiomegaly.').\n"
    "- effusion/atelectasis normally occur at a lung BASE.\n"
    "- only opacity may use ' and predict its type'.\n"
    "- change ONLY the intended slot; keep pattern + punctuation identical.\n\n"
    "LOCATION EDGE-CASE RULES:\n"
    "- If the original is location-free, set wrong_location text=null.\n"
    "- If the original location is BILATERAL ('right lung and left lung'), a disjoint location\n"
    "  does not exist: set wrong_location text=null and wrong_location_relaxed to a single-sided\n"
    "  location for the SAME finding.\n"
    "- Otherwise wrong_location must be SPATIALLY DISJOINT (no shared side or zone token).\n\n"
    "CORRUPTION DEFINITIONS:\n"
    "- wrong_disease: a DIFFERENT allowed finding, rendered with a location it naturally uses (or none).\n"
    "- wrong_location: SAME finding, disjoint location per rules above (may be null).\n"
    "- wrong_both: wrong_disease + a wrong location valid for the new finding.\n\n"
    "OUTPUT strict JSON with BOTH rendered string and parts:\n"
    '{"wrong_disease":{"text":"...","finding":"<allowed>","location":"<allowed|null>"},\n'
    ' "wrong_location":{"text":"...|null","finding":"<same>","location":"<allowed|null>"},\n'
    ' "wrong_location_relaxed":{"text":"...|null","finding":"<same>","location":"<allowed|null>"},\n'
    ' "wrong_both":{"text":"...","finding":"<allowed>","location":"<allowed|null>"}}\n'
    "Output ONLY the JSON.\n"
)
MEDGEMMA_FEWSHOT = (
    'INPUT: "Segment the opacity in the right lung and predict its type."\n'
    'OUTPUT: {"wrong_disease":{"text":"Segment the effusion in the right lung base.","finding":"effusion","location":"right lung base"},'
    '"wrong_location":{"text":"Segment the opacity in the left lung and predict its type.","finding":"opacity","location":"left lung"},'
    '"wrong_location_relaxed":{"text":"Segment the opacity in the left lung and predict its type.","finding":"opacity","location":"left lung"},'
    '"wrong_both":{"text":"Segment the effusion in the left lung base.","finding":"effusion","location":"left lung base"}}\n'
    'INPUT: "Segment the cardiomegaly."\n'
    'OUTPUT: {"wrong_disease":{"text":"Segment the edema in the right lung and left lung.","finding":"edema","location":"right lung and left lung"},'
    '"wrong_location":{"text":null,"finding":"cardiomegaly","location":null},'
    '"wrong_location_relaxed":{"text":null,"finding":"cardiomegaly","location":null},'
    '"wrong_both":{"text":"Segment the edema in the left lung base.","finding":"edema","location":"left lung base"}}\n'
    'INPUT: "Segment the edema in the right lung and left lung."\n'
    'OUTPUT: {"wrong_disease":{"text":"Segment the atelectasis in the right lung base.","finding":"atelectasis","location":"right lung base"},'
    '"wrong_location":{"text":null,"finding":"edema","location":null},'
    '"wrong_location_relaxed":{"text":"Segment the edema in the right lung base.","finding":"edema","location":"right lung base"},'
    '"wrong_both":{"text":"Segment the atelectasis in the left lung base.","finding":"atelectasis","location":"left lung base"}}\n'
)

def _medgemma_generate(instr, mdl, proc, tok):
    import torch
    prompt = MEDGEMMA_SYS + "\n" + MEDGEMMA_FEWSHOT + f'\nINPUT: "{instr}"\nOUTPUT:'
    msgs = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
    raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def _merge_with_fallback(instr, mg, base_a, strict):
    # Validate MedGemma fields; fall back to Option-A per field.
    # Returns dict of flavor -> {text, finding, location, source}.
    dis, loc, suf = parse_instruction(instr)

    result = {}
    avoid = loc_tokens(loc) if loc is not None else None
    specs = {
        "wrong_disease":  dict(forbid_finding=dis, allow_null_loc=True),
        "wrong_location": dict(require_finding=dis, avoid_tokens=avoid, allow_null_loc=True),
        "wrong_both":     dict(forbid_finding=dis, allow_null_loc=True),
    }
    # choose which wrong_location field to read from MedGemma
    wl_key = "wrong_location" if strict else "wrong_location_relaxed"
    key_map = {"wrong_disease":"wrong_disease","wrong_location":wl_key,"wrong_both":"wrong_both"}
    for flavor, spec in specs.items():
        a = base_a[flavor]
        used = "fallback_A"
        f = (mg or {}).get(key_map[flavor]) if isinstance(mg, dict) else None
        if isinstance(f, dict):
            ftext = f.get("text"); ffind = f.get("finding"); floc = f.get("location")
            if floc in ("null","", "None"): floc = None
            # a null wrong_location is a legitimate N/A answer
            if flavor == "wrong_location" and (ftext is None or ftext == "null"):
                result[flavor] = dict(text=None, finding=dis, location=None, source="medgemma"); continue
            if isinstance(ftext, str) and _valid_field(ffind, floc, suffix=suf, text=ftext, **spec):
                # ensure rendered text matches parts (defends against sneaky mismatches)
                exp = render(ffind, floc, ("predict its type" in ftext))
                if exp == ftext:
                    result[flavor] = dict(text=ftext, finding=ffind, location=floc, source="medgemma"); continue
        # fallback
        result[flavor] = dict(text=a["text"], finding=a["finding"], location=a["location"], source=used)
    return result

# ---- build the prompt table for every positive pair ----
if os.path.exists(CORRUPT_CSV):
    prompts_df = pd.read_csv(CORRUPT_CSV)
    print(f"[prompts] loaded cached {len(prompts_df)} rows from {CORRUPT_CSV}")
else:
    # 1) deterministic Option-A for every pair (always available)
    baseA = {}
    for r in pos.itertuples():
        baseA[r.pair_id] = corrupt_option_a(r.pair_id, r.pdis, r.ploc, r.psuf, LOCATION_STRICT)


    # 2) optional MedGemma pass (Option B), labelled once per unique instruction.
    mg_by_instr = {}
    if USE_MEDGEMMA_CORRUPTION:
        # RoSALIA (transformers 4.34) and MedGemma (>=4.50) CANNOT share a kernel
        # (swapping transformers mid-session breaks with 'cannot import name
        # PreTrainedConfig'). So run THIS cell in a FRESH runtime BEFORE Cell 5:
        #   run Cells 2, 3 (set USE_MEDGEMMA_CORRUPTION=True), then this cell.
        # It caches CORRUPT_CSV; then Runtime>Restart and run the RoSALIA pass,
        # where this cell just LOADS the cache.
        if "LISAForCausalLM" in globals() or "model" in globals():
            raise RuntimeError(
                "RoSALIA is loaded in this kernel, so MedGemma can't run here. Start a "
                "FRESH runtime (Runtime -> Disconnect and delete runtime), run Cells 2 & 3, "
                "then THIS cell to cache CORRUPT_CSV; restart and run the RoSALIA pass. "
                "Or keep USE_MEDGEMMA_CORRUPTION=False (deterministic Option A).")
        # ensure a MedGemma-capable transformers (4.5x). If we had to install it,
        # the already-imported transformers is stale -> require ONE restart+rerun.
        _need = True
        try:
            import transformers as _tf
            _v = tuple(int(x) for x in _tf.__version__.split(".")[:2])
            _need = not ((4, 50) <= _v < (5, 0))
        except Exception:
            _need = True
        if _need:
            print("[medgemma] installing transformers>=4.50,<5 (pinned <5 to avoid the "
                  "5.x API rename) ...")
            subprocess.run([sys.executable,"-m","pip","install","-q",
                            "transformers>=4.50,<5","accelerate>=0.30"], check=True)
            print("\n" + "="*72)
            print(">>> transformers for MedGemma is installed. NEXT STEPS:")
            print("    1) Runtime > Restart session   (menu, or Ctrl/Cmd+M .)")
            print("    2) Re-run Cell 2 (login) and Cell 3 (config, keep")
            print("       USE_MEDGEMMA_CORRUPTION=True), THEN re-run THIS cell.")
            print("    (The 'To exit: use quit' warning below is harmless.)")
            print("="*72)
            raise SystemExit

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText


        MODEL_NAME = "google/medgemma-4b-it"
        proc = AutoProcessor.from_pretrained(MODEL_NAME)
        tok = getattr(proc, "tokenizer", None) or proc
        tok.padding_side = "left"
        if getattr(tok,"pad_token",None) is None and getattr(tok,"eos_token",None):
            tok.pad_token = tok.eos_token
        mg_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()
        from tqdm.auto import tqdm
        uniq = sorted(pos["instruction"].unique())
        print(f"[medgemma] labelling {len(uniq)} unique instructions ...")
        for ins in tqdm(uniq):
            mg_by_instr[ins] = _medgemma_generate(ins, mg_model, proc, tok)
        import gc; del mg_model, proc; gc.collect(); torch.cuda.empty_cache()
        print("[medgemma] done")

    # 3) merge (validate + fallback) into a long prompt table
    rows = []
    src_counts = {}
    for r in pos.itertuples():
        merged = _merge_with_fallback(r.instruction, mg_by_instr.get(r.instruction),
                                      baseA[r.pair_id], LOCATION_STRICT) if USE_MEDGEMMA_CORRUPTION \
                 else {k: dict(**v, source="option_A") for k, v in baseA[r.pair_id].items()}
        rows.append(dict(pair_id=r.pair_id, flavor="correct", text=r.instruction,
                         finding=r.pdis, location=(r.ploc or ""), source="original"))

        for flavor, v in merged.items():
            rows.append(dict(pair_id=r.pair_id, flavor=flavor, text=(v["text"] or ""),
                             finding=v["finding"], location=(v["location"] or ""),
                             source=v["source"]))
            src_counts[(flavor, v["source"])] = src_counts.get((flavor, v["source"]), 0) + 1
    prompts_df = pd.DataFrame(rows)
    prompts_df.to_csv(CORRUPT_CSV, index=False)
    print(f"[prompts] wrote {len(prompts_df)} rows -> {CORRUPT_CSV}")
    if USE_MEDGEMMA_CORRUPTION:
        print("[prompts] source breakdown (flavor, source): ", src_counts)

# applicability + preview
_na = prompts_df[(prompts_df.flavor=="wrong_location") & (prompts_df.text=="")]
print(f"wrong_location N/A pairs: {len(_na)} / {(prompts_df.flavor=='wrong_location').sum()}")
print(prompts_df.head(8).to_string(index=False))
"""))

# ---- Cell 9: inference over all positives x flavors ------------------------
cells.append(md(r"""## Cell 9 — run RoSALIA over all positives × flavors (RESUMABLE)

For each pair we run the **correct** prompt plus each applicable wrong prompt,
score accuracy vs the silver mask, and — using the correct prediction as the
reference — compute how far each wrong prediction's mask/prob-map moved. Scalars
are appended to `ABL_CSV`; safe to re-run after a disconnect (skips finished
pairs). No masks are persisted (the gallery re-runs a handful for plotting).
"""))
cells.append(code(r"""import os, numpy as np, pandas as pd
from tqdm.auto import tqdm

WRONG_FLAVORS = ["wrong_disease","wrong_location","wrong_both"]

done_pairs = set()
if os.path.exists(ABL_CSV):
    _d = pd.read_csv(ABL_CSV)
    done_pairs = set(_d["pair_id"].unique())
    print(f"resuming; {len(done_pairs)} pairs already done")

def _append_rows(recs):
    pd.DataFrame(recs).to_csv(ABL_CSV, mode="a",
                              header=not os.path.exists(ABL_CSV), index=False)

# index prompts by pair
P_BY_PAIR = {pid: g.set_index("flavor") for pid, g in prompts_df.groupby("pair_id")}
pos_by_id = pos.set_index("pair_id")

for r in tqdm(pos.itertuples(), total=len(pos), desc="ablation infer"):
    if r.pair_id in done_pairs: continue
    try:
        pg = P_BY_PAIR.get(r.pair_id)
        if pg is None: continue
        img = load_image(r.image_path); gt = load_silver_mask(r.seg_mask_path)
        # correct first (reference)
        c_text = pg.loc["correct","text"]
        c_pred, c_log, c_out = segment(img, c_text)
        c_prob = _prob_grid(c_log)
        recs = []
        base = dict(pair_id=r.pair_id, study_id=r.study_id, disease=r.pdis,
                    gt_px=int(gt.sum()))

        cm = acc_metrics(c_pred, c_log, gt)
        recs.append({**base, "flavor":"correct", "prompt":c_text, "source":"original",
                     **cm, "model_text":c_out,
                     "mask_iou_vs_correct":1.0, "sym_change":0.0, "area_ratio":1.0,
                     "centroid_shift":0.0, "mean_abs_dprob":0.0, "rms_dprob":0.0, "prob_corr":1.0})
        for flavor in WRONG_FLAVORS:
            if flavor not in pg.index: continue
            text = pg.loc[flavor,"text"]
            src  = pg.loc[flavor,"source"]
            if not isinstance(text, str) or text == "":   # N/A flavor (e.g. cardiomegaly location)
                recs.append({**base, "flavor":flavor, "prompt":"", "source":src,
                             "iou":np.nan,"dice":np.nan,"p_in_gt":np.nan,"pred_px":np.nan,
                             "fired":np.nan,"model_text":"",
                             "mask_iou_vs_correct":np.nan,"sym_change":np.nan,"area_ratio":np.nan,
                             "centroid_shift":np.nan,"mean_abs_dprob":np.nan,"rms_dprob":np.nan,
                             "prob_corr":np.nan})
                continue
            w_pred, w_log, w_out = segment(img, text)
            w_prob = _prob_grid(w_log)
            am = acc_metrics(w_pred, w_log, gt)
            ch = mask_change(c_pred, c_prob, w_pred, w_prob)
            recs.append({**base, "flavor":flavor, "prompt":text, "source":src,
                         **am, "model_text":w_out, **ch})
        _append_rows(recs)
    except Exception as e:
        print(f"[skip] {r.pair_id}: {type(e).__name__}: {e}")

abl = pd.read_csv(ABL_CSV)
print(f"rows: {len(abl)} | pairs: {abl.pair_id.nunique()} | flavors: {abl.flavor.value_counts().to_dict()}")
"""))

# ---- Cell 10: analysis -----------------------------------------------------
cells.append(md(r"""## Cell 10 — did text conditioning matter? (paired accuracy + mask movement)

**Accuracy** answers "does a wrong prompt lower IoU vs the silver mask?" (paired
Wilcoxon + paired t across pairs). **Mask movement** answers "did the text change
the prediction at all?" A null accuracy effect *with* large mask movement means
the model uses text but is robust; a null effect *with* tiny movement means the
model largely ignores text.
"""))
cells.append(code(r"""import numpy as np, pandas as pd
from scipy import stats

abl = pd.read_csv(ABL_CSV)
wide = abl.pivot_table(index="pair_id", columns="flavor", values="iou")

def paired(colA, colB):
    d = wide[[colA, colB]].dropna()
    if len(d) < 5: return dict(n=len(d))
    a, b = d[colA].values, d[colB].values
    try: w_p = stats.wilcoxon(a, b).pvalue
    except Exception: w_p = float("nan")
    try: t_p = stats.ttest_rel(a, b).pvalue
    except Exception: t_p = float("nan")
    return dict(n=len(d), mean_correct=a.mean(), mean_wrong=b.mean(),
                delta=(b-a).mean(), wilcoxon_p=w_p, paired_t_p=t_p)

print("=== ACCURACY: correct IoU vs each wrong flavor (paired) ===")
acc_rows = []
for f in ["wrong_disease","wrong_location","wrong_both"]:
    if f in wide.columns:
        row = paired("correct", f); row["flavor"] = f; acc_rows.append(row)
acc_tab = pd.DataFrame(acc_rows).set_index("flavor")
print(acc_tab.round(4).to_string())

print("\n=== MASK MOVEMENT vs correct (mean over pairs; 1.0 mask_iou / 0 dprob = no change) ===")
mv = (abl[abl.flavor.isin(["wrong_disease","wrong_location","wrong_both"])]
      .groupby("flavor")[["mask_iou_vs_correct","sym_change","centroid_shift",
                           "mean_abs_dprob","prob_corr","area_ratio"]].mean())
print(mv.round(4).to_string())

print("\n=== %fired by flavor (did the model output any mask?) ===")
print(abl.groupby("flavor")["fired"].mean().round(3).to_string())

print("\n=== per-disease macro: mean IoU by flavor ===")
pd_iou = abl.pivot_table(index="disease", columns="flavor", values="iou", aggfunc="mean")
print(pd_iou.round(3).to_string())

# plain-English verdict
def verdict():
    msgs = []
    for f in ["wrong_disease","wrong_location","wrong_both"]:
        if f not in acc_tab.index: continue
        r = acc_tab.loc[f]
        sig = (r.get("wilcoxon_p", np.nan) < 0.05)
        moved = mv.loc[f,"mask_iou_vs_correct"] < 0.9 if f in mv.index else False
        if sig and r["delta"] < 0:
            m = f"{f}: wrong prompt LOWERS IoU by {abs(r['delta']):.3f} (p={r['wilcoxon_p']:.1e}) -> text MATTERS for accuracy."
        elif moved:
            m = f"{f}: IoU not significantly changed, but the mask MOVES (mask-IoU={mv.loc[f,'mask_iou_vs_correct']:.2f}) -> text is USED but accuracy is robust."
        else:
            m = f"{f}: IoU unchanged AND mask barely moves (mask-IoU={mv.loc[f,'mask_iou_vs_correct']:.2f}) -> model largely IGNORES this text."
        msgs.append(m)
    return "\n".join(msgs)

print("\n=== VERDICT ===")
print(verdict())
"""))

# ---- Cell 11: plots --------------------------------------------------------
cells.append(md(r"""## Cell 11 — plots: IoU by flavor, ΔIoU, and mask movement"""))
cells.append(code(r"""import numpy as np, pandas as pd, matplotlib.pyplot as plt

abl = pd.read_csv(ABL_CSV)
order = ["correct","wrong_disease","wrong_location","wrong_both"]
present = [f for f in order if f in abl.flavor.unique()]

fig, ax = plt.subplots(1, 3, figsize=(16,4.5))
# (1) IoU distribution by flavor
data = [abl[abl.flavor==f]["iou"].dropna().values for f in present]
ax[0].boxplot(data, labels=present, showmeans=True)
ax[0].set_ylabel("IoU vs silver mask"); ax[0].set_title("IoU by prompt flavor")
ax[0].tick_params(axis="x", rotation=20)

# (2) paired ΔIoU vs correct
wide = abl.pivot_table(index="pair_id", columns="flavor", values="iou")
deltas, labs = [], []
for f in ["wrong_disease","wrong_location","wrong_both"]:
    if f in wide.columns:
        d = (wide[f]-wide["correct"]).dropna().values
        deltas.append(d); labs.append(f)
ax[1].boxplot(deltas, labels=labs, showmeans=True); ax[1].axhline(0, color="red", ls="--")
ax[1].set_ylabel("IoU(wrong) - IoU(correct)"); ax[1].set_title("Paired ΔIoU (negative = wrong hurts)")
ax[1].tick_params(axis="x", rotation=20)

# (3) mask movement vs correct
mv = (abl[abl.flavor.isin(["wrong_disease","wrong_location","wrong_both"])]
      .groupby("flavor")["mask_iou_vs_correct"].apply(list))
ax[2].boxplot([mv[f] for f in mv.index], labels=list(mv.index), showmeans=True)
ax[2].set_ylabel("mask IoU (correct vs wrong)")
ax[2].set_title("Mask movement (1.0 = text changed nothing)")
ax[2].tick_params(axis="x", rotation=20)
plt.tight_layout()
_p = os.path.join(WORK_DIR, "text_cond_summary.png"); plt.savefig(_p, dpi=130); plt.show()
print("saved ->", _p)
"""))

# ---- Cell 12: qualitative gallery ------------------------------------------
cells.append(md(r"""## Cell 12 — qualitative gallery (≥30 examples)

Selects examples from three buckets — where a wrong prompt **hurt IoU most**,
where IoU was **most robust**, and **random** — then RE-RUNS RoSALIA for just
those pairs (cheap) to show, per row: **image | GT | correct pred | wrong_disease
| wrong_location | wrong_both**, with the prompts + stats printed above each row.
"""))
cells.append(code(r"""import os, textwrap, numpy as np, pandas as pd, matplotlib.pyplot as plt

abl = pd.read_csv(ABL_CSV)
wide = abl.pivot_table(index="pair_id", columns="flavor", values="iou")
# use wrong_both delta as the ranking signal (falls back to wrong_disease)
rank_flavor = "wrong_both" if "wrong_both" in wide.columns else "wrong_disease"
wide = wide.dropna(subset=["correct", rank_flavor]).copy()
wide["delta"] = wide[rank_flavor] - wide["correct"]

rng = np.random.default_rng(SEED)
n_each = max(10, (N_GALLERY + 2)//3)
hurt   = list(wide.sort_values("delta").head(n_each).index)              # most negative
robust = list(wide.reindex(wide["delta"].abs().sort_values().index).head(n_each).index)
rand   = list(rng.choice(wide.index.values, size=min(n_each, len(wide)), replace=False))
sel, seen = [], set()
for pid in hurt + robust + rand:
    if pid not in seen: sel.append(pid); seen.add(pid)
sel = sel[:N_GALLERY]
print(f"showing {len(sel)} examples (hurt / robust / random)")

prompts_df = pd.read_csv(CORRUPT_CSV)
P_BY_PAIR = {pid: g.set_index("flavor") for pid, g in prompts_df.groupby("pair_id")}
pos_by_id = pos.set_index("pair_id")
abl_by_pf = abl.set_index(["pair_id","flavor"])

def _overlay(ax, img, mask, title, color=(1,0,0)):
    ax.imshow(img, cmap="gray")
    if mask is not None and np.asarray(mask).sum() > 0:
        m = np.asarray(mask).astype(float)
        rgba = np.zeros((*m.shape,4)); rgba[...,0],rgba[...,1],rgba[...,2] = color
        rgba[...,3] = 0.45*m
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9); ax.axis("off")

FLAVORS = ["correct","wrong_disease","wrong_location","wrong_both"]
for pid in sel:
    r = pos_by_id.loc[pid]; pg = P_BY_PAIR.get(pid)
    if pg is None: continue
    img = np.array(load_image(r.image_path).convert("L"))
    gt  = load_silver_mask(r.seg_mask_path)
    fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("image", fontsize=9); ax[0].axis("off")
    _overlay(ax[1], img, gt, "GT (silver)", color=(0,1,0))
    col = 2
    print("\n" + "="*100)
    print(f"pair {pid} | disease={r.target} | gt_px={int(gt.sum())}")
    for flavor in FLAVORS:
        if flavor not in pg.index:
            ax[col].axis("off"); col+=1; continue
        text = pg.loc[flavor,"text"]
        try: iou = abl_by_pf.loc[(pid,flavor),"iou"]
        except Exception: iou = np.nan
        if flavor == "correct":
            title = f"correct\nIoU={iou:.2f}"
        elif not isinstance(text,str) or text=="":
            _overlay(ax[col], img, None, f"{flavor}\n(N/A)"); col+=1
            print(f"  {flavor:14s}: N/A")
            continue
        else:
            try:
                mi = abl_by_pf.loc[(pid,flavor),"mask_iou_vs_correct"]; dp = abl_by_pf.loc[(pid,flavor),"mean_abs_dprob"]
            except Exception: mi, dp = np.nan, np.nan
            title = f"{flavor}\nIoU={iou:.2f} | mIoU={mi:.2f}"
        pred, logits, _ = segment(load_image(r.image_path), text)
        color = (1,0,0) if flavor!="correct" else (0,0.5,1)
        _overlay(ax[col], img, pred, title, color=color); col += 1
        print(f"  {flavor:14s}: IoU={iou:.3f} | \"{text}\"")
    plt.tight_layout(); plt.show()
"""))

# =============================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU",
                  "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}

_out = Path(__file__).resolve().parent / "project" / "notebooks" / "text_conditioning_ablation_colab.ipynb"
_out.parent.mkdir(parents=True, exist_ok=True)
import os as _os
out_path = Path(_os.environ.get("NOTEBOOK_OUT", _out))
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")
