"""Build project/notebooks/uncertainty_balanced_kfold_colab.ipynb

A focused experiment notebook (reuses the ROSALIA setup from
mimic_ils_rosalia_reproduction_colab.ipynb) that answers, robustly:

    Do ROSALIA's segmentation quality + confidence differ between findings whose
    report language is CERTAIN vs UNCERTAIN, once we (a) balance the two groups
    to a similar sample size, (b) remove disease as a confounder, and (c) show
    the effect is stable across 5 folds?

Design (the important bits):
  * PER-PAIR METRICS CACHED ONCE. ROSALIA is run a single time over every
    positive pair; for each we store three metrics computed from the model's
    PRE-THRESHOLD sigmoid probability map (segment() already returns the raw
    `logits`, so no need to hack ROSALIA's internals):
        - IoU vs the silver GT mask,
        - mean predicted-mask probability  = mean sigmoid prob inside PRED mask,
        - mean prob inside the GT lesion    = mean sigmoid prob inside GT mask.
  * DISEASE-STRATIFIED BALANCING. For each disease we take n_d =
    min(#certain_d, #uncertain_d) and sample that many from each group, so the
    certain and uncertain arms have IDENTICAL disease composition + equal n.
    This is what stops disease (esp. cardiomegaly) from driving the contrast.
  * 5 DISJOINT FOLDS. The matched dataset is split into 5 stratified folds; the
    certain-vs-uncertain metrics are computed per fold and aggregated
    (mean +/- std across folds) -> a stability estimate.
  * MACRO + MICRO + EXCLUDE_CARDIOMEGALY. We report both micro (pool all) and
    macro (mean of per-disease means, cardiomegaly-robust) aggregates, and a
    toggle to drop cardiomegaly entirely.

Prereqs on Drive (same as the reproduction notebook):
  * MedGemma uncertainty labels cached at
    <WORK_DIR>/medgemma_uncertainty_subset.json  (from Phase A of the other nb),
  * the image zip(s) + mimic-cxr-ext-ils.zip in the shared Drive folder.
This notebook is ROSALIA-only (Phase B); run Phase A in the other notebook first
(or point UNC_LABELS_PATH at any cached certain/uncertain labels).
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
cells.append(md(r"""# ROSALIA certain-vs-uncertain: balanced, disease-matched, 5-fold

This notebook tests whether ROSALIA's **segmentation quality** and **confidence**
differ between findings with **certain** vs **uncertain** report language, while
controlling for the two things that dominated the earlier regression:
**class imbalance** and **disease** (esp. cardiomegaly).

**Three metrics per finding** (all from ROSALIA's *pre-threshold* sigmoid map,
which `segment()` returns as `logits` — no model surgery needed):
1. **IoU** vs the silver GT mask.
2. **mean predicted-mask probability** — mean sigmoid prob inside the PRED mask
   (how confident the model is where it did fire).
3. **mean prob inside the GT lesion** — mean sigmoid prob inside the GT mask
   (how much probability mass it put on the true lesion, fired or not).

**Controls:**
* **Disease-stratified balancing** — certain & uncertain arms are matched to the
  *same disease mix* with *equal n per disease* (removes disease confounding).
* **5 disjoint folds** — metrics aggregated mean±std across folds (stability).
* **Macro-average** (mean of per-disease means) + **EXCLUDE_CARDIOMEGALY** toggle
  so cardiomegaly can't dominate.

**Runtime:** GPU ≥16 GB (ROSALIA-7B bf16 ~15 GB). Run Phase A of the
reproduction notebook first so MedGemma certain/uncertain labels are cached.
"""))

# ---- Cell 1: install ROSALIA (copied from reproduction B1) -------------------
cells.append(md(r"""## Cell 1 — install ROSALIA deps + clone repo, then RESTART"""))
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
print("\n>>> RESTART THE RUNTIME NOW, then run Cell 2 onward. <<<")
"""))

# ---- Cell 2: HF login -------------------------------------------------------
cells.append(md(r"""## Cell 2 — Hugging Face login (ROSALIA/LISA weights are public)"""))
cells.append(code(r"""from huggingface_hub import notebook_login
notebook_login()
"""))

# ---- Cell 3: config (copied/trimmed from reproduction B3) -------------------
cells.append(md(r"""## Cell 3 — config + paths (post-restart)"""))
cells.append(code(r"""import os
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

# split must match the split whose MedGemma labels you cached
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

# ---- experiment knobs -------------------------------------------------------
N_FOLDS             = 5
EXCLUDE_CARDIOMEGALY = False   # True -> drop cardiomegaly entirely
MATCH_SEED          = 42       # seed for disease-stratified balancing
print(f"[config] {MANIFEST_CHOICE} | N_FOLDS={N_FOLDS} | "
      f"EXCLUDE_CARDIOMEGALY={EXCLUDE_CARDIOMEGALY}")
print("zips present?", {os.path.basename(z): os.path.exists(z) for z in IMG_ZIPS},
      "| ext:", os.path.exists(EXT_ZIP), "| unc labels:", os.path.exists(UNC_LABELS_PATH))
"""))

# ---- Cell 3b: extract zips (copied from reproduction B3b) -------------------
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

# ---- Cell 4: load manifest + labels + loaders ------------------------------
cells.append(md(r"""## Cell 4 — load manifest + MedGemma certain/uncertain labels; loaders

Positives only. `uncertainty_label` is the binary MedGemma label
(certain/uncertain); rows labelled `unknown` (no label) are dropped.
"""))
cells.append(code(r"""import os, json
import numpy as np, pandas as pd
from PIL import Image

manifest = pd.read_csv(SUBSET_MANIFEST)
unc = {}
if os.path.exists(UNC_LABELS_PATH):
    unc = {k: v["uncertainty_label"] for k, v in json.load(open(UNC_LABELS_PATH)).items()}
    print(f"loaded {len(unc)} MedGemma uncertainty labels")
else:
    print("[ERROR] no MedGemma labels; run Phase A of the reproduction notebook first.")
manifest["uncertainty_label"] = manifest.apply(
    lambda r: unc.get(f"{r.study_id}|{r.target}", "unknown"), axis=1)

pos = manifest[manifest.polarity == "positive"].reset_index(drop=True)
pos = pos[pos.uncertainty_label.isin(["certain", "uncertain"])].reset_index(drop=True)
print(f"positive pairs with a certain/uncertain label: {len(pos)}")
print("group counts:", pos.uncertainty_label.value_counts().to_dict())
print("disease x group:\n", pd.crosstab(pos.target, pos.uncertainty_label).to_string())

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
print(f"positive images present locally: {int(_present.sum())}/{len(pos)}")
pos = pos[_present].reset_index(drop=True)
"""))

# ---- Cell 5: load ROSALIA (copied from reproduction B5) --------------------
cells.append(md(r"""## Cell 5 — load ROSALIA (LISA-7B + SAM-H)

`segment(pil, instruction) -> (binary_mask, logits, text)`. `logits` is the raw
pre-threshold mask score map; sigmoid(logits) is the probability map we use for
the confidence metrics (mask = logits>0 == prob>0.5).
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
    raise RuntimeError("numpy upgraded to 2.x -> Runtime > Restart, then re-run Cell 2-5.")

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

# ---- Cell 6: metrics -------------------------------------------------------
cells.append(md(r"""## Cell 6 — per-example metrics (IoU + the two probability metrics)"""))
cells.append(code(r"""import numpy as np
from skimage.transform import resize as _skresize

def _grid(mask, grid=GRID):
    m = np.asarray(mask).astype(np.float32)
    if m.shape != (grid, grid):
        m = _skresize(m, (grid, grid), order=0, mode="edge",
                      anti_aliasing=False, preserve_range=True)
    return (m > 0.5).astype(np.uint8)

def iou_dice(pred, gt, grid=GRID):
    p, g = _grid(pred, grid), _grid(gt, grid)
    inter = int(np.logical_and(p, g).sum()); union = int(np.logical_or(p, g).sum())
    iou = inter/union if union>0 else float("nan")
    dice = (2*inter)/(p.sum()+g.sum()) if (p.sum()+g.sum())>0 else float("nan")
    return iou, dice, inter, union

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x, dtype=np.float64)))

def pair_metrics(pred, logits, gt):
    # returns iou, mean prob in PRED mask, mean prob in GT lesion, max prob, fired
    iou, dice, *_ = iou_dice(pred, gt)
    if logits is None:
        return dict(iou=iou, dice=dice, p_in_pred=np.nan, p_in_gt=np.nan,
                    max_prob=np.nan, fired=0)
    prob = _sigmoid(logits)
    predb = prob > 0.5                                   # == logits>0 == the pred mask
    p_in_pred = float(prob[predb].mean()) if predb.any() else np.nan
    # resize GT to the probability map's resolution, then average prob inside it
    g = _skresize(np.asarray(gt).astype(np.float32), prob.shape, order=0,
                  mode="edge", anti_aliasing=False, preserve_range=True) > 0.5
    p_in_gt = float(prob[g].mean()) if g.any() else np.nan
    return dict(iou=iou, dice=dice, p_in_pred=p_in_pred, p_in_gt=p_in_gt,
                max_prob=float(prob.max()), fired=int(predb.any()))
print("metrics ready: iou, p_in_pred (conf where it fired), p_in_gt (mass on true lesion).")
"""))

# ---- Cell 7: cache per-pair metrics ONCE -----------------------------------
cells.append(md(r"""## Cell 7 — run ROSALIA ONCE over all positives; cache per-pair metrics

Resumable. We infer each positive pair a single time and store IoU + the two
probability metrics. The 5-fold balanced analysis below reads only this cache,
so folds cost nothing extra.
"""))
cells.append(code(r"""import os, pandas as pd
from tqdm.auto import tqdm

PAIR_CSV = os.path.join(WORK_DIR, "balanced_kfold_per_pair.csv")
done = set()
if os.path.exists(PAIR_CSV):
    done = set(pd.read_csv(PAIR_CSV)["pair_id"]); print(f"resuming; {len(done)} cached")

def _append(rec):
    pd.DataFrame([rec]).to_csv(PAIR_CSV, mode="a",
                              header=not os.path.exists(PAIR_CSV), index=False)

for row in tqdm(pos.itertuples(), total=len(pos), desc="ROSALIA infer"):
    if row.pair_id in done: continue
    try:
        img = load_image(row.image_path); gt = load_silver_mask(row.seg_mask_path)
        pred, logits, txt = segment(img, row.instruction)
        m = pair_metrics(pred, logits, gt)
        _append({"pair_id":row.pair_id, "study_id":row.study_id,
                 "disease":row.target, "group":row.uncertainty_label,
                 "gt_px":int(gt.sum()), **m})
    except Exception as e:
        print(f"[skip] {row.pair_id}: {type(e).__name__}: {e}")

pair_df = pd.read_csv(PAIR_CSV).drop_duplicates("pair_id", keep="last")
pair_df.to_csv(PAIR_CSV, index=False)
print(f"cached {len(pair_df)} pairs -> {PAIR_CSV}")
print("group counts:", pair_df.group.value_counts().to_dict())
"""))

# ---- Cell 8: disease-stratified balanced 5-fold experiment -----------------
cells.append(md(r"""## Cell 8 — disease-stratified balancing + 5-fold split

* **Balance:** for each disease, take `n_d = min(#certain_d, #uncertain_d)` and
  sample `n_d` from each group -> certain & uncertain have identical disease mix.
* **Fold:** split the matched rows into `N_FOLDS` stratified folds (by
  disease×group) so every fold stays balanced.
"""))
cells.append(code(r"""import numpy as np, pandas as pd

df = pd.read_csv(PAIR_CSV)
df = df[df.group.isin(["certain","uncertain"])].copy()
if EXCLUDE_CARDIOMEGALY:
    df = df[df.disease != "cardiomegaly"].copy()
    print("[info] excluded cardiomegaly")

# ---- disease-stratified balancing (equal n per disease per group) ----------
rng = np.random.RandomState(MATCH_SEED)
matched_parts = []
print("per-disease matching (n kept per group):")
for dis, sub in df.groupby("disease"):
    c = sub[sub.group=="certain"]; u = sub[sub.group=="uncertain"]
    n_d = min(len(c), len(u))
    if n_d == 0:
        print(f"  {dis:14s} skipped (certain={len(c)}, uncertain={len(u)})"); continue
    matched_parts.append(c.sample(n_d, random_state=rng))
    matched_parts.append(u.sample(n_d, random_state=rng))
    print(f"  {dis:14s} n_d={n_d}  (from certain={len(c)}, uncertain={len(u)})")
matched = pd.concat(matched_parts).reset_index(drop=True)
print(f"\nmatched dataset: {len(matched)} rows | "
      f"{matched.group.value_counts().to_dict()}")
print("matched disease mix identical across groups?",
      (pd.crosstab(matched.disease, matched.group).diff(axis=1).iloc[:,1].abs().sum()==0))

# ---- assign stratified folds (by disease x group) --------------------------
matched["fold"] = -1
rng2 = np.random.RandomState(MATCH_SEED + 1)
for (_dis,_grp), idx in matched.groupby(["disease","group"]).groups.items():
    idx = np.array(idx); rng2.shuffle(idx)
    matched.loc[idx, "fold"] = np.arange(len(idx)) % N_FOLDS
print("\nfold sizes:", matched.fold.value_counts().sort_index().to_dict())
print("fold x group:\n", pd.crosstab(matched.fold, matched.group).to_string())
"""))

# ---- Cell 9: aggregate metrics across folds --------------------------------
cells.append(md(r"""## Cell 9 — metrics per fold, aggregated across folds

For each fold we compute, for **certain** and **uncertain**:
* **micro** = mean over all rows in the group,
* **macro** = mean of per-disease means (every disease weighted equally →
  cardiomegaly-robust).
Then we aggregate **mean ± std across the 5 folds** and report the
uncertain−certain gap with a paired t-test across folds.
"""))
cells.append(code(r"""import numpy as np, pandas as pd
from scipy import stats

METRICS = ["iou", "p_in_pred", "p_in_gt"]

def fold_group_stats(fold_df):
    out = {}
    for grp, g in fold_df.groupby("group"):
        for m in METRICS:
            out[(grp, m, "micro")] = g[m].mean()
            out[(grp, m, "macro")] = g.groupby("disease")[m].mean().mean()
    return out

rows = []
for f in range(N_FOLDS):
    fd = matched[matched.fold == f]
    s = fold_group_stats(fd)
    rec = {"fold": f}
    for (grp, m, agg), v in s.items():
        rec[f"{grp}_{m}_{agg}"] = v
    rows.append(rec)
folds = pd.DataFrame(rows)

def summarize(agg):
    print(f"\n================  {agg.upper()}-AVERAGED  (mean±std across {N_FOLDS} folds)  ================")
    print(f"{'metric':12s} {'certain':>18s} {'uncertain':>18s} {'gap(unc-cer)':>16s} {'paired t p':>12s}")
    for m in METRICS:
        c = folds[f"certain_{m}_{agg}"]; u = folds[f"uncertain_{m}_{agg}"]
        gap = u.values - c.values
        try:    p = stats.ttest_rel(u, c).pvalue
        except Exception: p = float("nan")
        print(f"{m:12s} {c.mean():8.4f}±{c.std():.4f}  {u.mean():8.4f}±{u.std():.4f}  "
              f"{gap.mean():+8.4f}±{gap.std():.4f}  {p:12.4g}")

summarize("micro")
summarize("macro")
folds.round(4)
"""))

# ---- Cell 10: per-disease stratified table ---------------------------------
cells.append(md(r"""## Cell 10 — per-disease certain vs uncertain (aggregated across folds)

Within each disease, the certain-vs-uncertain gap for every metric, averaged
across folds (mean±std). This is the stratified view you asked for.
"""))
cells.append(code(r"""import numpy as np, pandas as pd

def per_disease_table(metric):
    recs = []
    for dis in sorted(matched.disease.unique()):
        cvals, uvals, gaps = [], [], []
        for f in range(N_FOLDS):
            fd = matched[(matched.fold==f) & (matched.disease==dis)]
            c = fd[fd.group=="certain"][metric].mean()
            u = fd[fd.group=="uncertain"][metric].mean()
            if np.isfinite(c) and np.isfinite(u):
                cvals.append(c); uvals.append(u); gaps.append(u-c)
        if cvals:
            recs.append(dict(disease=dis, n_per_grp=int((matched.disease==dis).sum()//2),
                             certain=np.mean(cvals), uncertain=np.mean(uvals),
                             gap=np.mean(gaps), gap_std=np.std(gaps)))
    return pd.DataFrame(recs).set_index("disease")

for m in ["iou", "p_in_pred", "p_in_gt"]:
    print(f"\n===== {m}: per-disease certain vs uncertain (mean across folds) =====")
    print(per_disease_table(m).round(4).to_string())
"""))

# ---- Cell 11: plots --------------------------------------------------------
cells.append(md(r"""## Cell 11 — plots: per-fold gaps + per-disease bars"""))
cells.append(code(r"""import numpy as np, matplotlib.pyplot as plt, os

# (a) macro metric: certain vs uncertain across folds
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, m in zip(axes, ["iou","p_in_pred","p_in_gt"]):
    c = folds[f"certain_{m}_macro"]; u = folds[f"uncertain_{m}_macro"]
    x = np.arange(N_FOLDS); w = 0.38
    ax.bar(x-w/2, c, w, label="certain", color="tab:blue")
    ax.bar(x+w/2, u, w, label="uncertain", color="tab:orange")
    ax.axhline(c.mean(), color="tab:blue", ls="--", lw=1)
    ax.axhline(u.mean(), color="tab:orange", ls="--", lw=1)
    ax.set_title(f"{m} (macro) by fold"); ax.set_xlabel("fold"); ax.set_xticks(x)
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"balanced_kfold_macro.png"), dpi=120); plt.show()

# (b) per-disease IoU gap (uncertain - certain), averaged across folds
tbl = per_disease_table("iou").sort_values("gap")
fig, ax = plt.subplots(figsize=(8,4))
ax.barh(tbl.index, tbl["gap"], xerr=tbl["gap_std"], color="tab:purple", capsize=3)
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_xlabel("IoU gap (uncertain - certain)   negative = uncertain worse")
ax.set_title("Per-disease IoU gap, disease-matched, mean±std across folds")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR,"balanced_kfold_disease_gap.png"), dpi=120); plt.show()
print("saved plots ->", WORK_DIR)
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
_default_out = Path(__file__).resolve().parent / "project" / "notebooks" / "uncertainty_balanced_kfold_colab.ipynb"
out_path = Path(os.environ.get("NOTEBOOK_OUT", _default_out))
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")
