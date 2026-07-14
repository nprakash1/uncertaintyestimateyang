"""Patch project/notebooks/rosalia_iou_regression_colab.ipynb to use the 5-way
uncertainty TYPE scheme (certain / presence / spatial / diagnostic / borderline)
instead of the binary certain-vs-uncertain label.

- Cell "load & prepare": attach `unc_type` (from results CSV, a cached types CSV,
  else leave blank for the MedGemma cell to fill).
- New cell: label `unc_type` with MedGemma using the provided LABELS/DEFINITIONS/
  FEWSHOT (only runs if some positives are unlabeled).
- Cell "nested OLS": regress IoU on C(unc_type, Treatment('certain')) + disease + size.
- Coefficient plot: per-type IoU gap vs 'certain'.
"""
import json
from pathlib import Path

NB = Path("project/notebooks/rosalia_iou_regression_colab.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]


def find(sub):
    for i, c in enumerate(cells):
        if sub in "".join(c.get("source", [])):
            return i
    raise SystemExit(f"cell not found: {sub!r}")


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


# ---------------------------------------------------------------- load & prepare
LOAD = r'''# --- load & prepare the regression frame ---
res = pd.read_csv(RESULTS_CSV)
print("rows:", len(res), "| columns:", list(res.columns))

P = res[(res.get("polarity") == "positive") & res["iou"].notna()].copy()
P["log_area"] = np.log1p(P["gt_px"])          # silver-mask area (skewed -> log)
P["disease"]  = P["target"].astype("category")

# ---- attach the 5-way uncertainty TYPE (unc_type) ---------------------------
# Priority: (1) already a column in the results CSV; (2) merge a cached types CSV
# (produced by the uncertainty-type diagnostic notebook); (3) leave blank so the
# next cell can label it with MedGemma using LABELS/DEFINITIONS/FEWSHOT.
def _k(df):
    return df["study_id"].astype(str) + "|" + df["target"].astype(str)

if "unc_type" not in P.columns:
    P["unc_type"] = np.nan
    _TYPES_CANDIDATES = [
        os.path.join(OUT_DIR, "uncertainty_types_regression.csv"),
        os.path.join(OUT_DIR, "test_positives_with_unc_type.csv"),
        os.path.join(OUT_DIR, "uncertainty_types_test.csv"),
        "/content/drive/MyDrive/mimic_ils_rosalia/uncertainty_types_test.csv",
        "test_positives_with_unc_type.csv",
        "uncertainty_types_test.csv",
    ]
    for _tc in _TYPES_CANDIDATES:
        if os.path.exists(_tc):
            _t = pd.read_csv(_tc)
            if {"study_id", "target", "unc_type"}.issubset(_t.columns):
                _map = dict(zip(_k(_t), _t["unc_type"]))
                P["unc_type"] = _k(P).map(_map)
                print(f"[types] merged unc_type from {_tc}: "
                      f"{P['unc_type'].notna().sum()}/{len(P)} labeled")
                break
    else:
        print("[types] no cached unc_type found; run the MedGemma cell below.")
print("unc_type coverage:", int(P["unc_type"].notna().sum()), "/", len(P))
'''

# --------------------------------------------------- MedGemma 5-way type labeling
LABEL = r'''# --- label uncertainty TYPE with MedGemma (5-way scheme) ---------------------
# Runs ONLY if some positives still lack unc_type. Needs transformers>=4.50 and
# access to google/medgemma-4b-it. Report text is pulled from the manifest(s).
LABELS = ["certain", "presence", "spatial", "diagnostic", "borderline"]

DEFINITIONS = (
    "Definitions of the uncertainty TYPE for a specific finding:\n"
    "- certain: the report states the finding as a fact, no hedging.\n"
    "- presence: the report is unsure the finding EXISTS or is visible "
    "(possible, probable, questionable, cannot exclude, suggestive of, may be present, suspicious for).\n"
    "- spatial: the finding is accepted but its LOCATION, EXTENT, or BOUNDARY is unclear "
    "(ill-defined, poorly marginated, indistinct margins, hazy, extent/size uncertain).\n"
    "- diagnostic: the finding is seen but its IDENTITY is a differential between entities "
    "(X versus Y, could represent X or Y, may reflect atelectasis or consolidation).\n"
    "- borderline: AMBIGUOUS or only weakly hedged, so you cannot confidently decide "
    "between certain and one of the uncertain types (trace/tiny/subtle/minimal 'if any', "
    "very mild qualifiers, mixed or conflicting cues). Use this when it is a genuine toss-up.\n"
)

FEWSHOT = (
    "Examples:\n"
    "Finding: effusion | Report: 'Moderate right pleural effusion.' -> LABEL: certain; WHY: stated as fact.\n"
    "Finding: cardiomegaly | Report: 'The cardiac silhouette is enlarged.' -> LABEL: certain; WHY: definite.\n"
    "Finding: pneumonia | Report: 'Findings possibly represent pneumonia.' -> LABEL: presence; WHY: unsure it exists.\n"
    "Finding: effusion | Report: 'Cannot exclude a small pleural effusion.' -> LABEL: presence; WHY: existence hedged.\n"
    "Finding: opacity | Report: 'Ill-defined opacity at the right base.' -> LABEL: spatial; WHY: borders unclear.\n"
    "Finding: consolidation | Report: 'Hazy airspace opacity of indistinct extent.' -> LABEL: spatial; WHY: extent unclear.\n"
    "Finding: atelectasis | Report: 'Opacity may reflect atelectasis versus consolidation.' -> LABEL: diagnostic; WHY: differential.\n"
    "Finding: opacity | Report: 'Could represent aspiration or early pneumonia.' -> LABEL: diagnostic; WHY: competing diagnoses.\n"
    "Finding: effusion | Report: 'Trace pleural effusion, if any.' -> LABEL: borderline; WHY: 'if any' weakly hedges a stated finding, unclear if certain or presence.\n"
    "Finding: atelectasis | Report: 'Subtle bibasilar opacity, likely minimal atelectasis.' -> LABEL: borderline; WHY: mild/subtle qualifiers make certain-vs-uncertain a toss-up.\n"
)

_need = P["unc_type"].isna()
if _need.any():
    print(f"[types] {int(_need.sum())} positives need labeling -> loading MedGemma")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "--upgrade",
                    "transformers>=4.50", "accelerate>=0.30", "huggingface_hub"])
    from huggingface_hub import notebook_login; notebook_login()
    os.environ["HF_HUB_DISABLE_XET"] = "1"   # Xet backend can hang on Colab; use plain HTTPS
    import re, torch
    from transformers import AutoProcessor, AutoModelForImageTextToText


    # report text per (study_id, target): merge from the split manifests
    REPO_RAW = ("https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/"
                "rule-bag-of-words/project/data/mimic_ils")
    _mans = []
    for _f in ["mimic_ils_test_manifest.csv", "mimic_ils_val_manifest.csv",
               "mimic_ils_subset_manifest.csv"]:
        _p = f"/content/{_f}"
        if not os.path.exists(_p):
            os.system(f"curl -fsSL '{REPO_RAW}/{_f}' -o '{_p}'")
        if os.path.exists(_p):
            _mans.append(pd.read_csv(_p))
    man = pd.concat(_mans, ignore_index=True).drop_duplicates(["study_id", "target"])
    rep = dict(zip(_k(man), man["section_content"].astype(str)))

    MODEL_NAME = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tok = getattr(processor, "tokenizer", None) or processor
    tok.padding_side = "left"
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def classify(target, report_snip):
        prompt = (
            "You classify the TYPE of uncertainty in a radiology report about ONE finding.\n"
            + DEFINITIONS + "\n" + FEWSHOT + "\n"
            "Now classify this one. If it is a genuine toss-up between certain and an "
            "uncertain type, use borderline. Answer EXACTLY as "
            "'LABEL: <one of " + ", ".join(LABELS) + ">; WHY: <short reason>'.\n"
            f"Finding: {target} | Report: '{str(report_snip)[:600]}' ->")
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=40, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"label\s*:\s*([a-z]+)", raw.lower())
        lab = m.group(1) if m else "certain"
        return (lab if lab in LABELS else "certain"), raw.strip()

    TYPES_CSV = os.path.join(OUT_DIR, "uncertainty_types_regression.csv")
    done = set()
    if os.path.exists(TYPES_CSV):
        _d = pd.read_csv(TYPES_CSV)
        done = set(_d["study_id"].astype(str) + "|" + _d["target"].astype(str))
    from tqdm.auto import tqdm
    todo = P[_need][["study_id", "target"]].drop_duplicates()
    for r in tqdm(todo.itertuples(), total=len(todo), desc="unc_type"):
        k = f"{r.study_id}|{r.target}"
        if k in done:
            continue
        lab, raw = classify(r.target, rep.get(k, ""))
        pd.DataFrame([{"study_id": r.study_id, "target": r.target,
                       "unc_type": lab, "raw": raw[:200]}]).to_csv(
            TYPES_CSV, mode="a", header=not os.path.exists(TYPES_CSV), index=False)
    _t = pd.read_csv(TYPES_CSV)
    _map = dict(zip(_t["study_id"].astype(str) + "|" + _t["target"].astype(str), _t["unc_type"]))
    P["unc_type"] = _k(P).map(_map).fillna(P["unc_type"])
    import gc; del mdl, processor; gc.collect(); torch.cuda.empty_cache()
    print("[types] labeling done ->", TYPES_CSV)
else:
    print("[types] all positives already have unc_type; skipping MedGemma.")

# ---- finalize: categorical type (baseline='certain') + derived binary --------
P = P[P["unc_type"].isin(LABELS)].copy()
P["unc_type"]     = pd.Categorical(P["unc_type"], categories=LABELS, ordered=False)
P["is_uncertain"] = (P["unc_type"].astype(str) != "certain").astype(int)

tidy = P[["target", "gt_px", "log_area", "unc_type", "is_uncertain", "iou"]
         + (["split"] if "split" in P.columns else [])].copy()
tidy_path = os.path.join(OUT_DIR, "iou_regression_table.csv")
tidy.to_csv(tidy_path, index=False)
print("saved tidy table ->", tidy_path, "| n =", len(P))
print("\nunc_type distribution:")
print(P["unc_type"].value_counts().reindex(LABELS, fill_value=0).to_string())
print("\ncounts by lesion x unc_type:")
print(pd.crosstab(P["target"], P["unc_type"]).to_string())
'''

# --------------------------------------------------------------- nested OLS (type)
OLS = r'''# --- nested OLS models (robust HC3): 5-way uncertainty TYPE -------------------
# Baseline = 'certain'. Each beta = that type's IoU gap vs certain findings.
# Watch the per-type betas change as disease + silver-mask size are controlled.
_TYPES_NONBASE = [t for t in LABELS if t != "certain"]
_TERM = "C(unc_type, Treatment('certain'))"

def fit_report(df, tag):
    df = df.copy()
    df["unc_type"] = df["unc_type"].astype(str)   # drop unused categories
    m1 = smf.ols(f"iou ~ {_TERM}", data=df).fit(cov_type="HC3")
    m2 = smf.ols(f"iou ~ {_TERM} + C(disease)", data=df).fit(cov_type="HC3")
    m3 = smf.ols(f"iou ~ {_TERM} + C(disease) + log_area", data=df).fit(cov_type="HC3")
    print(f"\n===== {tag}  (n={len(df)}) =====")
    print("per-type IoU gap vs 'certain' (M3: +disease+size):")
    rows, ci = [], m3.conf_int()
    for t in _TYPES_NONBASE:
        name = f"{_TERM}[T.{t}]"
        if name in m3.params.index:
            b, p = m3.params[name], m3.pvalues[name]
            lo, hi = ci.loc[name]
            n_t = int((df["unc_type"] == t).sum())
            print(f"  {t:11s} beta={b:+.4f}  p={p:.4g}  95%CI=[{lo:+.3f},{hi:+.3f}]  n={n_t}")
            rows.append(dict(type=t, beta=b, p=p, lo=lo, hi=hi, n=n_t))
    print(f"M3 R2={m3.rsquared:.3f}")
    return m1, m2, m3, pd.DataFrame(rows)

m1, m2, m3, coefs_all = fit_report(P, "ALL positive findings")
'''

# ------------------------------------------------------------------- coef plot
PLOT = r'''# --- coefficient plot: per-type IoU gap vs 'certain' (M3, ALL findings) -------
fig, ax = plt.subplots(figsize=(6.5, 4))
if len(coefs_all):
    y = np.arange(len(coefs_all))[::-1]
    ax.errorbar(coefs_all["beta"], y,
                xerr=[coefs_all["beta"] - coefs_all["lo"], coefs_all["hi"] - coefs_all["beta"]],
                fmt="o", capsize=4, color="tab:blue")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(coefs_all["type"], coefs_all["n"])])
    for _, r in coefs_all.iterrows():
        ax.annotate(f"p={r['p']:.2g}",
                    (r["beta"], y[list(coefs_all["type"]).index(r["type"])]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_xlabel("beta vs 'certain' (effect on IoU)  —  negative = lower IoU than certain")
ax.set_title("Per-type uncertainty effect on IoU (M3: +disease+size)")
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "iou_regression_coefplot.png")
plt.savefig(plot_path, dpi=130); plt.show()
print("saved ->", plot_path)
'''

# apply edits (by content anchors so indices stay correct; idempotent)
cells[find("--- load & prepare the regression frame ---")]["source"] = LOAD
try:
    cells[find("label uncertainty TYPE with MedGemma")]["source"] = LABEL  # already patched
except SystemExit:
    cells.insert(find("nested OLS models"), code_cell(LABEL))              # first patch
cells[find("nested OLS models")]["source"] = OLS
cells[find("coefficient plot")]["source"] = PLOT


NB.write_text(json.dumps(nb, indent=1))
print(f"patched {NB} -> {len(cells)} cells")
