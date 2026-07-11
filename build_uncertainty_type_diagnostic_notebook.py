"""Build project/notebooks/uncertainty_type_diagnostic_colab.ipynb

A DIAGNOSTIC notebook (single runtime, modern transformers) that runs MedGemma
over EVERY positive finding in the MIMIC-ILS TEST split and assigns each one a
report-language uncertainty *type*:

    certain    - the report states the finding as fact
    presence   - hedges about whether the finding EXISTS / is visible
                 (possible, probable, cannot exclude, suggestive of, questionable)
    spatial    - hedges about LOCATION / EXTENT / BOUNDARY of the finding
                 (ill-defined, poorly marginated, indistinct, extent uncertain)
    diagnostic - hedges about WHAT the finding is: a differential between entities
                 (atelectasis versus consolidation, may represent X or Y)

The goal is transparency: after labelling, the notebook PRINTS several worked
examples per category (finding + the exact report sentence + MedGemma's label +
its one-line rationale) so you can see exactly what the classifier is doing,
then shows overall counts and a per-lesion breakdown, and saves a CSV.

Only needs the test manifest (pulled from the repo) + MedGemma (gated HF model).
No ROSALIA, no images, no Drive zips required. GPU runtime recommended.
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

cells.append(md(r"""# Uncertainty-TYPE diagnostic — MedGemma over MIMIC-ILS test positives

For every **positive** finding in the MIMIC-ILS **test** split, this notebook asks
**MedGemma** to classify the *type* of report-language uncertainty:

| label | meaning | example phrasing |
|---|---|---|
| **certain** | stated as fact | "Moderate right pleural effusion." |
| **presence** | unsure it EXISTS / is visible | "possible pneumonia", "cannot exclude effusion" |
| **spatial** | unsure of LOCATION / EXTENT / BOUNDARY | "ill-defined opacity", "indistinct margins" |
| **diagnostic** | unsure WHAT it is (differential) | "atelectasis versus consolidation" |

It then **prints several worked examples per category** — the finding, the exact
report sentence, the assigned label, and MedGemma's one-line reason — so you can
inspect what the model is actually doing before trusting the labels. Finally it
reports category counts, a per-lesion breakdown, and saves everything to CSV.

**Needs:** the test manifest (auto-downloaded) + MedGemma (gated). GPU runtime.
No images / Drive / ROSALIA required.
"""))

cells.append(md(r"""## Cell 1 — install deps (transformers>=4.50 for MedGemma)"""))

cells.append(code(r"""!pip -q install --upgrade 'transformers>=4.50' 'accelerate>=0.30' huggingface_hub pandas
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 50), \
    "Need transformers>=4.50 for MedGemma; if it just upgraded, restart the runtime."
"""))

cells.append(md(r"""## Cell 2 — Hugging Face login (MedGemma is gated)

Accept the license at https://huggingface.co/google/medgemma-4b-it first.
"""))

cells.append(code(r"""from huggingface_hub import notebook_login
notebook_login()
"""))

cells.append(md(r"""## Cell 3 — load the TEST-split positives

Pulls `mimic_ils_test_manifest.csv` from the repo and keeps positive pairs. One
classification per unique (study, finding) — report text is per study, so
duplicate (study, target) rows share a label.
"""))

cells.append(code(r"""import os, re
import pandas as pd

REPO_RAW_URL = "https://raw.githubusercontent.com/nprakash1/uncertaintyestimateyang/rule-bag-of-words"
MANIFEST = "/content/mimic_ils_test_manifest.csv"
if not os.path.exists(MANIFEST):
    rc = os.system(f"curl -fsSL '{REPO_RAW_URL}/project/data/mimic_ils/mimic_ils_test_manifest.csv' -o '{MANIFEST}'")
    if rc != 0 or not os.path.exists(MANIFEST):
        print("[warn] could not fetch manifest; upload mimic_ils_test_manifest.csv to", MANIFEST)

OUT_DIR = "/content/drive/MyDrive/mimic_ils_rosalia"
try:
    from google.colab import drive; drive.mount("/content/drive"); os.makedirs(OUT_DIR, exist_ok=True)
except Exception as e:
    OUT_DIR = "/content"; print("[info] Drive not mounted, saving to /content:", e)

m = pd.read_csv(MANIFEST)
m = m[m.split == "test"].copy()
pos = m[m.polarity == "positive"].copy()
pos["key"] = pos["study_id"] + "|" + pos["target"].astype(str)
uniq = pos.drop_duplicates("key")[["study_id","target","location","section_name","section_content"]].reset_index(drop=True)
print(f"test positives: {len(pos)} pairs | unique (study,target) to label: {len(uniq)}")
print("lesions:", pos.target.value_counts().to_dict())
"""))

cells.append(md(r"""## Cell 4 — helper: pull the report sentence that mentions the finding

The manifest's `section_content` is a numbered list of sentences. For display
(and to focus the model) we grab the sentence(s) that mention the finding, using
a small synonym map. If none match we fall back to the whole section.
"""))

cells.append(code(r"""import re

SYN = {
    "cardiomegaly": ["cardiomegaly","cardiac silhouette","heart size","cardiac enlarge","enlarged heart","cardiac contour"],
    "pneumonia":    ["pneumonia","infection","infectious"],
    "atelectasis":  ["atelectasis","atelectatic","collapse"],
    "opacity":      ["opacity","opacities","opacification"],
    "consolidation":["consolidation","consolidative","airspace"],
    "edema":        ["edema","oedema","vascular congestion","fluid overload"],
    "effusion":     ["effusion","pleural fluid"],
}

def split_sentences(text):
    if not isinstance(text, str): return []
    # sentences look like "(1) ... (2) ..."; also split on periods as fallback
    parts = re.split(r"\(\d+\)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]
    return parts

def relevant_sentences(target, text):
    sents = split_sentences(text)
    kws = SYN.get(str(target).lower(), [str(target).lower()])
    hits = [s for s in sents if any(k in s.lower() for k in kws)]
    return hits if hits else sents

# quick look at what the extractor pulls for the first few rows
for r in uniq.head(3).itertuples():
    hits = relevant_sentences(r.target, r.section_content)
    print(f"[{r.target}] ->", " | ".join(hits)[:220])
"""))

cells.append(md(r"""## Cell 5 — load MedGemma + define the 4-way classifier

The prompt gives MedGemma the finding and the relevant report sentence(s), the
4 definitions, and few-shot examples, and asks for a strict
`LABEL: <one>; WHY: <short reason>` response. We parse the label (defaulting to
`certain` if the report contains no hedge words) and keep the reason for display.
"""))

cells.append(code(r"""import torch, re
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_NAME = "google/medgemma-4b-it"
print("loading", MODEL_NAME, "...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tok = getattr(processor, "tokenizer", None) or processor
tok.padding_side = "left"
if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
    tok.pad_token = tok.eos_token
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()

LABELS = ["certain", "presence", "spatial", "diagnostic"]

DEFINITIONS = (
    "Definitions of the uncertainty TYPE for a specific finding:\n"
    "- certain: the report states the finding as a fact, no hedging.\n"
    "- presence: the report is unsure the finding EXISTS or is visible "
    "(possible, probable, questionable, cannot exclude, suggestive of, may be present, suspicious for).\n"
    "- spatial: the finding is accepted but its LOCATION, EXTENT, or BOUNDARY is unclear "
    "(ill-defined, poorly marginated, indistinct margins, hazy, extent/size uncertain).\n"
    "- diagnostic: the finding is seen but its IDENTITY is a differential between entities "
    "(X versus Y, could represent X or Y, may reflect atelectasis or consolidation).\n"
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
)

HEDGE = re.compile(r"possib|probab|question|cannot exclude|can't exclude|suggest|may |might|likely|"
                   r"versus| vs |could (represent|reflect|be)|ill-?defin|indistinct|poorly (defin|margin)|"
                   r"hazy|suspicious|concern for|worrisome|equivocal|uncertain", re.I)

def build_prompt(target, report_snip):
    return (
        "You classify the TYPE of uncertainty in a radiology report about ONE finding.\n"
        f"{DEFINITIONS}\n{FEWSHOT}\n"
        "Now classify this one. Answer EXACTLY as 'LABEL: <certain|presence|spatial|diagnostic>; WHY: <short reason>'.\n"
        f"Finding: {target} | Report: '{str(report_snip)[:700]}' ->"
    )

def classify(target, report_snip):
    prompt = build_prompt(target, report_snip)
    msgs = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    m = re.search(r"LABEL:\s*(certain|presence|spatial|diagnostic)", raw, re.I)
    label = m.group(1).lower() if m else None
    w = re.search(r"WHY:\s*(.+)", raw, re.I)
    why = w.group(1).strip()[:160] if w else raw[:160]
    # guardrail: if model says uncertain-type but there is literally no hedge word, call it certain
    if label in {"presence","spatial","diagnostic"} and not HEDGE.search(str(report_snip)):
        label, why = "certain", "(no hedge words found) " + why
    if label is None:
        label = "presence" if HEDGE.search(str(report_snip)) else "certain"
    return label, why, raw

# smoke test on 3 rows
for r in uniq.head(3).itertuples():
    snip = " ".join(relevant_sentences(r.target, r.section_content))
    lab, why, raw = classify(r.target, snip)
    print(f"[{r.target}] -> {lab} :: {why}")
"""))

cells.append(md(r"""## Cell 6 — run the classifier over ALL test positives (resumable)

Caches to CSV; re-running skips already-labelled (study, target) keys.
"""))

cells.append(code(r"""import os, pandas as pd
from tqdm.auto import tqdm

TYPES_CSV = os.path.join(OUT_DIR, "uncertainty_types_test.csv")
done = set()
if os.path.exists(TYPES_CSV):
    prev = pd.read_csv(TYPES_CSV)
    done = set(prev["key"].tolist())
    print(f"resuming; {len(done)} already labelled")

rows = []
for r in tqdm(uniq.itertuples(), total=len(uniq), desc="uncertainty-type"):
    key = f"{r.study_id}|{r.target}"
    if key in done: continue
    snip = " ".join(relevant_sentences(r.target, r.section_content))
    lab, why, raw = classify(r.target, snip)
    rec = {"key":key, "study_id":r.study_id, "target":r.target, "location":r.location,
           "section_name":r.section_name, "report_sentence":snip[:500],
           "unc_type":lab, "why":why, "raw":raw[:200]}
    rows.append(rec)
    pd.DataFrame([rec]).to_csv(TYPES_CSV, mode="a", header=not os.path.exists(TYPES_CSV), index=False)

types = pd.read_csv(TYPES_CSV)
print(f"\nDone. labelled {len(types)} unique (study,target).")
print(types["unc_type"].value_counts().to_string())
"""))

cells.append(md(r"""## Cell 7 — PRINT worked examples per category (the diagnostic view)

For each of the 4 labels, print `N_EXAMPLES` findings with the exact report
sentence and MedGemma's reason, so you can eyeball whether the labels make
sense. Increase `N_EXAMPLES` to see more.
"""))

cells.append(code(r"""import pandas as pd, textwrap
types = pd.read_csv(TYPES_CSV)

N_EXAMPLES = 8   # <-- bump this to inspect more per category

for lab in ["certain","presence","spatial","diagnostic"]:
    sub = types[types.unc_type == lab]
    print("\n" + "="*100)
    print(f"  {lab.upper()}   (n={len(sub)} of {len(types)})")
    print("="*100)
    for i, r in enumerate(sub.head(N_EXAMPLES).itertuples(), 1):
        print(f"\n[{i}] finding = {r.target}   (location={r.location}, section={r.section_name})")
        print("    report : " + "\n             ".join(textwrap.wrap(str(r.report_sentence), 90)))
        print(f"    -> {lab}  ::  {r.why}")
"""))

cells.append(md(r"""## Cell 8 — counts, per-lesion breakdown, and a plot"""))

cells.append(code(r"""import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
types = pd.read_csv(TYPES_CSV)
ORDER = ["certain","presence","spatial","diagnostic"]

print("overall counts:")
print(types["unc_type"].value_counts().reindex(ORDER, fill_value=0).to_string())

print("\nuncertainty TYPE x lesion (counts):")
ct = pd.crosstab(types["target"], types["unc_type"]).reindex(columns=ORDER, fill_value=0)
print(ct.to_string())

# stacked bar per lesion
fig, ax = plt.subplots(figsize=(11,5))
colors = {"certain":"tab:blue","presence":"tab:orange","spatial":"tab:green","diagnostic":"tab:red"}
bottom = np.zeros(len(ct))
for lab in ORDER:
    ax.bar(ct.index, ct[lab].values, bottom=bottom, label=lab, color=colors[lab])
    bottom += ct[lab].values
ax.set_ylabel("# findings"); ax.set_title("Uncertainty type by lesion (MIMIC-ILS test positives)")
ax.legend(); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
p = os.path.join(OUT_DIR, "uncertainty_types_by_lesion.png")
plt.savefig(p, dpi=120); plt.show(); print("saved ->", p)
print("\ntable saved ->", os.path.join(OUT_DIR, "uncertainty_types_test.csv"))
"""))

cells.append(md(r"""## Cell 9 — (optional) merge the fine type back onto every positive PAIR

The per-pair manifest can be joined on (study_id, target) so downstream analyses
(e.g. the IoU regression) can use `unc_type` instead of the binary label.
"""))

cells.append(code(r"""import pandas as pd, os
types = pd.read_csv(TYPES_CSV)[["study_id","target","unc_type","why","report_sentence"]]
pairs = m[m.polarity=="positive"].merge(types, on=["study_id","target"], how="left")
pairs["is_uncertain"] = (pairs["unc_type"] != "certain").astype(int)
out = os.path.join(OUT_DIR, "test_positives_with_unc_type.csv")
pairs.to_csv(out, index=False)
print("per-pair rows with fine type:", len(pairs))
print(pairs["unc_type"].value_counts(dropna=False).to_string())
print("saved ->", out)
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}
_default_out = Path(__file__).resolve().parent / "project" / "notebooks" / "uncertainty_type_diagnostic_colab.ipynb"
out_path = Path(os.environ.get("NOTEBOOK_OUT", _default_out))
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")
