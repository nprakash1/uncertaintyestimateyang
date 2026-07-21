"""Patch project/notebooks/rosalia_padchest_gr_colab.ipynb in place.

Fixes discussed for the low-IoU PadChest-GR run:
  (a) parse_location now emits ONLY ROSALIA's training location vocabulary
      (paper Table 9); anything unmappable -> None -> global instruction,
      instead of out-of-vocab strings ("lower lung", "lobe", "bilateral ...")
      that ROSALIA never saw and that produced empty masks.
  (b) INSTRUCTION_MODE gains an "auto" mode (default): bounded when a valid
      location parses, else global. "global"/"bounded" still available.
  (c) Cell B7 gains containment/recall metrics (mask-vs-box is a low-IoU ceiling
      by construction) + a normalized-box-coord assertion, and B10 aggregates
      the new columns.
Idempotent-ish: uses marker-based string surgery and asserts markers exist.
"""
import json, pathlib

NB = pathlib.Path("project/notebooks/rosalia_padchest_gr_colab.ipynb")
nb = json.loads(NB.read_text())


def cell_src(i):
    return "".join(nb["cells"][i]["source"])


def set_src(i, s):
    # store as a single string; nbformat accepts str or list
    nb["cells"][i]["source"] = s


# ---------------------------------------------------------------------------
# (b) Config cell (B3, index 16): switch default to "auto".
# ---------------------------------------------------------------------------
c16 = cell_src(16)
assert 'INSTRUCTION_MODE       = "bounded"' in c16, "config marker changed"
c16 = c16.replace(
    'INSTRUCTION_MODE       = "bounded"',
    'INSTRUCTION_MODE       = "auto"   # "auto" (bounded if a valid location\n'
    '                                  #  parses, else global) | "bounded" | "global"')
set_src(16, c16)

# ---------------------------------------------------------------------------
# (a) B4 cell (index 18): replace the location-parse + instruction block.
# ---------------------------------------------------------------------------
c18 = cell_src(18)
MARK = "# --- parse an anatomical location from the radiology sentence"
assert MARK in c18, "B4 parse-location marker not found"
head = c18[: c18.index(MARK)]

new_block = r'''# --- parse an anatomical location from the radiology sentence -----------------
# ROSALIA was trained on a FIXED location vocabulary (paper Table 9):
#   "right lung", "left lung",
#   "{right,left} {apical,upper,mid} zone lung", "{right,left} lung base"
# (and combinations joined by " and "). We map a phrase parsed from the report
# SENTENCE onto THESE EXACT strings. Anything we cannot map returns None, which
# yields a location-free GLOBAL instruction instead of an out-of-vocab string
# like "lower lung"/"lobe"/"bilateral ..." that the model never saw (those made
# ROSALIA answer "There is no <finding>." with an EMPTY mask). This is report
# text, NOT the GT boxes, so no annotation leakage.
import re as _re

_R_ZONES = {
    ("right", "apical"): "right apical zone lung",
    ("right", "upper"):  "right upper zone lung",
    ("right", "mid"):    "right mid zone lung",
    ("right", "base"):   "right lung base",
    ("left",  "apical"): "left apical zone lung",
    ("left",  "upper"):  "left upper zone lung",
    ("left",  "mid"):    "left mid zone lung",
    ("left",  "base"):   "left lung base",
}

def _zone_key(txt):
    # "lower"/"lobe-lower"/"costophrenic"/"base" -> ROSALIA's lowest zone = base
    if _re.search(r"base|basal|costophrenic|lower", txt): return "base"
    if _re.search(r"apex|apical|apic", txt):              return "apical"
    if _re.search(r"upper", txt):                          return "upper"
    if _re.search(r"\bmid", txt):                          return "mid"
    return None

def parse_location(sentence):
    """Return a ROSALIA-valid location string, or None (=> global instruction)."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    s = sentence.lower()
    # bilateral lung bases / bibasilar
    if _re.search(r"bibasal|bibasilar|bilateral\s+bas|both\s+bas", s):
        return "right lung base and left lung base"
    # explicit side (+ optional zone)
    for side in ("right", "left"):
        if _re.search(rf"\b{side}\b|\b{side}\s+hemithorax\b", s):
            zk = _zone_key(s)
            return _R_ZONES[(side, zk)] if zk else f"{side} lung"
    # retrocardiac ~ left lower field
    if "retrocardiac" in s:
        return "left lung base"
    # side-less zone -> both sides where ROSALIA supports a combination
    zk = _zone_key(s)
    if zk == "base":
        return "right lung base and left lung base"
    # bilateral / diffuse -> both broad lungs
    if _re.search(r"bilateral|diffuse|both\s+lung", s):
        return "right lung and left lung"
    return None

samples["rosalia_location"] = samples["sentence"].apply(parse_location)

def _build_instruction(target, mode=INSTRUCTION_MODE, location=None):
    if mode == "global":
        return INSTRUCTION_TEMPLATE_G.format(target=target)
    if location:  # "auto" and "bounded" both use a valid parsed location
        return INSTRUCTION_TEMPLATE_B.format(target=target, location=location)
    # no valid location parsed -> fall back to a global (location-free) prompt
    # rather than an out-of-vocab default like "lung".
    return INSTRUCTION_TEMPLATE_G.format(target=target)

samples["rosalia_instruction"] = samples.apply(
    lambda r: _build_instruction(r["rosalia_target"], location=r["rosalia_location"]),
    axis=1)

_cov = samples["rosalia_location"].notna().mean()
print(f"\nParsed a ROSALIA-valid location for {_cov*100:.1f}% of rows "
      f"(the rest use a global, location-free instruction).")
print("\nExample instructions:")
print(samples["rosalia_instruction"].value_counts().head(12).to_string())

samples.head(5)[["sample_id","image_id","finding_label","rosalia_target",
                 "rosalia_location","rosalia_instruction","uncertainty_label_rule"]]
'''
set_src(18, head + new_block)

# ---------------------------------------------------------------------------
# (c) B7 cell (index 28): containment/recall metrics + normalized-box assertion.
# ---------------------------------------------------------------------------
c28 = cell_src(28)
assert "def mask_iou(m1, m2):" in c28
# add containment/recall helpers right after mask_iou
anchor = ('def mask_iou(m1, m2):\n'
          '    inter=int(np.logical_and(m1,m2).sum()); union=int(np.logical_or(m1,m2).sum())\n'
          '    return inter/union if union>0 else float("nan")\n')
assert anchor in c28, "mask_iou anchor changed"
helpers = anchor + (
    '\n'
    'def mask_containment(pred_mask, gt_mask):\n'
    '    "Fraction of the PREDICTED mask inside the GT region (precision-like). "\n'
    '    "Robust to the mask-vs-box ceiling: a correct tight mask fully inside a "\n'
    '    "GT box scores 1.0 here even though IoU is低."\n'
    '    p=int(np.asarray(pred_mask).sum())\n'
    '    return int(np.logical_and(pred_mask,gt_mask).sum())/p if p>0 else float("nan")\n'
    '\n'
    'def mask_recall(pred_mask, gt_mask):\n'
    '    "Fraction of the GT region covered by the prediction (recall-like)."\n'
    '    g=int(np.asarray(gt_mask).sum())\n'
    '    return int(np.logical_and(pred_mask,gt_mask).sum())/g if g>0 else float("nan")\n')
c28 = c28.replace(anchor, helpers)

# fix the accidental non-ASCII char if it slipped in
c28 = c28.replace("IoU is低", "IoU is low")

# assertion + new bundle entries inside compute_iou_bundle
assert "all_boxes=(r1_boxes or [])+(r2_boxes or [])" in c28
c28 = c28.replace(
    "    all_boxes=(r1_boxes or [])+(r2_boxes or [])\n",
    "    all_boxes=(r1_boxes or [])+(r2_boxes or [])\n"
    "    for _b in all_boxes:\n"
    "        assert all(-0.01<=float(v)<=1.01 for v in _b), (\n"
    "            f'GT box coords must be normalized to [0,1], got {_b}. If these are '\n"
    "            'pixel coords, divide by image width/height before compute_iou_bundle().')\n"
    "    _gt_union=boxes_to_mask(all_boxes,grid_size)\n")
assert '"iou_with_pixel_or_union":mask_iou(pm,boxes_to_mask(all_boxes,grid_size)),' in c28
c28 = c28.replace(
    '        "num_reader2_boxes":len(r2_boxes or []),\n    }, pm',
    '        "num_reader2_boxes":len(r2_boxes or []),\n'
    '        "pred_frac_in_gt_union":mask_containment(pm,_gt_union),\n'
    '        "gt_union_frac_covered":mask_recall(pm,_gt_union),\n'
    '        "max_containment_per_box":float(np.max([mask_containment(pm,boxes_to_mask([b],grid_size)) for b in all_boxes])) if all_boxes else float("nan"),\n'
    '    }, pm')
set_src(28, c28)

# ---------------------------------------------------------------------------
# (c cont.) B10 cell (index 34): include new columns in numeric coercion.
# ---------------------------------------------------------------------------
c34 = cell_src(34)
if '"mean_pairwise_iou"]' in c34 and "pred_frac_in_gt_union" not in c34:
    c34 = c34.replace(
        '"mean_pairwise_iou"]',
        '"mean_pairwise_iou","pred_frac_in_gt_union","gt_union_frac_covered",\n'
        '          "max_containment_per_box"]')
    set_src(34, c34)

NB.write_text(json.dumps(nb, indent=1))
print("patched:", NB)

# quick compile check of the three edited code cells (strip !/% lines)
import re as _re2
for i in (16, 18, 28, 34):
    src = cell_src(i)
    lines = []
    for ln in src.split("\n"):
        lines.append("" if _re2.match(r"\s*[!%]", ln) else ln)
    compile("\n".join(lines), f"<cell{i}>", "exec")
print("compile OK for cells 16,18,28,34")
