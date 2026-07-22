"""Upgrade Cell B6b (index 24) of rosalia_padchest_gr_colab.ipynb.

Reasons (from the live diagnostic output):
  * The old cell built 'Segment the <t> in the None.' when a location did not
    parse (it passed rosalia_location literally). Fix: use the SAME fallback as
    Cell B4 (bounded if a valid location, else global).
  * Nothing fired even for a clean atelectasis@right-lung-base row, and windowed
    vs raw gave identical answers -> need the CARDIOMEGALY canary (ROSALIA's best
    class, ~89% IoU, emits the heart mask). If cardiomegaly ALSO returns empty,
    the failure is upstream (image prep / pipeline). If cardiomegaly fires but
    atelectasis etc. do not, it is the expected MIMIC->PadChest recall gap.
  * Print per-loader image intensity stats to rule out blank/inverted images.
  * Add a histogram-equalized loader (ROSALIA trained ~50% on hist-eq images).
"""
import json, pathlib, re

NB = pathlib.Path("project/notebooks/rosalia_padchest_gr_colab.ipynb")
nb = json.loads(NB.read_text())

new = r'''import io, numpy as np
from PIL import Image
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

# ---- loader variants --------------------------------------------------------
def _load_raw_rgb(image_id):
    "No windowing: mimic ROSALIA's cv2.imread 8-bit path (max-scale if 16-bit)."
    with fs.open(f"{BUCKET}/{IMG_PREFIX}/{image_id}", "rb") as f:
        arr = np.array(Image.open(io.BytesIO(f.read())))
    if arr.ndim == 3: arr = arr[..., 0]
    if arr.dtype != np.uint8:
        m = float(arr.max()) or 1.0
        arr = (arr.astype(np.float32) / m * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def _load_histeq_rgb(image_id):
    "Histogram-equalized 8-bit -> closest to MIMIC-CXR-JPG, which ROSALIA saw."
    img = np.array(_load_raw_rgb(image_id))[..., 0]
    if _HAVE_CV2:
        img = cv2.equalizeHist(img)
    else:
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum(); cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        img = np.ma.filled(cdf_m, 0).astype(np.uint8)[img]
    return Image.fromarray(img).convert("RGB")

_LOADERS = [("windowed", load_image_gcs),
            ("raw     ", _load_raw_rgb),
            ("histeq  ", _load_histeq_rgb)]

def _instr_for(row):
    "Same fallback as Cell B4: bounded if a valid location, else global."
    loc = row.rosalia_location
    if isinstance(loc, str) and loc.strip():
        return f"Segment the {row.rosalia_target} in the {loc}."
    return f"Segment the {row.rosalia_target}."

# ---- pick diagnostic rows: CARDIOMEGALY canary first, then located findings --
_card = samples[samples["rosalia_target"] == "cardiomegaly"].head(2)
_loc  = samples[samples["rosalia_location"].notna()].head(3)
import pandas as _pd
_diag = _pd.concat([_card, _loc]).drop_duplicates("sample_id").head(5)
if len(_diag) == 0:
    _diag = samples.head(4)

print("Loaders: windowed=1-99% percentile | raw=max-scaled 8-bit | histeq=CLAHE-like")
print("CANARY: cardiomegaly is ROSALIA's strongest class (~89% IoU, heart mask).")
print("  If EVEN cardiomegaly is empty -> upstream image-prep/pipeline problem.")
print("  If cardiomegaly fires but others don't -> expected MIMIC->PadChest recall gap.\n")

for row in _diag.itertuples():
    print("=" * 92)
    print(f"{row.sample_id} | finding={row.finding_label} -> target={row.rosalia_target} "
          f"| parsed_loc={row.rosalia_location}")
    print(f"  sentence: {str(row.sentence)[:120]}")
    auto_instr = _instr_for(row)
    for lname, loader in _LOADERS:
        img = loader(row.image_id)
        a = np.array(img)
        stats = f"shape={a.shape} min={a.min()} max={a.max()} mean={a.mean():.1f}"
        for itype, instr in [("auto  ", auto_instr),
                             ("global", f"Segment the {row.rosalia_target}.")]:
            m, t = segment_image_rosalia(model, tokenizer, clip_processor, transform, img, instr)
            print(f"  [{lname}|{itype}] px={int(m.sum()):>7d}  {stats}")
            print(f"       instr={instr!r} -> {t!r}")
print("=" * 92)
print("\nRead-out:")
print(" * cardiomegaly empty on ALL loaders  -> image prep / model load is the bug.")
print(" * cardiomegaly fires (px>0)          -> pipeline OK; empties elsewhere are the")
print("   real MIMIC->PadChest recall gap (ROSALIA paper atel. recall ~13%).")
print(" * one loader fires and others don't  -> switch Cell B5 to that loader.")
print(" * image mean≈0 or ≈255               -> blank/over-clipped image; fix windowing.")
'''

nb["cells"][24]["source"] = new

# sanity: cell must still reference cardiomegaly canary + no 'in the None'
NB.write_text(json.dumps(nb, indent=1))
src = "".join(nb["cells"][24]["source"])
assert "cardiomegaly" in src and "in the None" not in src
compile("\n".join("" if re.match(r"\s*[!%]", l) else l for l in src.split("\n")),
        "<b6b>", "exec")
print("B6b upgraded + compiles OK")
