import json, os, re, sys, hashlib, subprocess
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
# Guard: don't reuse an Option-A cache when the user asked for MedGemma. If the
# cached file has ZERO medgemma-sourced rows but USE_MEDGEMMA_CORRUPTION=True,
# rebuild it (this is what silently happened before -> every source == option_A).
_FLUSH = globals().get("FLUSH_PROMPT_CACHE", False)
_use_cache = os.path.exists(CORRUPT_CSV) and not _FLUSH
if os.path.exists(CORRUPT_CSV) and _FLUSH:
    print(f"[prompts] FLUSH_PROMPT_CACHE=True -> ignoring cached {CORRUPT_CSV}, rebuilding ...")
if _use_cache:
    _cached = pd.read_csv(CORRUPT_CSV)
    _srcs = set(map(str, _cached.get("source", pd.Series([], dtype=str)).unique()))
    if USE_MEDGEMMA_CORRUPTION and "medgemma" not in _srcs:
        print(f"[prompts] cached {CORRUPT_CSV} has NO medgemma rows "
              f"(sources={_srcs}); rebuilding WITH MedGemma ...")
        _use_cache = False

if _use_cache:
    prompts_df = _cached
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
        # Load MedGemma on WHATEVER transformers this fresh runtime already has
        # (Colab ships a recent one that supports MedGemma). We do NOT gate on a
        # version number -- that caused an infinite "install -> restart" loop.
        # Only if the load actually fails do we do a ONE-TIME pinned reinstall,
        # guarded by a sentinel file so it can never loop.
        import torch
        _SENTINEL = "/content/.medgemma_tf_reinstalled"
        MODEL_NAME = "google/medgemma-4b-it"
        try:
            import transformers as _tf
            print(f"[medgemma] using transformers {_tf.__version__}")
            from transformers import AutoProcessor, AutoModelForImageTextToText
            proc = AutoProcessor.from_pretrained(MODEL_NAME)
            tok = getattr(proc, "tokenizer", None) or proc
            tok.padding_side = "left"
            if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            mg_model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto").eval()
        except Exception as _e:
            if os.path.exists(_SENTINEL):
                # already reinstalled once and it STILL fails -> surface the real error
                raise RuntimeError(
                    f"MedGemma still won't load after a pinned transformers reinstall: "
                    f"{type(_e).__name__}: {_e}") from _e
            open(_SENTINEL, "w").close()
            print(f"[medgemma] load failed ({type(_e).__name__}: {_e})")
            print("[medgemma] installing a known-good transformers==4.53.2 (one time) ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                            "transformers==4.53.2", "accelerate>=0.30"], check=True)
            print("\n" + "="*72)
            print(">>> Installed a MedGemma-compatible transformers. NEXT STEPS:")
            print("    1) Runtime > Restart session   (menu, or Ctrl/Cmd+M .)")
            print("    2) Re-run Cell 2 (login) and Cell 3 (config, keep")
            print("       USE_MEDGEMMA_CORRUPTION=True), THEN re-run THIS cell.")
            print("    (The 'To exit: use quit' warning below is harmless.)")
            print("="*72)
            raise SystemExit

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
