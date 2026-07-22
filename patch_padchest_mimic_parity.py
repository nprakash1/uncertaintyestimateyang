"""Bring rosalia_padchest_gr_colab.ipynb to MIMIC-ILS stats parity.

Changes:
  * B9 (cell 32): capture confidence/model-activation stats (max_prob, mean_prob,
    mean_prob_in_gt over reader-union), fired, pred_px, gt_px, inter, union, dice.
    Uses a self-contained logits-returning segmenter. Cache-version aware:
    cached JSONs missing the v2 keys are recomputed; complete ones are reused.
  * B10 (cell 34): MIMIC-style block() -> overall / per-lesion / by-uncertainty /
    lesion x uncertainty pivot, PLUS measured reader-agreement stratification and
    fire-rate/abstention. Keeps the detailed per-target/per-finding CSVs.
  * B11 (cell 36): per-lesion gIoU bar, IoU-by-uncertainty hist, certain-vs-
    uncertain counts per lesion, fire-rate per lesion, agreement-vs-IoU scatter.
  * NEW cells: B11b confidence distributions + Mann-Whitney U + Cohen's d;
    B11c measured-spatial vs linguistic cross-check (+ Spearman vs ROSALIA IoU);
    B11d probability heat-map galleries (uncertain vs certain) with reader boxes.
"""
import json, pathlib, re

NB = pathlib.Path("project/notebooks/rosalia_padchest_gr_colab.ipynb")
nb = json.loads(NB.read_text())

# ---------------------------------------------------------------- B9 (cell 32)
cell32 = r'''import io, json
import numpy as np
from tqdm.auto import tqdm

# ---- logits-returning segmenter (self-contained; uses B6 globals) -----------
def _segment_logits(model, tokenizer, clip_processor, transform, pil_image, instruction):
    image_np = np.array(pil_image)
    if image_np.ndim == 2: image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[-1] == 4: image_np = image_np[..., :3]
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
        return None, text_out
    lm = pred_masks[0]
    if lm.ndim == 3: lm = lm[0]
    return lm.float().detach().cpu().numpy(), text_out

# expose under the B6c name too, so the heat-map gallery works even if B6c was skipped
segment_image_rosalia_logits = _segment_logits

def _confidence(logits, r1, r2):
    "Peak/mean sigmoid prob, and mean prob inside the reader-union GT region."
    if logits is None:
        return dict(max_prob=float("nan"), mean_prob=float("nan"), mean_prob_in_gt=float("nan"))
    p = 1.0/(1.0+np.exp(-logits.astype(np.float64)))
    H, W = p.shape
    gt = np.zeros((H, W), bool)
    for b in list(r1 or []) + list(r2 or []):
        x1,y1,x2,y2 = b
        gt[int(np.floor(y1*H)):int(np.ceil(y2*H)), int(np.floor(x1*W)):int(np.ceil(x2*W))] = True
    return dict(max_prob=float(p.max()), mean_prob=float(p.mean()),
                mean_prob_in_gt=(float(p[gt].mean()) if gt.any() else float("nan")))

def exists_in_bucket(path):
    try: return fs.exists(path)
    except Exception: return False
def upload_json(path,obj):
    with fs.open(path,"w") as f: f.write(json.dumps(obj))
def upload_npz(path,mask):
    buf=io.BytesIO(); np.savez_compressed(buf,mask=mask); buf.seek(0)
    with fs.open(path,"wb") as f: f.write(buf.read())

# v2 result schema: rows lacking these keys are recomputed (adds confidence/fired/dice)
REQUIRED_KEYS = {"max_prob","fired","inter","union","dice","pred_px","gt_px"}

all_records=[]; n_skipped=0; n_new=0; n_recomputed=0
for row in tqdm(samples.itertuples(), total=len(samples), desc="full"):
    out_json=f"{BUCKET}/{OUT_PREFIX}/{row.sample_id}.json"
    out_npz =f"{BUCKET}/{MASK_PREFIX}/{row.sample_id}.npz"
    if exists_in_bucket(out_json):
        with fs.open(out_json,"r") as f: rec=json.loads(f.read())
        if REQUIRED_KEYS.issubset(rec.keys()):
            all_records.append(rec); n_skipped+=1; continue    # already v2
        n_recomputed+=1                                         # upgrade old row
    try:
        img=load_image_gcs(row.image_id)
        logits,text_out=_segment_logits(model,tokenizer,clip_processor,transform,img,row.rosalia_instruction)
        mask=(logits>0).astype(np.uint8) if logits is not None else np.zeros((1,1),np.uint8)
        r1=json.loads(row.reader1_boxes) if isinstance(row.reader1_boxes,str) else row.reader1_boxes
        r2=json.loads(row.reader2_boxes) if isinstance(row.reader2_boxes,str) else row.reader2_boxes
        bundle,pm=compute_iou_bundle(mask,r1,r2)
        gtu=boxes_to_mask((r1 or [])+(r2 or []))
        inter=int(np.logical_and(pm,gtu).sum()); union=int(np.logical_or(pm,gtu).sum())
        psum=int(pm.sum()); gsum=int(gtu.sum())
        dice=(2*inter/(psum+gsum)) if (psum+gsum)>0 else float("nan")
        conf=_confidence(logits,r1,r2)
        rec={"sample_id":row.sample_id,"image_id":row.image_id,"finding_label":row.finding_label,
             "rosalia_target":row.rosalia_target,"rosalia_instruction":row.rosalia_instruction,
             "rosalia_text_output":text_out,"uncertainty_label_rule":row.uncertainty_label_rule,
             "uncertainty_label_medgemma":row.uncertainty_label_medgemma,"sentence_raw":row.sentence,
             "fired":int(psum>0),"pred_px":psum,"gt_px":gsum,"inter":inter,"union":union,"dice":dice,
             **conf,**bundle}
        upload_json(out_json,rec); upload_npz(out_npz,pm); all_records.append(rec); n_new+=1
    except Exception as e:
        print(f"[skip] {row.sample_id}: {type(e).__name__}: {e}"); continue

results_df=pd.DataFrame(all_records)
print(f"\nDone. skipped(v2 cached)={n_skipped} recomputed={n_recomputed} new={n_new} total={len(results_df)}")
results_df.to_csv("/content/rosalia_per_sample_ious.csv", index=False)
'''

# --------------------------------------------------------------- B10 (cell 34)
cell34 = r'''import json, numpy as np, pandas as pd

PRIMARY = "iou_with_pixel_or_union"   # IoU vs pixel-OR union of both readers' boxes
for c in [PRIMARY,"inter","union","dice","fired","reader_iou_union",
          "max_prob","mean_prob","mean_prob_in_gt","pred_frac_in_gt_union",
          "gt_union_frac_covered","max_containment_per_box"]:
    if c in results_df: results_df[c]=pd.to_numeric(results_df[c],errors="coerce")

def block(df):
    "MIMIC-parity summary: n, fire_rate, gIoU(mean IoU), cIoU(sum inter/sum union), mDice."
    d=df.dropna(subset=[PRIMARY])
    giou=d[PRIMARY].mean()
    ciou=(d["inter"].sum()/d["union"].sum()) if ("union" in d.columns and d["union"].sum()>0) else float("nan")
    dice=d["dice"].mean() if "dice" in d.columns else float("nan")
    fire=df["fired"].mean() if "fired" in df.columns else float("nan")
    return pd.Series({"n":len(df),"fire_rate":fire,"gIoU":giou,"cIoU":ciou,"mDice":dice})

print("="*66,"\nOVERALL (primary = IoU vs pixel-OR reader union):")
print(block(results_df).round(4).to_string())

print("\nPER LESION (disease-level):")
print(results_df.groupby("rosalia_target").apply(block).round(4).to_string())

print("\nBY LINGUISTIC UNCERTAINTY (rule):")
print(results_df.groupby("uncertainty_label_rule").apply(block).round(4).to_string())

print("\nLESION x UNCERTAINTY (gIoU):")
print(results_df.dropna(subset=[PRIMARY]).pivot_table(
      index="rosalia_target", columns="uncertainty_label_rule",
      values=PRIMARY, aggfunc="mean").round(3).to_string())

# ---- MEASURED spatial uncertainty: inter-reader box agreement (PadChest-only) ----
if "reader_iou_union" in results_df.columns and results_df["reader_iou_union"].notna().any():
    med=results_df["reader_iou_union"].median()
    results_df["reader_agreement_bin"]=np.where(results_df["reader_iou_union"]>=med,
                                                 "high_agreement","low_agreement")
    print("\n"+"="*66,f"\nBY MEASURED READER AGREEMENT (reader_iou_union median={med:.3f}):")
    print(results_df.groupby("reader_agreement_bin").apply(block).round(4).to_string())

# ---- abstention / fire-rate view (headline given the domain gap) ----
print("\n"+"="*66,"\nABSTENTION / FIRE RATE:")
fr=results_df["fired"].mean()
print(f"  overall fire rate = {fr:.3f}  (abstention = {1-fr:.3f})")
print("  fire rate per lesion:");      print(results_df.groupby("rosalia_target")["fired"].mean().round(3).to_string())
print("  fire rate per uncertainty:"); print(results_df.groupby("uncertainty_label_rule")["fired"].mean().round(3).to_string())

# ---- keep the detailed per-target / per-finding CSVs ----
IOU_COLS=[c for c in ["iou_with_pixel_or_union","iou_with_outer_bbox_union","iou_with_intersection",
    "iou_with_reader1_union","iou_with_reader2_union","reader_iou_union","mean_per_box_iou",
    "max_per_box_iou","pred_frac_in_gt_union","gt_union_frac_covered","max_containment_per_box"]
    if c in results_df.columns]
results_df.groupby(["rosalia_target","uncertainty_label_rule"])[IOU_COLS].agg(
    ["count","mean","median"]).to_csv("/content/rosalia_per_target_ious.csv")
results_df.groupby(["uncertainty_label_rule","finding_label"])[IOU_COLS].agg(
    ["count","mean","median"]).to_csv("/content/rosalia_per_finding_ious.csv")

# ---- compact JSON summary ----
summary={"overall":block(results_df).round(4).to_dict(),
    "per_lesion":{k:block(v).round(4).to_dict() for k,v in results_df.groupby("rosalia_target")},
    "by_uncertainty":{k:block(v).round(4).to_dict() for k,v in results_df.groupby("uncertainty_label_rule")},
    "overall_fire_rate":float(results_df["fired"].mean())}
if "reader_agreement_bin" in results_df.columns:
    summary["by_reader_agreement"]={k:block(v).round(4).to_dict()
                                    for k,v in results_df.groupby("reader_agreement_bin")}
json.dump(summary, open("/content/rosalia_aggregate_ious.json","w"), indent=2)
print("\nsaved -> /content/rosalia_aggregate_ious.json")
summary
'''

# --------------------------------------------------------------- B11 (cell 36)
cell36 = r'''import matplotlib.pyplot as plt, numpy as np, pandas as pd
PRIMARY="iou_with_pixel_or_union"
P=results_df.dropna(subset=[PRIMARY])

# 1) per-lesion gIoU bar
fig,ax=plt.subplots(figsize=(9,4))
P.groupby("rosalia_target")[PRIMARY].mean().sort_values(ascending=False).plot(kind="bar",ax=ax,color="teal")
ax.set_ylabel("gIoU (mean IoU vs reader union)"); ax.set_title("ROSALIA gIoU by lesion (PadChest-GR)")
plt.tight_layout(); plt.savefig("/content/padchest_giou_by_lesion.png",dpi=120); plt.show()

# 2) IoU distribution by uncertainty
fig,ax=plt.subplots(figsize=(7,4))
for lab,c in [("certain","tab:blue"),("uncertain","tab:orange")]:
    s=P[P.uncertainty_label_rule==lab][PRIMARY]
    if len(s): ax.hist(s,bins=30,alpha=0.55,label=f"{lab} (n={len(s)})",color=c)
ax.set_xlabel("IoU vs reader union"); ax.set_ylabel("count"); ax.legend()
ax.set_title("ROSALIA IoU by linguistic uncertainty"); plt.tight_layout()
plt.savefig("/content/padchest_iou_by_uncertainty.png",dpi=120); plt.show()

# 3) certain vs uncertain COUNTS per lesion
ct=results_df.groupby(["rosalia_target","uncertainty_label_rule"]).size().unstack(fill_value=0)
for lab in ["certain","uncertain"]:
    if lab not in ct.columns: ct[lab]=0
labels=list(ct.index); x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(11,4.5))
b1=ax.bar(x-w/2,ct["certain"],w,label="certain",color="tab:blue")
b2=ax.bar(x+w/2,ct["uncertain"],w,label="uncertain",color="tab:orange")
ax.bar_label(b1,fontsize=8); ax.bar_label(b2,fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha="right"); ax.legend()
ax.set_ylabel("# findings"); ax.set_title("Linguistic uncertainty per lesion (counts)")
plt.tight_layout(); plt.savefig("/content/padchest_uncertainty_counts_by_lesion.png",dpi=120); plt.show()
print(ct[["certain","uncertain"]].to_string())

# 4) fire rate per lesion (abstention view)
fig,ax=plt.subplots(figsize=(9,4))
results_df.groupby("rosalia_target")["fired"].mean().sort_values().plot(kind="bar",ax=ax,color="indianred")
ax.set_ylabel("fire rate (1 - abstention)"); ax.set_title("ROSALIA fire rate by lesion")
plt.tight_layout(); plt.savefig("/content/padchest_fire_rate_by_lesion.png",dpi=120); plt.show()

# 5) measured reader agreement vs ROSALIA IoU
fig,ax=plt.subplots(figsize=(6,5))
for lab,c in [("certain","tab:blue"),("uncertain","tab:orange")]:
    sub=results_df[results_df.uncertainty_label_rule==lab]
    ax.scatter(sub["reader_iou_union"],sub[PRIMARY],s=10,alpha=0.5,label=lab,color=c)
ax.set_xlabel("reader-reader IoU (measured agreement)"); ax.set_ylabel("ROSALIA IoU")
ax.set_title("ROSALIA quality vs annotator agreement"); ax.legend()
plt.tight_layout(); plt.savefig("/content/padchest_agreement_vs_iou.png",dpi=120); plt.show()
'''

# ---------------------------------------------------- NEW cells (B11b/c/d)
md_b11b = "## Cell B11b — model-activation confidence: certain vs uncertain (Mann-Whitney U + Cohen's d)"
code_b11b = r'''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

if "max_prob" not in results_df.columns or results_df["max_prob"].notna().sum()==0:
    print("[skip] no confidence stats in results_df -> re-run B9 to populate max_prob/mean_prob_in_gt.")
else:
    P=results_df.copy()
    def cohens_d(a,b):
        a,b=np.asarray(a),np.asarray(b); na,nb=len(a),len(b)
        if na<2 or nb<2: return float("nan")
        sp=np.sqrt(((na-1)*a.std(ddof=1)**2+(nb-1)*b.std(ddof=1)**2)/(na+nb-2))
        return (a.mean()-b.mean())/sp if sp>0 else float("nan")
    METRICS=[("max_prob","peak sigmoid probability"),
             ("mean_prob_in_gt","mean prob inside reader union")]
    fig,axes=plt.subplots(1,len(METRICS),figsize=(7*len(METRICS),4.5)); axes=np.atleast_1d(axes)
    for ax,(m,nice) in zip(axes,METRICS):
        cer=P[P.uncertainty_label_rule=="certain"][m].dropna().values
        unc=P[P.uncertainty_label_rule=="uncertain"][m].dropna().values
        bins=np.linspace(0,1,31)
        ax.hist(cer,bins=bins,alpha=0.55,density=True,color="tab:blue",
                label=f"certain (n={len(cer)}, med={np.median(cer):.2f})" if len(cer) else "certain (n=0)")
        ax.hist(unc,bins=bins,alpha=0.55,density=True,color="tab:orange",
                label=f"uncertain (n={len(unc)}, med={np.median(unc):.2f})" if len(unc) else "uncertain (n=0)")
        if len(cer): ax.axvline(np.median(cer),color="tab:blue",ls="--")
        if len(unc): ax.axvline(np.median(unc),color="tab:orange",ls="--")
        if len(cer)>=3 and len(unc)>=3:
            U,p=mannwhitneyu(unc,cer,alternative="less"); d=cohens_d(unc,cer)
            ax.set_title(f"{nice}\nMann-Whitney (uncertain<certain): p={p:.3g}, d={d:.2f}")
            print(f"[{m}] certain med={np.median(cer):.3f} mean={cer.mean():.3f} | "
                  f"uncertain med={np.median(unc):.3f} mean={unc.mean():.3f} | p={p:.3g} d={d:.3f}")
        else:
            ax.set_title(f"{nice}\n(insufficient n for test)")
        ax.set_xlabel(m); ax.set_ylabel("density"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("/content/padchest_prob_dist_by_uncertainty.png",dpi=120); plt.show()

    # peak-confidence boxplot
    fig,ax=plt.subplots(figsize=(5,4))
    ax.boxplot([P[P.uncertainty_label_rule=="certain"]["max_prob"].dropna(),
                P[P.uncertainty_label_rule=="uncertain"]["max_prob"].dropna()],
               labels=["certain","uncertain"], showmeans=True)
    ax.set_ylabel("max_prob (peak sigmoid)"); ax.set_title("ROSALIA peak confidence by uncertainty")
    plt.tight_layout(); plt.savefig("/content/padchest_maxprob_box.png",dpi=120); plt.show()
'''

md_b11c = "## Cell B11c — cross-check: does linguistic uncertainty track MEASURED reader disagreement?"
code_b11c = r'''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

R=results_df.dropna(subset=["reader_iou_union"]).copy()
cer=R[R.uncertainty_label_rule=="certain"]["reader_iou_union"].values
unc=R[R.uncertainty_label_rule=="uncertain"]["reader_iou_union"].values

fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].boxplot([cer,unc], labels=[f"certain\n(n={len(cer)})",f"uncertain\n(n={len(unc)})"], showmeans=True)
ax[0].set_ylabel("reader-reader IoU (measured spatial agreement)")
if len(cer)>=3 and len(unc)>=3:
    U,p=mannwhitneyu(unc,cer,alternative="less")
    ax[0].set_title(f"Linguistically-uncertain -> LOWER reader agreement?\nMann-Whitney p={p:.3g}")
    print(f"reader_iou_union: certain med={np.median(cer):.3f} | uncertain med={np.median(unc):.3f} | "
          f"p(uncertain<certain)={p:.3g}")
else:
    ax[0].set_title("reader agreement by linguistic uncertainty (insufficient n)")

rho,pp=spearmanr(R["reader_iou_union"],R["iou_with_pixel_or_union"],nan_policy="omit")
ax[1].scatter(R["reader_iou_union"],R["iou_with_pixel_or_union"],s=8,alpha=0.4,color="purple")
ax[1].set_xlabel("reader-reader IoU (agreement)"); ax[1].set_ylabel("ROSALIA IoU")
ax[1].set_title(f"ROSALIA IoU vs measured agreement\nSpearman rho={rho:.3f}, p={pp:.3g}")
print(f"Spearman(reader_iou_union, ROSALIA IoU) rho={rho:.3f} p={pp:.3g}")
plt.tight_layout(); plt.savefig("/content/padchest_reader_agreement_crosscheck.png",dpi=120); plt.show()
'''

md_b11d = "## Cell B11d — probability heat-map galleries (uncertain vs certain) with reader boxes"
code_b11d = r'''import numpy as np, json, matplotlib.pyplot as plt
from matplotlib import patches
from skimage.transform import resize as _skresize

def _draw_boxes(ax,boxes,color,H,W):
    for x1,y1,x2,y2 in (boxes or []):
        ax.add_patch(patches.Rectangle((x1*W,y1*H),(x2-x1)*W,(y2-y1)*H,
                                        fill=False,edgecolor=color,lw=1.5))

def gallery(df_sel,title,fname):
    n=len(df_sel)
    if n==0: print("no examples for",title); return
    cols=5; rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(4*cols,4*rows)); axes=np.atleast_1d(axes).ravel()
    for k,row in enumerate(df_sel.itertuples()):
        ax=axes[k]; img=load_image_gcs(row.image_id); a=np.array(img); H,W=a.shape[:2]
        logits,txt=_segment_logits(model,tokenizer,clip_processor,transform,img,row.rosalia_instruction)
        ax.imshow(a,cmap="gray")
        if logits is not None:
            prob=1.0/(1.0+np.exp(-logits))
            pr=_skresize(prob,(H,W),order=1,mode="edge",anti_aliasing=True,preserve_range=True)
            ax.imshow(pr,cmap="jet",alpha=0.45,vmin=0,vmax=1); mp=float(prob.max())
        else: mp=float("nan")
        r1=getattr(row,"reader1_boxes",None)
        if isinstance(r1,str): r1=json.loads(r1)
        _draw_boxes(ax,r1,"lime",H,W)
        iou=getattr(row,"iou_with_pixel_or_union",float("nan"))
        ax.set_title(f"{row.rosalia_target} | IoU={iou:.2f}\nmax_p={mp:.2f}",fontsize=9); ax.axis("off")
    for k in range(n,len(axes)): axes[k].axis("off")
    fig.suptitle(title,fontsize=13); plt.savefig(fname,dpi=110,bbox_inches="tight"); plt.show()

# ensure reader boxes are available for overlay
_g=results_df.copy()
if "reader1_boxes" not in _g.columns:
    _g=_g.merge(samples[["sample_id","reader1_boxes"]],on="sample_id",how="left")
_fired=_g[_g["fired"]==1] if "fired" in _g.columns else _g
gallery(_fired[_fired.uncertainty_label_rule=="uncertain"].head(10),
        "UNCERTAIN findings — ROSALIA probability heat-map","/content/padchest_heatmaps_uncertain.png")
gallery(_fired[_fired.uncertainty_label_rule=="certain"].head(10),
        "CERTAIN findings — ROSALIA probability heat-map","/content/padchest_heatmaps_certain.png")
'''

def mk_md(src):   return {"cell_type":"markdown","metadata":{},"source":src}
def mk_code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src}

# apply cell replacements
nb["cells"][32]["source"]=cell32
nb["cells"][34]["source"]=cell34
nb["cells"][36]["source"]=cell36

# insert new cells right after B11 code (index 36) -> before B12 (index 37)
new_cells=[mk_md(md_b11b),mk_code(code_b11b),
           mk_md(md_b11c),mk_code(code_b11c),
           mk_md(md_b11d),mk_code(code_b11d)]
nb["cells"][37:37]=new_cells

NB.write_text(json.dumps(nb, indent=1))

# ---- validation: every code cell compiles (strip !/% magics) ----
def _clean(src):
    return "\n".join("" if re.match(r"\s*[!%]",l) else l for l in src.split("\n"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]=="code":
        compile(_clean("".join(c["source"])), f"<cell {i}>", "exec")
    print(f"OK: {len(nb['cells'])} cells; all code cells compile.")

