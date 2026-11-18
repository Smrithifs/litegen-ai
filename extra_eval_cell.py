# ============================================================================
# LiteGen AI - EXTRA EVALUATION CELL (run in the same Colab, AFTER the frontend cell;
# the demo keeps running). ~6-8 min on a T4.
# Adds: (1) image DIVERSITY metric across 4 seeds (distribution-level metric instead of FID),
#       (2) re-generated images so the Gallery tab works again,
#       (3) CV before/after + CLIP guard on LOW, QUALITY best-of-4 from the same seeds,
#       (4) a visual comparison figure for the report, (5) outputs.zip to download.
# ============================================================================
import itertools, zipfile, glob
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEEDS = [12345, 101, 202, 303]
CFG = {  # order chosen so the already-loaded SD-Turbo is used first
    "SD1":          ("sdturbo", dict(steps=1,  guidance=0.0, size=512), "SD-Turbo 1 step (FAST)"),
    "E6_lcm_4_384": ("lcm",     dict(steps=4,  guidance=1.5, size=384), "LCM-LoRA 4 steps 384px (LOW)"),
    "E2_fp16_15":   ("fp16",    dict(steps=15, guidance=7.5, size=512), "FP16 15 steps (BALANCED)"),
}
os.makedirs(SDTURBO_IMG_DIR, exist_ok=True)

@torch.inference_mode()
def clip_both(prompt, img):
    dev = next(cmodel.parameters()).device
    x = cproc(text=[prompt], images=[img.convert("RGB")], return_tensors="pt",
              padding=True, truncation=True).to(dev)
    out = cmodel(**x)
    t = F.normalize(out.text_embeds, dim=-1); e = F.normalize(out.image_embeds, dim=-1)
    return float((t * e).sum()), e[0].cpu()

rows, EMB = [], {}
for rid, (kind, kw, label) in CFG.items():
    pipe, load_s = get_pipe(kind)
    gen(pipe, PROMPTS[0], steps=min(2, kw["steps"]), guidance=kw["guidance"], size=kw["size"])  # warm-up
    d = OUTPUT_DIR / rid; d.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PROMPTS, 1):
        for s in SEEDS:
            img, lat, mem = gen(pipe, p, seed=s, **kw)
            c, e = clip_both(p, img); EMB[(rid, i, s)] = e
            path = d / f"p{i:02d}_s{s}.png"; img.save(path)
            if s == 12345:
                img.save(d / f"p{i:02d}.png")
                if rid == "SD1": img.save(os.path.join(SDTURBO_IMG_DIR, f"P{i:02d}_1steps.png"))
            rows.append(dict(run_id=rid, config=label, prompt_id=i, seed=s, latency_s=lat,
                             peak_mem_mb=mem, clip=c, path=str(path)))
    print(f"{label}: done")

R = pd.DataFrame(rows)

# ---- diversity: mean pairwise CLIP-image cosine distance between the 4 seeds of each prompt ----
def diversity(rid, pid):
    es = [EMB[(rid, pid, s)] for s in SEEDS]
    return float(np.mean([1 - float(a @ b) for a, b in itertools.combinations(es, 2)]))

div = pd.DataFrame([dict(run_id=r, prompt_id=i, diversity=diversity(r, i)) for r in CFG for i in range(1, 13)])

# ---- QUALITY = best-of-4 by CLIP from the BALANCED seeds (same as the original benchmark) ----
cd = R[R.run_id == "E2_fp16_15"].rename(columns={"clip": "clip_score"})[["prompt_id", "seed", "path", "clip_score", "latency_s"]]
best = cd.loc[cd.groupby("prompt_id")["clip_score"].idxmax()]

# ---- CV pipeline + CLIP guard on LOW outputs (seed 12345) ----
FINAL = OUTPUT_DIR / "cv_final"; FINAL.mkdir(exist_ok=True); cvrows = []
for i, p in enumerate(PROMPTS, 1):
    raw = Image.open(OUTPUT_DIR / "E6_lcm_4_384" / f"p{i:02d}.png")
    base = raw.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    proc, m = postprocess(raw, target_size=512)
    c0, c1 = _clip(p, base), _clip(p, proc); keep = c1 >= c0 - 0.002
    (proc if keep else base).save(FINAL / f"p{i:02d}.png")
    cvrows.append(dict(prompt_id=i, clip_before=round(c0, 4), clip_after=round(c1, 4), kept_processed=keep,
                       sharp_before=round(m["before"]["sharpness"], 1), sharp_after=round(m["after"]["sharpness"], 1),
                       flags_before=",".join(m["flags_before"]) or "-", sharpened=m["sharpened"]))
cv = pd.DataFrame(cvrows)

# gallery labels
df = R[R.seed == 12345][["run_id", "prompt_id", "clip"]].rename(columns={"clip": "clip_similarity"})

# ---- summary table ----
s12 = R[R.seed == 12345].groupby("run_id").agg(config=("config", "first"), median_latency_s=("latency_s", "median"),
                                               peak_mem_mb=("peak_mem_mb", "max"), clip_seed12345=("clip", "mean"))
s12["clip_all_4_seeds"] = R.groupby("run_id")["clip"].mean()
s12["diversity_4_seeds"] = div.groupby("run_id")["diversity"].mean()
qual = dict(config="QUALITY best-of-4 (CLIP-selected)",
            median_latency_s=float(cd.groupby("prompt_id")["latency_s"].sum().median()),
            peak_mem_mb=float(s12.loc["E2_fp16_15", "peak_mem_mb"]), clip_seed12345=float(best.clip_score.mean()),
            clip_all_4_seeds=float("nan"), diversity_4_seeds=float("nan"))
s12.loc["bestofN"] = qual
s12 = s12.round(4)
display(s12)
print("CV step: delivered CLIP", round(cv.apply(lambda r: r.clip_after if r.kept_processed else r.clip_before, axis=1).mean(), 4),
      "vs resize-only", round(cv.clip_before.mean(), 4), "| processed kept", int(cv.kept_processed.sum()), "of 12")

R.to_csv("rerun_per_image_metrics.csv", index=False); div.to_csv("diversity_per_prompt.csv", index=False)
s12.to_csv("rerun_summary.csv"); cv.to_csv("cv_metrics_rerun.csv", index=False)

# ---- visual comparison figure (complex prompts) ----
PIDS = [1, 5, 11, 12]
cols = [("LOW raw", lambda i: OUTPUT_DIR / "E6_lcm_4_384" / f"p{i:02d}.png"),
        ("LOW + CV (guarded)", lambda i: FINAL / f"p{i:02d}.png"),
        ("FAST SD-Turbo", lambda i: OUTPUT_DIR / "SD1" / f"p{i:02d}.png"),
        ("BALANCED FP16/15", lambda i: OUTPUT_DIR / "E2_fp16_15" / f"p{i:02d}.png"),
        ("QUALITY best-of-4", lambda i: best[best.prompt_id == i].path.iloc[0])]
fig, ax = plt.subplots(len(PIDS), len(cols), figsize=(3.2 * len(cols), 3.4 * len(PIDS)))
for r, pid in enumerate(PIDS):
    for c, (name, f) in enumerate(cols):
        ax[r, c].imshow(Image.open(f(pid)).resize((512, 512))); ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(name, fontsize=10)
    ax[r, 0].text(-20, 256, f"P{pid}", fontsize=12, ha="right", va="center")
plt.tight_layout(); plt.savefig("visual_comparison.png", dpi=110); plt.show()

# ---- diversity grid (4 seeds) for one prompt ----
fig, ax = plt.subplots(3, 4, figsize=(13, 10))
for r, rid in enumerate(CFG):
    for c, s in enumerate(SEEDS):
        ax[r, c].imshow(Image.open(OUTPUT_DIR / rid / f"p11_s{s}.png").resize((384, 384))); ax[r, c].axis("off")
        ax[r, c].set_title(f"{CFG[rid][2].split(' (')[0]} | seed {s}", fontsize=8)
plt.tight_layout(); plt.savefig("diversity_prompt11.png", dpi=100); plt.show()

with zipfile.ZipFile("extra_results.zip", "w") as z:
    for f in ["rerun_per_image_metrics.csv", "diversity_per_prompt.csv", "rerun_summary.csv",
              "cv_metrics_rerun.csv", "visual_comparison.png", "diversity_prompt11.png"]:
        z.write(f)
    for f in glob.glob(str(OUTPUT_DIR / "*" / "p??.png")) + glob.glob(str(OUTPUT_DIR / "sdturbo" / "*.png")):
        z.write(f)
try:
    from google.colab import files; files.download("extra_results.zip")
except Exception as e:
    print("Download extra_results.zip from the Colab file panel:", e)
print("Done. Gallery tab now has images (re-select a prompt to refresh).")
