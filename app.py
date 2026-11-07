# ============================================================================
# LiteGen AI - improved frontend (paste into ONE Colab cell, run it LAST).
# It REPLACES the old Gradio cell. Run `%pip install -q gradio` in its own cell first.
# Needs these names from the notebook: torch, time, pd, PROMPTS, MODEL_ID, OUTPUT_DIR,
# StableDiffusionPipeline, gen, cmodel, cproc, choose_profile, stats, PROFILE_RUNS,
# df (per-prompt CLIP table), cd and best (best-of-4 tables), cv (CV table, optional).
# ============================================================================
import gc, os, time, importlib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from diffusers import LCMScheduler, AutoPipelineForText2Image
import cv_postprocess; importlib.reload(cv_postprocess)
from cv_postprocess import postprocess

# ---- EDIT THESE -------------------------------------------------------------
TEAM_NAME = "LiteGen AI"
MEMBERS = ["Member 1: <name>", "Member 2: <name> (SD-Turbo benchmark)",
           "Member 3: <name>", "Member 4: <name> (DreamShaper, CV, controller, demo)"]
# Optional: copy Member 2's images to outputs/sdturbo/ as P01_1steps.png ... P12_4steps.png
SDTURBO_IMG_DIR = os.path.join(str(OUTPUT_DIR), "sdturbo")
# -----------------------------------------------------------------------------

try: demo.close()          # stop the previous demo if this cell is re-run
except Exception: pass
if "CUR" in globals(): CUR["pipe"] = None
gc.collect(); torch.cuda.empty_cache()
cmodel.to("cpu")           # keep CLIP off the GPU so memory numbers stay clean

CUR = {"kind": None, "pipe": None}
LOG = []

# ---- Member 2 (SD-Turbo): 12 prompts, seed 12345, FP16, 512px, guidance 0, Kaggle T4 ----
# source: prompt_metrics.csv / SDTurbo_Model_Size_Peak_Memory.csv (median latency over 12 prompts)
SD_TURBO = {1: dict(latency=0.339, clip=0.3367), 2: dict(latency=0.432, clip=0.3269),
            4: dict(latency=0.618, clip=0.3252)}
SD_MEM_MB = 3.0256 * 1024
if "sdturbo_1" not in stats.index:
    _r = dict(steps=1, size=512, precision="fp16", latency=SD_TURBO[1]["latency"], mem=SD_MEM_MB,
              clip_mean=SD_TURBO[1]["clip"], clip=SD_TURBO[1]["clip"])
    stats.loc["sdturbo_1"] = {c: _r.get(c) for c in stats.columns}
PROFILE_RUNS["QUALITY"] = "bestofN" if "bestofN" in stats.index else "E1_fp16_30"
PROFILES = {"QUALITY": PROFILE_RUNS["QUALITY"], "BALANCED": PROFILE_RUNS["BALANCED"],
            "LOW": PROFILE_RUNS["LOW"], "FAST": "sdturbo_1"}

def _clip_of(s):
    return float(s["clip_mean"]) if "clip_mean" in s.index else float(s["clip"])

@torch.inference_mode()
def _clip(prompt, img):
    dev = next(cmodel.parameters()).device
    x = cproc(text=[prompt], images=[img.convert("RGB")], return_tensors="pt",
              padding=True, truncation=True).to(dev)
    out = cmodel(**x)
    return float((F.normalize(out.text_embeds, dim=-1) * F.normalize(out.image_embeds, dim=-1)).sum())

# ---------------------------------------------------------------- pipelines ----
def get_pipe(kind):
    if CUR["kind"] == kind: return CUR["pipe"], 0.0
    CUR["pipe"] = None; CUR["kind"] = None; gc.collect(); torch.cuda.empty_cache()
    t = time.perf_counter()
    if kind == "sdturbo":
        p = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo",
                torch_dtype=torch.float16, variant="fp16").to("cuda")
    else:
        p = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                use_safetensors=True, safety_checker=None).to("cuda")
        if kind == "lcm":
            p.scheduler = LCMScheduler.from_config(p.scheduler.config)
            p.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); p.fuse_lora(); p.unload_lora_weights()
    p.set_progress_bar_config(disable=True)
    CUR.update(kind=kind, pipe=p)
    return p, time.perf_counter() - t

def run_profile(prof, prompt, seed):
    if prof == "LOW":
        pipe, load_s = get_pipe("lcm")
        img, lat, mem = gen(pipe, prompt, steps=4, guidance=1.5, size=384, seed=seed)
        return img, lat, mem, load_s, "DreamShaper 8 + LCM-LoRA, 4 steps, 384px (upscaled to 512)"
    if prof == "FAST":
        pipe, load_s = get_pipe("sdturbo")
        img, lat, mem = gen(pipe, prompt, steps=1, guidance=0.0, size=512, seed=seed)
        return img, lat, mem, load_s, "SD-Turbo (distilled), 1 step, 512px"
    pipe, load_s = get_pipe("fp16")
    if prof == "BALANCED":
        img, lat, mem = gen(pipe, prompt, steps=15, guidance=7.5, size=512, seed=seed)
        return img, lat, mem, load_s, "DreamShaper 8 FP16, 15 steps, 512px"
    cands, lat, mem = [], 0.0, 0.0
    for s in [seed, 101, 202, 303]:
        im, l, m = gen(pipe, prompt, steps=15, guidance=7.5, size=512, seed=s)
        cands.append(im); lat += l; mem = max(mem, m)
    scores = [_clip(prompt, im) for im in cands]
    return (cands[scores.index(max(scores))], lat, mem, load_s,
            "DreamShaper 8 FP16, best-of-4 seeds (CLIP-selected), 15 steps, 512px")

# --------------------------------------------------------------- controller ----
def _budget(v):
    return float(v) if v and float(v) > 0 else None

def plan_profile(requested, free_mb, max_lat):
    max_lat = _budget(max_lat)
    if requested == "AUTO": requested = "QUALITY"
    if requested == "FAST":
        s = stats.loc["sdturbo_1"]
        too_big = s["mem"] > free_mb * 0.9
        too_slow = max_lat is not None and s["latency"] > max_lat
        if not (too_big or too_slow):
            return dict(profile="FAST", est_latency_s=round(float(s["latency"]), 2),
                        est_mem_mb=round(float(s["mem"])), notes="requested profile fits")
        plan = choose_profile("LOW", free_mb, max_lat)
        why = f"FAST -> LOW ({'memory' if too_big else 'latency'} budget exceeded)"
        plan["notes"] = why if plan["notes"] == "requested profile fits" else why + "; " + plan["notes"]
        return plan
    return choose_profile(requested, free_mb, max_lat)

def profile_table(free_mb, max_lat):
    max_lat = _budget(max_lat); rows = []
    for p, rid in PROFILES.items():
        s = stats.loc[rid]
        rows.append({"profile": p, "est. latency (s)": round(float(s["latency"]), 2),
                     "est. peak memory (MB)": round(float(s["mem"])), "CLIP (12-prompt mean)": round(_clip_of(s), 3),
                     "fits memory budget": "yes" if s["mem"] <= free_mb * 0.9 else "NO",
                     "fits latency budget": "yes" if (max_lat is None or s["latency"] <= max_lat) else "NO"})
    return pd.DataFrame(rows)

def decision_md(requested, plan, free_mb, max_lat):
    lim = f"{_budget(max_lat):.1f} s" if _budget(max_lat) else "none"
    return (f"### Controller decision\n**Requested:** {requested}  ->  **Selected:** {plan['profile']}  \n"
            f"**Budget:** {free_mb:.0f} MB memory (a profile is allowed if its estimated peak is at most 90% of this), latency limit: {lim}  \n"
            f"**Reasoning:** {plan['notes']}  \n"
            f"**Estimate for the selected profile:** {plan['est_latency_s']} s, {plan['est_mem_mb']} MB (measured on a Tesla T4)")

def preview(requested, free_mb, max_lat):
    plan = plan_profile(requested, free_mb, max_lat)
    return decision_md(requested, plan, free_mb, max_lat), profile_table(free_mb, max_lat)

# ----------------------------------------------------------------- generate ----
def generate(prompt, requested, free_mb, max_lat, use_cv, seed):
    try:
        prompt = (prompt or "").strip()
        if not prompt:
            return None, None, "**Please enter a prompt.**", pd.DataFrame(LOG), profile_table(free_mb, max_lat)
        seed = int(seed) if seed is not None else 12345
        plan = plan_profile(requested, free_mb, max_lat)
        prof, notes = plan["profile"], plan["notes"]
        try:
            img, lat, mem, load_s, desc = run_profile(prof, prompt, seed)
        except Exception as e:                     # OOM, download failure, ...
            gc.collect(); torch.cuda.empty_cache()
            kind = "out of memory" if "out of memory" in str(e).lower() else type(e).__name__
            prof, notes = "LOW", notes + f"; {kind} -> retried with LOW"
            img, lat, mem, load_s, desc = run_profile("LOW", prompt, seed)
        base = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        final, cv_note = base, "CV step off"
        if use_cv:
            proc, m = postprocess(img, target_size=512)
            c0, c1 = _clip(prompt, base), _clip(prompt, proc)
            keep = c1 >= c0 - 0.002
            final = proc if keep else base
            cv_note = (f"CV step {'applied' if keep else 'rejected by the CLIP guard'} "
                       f"(CLIP {c0:.3f} -> {c1:.3f}; sharpness {m['before']['sharpness']:.0f} -> {m['after']['sharpness']:.0f}; "
                       f"flags before: {', '.join(m['flags_before']) or 'none'})")
        clip_final = _clip(prompt, final)
        LOG.append({"prompt": prompt[:60], "requested": requested, "used": prof, "latency_s": round(lat, 2),
                    "peak_mem_mb": round(mem), "clip": round(clip_final, 3)})
        pd.DataFrame(LOG).to_csv("demo_log.csv", index=False)
        card = (f"### Result\n**Profile used:** {prof} (requested {requested})  \n**Model/config:** {desc}  \n"
                f"**Generation time:** {lat:.2f} s (one-off model load {load_s:.1f} s not included)  \n"
                f"**Peak GPU memory:** {mem:.0f} MB  \n**CLIP similarity to prompt:** {clip_final:.3f}  \n"
                f"**Controller:** {notes}  \n**Post-processing:** {cv_note}  \n\n"
                f"*Budgets are simulated: the model runs on the Colab GPU; the budget only steers the controller. "
                f"CLIP does not catch every colour or spatial error, so check the image yourself.*")
        return final, base, card, pd.DataFrame(LOG[-10:]), profile_table(free_mb, max_lat)
    except Exception as e:
        return None, None, f"**Error:** {type(e).__name__}: {e}", pd.DataFrame(LOG), profile_table(free_mb, max_lat)

# --------------------------------------------------------------- benchmarks ----
NAMES = {"E0_fp32_30": "FP32, 30 steps (baseline)", "E1_fp16_30": "FP16, 30 steps",
         "E2_fp16_15": "FP16, 15 steps (BALANCED)", "E3_fp16_8": "FP16, 8 steps",
         "E4_lcm_4": "LCM-LoRA, 4 steps", "E5_lcm_6": "LCM-LoRA, 6 steps",
         "E6_lcm_4_384": "LCM-LoRA, 4 steps, 384px (LOW)",
         "bestofN": "FP16, best-of-4 x 15 steps (QUALITY, CLIP-selected)"}
SHORT = {"E0_fp32_30": "FP32/30", "E1_fp16_30": "FP16/30", "E2_fp16_15": "FP16/15", "E3_fp16_8": "FP16/8",
         "E4_lcm_4": "LCM/4", "E5_lcm_6": "LCM/6", "E6_lcm_4_384": "LCM/4@384", "bestofN": "Best-of-4",
         "SD1": "Turbo/1", "SD2": "Turbo/2", "SD4": "Turbo/4"}

def build_bench():
    rows = []
    for rid, label in NAMES.items():
        if rid in stats.index:
            s = stats.loc[rid]
            rows.append(dict(rid=rid, model="DreamShaper 8 (SD 1.5)", config=label, source="Member 4, Colab T4",
                             latency_s=float(s["latency"]), peak_mem_mb=float(s["mem"]), clip_sim=_clip_of(s)))
    for k, v in SD_TURBO.items():
        rows.append(dict(rid=f"SD{k}", model="SD-Turbo (distilled, SD 2.1)", config=f"FP16, {k} step(s), guidance 0",
                         source="Member 2, Kaggle T4", latency_s=v["latency"], peak_mem_mb=SD_MEM_MB, clip_sim=v["clip"]))
    b = pd.DataFrame(rows)
    base = b[b["rid"] == "E0_fp32_30"].iloc[0] if (b["rid"] == "E0_fp32_30").any() else b.iloc[0]
    b["latency gain vs FP32 (%)"] = ((base.latency_s - b.latency_s) / base.latency_s * 100).round(1)
    b["memory gain vs FP32 (%)"] = ((base.peak_mem_mb - b.peak_mem_mb) / base.peak_mem_mb * 100).round(1)
    return b

def bench_table(b):
    t = b.drop(columns=["rid"]).copy()
    t["latency_s"] = t["latency_s"].round(2); t["peak_mem_mb"] = t["peak_mem_mb"].round(0).astype(int); t["clip_sim"] = t["clip_sim"].round(3)
    return t.rename(columns={"latency_s": "latency (s)", "peak_mem_mb": "peak memory (MB)", "clip_sim": "CLIP (12-prompt mean)"})

def make_plot(b):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    pareto = [i for i, x in b.iterrows() if not any(
        (o["latency_s"] <= x["latency_s"] and o["clip_sim"] >= x["clip_sim"] and (o["latency_s"] < x["latency_s"] or o["clip_sim"] > x["clip_sim"]))
        for _, o in b.iterrows())]
    for i, x in b.iterrows():
        c = "tab:orange" if x["rid"].startswith("SD") else "tab:blue"
        star = i in pareto
        for a, xv in ((ax[0], x["latency_s"]), (ax[1], x["peak_mem_mb"])):
            a.scatter(xv, x["clip_sim"], s=130 if star else 45, marker="*" if star else "o", color=c, zorder=3)
            a.annotate(SHORT.get(x["rid"], x["rid"]), (xv, x["clip_sim"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Median latency (s, log scale)"); ax[0].set_title("Quality vs latency (stars = Pareto-optimal on latency/CLIP)")
    ax[1].set_xlabel("Peak GPU memory (MB)"); ax[1].set_title("Quality vs memory")
    for a in ax: a.set_ylabel("Mean CLIP similarity"); a.grid(alpha=0.3)
    ax[0].scatter([], [], color="tab:blue", label="DreamShaper 8"); ax[0].scatter([], [], color="tab:orange", label="SD-Turbo")
    ax[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.close(fig)
    return fig

BENCH = build_bench()

# ------------------------------------------------------------------ gallery ----
def gallery(pid):
    pid = int(pid); items = []
    def add(path, label):
        if path and os.path.exists(str(path)): items.append((str(path), label))
    def clip_run(rid):
        try: return f" | CLIP {df[(df.run_id == rid) & (df.prompt_id == pid)].clip_similarity.iloc[0]:.3f}"
        except Exception: return ""
    for rid, label in [("E0_fp32_30", "FP32 / 30 steps (baseline)"), ("E2_fp16_15", "FP16 / 15 steps (BALANCED)"),
                       ("E6_lcm_4_384", "LCM-LoRA / 4 steps / 384px (LOW)")]:
        add(OUTPUT_DIR / rid / f"p{pid:02d}.png", label + clip_run(rid))
    try:
        r = cv[cv.prompt_id == pid].iloc[0]
        add(OUTPUT_DIR / "cv_final" / f"p{pid:02d}.png",
            f"LOW + CV pipeline ({'processed' if r.kept_processed else 'guard kept original'}) | CLIP {r.clip_after if r.kept_processed else r.clip_before:.3f}")
    except Exception: add(OUTPUT_DIR / "cv_final" / f"p{pid:02d}.png", "LOW + CV pipeline")
    try:
        r = best[best.prompt_id == pid].iloc[0]; add(r["path"], f"QUALITY best-of-4 | CLIP {r['clip_score']:.3f} (selected by CLIP)")
    except Exception: pass
    sd_clip = {}
    for f in ("SDTurbo_CLIP_Similarity_12_Prompts.csv", "prompt_metrics.csv"):
        if os.path.exists(f):
            t = pd.read_csv(f); sd_clip = {(r.prompt_id, int(r.steps)): r.clip_score for r in t.itertuples()}; break
    for st in (1, 2, 4):
        c = sd_clip.get((f"P{pid:02d}", st)); add(os.path.join(SDTURBO_IMG_DIR, f"P{pid:02d}_{st}steps.png"),
                                                   f"SD-Turbo / {st} step(s)" + (f" | CLIP {c:.3f}" if c is not None else ""))
    return items

def gallery_view(pid):
    pid = int(pid)
    return f"**Prompt {pid}:** {PROMPTS[pid - 1]}", gallery(pid)

# ------------------------------------------------------------------- the UI ----
try: gpu_name = torch.cuda.get_device_name(0)
except Exception: gpu_name = "no GPU detected"

ABOUT = f"""
### What this app demonstrates
| Brief item | Where to see it |
|---|---|
| Complex multi-object prompts | The 12 benchmark prompts (Generate tab examples, Gallery tab) |
| Semantic consistency | CLIP similarity on every result; manual rubric in the repository |
| Lower latency and memory | Benchmarks tab: FP32 baseline vs FP16, fewer steps, LCM-LoRA, 384px |
| Quality degradation from compression | Gallery tab (compare configs per prompt); Benchmarks tab chart |
| Adaptive generation by budget | Generate tab: AUTO profile, memory and latency sliders |
| Fallback for constrained devices | Lower the memory slider and press *Preview controller decision* |

### How the controller works
Each profile was benchmarked once (latency, peak memory, CLIP). The controller never guesses: it only uses these measured values. If the requested profile needs more than 90% of the memory budget, or exceeds the latency limit, it steps down QUALITY -> BALANCED -> LOW. FAST (SD-Turbo) falls back to LOW. If even LOW does not fit, it warns and still uses LOW.

### Honest limitations
- All numbers are measured on a Tesla T4 GPU. **Nothing was measured on a phone**; the memory slider only simulates a budget.
- No training or fine-tuning was done by us: we fused the published LCM-LoRA weights and used the published SD-Turbo checkpoint.
- CLIP is a rough alignment score and misses some colour, count and spatial errors. The QUALITY CLIP value is optimistic because CLIP also selects the best of 4 seeds.
- Only 12 prompts and one seed per configuration: small CLIP differences (under about 0.01) should not be over-interpreted.
- SD-Turbo results (Member 2) come from a separate Kaggle run with different library versions; the models also differ in base model, steps and guidance, so the comparison is indicative.
- FID was not computed: 12 images per configuration are too few for a reliable estimate.

### Team
{chr(10).join('- ' + m for m in MEMBERS)}
"""

HEADER = f"""# {TEAM_NAME}: resource-aware text-to-image
Pick a compute profile or let the controller choose from a memory and latency budget. **GPU:** {gpu_name} (Colab). First request after switching profile loads the model (10-60 s)."""

with gr.Blocks(title=f"{TEAM_NAME} - resource-aware text-to-image") as demo:
    gr.Markdown(HEADER)
    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt_box = gr.Textbox(label="Prompt", value=PROMPTS[10], lines=3)
                    gr.Examples(examples=[[p] for p in PROMPTS], inputs=[prompt_box], label="12 benchmark prompts (click to load)")
                    profile = gr.Radio(["AUTO", "QUALITY", "BALANCED", "LOW", "FAST"], value="AUTO", label="Compute profile",
                                       info="AUTO = best quality that fits the budget. FAST = SD-Turbo 1 step (distilled model).")
                    mem_budget = gr.Slider(1500, 14000, value=8000, step=100, label="Simulated device memory budget (MB)")
                    with gr.Row():
                        for lab, val in [("Server GPU 14000", 14000), ("Laptop 8000", 8000),
                                         ("High-end phone 4000", 4000), ("Low-end phone 2000", 2000)]:
                            gr.Button(lab, size="sm").click(lambda v=val: v, None, mem_budget)
                    lat_budget = gr.Slider(0, 15, value=0, step=0.1, label="Latency limit (seconds, 0 = no limit)")
                    use_cv = gr.Checkbox(value=True, label="Apply classical CV enhancement (CLIP-guarded)")
                    seed_box = gr.Number(value=12345, precision=0, label="Seed")
                    with gr.Row():
                        prev_btn = gr.Button("Preview controller decision (instant)")
                        gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=5):
                    out_img = gr.Image(type="pil", label="Delivered image")
                    card = gr.Markdown()
                    with gr.Accordion("Before the CV step (resize only)", open=False):
                        raw_img = gr.Image(type="pil", label="Raw output")
                    plan_tbl = gr.Dataframe(label="Profiles vs your budget (measured values)", interactive=False)
                    hist = gr.Dataframe(label="Session log (last 10 generations)", interactive=False)
            ins = [profile, mem_budget, lat_budget]
            prev_btn.click(preview, ins, [card, plan_tbl])
            for c in ins: c.change(preview, ins, [card, plan_tbl])
            gen_btn.click(generate, [prompt_box, profile, mem_budget, lat_budget, use_cv, seed_box],
                          [out_img, raw_img, card, hist, plan_tbl])
            demo.load(preview, ins, [card, plan_tbl])
        with gr.Tab("Benchmarks"):
            gr.Markdown("**12 prompts, seed 12345, warm-up then median latency, peak allocated GPU memory, CLIP ViT-B/32.** "
                        "Gains are relative to the FP32 / 30-step baseline. Rows from two sources are labelled; do not read them as one controlled experiment.")
            gr.Dataframe(value=bench_table(BENCH), interactive=False, wrap=True)
            gr.Plot(value=make_plot(BENCH), label="Quality vs cost")
        with gr.Tab("Gallery (no GPU needed)"):
            gr.Markdown("Saved outputs from the benchmark runs, same prompt and seed across configurations.")
            pid_dd = gr.Dropdown(choices=[(f"P{i}: {PROMPTS[i-1][:70]}", i) for i in range(1, 13)], value=1, label="Prompt")
            pid_txt = gr.Markdown()
            gal = gr.Gallery(label="Results by configuration", columns=3)
            pid_dd.change(gallery_view, pid_dd, [pid_txt, gal])
            demo.load(gallery_view, pid_dd, [pid_txt, gal])
        with gr.Tab("About and limitations"):
            gr.Markdown(ABOUT)

try: demo.queue(default_concurrency_limit=1)
except TypeError: demo.queue()
demo.launch(share=True)