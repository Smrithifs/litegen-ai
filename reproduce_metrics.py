"""
LiteGen AI – reproduce all evaluation metrics from scratch.

Measures, for every configuration on the 12 benchmark prompts:
  median latency (s), peak allocated GPU memory (MB), model weight size (MB),
  mean CLIP ViT-B/32 similarity, and 4-seed diversity (mean pairwise CLIP-image distance).

Usage (Google Colab or any CUDA GPU; results in the report were measured on an NVIDIA T4):
    pip install -r requirements.txt
    python reproduce_metrics.py                 # all configs, ~30-40 min on a T4
    python reproduce_metrics.py --quick         # skip FP32 baseline + diversity, ~10 min
    python reproduce_metrics.py --lora lora_colour/pytorch_lora_weights.safetensors  # + our fine-tuned LoRA

Outputs (in ./repro_results/):
    per_image.csv   – one row per generated image
    summary.csv     – one row per configuration (the report's main table)
    images/<config>/pXX_sYYYYY.png
"""
import argparse, gc, itertools, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, LCMScheduler
from transformers import CLIPModel, CLIPProcessor

PROMPTS = [
 "A red sports car parked beside a yellow bicycle under a tree, with a mountain in the background.",
 "A wooden table with a blue ceramic vase, three red apples and an open book, photographed in natural light.",
 "A small white dog sitting on a green sofa, with a brown cat sleeping beside it.",
 "A futuristic city with flying vehicles, glass skyscrapers and people walking on a wet street at night.",
 "A woman wearing a red jacket standing to the left of a blue motorcycle, with a lake behind them.",
 "A wooden cabin surrounded by pine trees, snow-covered mountains in the background and a river flowing in front.",
 "A black laptop placed on a wooden desk, with a coffee cup on its right and a notebook on its left.",
 "A realistic garden containing yellow sunflowers, purple tulips and a small stone fountain.",
 "A modern living room with a grey sofa, a round coffee table and a large window overlooking a city.",
 "A golden retriever running on a beach while two birds fly above the ocean.",
 "A red umbrella leaning against a blue chair, both placed under a large tree in a quiet park.",
 "A glass bottle filled with orange juice placed behind a white plate containing two green pears and one sliced orange.",
]
SEED = 12345
DIV_SEEDS = [12345, 101, 202, 303]
DS = "Lykon/dreamshaper-8"

# name: (loader, steps, guidance, size, label)
CONFIGS = {
    "E0_fp32_30":   ("ds_fp32",   30, 7.5, 512, "Baseline FP32, 30 steps"),
    "E1_fp16_30":   ("ds_fp16",   30, 7.5, 512, "FP16, 30 steps"),
    "E2_fp16_15":   ("ds_fp16",   15, 7.5, 512, "BALANCED: FP16, 15 steps"),
    "E3_fp16_8":    ("ds_fp16",    8, 7.5, 512, "FP16, 8 steps"),
    "E4_lcm_4":     ("ds_lcm",     4, 1.5, 512, "LCM-LoRA, 4 steps"),
    "E5_lcm_6":     ("ds_lcm",     6, 1.5, 512, "LCM-LoRA, 6 steps"),
    "E6_lcm_4_384": ("ds_lcm",     4, 1.5, 384, "LOW: LCM-LoRA, 4 steps, 384 px"),
    "SD1":          ("sdturbo",    1, 0.0, 512, "FAST: SD-Turbo, 1 step"),
    "SD2":          ("sdturbo",    2, 0.0, 512, "SD-Turbo, 2 steps"),
    "SD4":          ("sdturbo",    4, 0.0, 512, "SD-Turbo, 4 steps"),
}
DIVERSITY_FOR = {"E2_fp16_15", "E6_lcm_4_384", "SD1"}   # also used for QUALITY best-of-4


def load(kind, lora=None):
    if kind == "sdturbo":
        p = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    else:
        dt = torch.float32 if kind == "ds_fp32" else torch.float16
        p = StableDiffusionPipeline.from_pretrained(DS, torch_dtype=dt, use_safetensors=True, safety_checker=None)
        if kind == "ds_lcm":
            p.scheduler = LCMScheduler.from_config(p.scheduler.config)
            p.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); p.fuse_lora(); p.unload_lora_weights()
        if kind == "ds_lora" and lora:
            p.load_lora_weights(lora)
    p = p.to("cuda"); p.set_progress_bar_config(disable=True)
    return p


def weight_mb(p):
    mods = [p.unet, p.vae, p.text_encoder]
    return sum(x.numel() * x.element_size() for m in mods for x in m.parameters()) / 1024**2


def gen(p, prompt, steps, guidance, size, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter()
    img = p(prompt, num_inference_steps=steps, guidance_scale=guidance, height=size, width=size, generator=g).images[0]
    torch.cuda.synchronize()
    return img, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 1024**2


@torch.inference_mode()
def clip_score(cm, cp, prompt, img):
    x = cp(text=[prompt], images=[img.convert("RGB")], return_tensors="pt", padding=True, truncation=True).to("cpu")
    o = cm(**x)
    t, e = F.normalize(o.text_embeds, dim=-1), F.normalize(o.image_embeds, dim=-1)
    return float((t * e).sum()), e[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip FP32 baseline and diversity seeds")
    ap.add_argument("--lora", default=None, help="path to our fine-tuned LoRA .safetensors (optional)")
    ap.add_argument("--out", default="repro_results")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "A CUDA GPU is required (Colab: Runtime -> Change runtime type -> T4 GPU)."
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

    cfgs = dict(CONFIGS)
    if a.quick: cfgs.pop("E0_fp32_30")
    if a.lora: cfgs["LORA_fp16_15"] = ("ds_lora", 15, 7.5, 512, "FP16, 15 steps + our colour LoRA")
    if a.lora: DIVERSITY_FOR.add("LORA_fp16_15")

    # CLIP stays on CPU so it never affects GPU memory measurements
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rows, emb, sizes = [], {}, {}
    for kind in dict.fromkeys(v[0] for v in cfgs.values()):          # load each model once
        p = load(kind, a.lora); size_mb = weight_mb(p)
        for rid, (k, steps, guid, res, label) in cfgs.items():
            if k != kind: continue
            sizes[rid] = size_mb
            gen(p, PROMPTS[0], min(2, steps), guid, res, SEED)         # warm-up, not measured
            seeds = DIV_SEEDS if (rid in DIVERSITY_FOR and not a.quick) else [SEED]
            d = os.path.join(a.out, "images", rid); os.makedirs(d, exist_ok=True)
            for i, prompt in enumerate(PROMPTS, 1):
                for s in seeds:
                    img, lat, mem = gen(p, prompt, steps, guid, res, s)
                    c, e = clip_score(cm, cp, prompt, img); emb[(rid, i, s)] = e
                    img.save(os.path.join(d, f"p{i:02d}_s{s}.png"))
                    rows.append(dict(config=rid, label=label, prompt_id=i, seed=s, steps=steps, size=res,
                                     latency_s=lat, peak_mem_mb=mem, clip=c))
            r = pd.DataFrame([x for x in rows if x["config"] == rid and x["seed"] == SEED])
            print(f"{label:38s} latency {r.latency_s.median():6.2f}s  mem {r.peak_mem_mb.max():6.0f}MB  CLIP {r.clip.mean():.4f}")
        del p; gc.collect(); torch.cuda.empty_cache()

    R = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "per_image.csv"), index=False)

    main_ = R[R.seed == SEED]
    S = main_.groupby("config", sort=False).agg(label=("label", "first"), median_latency_s=("latency_s", "median"),
                                                peak_mem_mb=("peak_mem_mb", "max"), clip_mean=("clip", "mean"))
    S["weights_mb"] = pd.Series(sizes)

    def diversity(rid):
        if not all((rid, i, s) in emb for i in range(1, 13) for s in DIV_SEEDS): return np.nan
        return float(np.mean([np.mean([1 - float(emb[(rid, i, x)] @ emb[(rid, i, y)])
                                       for x, y in itertools.combinations(DIV_SEEDS, 2)]) for i in range(1, 13)]))
    S["diversity_4seeds"] = [diversity(r) for r in S.index]

    if "E2_fp16_15" in DIVERSITY_FOR and not a.quick:             # QUALITY = best-of-4 by CLIP
        b = R[R.config == "E2_fp16_15"]
        best = b.loc[b.groupby("prompt_id").clip.idxmax()]
        S.loc["bestofN"] = dict(label="QUALITY: best-of-4 x 15 steps (CLIP-selected)",
                                median_latency_s=float(b.groupby("prompt_id").latency_s.sum().median()),
                                peak_mem_mb=float(b.peak_mem_mb.max()), clip_mean=float(best.clip.mean()),
                                weights_mb=sizes.get("E2_fp16_15"), diversity_4seeds=np.nan)

    if "E0_fp32_30" in S.index:
        base = S.loc["E0_fp32_30"]
        S["latency_gain_%"] = (100 * (base.median_latency_s - S.median_latency_s) / base.median_latency_s).round(1)
        S["memory_gain_%"] = (100 * (base.peak_mem_mb - S.peak_mem_mb) / base.peak_mem_mb).round(1)
    S = S.round(4); S.to_csv(os.path.join(a.out, "summary.csv"))
    pd.set_option("display.width", 200); print("\n", S)
    print(f"\nSaved: {a.out}/summary.csv, {a.out}/per_image.csv, {a.out}/images/")


if __name__ == "__main__":
    main()
