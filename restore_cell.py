# ============================================================================
# LiteGen AI - RESTORE CELL (run after reconnecting; replaces all benchmark cells)
# Cell A (run first, separately):
#   %pip uninstall -y -q torchao
#   %pip install -q diffusers transformers accelerate safetensors peft opencv-python-headless pandas matplotlib psutil gradio
# Then run THIS cell, then the frontend cell unchanged.
# All numbers below are copied from the measured outputs of M4_DreamShaper_Adaptive_CV (Tesla T4).
# ============================================================================
import torch, time, gc, os
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "Lykon/dreamshaper-8"
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

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

def gen(pipe, prompt, steps=30, guidance=7.5, size=512, seed=12345):
    g = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance,
               height=size, width=size, generator=g).images[0]
    torch.cuda.synchronize()
    return img, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 1024**2

# CLIP ViT-B/32 (kept on CPU; frontend moves it as needed)
cmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- measured stats (12 prompts, seed 12345, median latency, peak allocated MB, mean CLIP) ----
stats = pd.DataFrame([
    ("E0_fp32_30",   30, 512, "fp32",          15.003, 4870.4, 0.361),
    ("E1_fp16_30",   30, 512, "fp16",           4.872, 2675.8, 0.361),
    ("E2_fp16_15",   15, 512, "fp16",           2.518, 2675.8, 0.356),
    ("E3_fp16_8",     8, 512, "fp16",           1.473, 2675.8, 0.349),
    ("E4_lcm_4",      4, 512, "fp16+lcm-lora",  0.876, 2675.9, 0.334),
    ("E5_lcm_6",      6, 512, "fp16+lcm-lora",  1.166, 2675.9, 0.332),
    ("E6_lcm_4_384",  4, 384, "fp16+lcm-lora",  0.581, 2409.2, 0.329),
    ("bestofN",      15, 512, "fp16",          10.20,  2676.0, 0.3726),
], columns=["run_id", "steps", "size", "precision", "latency", "mem", "clip_mean"]).set_index("run_id")

PROFILE_RUNS = {"LOW": "E6_lcm_4_384", "BALANCED": "E2_fp16_15", "QUALITY": "bestofN"}
ORDER = ["QUALITY", "BALANCED", "LOW"]

def choose_profile(requested, free_mem_mb, max_latency_s=None):
    """Rule-based controller: only uses configs already benchmarked."""
    prof, msgs = requested, []
    while True:
        s = stats.loc[PROFILE_RUNS[prof]]
        too_big = s.mem > free_mem_mb * 0.9
        too_slow = max_latency_s is not None and s.latency > max_latency_s
        if not (too_big or too_slow) or prof == "LOW":
            if too_big or too_slow:
                msgs.append("WARNING: even LOW exceeds the budget; using LOW anyway")
            break
        nxt = ORDER[ORDER.index(prof) + 1]
        msgs.append(f"{prof} -> {nxt} ({'memory' if too_big else 'latency'} budget exceeded)")
        prof = nxt
    s = stats.loc[PROFILE_RUNS[prof]]
    return dict(profile=prof, run_id=PROFILE_RUNS[prof], precision=s.precision, steps=int(s.steps),
                size=int(s["size"]), est_latency_s=round(float(s.latency), 2), est_mem_mb=round(float(s.mem)),
                notes="; ".join(msgs) or "requested profile fits")

# Gallery tables: empty because saved images were lost with the runtime (gallery shows what exists).
df = pd.DataFrame(columns=["run_id", "prompt_id", "clip_similarity"])
cd = pd.DataFrame(columns=["prompt_id", "seed", "path", "clip_score"]); best = cd.copy()
cv = pd.DataFrame(columns=["prompt_id", "kept_processed", "clip_before", "clip_after"])

# ---- classical CV module (identical to the benchmarked version) ----
CV_SRC = '''import cv2, numpy as np
from PIL import Image

def quality_metrics(img):
    rgb = np.array(img.convert("RGB")); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return dict(sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                brightness_mean=float(gray.mean()), contrast_std=float(gray.std()),
                edge_density=float((edges > 0).mean()))

def artifact_flags(m, blur_thresh=100.0):
    f = []
    if m["sharpness"] < blur_thresh: f.append("blurry")
    if m["contrast_std"] < 35: f.append("low_contrast")
    if m["brightness_mean"] < 60: f.append("dark")
    if m["brightness_mean"] > 200: f.append("overexposed")
    return f

def postprocess(img, target_size=None, blur_thresh=100.0):
    out = img.convert("RGB")
    if target_size and out.size != (target_size, target_size):
        out = out.resize((target_size, target_size), Image.Resampling.LANCZOS)
    before = quality_metrics(out)
    rgb = np.array(out)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=40, sigmaSpace=40)
    sharpened = False
    if quality_metrics(Image.fromarray(rgb))["sharpness"] < blur_thresh:
        soft = cv2.GaussianBlur(rgb, (0, 0), 1.2)
        rgb = cv2.addWeighted(rgb, 1.4, soft, -0.4, 0); sharpened = True
    out2 = Image.fromarray(rgb)
    after = quality_metrics(out2)
    return out2, dict(before=before, after=after, flags_before=artifact_flags(before, blur_thresh),
                      flags_after=artifact_flags(after, blur_thresh), sharpened=sharpened)
'''
open("cv_postprocess.py", "w").write(CV_SRC)

# SD-Turbo per-prompt CLIP CSV for the gallery labels (optional: upload Member 2's CSV to /content instead)
print("Restored:", list(stats.index))
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU - set Runtime > T4 GPU")
