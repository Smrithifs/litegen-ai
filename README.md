# LiteGen AI – Resource-Efficient Text-to-Image Generation for Mobile Devices
**Problem P3 · Technical Solution & Demonstration Report**

| Field | Details |
|---|---|
| Problem ID | P3 |
| Problem Title | Resource-Efficient Text-to-Image Generation for Mobile Devices |
| Team Name | Super Mario |
| Team Members | Shamant Nagraj – 1RUA24CSE0420 · Srujan R Bhat – 1RUA24CSE0477 · Tamidela Leha Reddy – 1RUA24CSE0491 · Smrithi S – 1RUA24CSE0464 |
| GitHub Repository | https://github.com/Smrithifs/litegen-ai – commit `[hash]` |
| Live Demo / Deployment | Gradio on Google Colab (T4): `[gradio.live link]` · Video: `[link]` |

## Project Overview
LiteGen AI is an adaptive text-to-image system that cuts inference latency and memory while keeping semantic quality measurable. It combines accelerated diffusion models (FP16, fewer steps, LCM-LoRA, reduced resolution, SD-Turbo), a resource-aware profile controller with fallback, and a CLIP-guarded classical computer-vision post-processing stage.

**Headline result (Tesla T4):** the LOW profile generates an image in **0.56 s vs 15.00 s** for the FP32 baseline (**26.8× faster, −96.3% latency**) using **2,434 MB vs 4,870 MB** peak GPU memory (**−50.0%**), at a CLIP cost of 0.032 (0.361 → 0.329). FAST (SD-Turbo) reaches **0.32 s**.

---

## 1. Problem Understanding & Solution Strategy

### 1.1 Core Technical Challenge
Diffusion models run many iterative denoising steps, so generation is slow and memory-heavy, which blocks mobile and low-resource deployment. Cutting steps, precision or resolution saves cost but can cause missing objects, wrong colour–object binding and weaker spatial prompt-following, especially for complex prompts.

| Parameter | Baseline / target |
|---|---|
| Baseline latency | 15.00 s (DreamShaper 8, FP32, 30 steps) |
| Baseline peak GPU memory | 4,870 MB |
| Target | Sub-second generation, ≤ ~2.5 GB peak memory in the low-resource profile |
| Inference precision | FP16 |
| Accelerated generation | 1–4 denoising steps for fast paths |
| Adaptive resolution | 384 × 384 in LOW profile |
| Quality evaluation | CLIP ViT-B/32 similarity, 4-seed diversity, manual rubric |
| Resource management | Configurable latency and memory budgets |

### 1.2 Proposed Methodology
LiteGen AI benchmarks a baseline and several optimized configurations of DreamShaper 8 and SD-Turbo on 12 complex multi-object prompts, then lets a rule-based controller choose the best-quality profile (QUALITY, BALANCED, LOW, FAST, AUTO) that fits a memory and latency budget, with automatic fallback. Generated images pass through conditional classical CV enhancement that is kept only if CLIP similarity does not drop. A single static configuration cannot serve both a server GPU and a constrained device; adaptive selection can.

---

## 2. Technical Architecture & Implementation

```
Prompt + profile / budget
        │
Budget-aware controller ──► QUALITY | BALANCED | LOW | FAST
        │
Diffusion generation (DreamShaper 8 / LCM-LoRA / SD-Turbo)
        │
Classical CV post-processing (optional)
        │
CLIP guard: keep or reject enhancement
        │
Image + latency / memory / CLIP shown in Gradio UI
```

### 2.1 Classical Algorithmic Baseline

| Technique | Role in LiteGen AI | Requirement covered |
|---|---|---|
| Lanczos resampling / INTER_AREA | Resizes 384 px output to 512 px; resolution reduction experiments | Resizing & resolution optimization |
| LAB colour space + CLAHE (clip 1.0–1.5, 8×8 tiles) | Local contrast and colour correction on luminance only | Colour correction & enhancement |
| Bilateral filtering (d=5, σ=40) | Noise reduction while preserving edges | Edge-aware post-processing |
| Laplacian-variance blur detection, brightness, contrast, Canny edge density, SSIM, PSNR | Measures image quality before/after | Image quality assessment |
| Artifact flags (blurry, dark, overexposed, low contrast) + edge-aware artifact mask + inpainting | Detects and repairs local artifacts | Artifact detection & correction |
| Conditional unsharp masking | Sharpens only when the image is flagged blurry | Artifact correction |
| CLIP similarity guard | Rejects any enhancement that lowers prompt alignment by > 0.002 | Quality control |

**Adaptive selection logic:** operations are conditional, not blind. Sharpening runs only if Laplacian variance < 100. The CLAHE clip limit was chosen by measurement (Member 1): clip 1.0 / 1.5 / 2.0 gave SSIM 0.948 / 0.908 / 0.871 to the original, so the lower limit preserves structure best. Artifact correction (Member 1) found 17 candidate pixels via an edge-aware mask and repaired them with inpainting at **SSIM 0.99999, PSNR 57.5 dB**, so the correction is local and does not damage the image. Every processed result goes through the CLIP guard; otherwise the original is delivered.

**Measured effect:** mean delivered CLIP 0.3289 → 0.3309, processed image kept in 6 of 12 cases. In one demo example CLAHE lowered CLIP 0.362 → 0.344 and the guard correctly rejected it.

### 2.2 Deep Learning Pipeline

**Model architecture**
- **DreamShaper 8 (Stable Diffusion 1.5 family):** latent diffusion with a UNet denoiser, CLIP ViT-L text encoder and VAE decoder. Used for FP32/FP16, reduced-step and quality configurations.
- **LCM-LoRA path:** published LCM-LoRA SD 1.5 weights fused into the DreamShaper UNet, with the LCM scheduler, for 4-step generation; LOW additionally uses 384 × 384.
- **SD-Turbo path:** separate adversarially distilled SD 2.1 model for 1-step generation (FAST).
- **Stable Diffusion 1.5 (Member 1):** reference baseline and scheduler study (PNDM vs DDIM).

**Prompt encoding and conditional generation:** the prompt is tokenised by the CLIP tokenizer (77 tokens), encoded by the CLIP text encoder, and injected into the UNet through cross-attention at every denoising step. Classifier-free guidance is set per model: 7.5 (SD 1.5), 1.5 (LCM), 0 (SD-Turbo).

**Why this design:** the goal is inference efficiency, so we use pretrained diffusion models with inference-time acceleration rather than training a network from scratch. Keeping separate paths lets the controller expose different speed / memory / quality trade-offs.

**Loss Functions & Training:** we did not train or fine-tune any model. LiteGen AI is an inference-optimization system built on pretrained weights, so the low-light losses in the template (spatial consistency, exposure control, colour constancy, perceptual) do not apply.

Pretrained components: DreamShaper 8; LCM-LoRA SD 1.5; SD-Turbo; Stable Diffusion 1.5 (baseline study); CLIP ViT-B/32 (evaluation and guard only).

**How we adapted them (no training):** fused LoRA weights, swapped schedulers (LCM, PNDM, DDIM), tuned guidance per model, FP16 inference, 384 px latent resolution, and CLIP-based best-of-4 seed selection. These change how the models run, not their weights.

**Evaluation data:** a fixed set of 12 complex prompts with multiple objects, colours, counts and spatial relations; seed 12345, plus 4 seeds per prompt for diversity. Used for benchmarking only.

### 2.3 Hybrid Integration & Adaptive Mechanism

**Integration strategy (sequential hybrid):**
1. The controller selects a generation profile.
2. The selected diffusion path generates the image.
3. The optional CV stage resizes / enhances it.
4. Blur and CLIP checks decide whether the processed output is kept.
5. The UI returns the image with profile, latency, memory and CLIP.

The classical algorithms act after generation (refinement and quality control); they are not fed into the UNet as extra inputs.

| Profile | Model path and settings | Trade-off |
|---|---|---|
| QUALITY | DreamShaper 8, FP16, 15 steps × 4 seeds, CLIP best-of-4 | Highest CLIP, highest latency |
| BALANCED | DreamShaper 8, FP16, 15 steps, 512 px | General quality / speed balance |
| LOW | DreamShaper + LCM-LoRA, 4 steps, 384 px | Lowest memory, sub-second |
| FAST | SD-Turbo, 1 step, 512 px | Lowest latency |
| AUTO | Rule-based choice of the best feasible profile | Respects memory and latency limits |

**Decision logic:**
- A profile is eligible if its measured peak memory ≤ 90% of the memory budget and its measured latency ≤ the latency limit.
- If not eligible, it steps down QUALITY → BALANCED → LOW; FAST falls back to LOW.
- On a CUDA out-of-memory error at run time, it retries with LOW.
- If even LOW does not fit, it warns ("even LOW exceeds the budget") and still uses LOW.

**Tested in the demo:** BALANCED with a 2,800 MB budget → LOW (memory budget exceeded); AUTO with a 3 s limit → BALANCED; 1 s limit → LOW; 2,000 MB budget → LOW with warning. Device presets: Server GPU 14,000 MB, Laptop 8,000 MB, High-end phone 4,000 MB, Low-end phone 2,000 MB.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks
NVIDIA Tesla T4, 12 complex prompts, seed 12345, warm-up then median latency, peak allocated GPU memory, CLIP ViT-B/32. PSNR / SSIM / LPIPS need paired ground-truth images, which text-to-image generation does not have, so CLIP, diversity, latency, memory and model size are reported instead.

| Approach / Model Variant | CLIP ↑ | Diversity ↑ (4 seeds) | Median latency ↓ | Peak GPU memory ↓ | Model size |
|---|---|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | – | 15.00 s | 4,870 MB | 4,067 MB |
| FP16, 30 steps | 0.361 | – | 4.87 s | 2,676 MB | 2,034 MB |
| **BALANCED** FP16, 15 steps | 0.356 | 0.091 | 2.54 s | 2,693 MB | 2,034 MB |
| FP16, 8 steps | 0.349 | – | 1.47 s | 2,676 MB | 2,034 MB |
| LCM-LoRA, 4 steps | 0.334 | – | 0.88 s | 2,676 MB | 2,034 MB |
| **LOW** LCM-LoRA, 4 steps, 384 px | 0.329 | 0.104 | 0.56 s | 2,434 MB | 2,034 MB |
| **FAST** SD-Turbo, 1 step | 0.337 | 0.088 | 0.32 s | 3,106 MB | 4.8 GB (download snapshot) |
| **QUALITY** best-of-4 × 15 steps | 0.373* | – | 10.15 s | 2,693 MB | 2,034 MB |

*Selection-biased: CLIP both selects and scores the best of 4 candidates.

**Relative to the FP32 baseline**

| Configuration | Latency reduction | Peak memory reduction |
|---|---|---|
| FP16, 30 steps | 67.5% | 45.1% |
| BALANCED, 15 steps | 83.1% | 44.7% |
| FP16, 8 steps | 90.2% | 45.1% |
| LCM-LoRA, 4 steps | 94.1% | 45.1% |
| LOW, 4 steps, 384 px | 96.3% | 50.0% |
| FAST, SD-Turbo, 1 step | 97.9% | 36.2% |
| QUALITY, best-of-4 | 32.3% | 44.7% |

**Stable Diffusion 1.5 reference study (Member 1, Kaggle T4)**

| Configuration | CLIP | Latency | Peak memory |
|---|---|---|---|
| SD 1.5 FP32, PNDM, 30 steps | 0.329 | 20.78 s | 6,033 MB |
| SD 1.5 FP16, PNDM, 30 steps | 0.328 | 5.20 s | 3,258 MB |
| SD 1.5 FP16, DDIM, 30 steps | 0.330 | 4.97 s | 3,258 MB |
| SD 1.5 FP16, DDIM, 5 steps | – | 1.04 s | 3,258 MB |
| SD 1.5 FP16, PNDM, 30 steps, 384 px | 0.305 | 3.38 s | 2,991 MB |
| SD 1.5 FP16, PNDM, 10 steps, 384 px | 0.298 | 1.29 s | 2,991 MB |

This confirms the same pattern on a second base model (FP16 ≈ 4× faster, ~46% less memory, no CLIP loss) and motivated choosing DreamShaper 8, which scored higher CLIP at lower memory.

**Distribution-level metric:** FID needs thousands of images per configuration; with 12 prompts it is unreliable. Instead we report **diversity**: the mean pairwise cosine distance between CLIP image embeddings of 4 seeds per prompt. SD-Turbo is the least diverse (0.088), a known effect of adversarial distillation; LCM-LoRA keeps the most variety (0.104). A second independent run reproduced latency within ±0.03 s and CLIP within ±0.001.

**Analysis:** FP16 alone cuts latency 67.5% and memory 45% with no CLIP loss. Fewer steps and LCM-LoRA give a further large speed-up for a 0.032 CLIP cost and no diversity loss. FAST is the fastest but uses more memory than LOW and is less diverse. QUALITY gives the highest CLIP at 4× BALANCED's latency, but that score is selection-biased.

### 3.2 Visual Outputs & Intermediate Representations
Figures are in the repository under `results/`.

- **Figure 1 – Profile comparison on complex prompts** (`visual_comparison.png`): prompts P1, P5, P11, P12 × LOW raw / LOW + CV / FAST / BALANCED / QUALITY, same seed.
- **Figure 2 – Quality degradation (P10, Gallery tab):** FP32/30 vs FP16/15 vs LCM/4/384. FP16 with half the steps is visually almost identical to the baseline (CLIP 0.357); LOW keeps the main subject but loses the small secondary objects (the two birds).
- **Figure 3 – CLIP guard in action (demo screenshot):** FAST output, CV step rejected (CLIP 0.362 → 0.344, sharpness 1255 → 2303), original delivered.
- **Figure 4 – Diversity** (`diversity_prompt11.png`): 4 seeds × 3 configurations for P11.
- **Figure 5 – Controller fallback (demo screenshot):** BALANCED requested at a 2,000 MB budget → LOW, with warning.
- **Intermediate CV outputs (Member 1):** LAB/CLAHE result at clip limits 1.0 / 1.5 / 2.0, bilateral-filter result, Laplacian sharpness response, edge-aware artifact mask and inpainted correction.

### 3.3 Ablation Study

| Configuration change | Measured result | Interpretation |
|---|---|---|
| FP32 → FP16, 30 steps | 15.00 → 4.87 s; 4,870 → 2,676 MB; CLIP 0.361 → 0.361 | Precision reduction is almost free in quality |
| 30 → 15 FP16 steps | 4.87 → 2.54 s; CLIP 0.361 → 0.356 | Small quality cost |
| 30 → 8 FP16 steps | 4.87 → 1.47 s; CLIP 0.361 → 0.349 | Larger quality cost |
| Standard → 4-step LCM-LoRA | 0.88 s; CLIP 0.334 | Distillation enables few-step generation |
| LCM-LoRA 4 vs 6 steps | CLIP 0.334 vs 0.332 | More steps did not help |
| LCM 512 → 384 px (LOW) | 0.88 → 0.56 s; 2,676 → 2,434 MB; CLIP −0.005 | Resolution cuts cost with small CLIP loss |
| SD-Turbo 1 / 2 / 4 steps | CLIP 0.337 / 0.327 / 0.325 | 1 step is best for this model |
| PNDM → DDIM scheduler (SD 1.5, 30 steps) | 5.20 → 4.97 s; CLIP 0.328 → 0.330 | Scheduler choice gives a small free gain |
| Best-of-4 vs single seed | CLIP 0.356 → 0.373; ~4× latency | Improves its own (CLIP) metric |
| CLAHE clip limit 1.0 / 1.5 / 2.0 | SSIM 0.948 / 0.908 / 0.871 | Lower limit preserves structure |
| CV with CLIP guard | Delivered CLIP 0.3289 → 0.3309; kept 6/12 | Conditional enhancement avoids harm |

### 3.4 Semantic Consistency – Manual Rubric (seed 12345)

| Prompt | Config | Objects present | Colours bound correctly | Spatial relation correct |
|---|---|---|---|---|
| P1 car + bicycle + tree + mountain | LOW | 3/4 (no bicycle) | yes | partial |
| | FAST | 3/4 (no bicycle) | yes | partial |
| | BALANCED | 3/4 (no bicycle) | partial (yellow leaks to tree) | partial |
| | QUALITY | 4/4 | **no – car yellow** | partial |
| P5 woman in red jacket, left of blue motorcycle, lake | LOW | 2/3 (no motorcycle) | yes | no |
| | FAST | 3/3 | no (motorcycle red) | no (sitting on it) |
| | BALANCED / QUALITY | 3/3 | partial (red-blue motorcycle) | no (woman on the right) |
| P11 red umbrella against blue chair, tree, park | LOW | 4/4 | **yes** | partial |
| | FAST | 4/4 | no (chair red) | partial |
| | BALANCED / QUALITY | 4/4 | **no – colours swapped** | partial |
| P12 bottle behind plate, 2 green pears, sliced orange | LOW | 3/4 (no pears) | yes | yes |
| | FAST | 2/4 (glass, not bottle) | partial | no |
| | BALANCED | 4/4 (1 pear, not 2) | yes | yes |
| | QUALITY | 4/4 (1 pear, not 2) | yes | partial |

**Finding:** every configuration captures the main objects, but attribute binding, counts and relations fail at every compute level; more compute does not fix them, and CLIP does not detect them.

---

## 4. Failure Modes & Limitations

### 4.1 Observed Failure Cases

| Failure | Example | Profile |
|---|---|---|
| Attribute swap chosen by CLIP | P1 "red car beside yellow bicycle" → yellow car, red bicycle | QUALITY |
| Colour swap | P11 → blue umbrella, red chair | BALANCED, QUALITY |
| Colour + spatial error | P5 → red motorcycle; woman sitting on it, not left of it | FAST |
| Count error | P12 "two green pears" → one pear | BALANCED, QUALITY |
| Object omission | P1 bicycle missing; P10 two birds missing | LOW, FAST, BALANCED |
| Harmful enhancement | CLAHE lowered CLIP 0.362 → 0.344 | CV stage (rejected by guard) |
| Memory budget below LOW | 2,000 MB budget | Controller warns, uses LOW |

### 4.2 Root Cause & Mitigation
**Root cause:** the CLIP text encoder represents a prompt close to a bag of words, so "red car + yellow bicycle" and "yellow car + red bicycle" get almost the same embedding. The UNet's cross-attention therefore mixes attributes between objects, and CLIP-based scoring (including best-of-4 selection) cannot tell the difference. Fewer steps and lower resolution additionally drop small secondary objects.

**Mitigation:** attention-guided sampling (Attend-and-Excite) or regional / layout prompting for attribute binding; a VQA-based score (e.g. TIFA) instead of CLIP for best-of-N selection; a stronger text encoder (T5); a LoRA fine-tune on captioned compositional data (e.g. COCO) with the standard noise-prediction MSE loss.

### 4.3 Overall Limitations
- All quantitative results are from a Tesla T4 GPU. On-device phone latency, memory, battery and thermals were not measured.
- LOW still needs ~2.4 GB peak memory; no INT8 quantization yet.
- No training or fine-tuning by the team; published weights only.
- 12 prompts, 4 seeds; FID not computed; diversity reported for BALANCED, LOW and FAST only.
- CLIP misses colour, count and spatial errors; QUALITY's CLIP is selection-biased.
- SD-Turbo and SD 1.5 results come from separate Kaggle runs; model sizes use different conventions (snapshot vs in-memory weights).

**Mobile deployment target:** a mid/high-end phone with 8 GB RAM (e.g. iPhone 15 / Snapdragon 8 Gen 2 class), where an app can use about 3 GB. LOW's 2,034 MB FP16 weights and ~2.4 GB peak fit this budget on paper; FP32 (4,870 MB) does not. INT8 weight quantization would cut weights to ~1 GB (estimate). On-device path: Core ML Stable Diffusion (iOS) or MediaPipe / Qualcomm AI Hub (Android) with the LOW profile.

---

## 5. Deployment & Quickstart Guide

### 5.1 Deployment Format
Gradio web application running in Google Colab on an NVIDIA T4 GPU (server-side inference), with **Generate**, **Benchmarks**, **Gallery** and **About** tabs. The link works from any laptop or phone browser; the demo was opened and used on an **iPhone 15** (inference runs on the server GPU).

| Component | Current demonstration |
|---|---|
| User interface | Gradio web interface |
| Execution environment | Google Colab |
| Compute | NVIDIA Tesla T4 GPU |
| Inference | Server-side GPU |
| Generation paths | DreamShaper 8, LCM-LoRA, SD-Turbo |
| Precision / acceleration | FP16, reduced steps, 384 px, distillation |
| Mobile status | Browser-accessible; demo used on an iPhone 15 (server-side inference) |

### 5.2 Quickstart Commands
```bash
# 1. Clone the project
git clone https://github.com/Smrithifs/litegen-ai.git
cd litegen-ai

# 2. Install dependencies
pip install -r requirements.txt
```

**Run in Google Colab (recommended, T4 GPU required):**
1. New Colab notebook → Runtime → Change runtime type → **T4 GPU**.
2. Cell 1 – install:
   ```python
   %pip uninstall -y -q torchao
   %pip install -q -r requirements.txt
   ```
3. Cell 2 – run `restore_cell.py` (loads measured profile stats, CLIP model and the CV module).
4. Cell 3 – run `app.py` → prints a public `https://xxxx.gradio.live` link.
5. Cell 4 (optional) – run `extra_eval_cell.py` to regenerate gallery images, the diversity metric and `visual_comparison.png`.

The first generation after switching profile downloads and loads the model (1–2 min); after that, FAST takes ~0.3 s and LOW ~0.6 s per image.

### 5.3 Runtime Workflow
1. The user enters a text prompt.
2. The user selects QUALITY, BALANCED, LOW, FAST or AUTO, and optionally a memory budget and latency limit.
3. The controller checks the profile against the budget and falls back if needed.
4. The selected model path generates the image.
5. Optional CV post-processing runs.
6. Blur and CLIP checks decide whether the processed output is kept.
7. The UI shows the image, profile used, generation time, peak GPU memory, CLIP score, controller reasoning and post-processing decision.

### 5.4 Repository Contents
| Path | Contents |
|---|---|
| `notebooks/M1_SD15.ipynb` | SD 1.5 baseline, PNDM/DDIM schedulers, resolution, CLAHE tuning, artifact detection + inpainting |
| `notebooks/M2_sdturbo.ipynb` | SD-Turbo benchmark (1/2/4 steps) |
| `notebooks/M3_Cv_CIE3.ipynb` | DreamShaper FP32/FP16, steps, resolution, scheduler, combined optimization |
| `notebooks/M4_DreamShaper_Adaptive_CV.ipynb` | Full benchmark, LCM-LoRA, controller, CV pipeline |
| `restore_cell.py`, `app.py`, `extra_eval_cell.py` | Demo and evaluation |
| `results/` | CSV/JSON metrics, figures, per-config images |

---

## Student Submission Checklist
- [x] Working GitHub repository with updated README.md
- [x] No raw code pasted in this report
- [x] Table metrics match runs reproducible from the repository
- [x] All team member details recorded
