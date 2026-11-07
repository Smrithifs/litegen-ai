# Hackathon Technical Solution Report

| Field | Details |
|---|---|
| Problem ID | P3 |
| Problem Title | Resource-Efficient Text-to-Image Generation for Mobile Devices |
| Team Name | LiteGen AI |
| Team Members | [Name 1 – ID], [Name 2 – ID], [Name 3 – ID], [Name 4 – ID] |
| GitHub Repository | [URL] – commit [hash] |
| Live Demo / Deployment | [Gradio share link] / [video link] |

## 1. Problem Understanding & Solution Strategy
**Core Technical Challenge:** Standard Stable Diffusion needs about 4.9 GB of GPU memory and 15 s per image (FP32, 30 steps), which is too heavy for mobile devices. Cutting precision, steps and resolution saves cost but can reduce prompt alignment, especially for prompts with many objects, colours and spatial relations. Target: generation under 1 s and peak memory under 2.5 GB, with as little CLIP loss as possible.

**Proposed Methodology:** We benchmarked a baseline and several optimized configurations of DreamShaper 8 (SD 1.5) and SD-Turbo on 12 complex multi-object prompts, measuring latency, peak memory, model size and CLIP similarity. A rule-based controller then picks the best-quality profile that fits a given memory and latency budget, with automatic fallback, and a CLIP-guarded classical CV step enhances the result. A static configuration cannot serve both a server GPU and a low-end phone; the adaptive controller can.

## 2. Technical Architecture & Implementation
### 2.1 Classical Computer Vision
**Algorithms:** Lanczos resizing (384 → 512 px), LAB colour space + CLAHE (clip limit 1.5, 8×8 tiles) for colour and contrast correction, bilateral filtering (d=5, σ=40) for edge-aware denoising, Laplacian-variance blur detection with conditional unsharp masking, and quality metrics (sharpness, brightness, contrast, Canny edge density) for artifact flags (blurry, dark, overexposed, low contrast). Module: `cv_postprocess.py`.

**Adaptive Selection Logic:** Sharpening is applied only when Laplacian variance is below 100. Every processed image passes a CLIP guard: it is kept only if CLIP does not drop by more than 0.002, otherwise the resize-only image is delivered.

### 2.2 Deep Learning Pipeline
**Model Architecture:** DreamShaper 8 (latent diffusion, SD 1.5 UNet, CLIP ViT-L text encoder, VAE), chosen for strong photorealism at SD 1.5 cost. SD-Turbo (distilled SD 2.1) for single-step generation. Prompts are tokenised by the CLIP tokenizer (77 tokens) and condition the UNet through cross-attention; classifier-free guidance 7.5 (1.5 for LCM, 0 for SD-Turbo).

**Adaptation & Optimization:** No training was done by us. Adaptation: we fused the published LCM-LoRA weights into the UNet and switched to the LCM scheduler, and used the published adversarially distilled SD-Turbo checkpoint. Optimizations: FP16 inference, reduced sampling steps (30 → 15 → 8 → 4 → 1), latent-resolution reduction (512 → 384), knowledge distillation (LCM-LoRA, SD-Turbo), and CLIP-based best-of-4 seed selection for the quality profile.

### 2.3 Hybrid Integration & Adaptive Mechanism
**Profiles:** QUALITY (FP16, best-of-4 × 15 steps), BALANCED (FP16, 15 steps), LOW (LCM-LoRA, 4 steps, 384 px), FAST (SD-Turbo, 1 step). AUTO = best quality that fits.

**Decision Logic:** The controller only uses measured values. If a profile's peak memory exceeds 90% of the memory budget, or its latency exceeds the limit, it steps down QUALITY → BALANCED → LOW; FAST falls back to LOW. A run-time out-of-memory error triggers a retry with LOW. The CV step then runs on the output, guarded by CLIP.

## 3. Experimental Benchmarks & Evaluation
### 3.1 Quantitative Benchmarks (Tesla T4, 12 prompts, seed 12345, median latency)
| Variant | CLIP ↑ | Latency | Peak memory | Model size |
|---|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | 15.00 s | 4870 MB | 4067 MB |
| BALANCED FP16, 15 steps | 0.356 | 2.52 s | 2676 MB | 2034 MB |
| LOW LCM-LoRA, 4 steps, 384 px | 0.329 | 0.58 s | 2409 MB | 2034 MB |
| FAST SD-Turbo, 1 step | 0.337 | 0.34 s | 3098 MB | 4.8 GB snapshot |
| QUALITY best-of-4 | 0.373* | 10.20 s | 2676 MB | 2034 MB |

*Optimistic because CLIP also selects the seed.

**Analysis:** FP16 alone cut latency by 67.5% and memory by 45% with no CLIP loss. Fewer steps and LCM-LoRA give a further 96% latency reduction at a CLIP cost of 0.032. SD-Turbo is the fastest (0.34 s, 97.7% faster) with CLIP close to LCM, but uses more memory than LOW. FID was not computed because 12 images per configuration are too few for a reliable estimate.

### 3.2 Visual Outputs
See repository: per-prompt images for each configuration, `cv_before_after.png` (prompts 1, 5, 8), `bestofN_prompt11.png`, and the Pareto chart (quality vs latency/memory) in the demo's Benchmarks tab.

### 3.3 Ablation Study
| Change | Effect |
|---|---|
| FP32 → FP16 (30 steps) | latency −67.5%, memory −45.1%, CLIP unchanged (0.361) |
| 30 → 15 → 8 steps (FP16) | CLIP 0.361 → 0.356 → 0.349 |
| LCM-LoRA 4 vs 6 steps | CLIP 0.334 vs 0.332, so 4 steps is enough |
| 512 → 384 px (LCM, 4 steps) | latency 0.88 → 0.58 s, memory −266 MB, CLIP −0.005 |
| SD-Turbo 1 / 2 / 4 steps | CLIP 0.337 / 0.327 / 0.325: more steps do not help |
| Best-of-4 vs single seed (15 steps) | CLIP 0.356 → 0.373, at 4× latency |
| CV post-processing with CLIP guard | delivered CLIP 0.3289 → 0.3309; processed kept 6 of 12 |

## 4. Failure Modes & Limitations
**Observed Failure Case:** Count and spatial errors on prompts 1, 7 and 12 (e.g. "two green pears", "bottle behind the plate", "cup on its right"); prompts 1, 7 and 12 had the lowest SD-Turbo CLIP (≈0.27–0.30). CLAHE also lowered CLIP on prompt 1 (0.294 → 0.271), so the guard rejected it.

**Root Cause & Mitigation:** CLIP text conditioning pools the prompt weakly and does not bind attributes to objects or encode spatial relations; fewer steps leave less time to refine layout. CLIP as a metric also misses these errors. Mitigation: layout or attention guidance (e.g. Attend-and-Excite, regional prompting), a stronger metric such as VQA-based scoring, and INT8 quantization or a smaller UNet for real phone deployment. All numbers were measured on a T4, not on a phone; even LOW needs ~2.4 GB.

## 5. Deployment & Quickstart Guide
```
git clone [repo-url]
cd [repo-folder]
pip install -r requirements.txt
# In Google Colab (T4 GPU): run restore_cell.py, then app.py -> public Gradio link
```
**Deployment Format:** Gradio web UI in Google Colab (T4 GPU) with generate, benchmarks, gallery and about tabs.
