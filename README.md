# LiteGen AI
**Problem P3 – Resource-Efficient Text-to-Image Generation for Mobile Devices**
*Technical Solution & Demonstration Report*

## Project Overview
LiteGen AI is an adaptive text-to-image generation framework designed to reduce inference latency and memory consumption while maintaining semantic quality. It combines accelerated diffusion models, resource-aware profile selection, and quality-guarded computer-vision post-processing to support different computational budgets and generation requirements.

**Demonstration focus**
- Adaptive text-to-image generation across multiple performance profiles.
- Measurable improvements in inference speed and GPU memory efficiency.
- Evaluation of semantic quality and prompt-following trade-offs.
- Intelligent resource management, fallback mechanisms and failure handling.
- Deployment considerations for resource-constrained and mobile environments.

| Field | Details |
|---|---|
| Problem ID | P3 |
| Problem Title | Resource-Efficient Text-to-Image Generation for Mobile Devices |
| Team Name | Super Mario |
| Team Members | Shamant Nagraj – 1RUA24CSE0420<br>Srujan R Bhat – 1RUA24CSE0477<br>Tamidela Leha Reddy – 1RUA24CSE0491<br>Smrithi S – 1RUA24CSE0464 |
| GitHub Repository | https://github.com/Smrithifs/litegen-ai |
| Live Demo / Deployment | `[gradio.live link]` / `[video link]` |

---

## 1. Problem Understanding & Solution Strategy

### 1.1 Core Technical Challenge
The primary challenge in text-to-image generation on resource-constrained devices is balancing image quality, generation latency and memory consumption. Conventional diffusion models require multiple iterative denoising steps, resulting in high computational overhead and slow inference, which makes deployment on mobile and low-resource hardware difficult.

LiteGen AI addresses the following challenges:
- **High inference latency:** conventional 30-step diffusion generation takes 15 seconds in the measured FP32 baseline.
- **High memory consumption:** the baseline requires 4,870 MB of peak allocated GPU memory.
- **Quality degradation:** reducing denoising steps can lead to missing objects, incorrect colour associations and weaker prompt adherence.
- **Resource constraints:** different devices and execution environments have different memory budgets and latency requirements.
- **Generation consistency:** faster generation paths may produce visually plausible images while missing specific objects or relationships described in the prompt.

**Target constraints**

| Parameter | Baseline / target consideration |
|---|---|
| Baseline latency | 15.00 seconds |
| Baseline peak GPU memory | 4,870 MB |
| Inference precision | FP16 |
| Accelerated generation | 1–4 denoising steps for fast paths |
| Adaptive resolution | 384 × 384 in LOW profile |
| Quality evaluation | CLIP ViT-B/32 similarity |
| Resource management | Configurable latency and memory budgets |

The objective is to reduce inference time and memory consumption while maintaining acceptable semantic alignment with the input text prompt.

### 1.2 Proposed Methodology
LiteGen AI uses an adaptive text-to-image generation pipeline combining DreamShaper 8, FP16 inference, published LCM-LoRA acceleration and a separate SD-Turbo fast-generation path. A rule-based controller evaluates latency and memory constraints to select between QUALITY, BALANCED, LOW, FAST and AUTO profiles, while optional classical computer-vision post-processing and CLIP-based quality checks help preserve output quality.

This approach was chosen over a single static diffusion configuration because different generation requests have different speed and quality requirements. Instead of forcing every request through an expensive multi-step pipeline, LiteGen AI dynamically selects a suitable generation strategy and provides fallback handling when resource constraints are exceeded.

---

## 2. Technical Architecture & Implementation
LiteGen AI is organised as a generation pipeline with three cooperating parts: a diffusion-model generation path, a resource-aware profile controller, and an optional classical computer-vision post-processing and quality-check stage. The system offers multiple generation configurations rather than forcing every prompt through one fixed model configuration.

### 2.1 Classical Algorithmic Baseline

**Algorithms utilised**

The classical computer-vision stage operates on generated images after diffusion inference. It does not generate images independently and does not replace the diffusion model.

| Technique | Role in LiteGen AI | When it is useful |
|---|---|---|
| Lanczos resampling | Resizes generated images to the required output dimensions | Resolution adjustment |
| LAB colour-space conversion with CLAHE | Improves local contrast while operating on the luminance-oriented representation | Images where local contrast can be improved |
| Bilateral filtering | Reduces noise while attempting to preserve edges | Outputs with visible local noise |
| Laplacian-variance blur detection | Estimates sharpness using image-gradient variation | Detecting potentially blurry outputs |
| Conditional unsharp masking | Applies sharpening only when the quality checks permit it | Outputs flagged as needing additional sharpness |
| CLIP similarity guard | Compares semantic similarity before and after processing | Rejecting post-processing that harms prompt alignment |

**Adaptive selection logic**

The post-processing stage is conditional rather than applying every operation blindly. The system uses a blur/sharpness signal from Laplacian variance to identify potentially blurry outputs, and a CLIP-based guard to check whether a processed image has lost semantic similarity. A processed result is retained only when it passes the quality checks; otherwise, the original generated image is delivered.

In the evaluation, post-processing changed mean delivered CLIP similarity from 0.3289 to 0.3309, and the processed output was retained in 6 of 12 cases. One observed CLAHE example reduced CLIP similarity from 0.362 to 0.344, illustrating why the guard is necessary.

### 2.2 Deep Learning Pipeline

**Model architecture**

LiteGen AI uses pretrained text-to-image diffusion models rather than a custom low-light CNN or Transformer.
- **DreamShaper 8 / Stable Diffusion 1.5-family pipeline:** the primary generation path. It provides the standard diffusion pipeline used for FP16, reduced-step and quality-oriented configurations.
- **LCM-LoRA accelerated path:** published LCM-LoRA weights are integrated with the DreamShaper UNet, and an LCM scheduler is used to enable few-step generation. The evaluated LCM configuration uses four denoising steps. The LOW profile additionally uses 384 × 384 resolution.
- **SD-Turbo fast path:** a separate distilled text-to-image model path used for the FAST profile, evaluated at one denoising step. It is a separate model path, not the same as applying LCM-LoRA to DreamShaper.

**Why this design was selected**

The project targets inference efficiency, so it uses pretrained diffusion models and inference-time acceleration rather than training a new image-generation network from scratch. FP16 reduces inference cost, LCM-LoRA supports few-step generation, and SD-Turbo provides a distinct low-latency option. Keeping these paths separate allows the controller to expose different speed, memory and quality trade-offs.

**Loss Functions & Training**

We did not train or fine-tune any model. LiteGen AI is an inference-optimization system built on pretrained weights, so the low-light enhancement losses in the template (spatial consistency, exposure control, colour constancy, perceptual) don't apply.

Pretrained components used:
- **DreamShaper 8** (Stable Diffusion 1.5 latent diffusion; UNet, CLIP ViT-L text encoder, VAE)
- **LCM-LoRA SD 1.5** (published latent-consistency-distillation adapter), fused into the UNet and run with the LCM scheduler for 4-step generation
- **SD-Turbo** (published adversarially distilled SD 2.1 model) for 1-step generation
- **CLIP ViT-B/32**, used only for evaluation and the CLIP guard

**How we adapted them (no training):** we fused the LoRA weights, swapped the scheduler, tuned guidance per model (7.5 / 1.5 / 0), used FP16 inference and 384 px latent resolution, and selected the best of 4 seeds with CLIP. All of this changes how the models run, not their weights.

**Evaluation data:** a fixed set of 12 complex prompts with multiple objects, attributes and spatial relations (seed 12345, plus 4 seeds for diversity). This set is used for benchmarking only, not for training.

**Training-based next step:** a LoRA fine-tune on captioned compositional data (e.g. COCO captions), trained with the standard diffusion noise-prediction MSE loss, to improve colour binding and counting.

### 2.3 Hybrid Integration & Adaptive Mechanism

**Integration strategy**

LiteGen uses a sequential hybrid design:
1. The controller selects a generation profile.
2. The selected diffusion path generates an image from the text prompt.
3. The optional classical CV stage resizes or enhances the generated image.
4. Blur and semantic-quality checks determine whether the processed output should be retained.
5. The system returns the accepted output and its profile/benchmark information.

The classical algorithms do not currently provide illumination maps, edge maps or Retinex priors as additional inputs to the diffusion UNet. Their role is post-generation refinement and quality control. The implementation is a hybrid generation-plus-post-processing pipeline, not a feature-level fusion network.

**Generation profiles**

| Profile | Model path and settings | Intended trade-off |
|---|---|---|
| QUALITY | DreamShaper 8, FP16, 15 steps per candidate, best-of-four CLIP selection | Higher candidate-selection quality at higher latency |
| BALANCED | DreamShaper 8, FP16, 15 steps | General quality/speed balance |
| LOW | DreamShaper + LCM-LoRA, 4 steps, 384 × 384 | Lower latency and memory |
| FAST | SD-Turbo, 1 step | Lowest measured latency |
| AUTO | Rule-based selection among feasible profiles | Respect configured latency and memory limits |

**Decision logic**

The AUTO controller uses measured profile requirements and configured resource constraints:
- A profile is eligible when its measured peak memory is no more than 90% of the configured memory budget and its measured latency is within the configured latency limit.
- When a selected profile is not feasible, the controller steps down through QUALITY → BALANCED → LOW.
- If FAST is not feasible, the controller falls back to LOW.
- If CUDA out-of-memory occurs, the system retries using LOW.
- If LOW also exceeds the available budget, the system reports a resource warning rather than claiming successful generation.

**Measured evidence supporting the design (NVIDIA Tesla T4)**

| Configuration | CLIP ViT-B/32 | Median latency | Peak allocated GPU memory |
|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | 15.00 s | 4,870 MB |
| FP16, 30 steps | 0.361 | 4.87 s | 2,676 MB |
| BALANCED FP16, 15 steps | 0.356 | 2.54 s | 2,693 MB |
| LCM-LoRA, 4 steps | 0.334 | 0.88 s | 2,676 MB |
| LOW: LCM-LoRA, 4 steps, 384 px | 0.329 | 0.56 s | 2,434 MB |
| FAST: SD-Turbo, 1 step | 0.337 | 0.32 s | 3,106 MB |
| QUALITY: best-of-4 × 15 steps | 0.373* | 10.15 s | 2,693 MB |

FP16 reduced latency by 67.5% and peak allocated GPU memory by 45.1% relative to the FP32 baseline, while the CLIP score remained 0.361. LOW reduced latency by 96.3% and peak memory by 50.0% relative to the baseline. FAST achieved the lowest measured latency, but it did not have the lowest peak memory.

\* QUALITY uses CLIP to select the best candidate and the same CLIP model for evaluation, so its score is selection-biased and should not be treated as an independent quality comparison.

**Benchmark scope:** these are NVIDIA Tesla T4 measurements. They do not establish on-device iPhone latency, memory consumption, battery use or thermal performance.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks
The system was evaluated on an NVIDIA Tesla T4 GPU using 12 complex text prompts and seed 12345. Latency is reported as median generation time, memory as peak allocated GPU memory, and semantic similarity using CLIP ViT-B/32.

Since LiteGen AI performs text-to-image generation, PSNR, SSIM and LPIPS/NIQE are not the primary metrics: the experiment has no paired ground-truth target images. CLIP similarity, latency and peak GPU memory are reported instead.

| Approach / Model Variant | CLIP ViT-B/32 ↑ | Median latency (s) ↓ | Peak GPU memory (MB) ↓ | Model size |
|---|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | 15.00 | 4,870 | 4,067 MB |
| FP16, 30 steps | 0.361 | 4.87 | 2,676 | 2,034 MB |
| BALANCED FP16, 15 steps | 0.356 | 2.54 | 2,693 | 2,034 MB |
| FP16, 8 steps | 0.349 | 1.47 | 2,676 | 2,034 MB |
| LCM-LoRA, 4 steps | 0.334 | 0.88 | 2,676 | 2,034 MB |
| LOW: LCM-LoRA, 4 steps, 384 px | 0.329 | 0.56 | 2,434 | 2,034 MB |
| FAST: SD-Turbo, 1 step | 0.337 | 0.32 | 3,106 | 4.8 GB snapshot |
| QUALITY: best-of-4 × 15 steps | 0.373* | 10.15 | 2,693 | 2,034 MB |

**Relative improvement over the FP32 baseline**

| Configuration | Latency reduction | Peak memory reduction |
|---|---|---|
| FP16, 30 steps | 67.5% | 45.1% |
| BALANCED, 15 steps | 83.1% | 44.7% |
| FP16, 8 steps | 90.2% | 45.1% |
| LCM-LoRA, 4 steps | 94.1% | 45.1% |
| LOW, 4 steps, 384 px | 96.3% | 50.0% |
| FAST, SD-Turbo, 1 step | 97.9% | 36.2% |
| QUALITY, best-of-4 | 32.3% | 44.7% |

**Analysis**

FP16 inference reduced median latency from 15.00 s to 4.87 s and peak GPU memory from 4,870 MB to 2,676 MB, while retaining the same CLIP score of 0.361. The LOW profile achieved 0.56 s and 2,434 MB, while FAST achieved the lowest latency at 0.32 s but used more peak memory than LOW. QUALITY recorded the highest CLIP score (0.373) but required 10.15 s, and its score is selection-biased because CLIP was used both to select and evaluate the best candidate.

**Additional quality observations**
- Mean delivered CLIP similarity after the post-processing and quality-guard stage changed from 0.3289 to 0.3309.
- The processed output was retained in 6 of 12 cases.
- Diversity, defined as mean pairwise cosine distance between CLIP image embeddings over four seeds, was 0.091 for BALANCED, 0.104 for LOW and 0.088 for FAST.

### 3.2 Visual Outputs & Intermediate Representations

**Figure 1 – Same-prompt profile comparison**
Outputs for the same prompts and seed from QUALITY, BALANCED, LOW and FAST (`results/visual_comparison.png`).
*Caption: Comparison of image-generation profiles under different inference configurations, illustrating the speed–quality trade-off.*

**Figure 2 – High-detail prompt-following example**
P1: "A red sports car parked beside a yellow bicycle under a tree, with a mountain in the background." LOW, FAST and BALANCED omitted the bicycle; QUALITY included both objects but rendered the car yellow (colour-binding error).

**Figure 3 – Post-processing quality guard**
Original generated image, candidate post-processed image and final accepted output.
*Caption: The quality guard retains post-processing only when the measured quality checks permit it. In this example CLIP similarity dropped from 0.362 to 0.344, so the enhancement was rejected and the original was delivered.*

**Intermediate representations**
LiteGen's pipeline does not estimate Retinex illumination maps, reflectance maps or edge-attention maps. The intermediate CV outputs shown are: the LAB/CLAHE result, the bilateral-filter result, the Laplacian blur/sharpness response, and the CLIP guard's accept/reject decision.

### 3.3 Ablation Study

| Configuration change | Measured result | Interpretation |
|---|---|---|
| FP32 → FP16, 30 steps | Latency 15.00 → 4.87 s; memory 4,870 → 2,676 MB; CLIP 0.361 → 0.361 | Precision reduction produced substantial efficiency gains with unchanged CLIP similarity |
| 30 → 15 FP16 steps | Latency 4.87 → 2.54 s; CLIP 0.361 → 0.356 | Fewer denoising steps reduced latency with a small reduction in CLIP similarity |
| 30 → 8 FP16 steps | Latency 4.87 → 1.47 s; CLIP 0.361 → 0.349 | Further step reduction improved speed with a larger quality-score trade-off |
| Standard diffusion → 4-step LCM-LoRA | Latency 0.88 s; CLIP 0.334 | Few-step generation substantially reduced latency |
| LCM-LoRA 4-step → LOW 384 px | Latency 0.88 → 0.56 s; memory 2,676 → 2,434 MB | Reduced resolution further reduced latency and memory |
| DreamShaper path → SD-Turbo FAST | Latency 0.32 s; memory 3,106 MB; CLIP 0.337 | Separate distilled path delivered the lowest latency, but not the lowest memory |
| Post-processing quality guard | Mean delivered CLIP 0.3289 → 0.3309; processed result retained in 6/12 cases | Conditional acceptance avoids keeping every processed image |

**Ablation interpretation:** precision, denoising-step count, resolution and model path each correspond to different measured efficiency/quality trade-offs. The post-processing result supports using a quality guard instead of unconditionally applying every enhancement.

---

## 4. Failure Modes & Limitations

### 4.1 Observed Failure Cases
The manual prompt-following review identified failures in object presence, colour binding and spatial relationships.

| Prompt / Test Case | Profile | Observed Failure |
|---|---|---|
| P1: Red car and yellow bicycle | QUALITY | Both objects appeared, but the car was yellow. |
| P5: Woman in red jacket, blue motorcycle and lake | FAST | Incorrect colour and spatial relation. |

### 4.2 Root Cause & Mitigation
**Failure: incorrect colour binding.** The model may associate a colour with the wrong object when a prompt contains several coloured objects. This is a compositional prompt-following limitation of the diffusion model's text conditioning, not a failure of the image post-processing stage.

**Mitigation:** use clearer object–attribute phrasing, evaluate colour/object binding separately, and add region- or object-aware conditioning in future versions. The current controller does not solve colour binding.

### 4.3 Overall Limitations
- CLIP similarity is a semantic similarity measure, not a percentage accuracy score or a complete measure of visual quality.
- QUALITY's CLIP result is selection-biased because CLIP is used for both selection and evaluation.
- Diversity was reported only for BALANCED, LOW and FAST.

---

## 5. Deployment & Quickstart Guide

### 5.1 Deployment Format
LiteGen AI is deployed as a Gradio-based demonstration hosted in Google Colab with an NVIDIA T4 runtime. This is a browser-accessible demonstration backed by server-side GPU inference. LiteGen AI was also opened and used on an iPhone 15 to demonstrate mobile accessibility.

| Component | Current demonstration |
|---|---|
| User interface | Gradio web interface |
| Execution environment | Google Colab |
| Compute | NVIDIA Tesla T4 GPU |
| Inference | Server-side GPU execution |
| Generation paths | DreamShaper 8, LCM-LoRA accelerated path, SD-Turbo |
| Precision / acceleration | FP16 and reduced-step inference where configured |
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
1. Open a new Colab notebook → Runtime → Change runtime type → **T4 GPU**.
2. Cell 1 – install:
   ```python
   %pip uninstall -y -q torchao
   %pip install -q -r requirements.txt
   ```
3. Cell 2 – paste and run `restore_cell.py` (loads measured profile stats, CLIP model and the CV module).
4. Cell 3 – paste and run `app.py` → prints a public `https://xxxx.gradio.live` link.
5. Cell 4 (optional) – run `extra_eval_cell.py` to regenerate gallery images, the diversity metric and `visual_comparison.png`.

The first generation after switching profile downloads and loads the model (1–2 min). After that, FAST takes about 0.3 s and LOW about 0.6 s per image.

**Deployment format:** Gradio web UI on a Colab T4 GPU, with Generate, Benchmarks, Gallery and About tabs. Any phone or laptop browser can open the link; inference runs on the server GPU.

### 5.3 Runtime Workflow
1. The user enters a text prompt.
2. The user selects QUALITY, BALANCED, LOW, FAST or AUTO.
3. The controller checks the selected profile against the configured latency and memory constraints when AUTO is used.
4. The selected model path generates the image.
5. Optional CV post-processing runs.
6. Blur and CLIP quality checks decide whether the processed output is retained.
7. The UI displays the generated image with the profile used and its performance information.

---

## Student Submission Checklist
- [x] Working GitHub repository link included with an updated README.md
- [x] No raw code pasted into this report
- [x] All table metrics match test runs reproducible via the repository
- [x] All team member details accurately recorded
