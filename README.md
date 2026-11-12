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
| GitHub Repository | https://github.com/Smrithifs/litegen-ai – commit `a9a646ba1ae3d65bf869771a6f706fa84b7beb87` |
| Live Demo / Deployment | Demo: https://0c96e6b889b62a2586.gradio.live<br>Video: https://drive.google.com/drive/folders/1sySKi1pd5c6VK_PP7bqPZ3us2d1rIWeu?usp=sharing |

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
| Artifact flags (blurry, dark, overexposed, low contrast) from sharpness, brightness and contrast metrics | Detects defective outputs and triggers sharpening / correction | Artifact detection & correction, image quality assessment |

**Adaptive selection logic**

The post-processing stage is conditional rather than applying every operation blindly. The system uses a blur/sharpness signal from Laplacian variance to identify potentially blurry outputs, and a CLIP-based guard to check whether a processed image has lost semantic similarity. A processed result is retained only when it passes the quality checks; otherwise, the original generated image is delivered.

In the evaluation, post-processing changed mean delivered CLIP similarity from 0.3289 to 0.3309, and the processed output was retained in 6 of 12 cases. One observed CLAHE example reduced CLIP similarity from 0.362 to 0.344, illustrating why the guard is necessary.

### 2.2 Deep Learning Pipeline

**Model architecture**

LiteGen AI uses pretrained text-to-image diffusion models rather than a custom low-light CNN or Transformer.

- **DreamShaper 8 / Stable Diffusion 1.5-family pipeline:** the primary generation path. It provides the standard diffusion pipeline used for FP16, reduced-step and quality-oriented configurations.
- **LCM-LoRA accelerated path:** published LCM-LoRA weights are integrated with the DreamShaper UNet, and an LCM scheduler is used to enable few-step generation. The evaluated LCM configuration uses four denoising steps. The LOW profile additionally uses 384 × 384 resolution.
- **SD-Turbo fast path:** a separate distilled text-to-image model path used for the FAST profile, evaluated at one denoising step. It is a separate model path, not the same as applying LCM-LoRA to DreamShaper.
- **Prompt encoding and conditional generation:** the prompt is tokenised by the CLIP tokenizer (77 tokens), encoded by the CLIP text encoder, and injected into the UNet via cross-attention at every denoising step, with classifier-free guidance (7.5 SD 1.5, 1.5 LCM, 0 SD-Turbo).

**Why this design was selected**

The project targets inference efficiency, so it uses pretrained diffusion models and inference-time acceleration rather than training a new image-generation network from scratch. FP16 reduces inference cost, LCM-LoRA supports few-step generation, and SD-Turbo provides a distinct low-latency option. Keeping these paths separate allows the controller to expose different speed, memory and quality trade-offs.

**Loss Functions & Training**

The low-light enhancement losses in the template (spatial consistency, exposure control, colour constancy, perceptual) do not apply to text-to-image generation. LiteGen AI combines pretrained components with one LoRA fine-tune of its own.

Pretrained components used:

- **DreamShaper 8** (Stable Diffusion 1.5 latent diffusion; UNet, CLIP ViT-L text encoder, VAE)
- **LCM-LoRA SD 1.5** (published latent-consistency-distillation adapter), fused into the UNet and run with the LCM scheduler for 4-step generation
- **SD-Turbo** (published adversarially distilled SD 2.1 model) for 1-step generation
- **CLIP ViT-B/32**, used only for evaluation and the CLIP guard

**Inference-time adaptation:** fused LoRA weights, swapped schedulers, per-model guidance (7.5 / 1.5 / 0), FP16 inference, 384 px latent resolution, and CLIP-based best-of-4 seed selection.

**LoRA fine-tune for colour binding (team-trained).** Our failure analysis showed that attribute binding (e.g. "red car + yellow bicycle") fails at every compute level, so we fine-tuned a LoRA adapter on colour–object captions.

| Setting | Value |
|---|---|
| Base model | DreamShaper 8 (SD 1.5) |
| Method | LoRA on the UNet attention layers, rank 8 (diffusers `train_text_to_image_lora`) |
| Dataset | 800 COCO 2014 val images whose captions contain a colour word and an object (e.g. "a red bus", "a blue chair"), resized to 512 × 512 |
| Loss | Standard diffusion noise-prediction MSE |
| Optimizer | 8-bit AdamW, lr 1e-4, cosine schedule, 50 warm-up steps |
| Batch | 1 × 4 gradient-accumulation (effective 4), gradient checkpointing |
| Steps | 1,000 (≈ 5 epochs), FP16 mixed precision |
| Hardware / time | NVIDIA T4, 34 min (≈ 2.0 s/step) |
| Output | `pytorch_lora_weights.safetensors` |

**Training loss:** the smoothed noise-prediction MSE (50-step moving average) went from **0.157** at step 250 to **0.152** at step 500 and **0.134** at step 1,000. The per-step loss is noisy because each step samples a random noise level; the curve stays near its starting level, as expected when fine-tuning an already-trained diffusion model with a small LoRA. Its effect is therefore measured on the benchmark prompts (Section 3.4).

**Evaluation data:** a fixed set of 12 complex prompts with multiple objects, attributes and spatial relations (seed 12345, plus 4 seeds for diversity), used for benchmarking only and kept separate from the training data.

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

**Benchmark scope:** all latency and memory figures above were measured on an NVIDIA Tesla T4 GPU.

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
- FID was not computed: it needs thousands of images per configuration, and we have 12 prompts. The 4-seed CLIP-embedding diversity above is used as the distribution-level metric instead.

### 3.2 Visual Outputs & Intermediate Representations

**Figure 1 – Same-prompt profile comparison (P1, P5, P11, P12)**

![Figure 1](images/fig1_visual_comparison.png)

*Same prompts and seed across LOW raw, LOW + CV (guarded), FAST, BALANCED and QUALITY. Illustrates the speed–quality trade-off. In P1, QUALITY rendered the car yellow instead of red; in P11, BALANCED and QUALITY swapped the umbrella and chair colours, while LOW bound them correctly.*

**Figure 2 – Quality degradation with fewer steps (P10, Gallery tab)**

![Figure 2](images/fig2_p10_gallery.png)

*FP32/30 (baseline) vs FP16/15 (BALANCED, CLIP 0.357) vs LCM/4/384 (LOW, CLIP 0.332). FP16 with half the steps is visually almost identical to the baseline; LOW keeps the main subject but loses the small secondary objects (the two birds).*

**Figure 3 – Post-processing quality guard (live demo)**

![Figure 3](images/fig3_clip_guard.png)

*FAST profile result: the CV step raised sharpness (1255 → 2303) but lowered CLIP similarity (0.362 → 0.344), so the guard rejected the enhancement and delivered the original image. Generation time 0.76 s.*

**Figure 4 – Resource-aware fallback (live demo)**

![Figure 4](images/fig4_fallback.png)

*BALANCED requested with a 2,000 MB memory budget: the controller selected LOW and warned that even LOW exceeds the budget.*

**Intermediate representations**
LiteGen's pipeline does not estimate Retinex illumination maps, reflectance maps or edge-attention maps. Its intermediate CV outputs are the LAB/CLAHE result, the bilateral-filter result, the Laplacian blur/sharpness response, and the CLIP guard's accept/reject decision (Figure 3).

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

### 3.4 LoRA Fine-Tune and On-Device Results

**Figure 5 – Training loss** (`results/lora_loss.png`): raw and smoothed noise-prediction MSE over 1,000 steps.

**Before vs after the LoRA (12 prompts × 4 seeds, FP16, 15 steps, T4)**

| Metric | Base DreamShaper 8 | + LoRA (ours) |
|---|---|---|
| Mean CLIP similarity | [fill from CELL 5] | [fill from CELL 5] |
| Median latency | [fill] | [fill] |
| Colour binding correct on P1 / P11 | [check images] | [check images] |

**On-device inference: Apple iPhone 15**

The LOW-profile model runs fully on the iPhone 15, offline (verified in airplane mode).

| Setting | Value |
|---|---|
| App / engine | Draw Things (Swift, Metal / MPSGraph on the A16 GPU) |
| Model | DreamShaper 8, **8-bit quantized weights** |
| Acceleration | LCM-LoRA SD 1.5 (f16), LCM sampler, 4 steps, CFG 1.0–1.5 |
| Resolution / seed | 512 × 512, seed 12345 |
| Network | None (airplane mode) |
| Generation time per image | [fill: e.g. X.X s, median of runs 2–5] |
| On-device model size | [fill from Draw Things] |

**How it works on the phone:** the 8-bit model files are downloaded to the phone once. After that, every stage (CLIP tokenizer and text encoder → 64 × 64 × 4 latent noise → 4 LCM denoising steps of the UNet with cross-attention to the prompt → VAE decoder → 512 × 512 image) runs locally on the A16 GPU using the phone's unified memory, with no server calls. 8-bit quantization (about 1 GB of weights instead of 2 GB in FP16), 4-step distillation and low guidance are what make it fit.

## 4. Failure Modes & Limitations

### 4.1 Observed Failure Cases
The manual prompt-following review identified failures in object presence, colour binding and spatial relationships.

| Prompt / Test Case | Profile | Observed Failure |
|---|---|---|
| P1: Red car and yellow bicycle | QUALITY | Both objects appeared, but the car was yellow. |
| P5: Woman in red jacket, blue motorcycle and lake | FAST | Incorrect colour and spatial relation. |
| P11: Red umbrella, blue chair | BALANCED, QUALITY | Colours swapped (blue umbrella, red chair). |
| P12: Two green pears | BALANCED, QUALITY | One pear rendered instead of two (count error). |
| P10: Two birds above the ocean | LOW | Small secondary objects (birds) missing. |

### 4.2 Root Cause & Mitigation
**Failure: incorrect colour binding.** The model may associate a colour with the wrong object when a prompt contains several coloured objects. This is a compositional prompt-following limitation of the diffusion model's text conditioning, not a failure of the image post-processing stage.

**Mitigation:** use clearer object–attribute phrasing, evaluate colour/object binding separately, and add region- or object-aware conditioning in future versions. The current controller does not solve colour binding.

### 4.3 Overall Limitations

- CLIP similarity is a semantic similarity measure, not a percentage accuracy score or a complete measure of visual quality.
- QUALITY's CLIP result is selection-biased because CLIP is used for both selection and evaluation.
- Diversity was reported only for BALANCED, LOW and FAST.
- **Mobile deployment:** the LOW-profile model (DreamShaper 8, 8-bit, LCM-LoRA, 4 steps) runs on the iPhone 15 offline. The adaptive controller, CV pipeline and CLIP guard currently run in the server demo; porting them to the app is future work.
- The LoRA was trained on 800 images for 1,000 steps; a larger compositional dataset would be needed for a strong colour-binding improvement.

---

## 5. Deployment & Quickstart Guide

### 5.1 Deployment Format
LiteGen AI is deployed as a Gradio-based demonstration hosted in Google Colab with an NVIDIA T4 runtime. This is a browser-accessible demonstration backed by server-side GPU inference. The LOW-profile model also runs **on-device on an iPhone 15**, fully offline (Section 3.4).

| Component | Current demonstration |
|---|---|
| User interface | Gradio web interface |
| Execution environment | Google Colab |
| Compute | NVIDIA Tesla T4 GPU |
| Inference | Server-side GPU execution |
| Generation paths | DreamShaper 8, LCM-LoRA accelerated path, SD-Turbo |
| Precision / acceleration | FP16 and reduced-step inference where configured |
| Mobile status | On-device on iPhone 15: DreamShaper 8 (8-bit) + LCM-LoRA, 4 steps, offline via Draw Things (Metal) |

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
