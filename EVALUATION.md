# LiteGen AI – Evaluation Results & Reproduction Guide

All metrics below were measured on an **NVIDIA Tesla T4 GPU** (Google Colab) on a fixed set of **12 complex prompts** with multiple objects, colours, counts and spatial relations, using **seed 12345** (plus seeds 101, 202, 303 for diversity).

## 1. Metrics Used

| Metric | What it measures | How it is computed |
|---|---|---|
| **CLIP similarity** ↑ | How well the image matches the prompt | Cosine similarity of CLIP ViT-B/32 text and image embeddings, mean over 12 prompts |
| **Diversity** ↑ | How different the images are across seeds (distribution-level metric) | Mean pairwise cosine distance between CLIP image embeddings of 4 seeds per prompt |
| **Latency** ↓ | Time to generate one image | Median over 12 prompts after one warm-up run, CUDA-synchronised |
| **Peak GPU memory** ↓ | Memory needed during generation | `torch.cuda.max_memory_allocated()` per image, max over prompts |
| **Model size** | Storage / weight footprint | Sum of UNet + VAE + text-encoder parameter bytes |
| **Quality vs cost** | Trade-off between the above | Latency and memory gain relative to the FP32 baseline |

**Why not FID:** FID needs thousands of images per configuration to be reliable. With 12 prompts it is not meaningful, so the 4-seed CLIP-embedding diversity is used as the distribution-level metric. **Why not PSNR / SSIM / LPIPS:** text-to-image generation has no paired ground-truth image.

## 2. Main Results

| Configuration | CLIP ↑ | Diversity ↑ | Latency ↓ | Peak memory ↓ | Model size |
|---|---|---|---|---|---|
| Baseline: DreamShaper 8, FP32, 30 steps | 0.361 | – | 15.00 s | 4,870 MB | 4,067 MB |
| FP16, 30 steps | 0.361 | – | 4.87 s | 2,676 MB | 2,034 MB |
| **BALANCED**: FP16, 15 steps | 0.356 | 0.091 | 2.54 s | 2,693 MB | 2,034 MB |
| FP16, 8 steps | 0.349 | – | 1.47 s | 2,676 MB | 2,034 MB |
| LCM-LoRA, 4 steps | 0.334 | – | 0.88 s | 2,676 MB | 2,034 MB |
| LCM-LoRA, 6 steps | 0.332 | – | 1.17 s | 2,676 MB | 2,034 MB |
| **LOW**: LCM-LoRA, 4 steps, 384 px | 0.329 | 0.104 | 0.56 s | 2,434 MB | 2,034 MB |
| **FAST**: SD-Turbo, 1 step | 0.337 | 0.088 | 0.32 s | 3,106 MB | 4.8 GB (download snapshot) |
| SD-Turbo, 2 steps | 0.327 | – | 0.43 s | 3,098 MB | 4.8 GB (download snapshot) |
| SD-Turbo, 4 steps | 0.325 | – | 0.62 s | 3,098 MB | 4.8 GB (download snapshot) |
| **QUALITY**: FP16, best-of-4 × 15 steps | 0.373* | – | 10.15 s | 2,693 MB | 2,034 MB |

\*Selection-biased: CLIP both picks the best of 4 candidates and scores it.

## 3. Quality vs Cost (relative to the FP32 baseline)

| Configuration | Speed-up | Latency reduction | Memory reduction | CLIP change |
|---|---|---|---|---|
| FP16, 30 steps | 3.1× | 67.5% | 45.1% | 0.000 |
| BALANCED | 5.9× | 83.1% | 44.7% | −0.005 |
| FP16, 8 steps | 10.2× | 90.2% | 45.1% | −0.012 |
| LCM-LoRA, 4 steps | 17.0× | 94.1% | 45.1% | −0.027 |
| **LOW** | **26.8×** | **96.3%** | **50.0%** | −0.032 |
| **FAST** | **46.9×** | **97.9%** | 36.2% | −0.024 |
| QUALITY | 1.5× | 32.3% | 44.7% | +0.012* |

**Key findings**
- **FP16 is nearly free:** 67.5% faster and 45% less memory with no CLIP loss.
- **LOW** is the best memory option: 27× faster and half the memory for a 0.032 CLIP drop, while keeping the highest diversity.
- **FAST** is the fastest (0.32 s) but uses more memory than LOW and is the least diverse, a known effect of adversarial distillation.
- More steps did not help the distilled models (LCM 4 vs 6 steps; SD-Turbo 1 vs 2 vs 4 steps).

## 4. Supporting Evaluations

**Stable Diffusion 1.5 reference study (Member 1, Kaggle T4)**

| Configuration | CLIP | Latency | Peak memory |
|---|---|---|---|
| SD 1.5 FP32, PNDM, 30 steps | 0.329 | 20.78 s | 6,033 MB |
| SD 1.5 FP16, PNDM, 30 steps | 0.328 | 5.20 s | 3,258 MB |
| SD 1.5 FP16, DDIM, 30 steps | 0.330 | 4.97 s | 3,258 MB |
| SD 1.5 FP16, DDIM, 5 steps | – | 1.04 s | 3,258 MB |
| SD 1.5 FP16, PNDM, 10 steps, 384 px | 0.298 | 1.29 s | 2,991 MB |

**Classical CV post-processing (CLIP-guarded)**

| Measure | Result |
|---|---|
| Mean delivered CLIP, LOW outputs | 0.3289 (resize only) → 0.3309 (with CV + guard) |
| Enhancement kept by the guard | 6 of 12 images |
| Example rejection | CLIP 0.362 → 0.344, original delivered |
| CLAHE clip limit 1.0 / 1.5 / 2.0 | SSIM 0.948 / 0.908 / 0.871 |
| Artifact inpainting (Member 1) | 17 pixels repaired, SSIM 0.99999, PSNR 57.5 dB |

**LoRA fine-tune (colour binding)**

| Setting | Value |
|---|---|
| Data | 800 COCO 2014 val images with colour–object captions |
| Training | LoRA rank 8, 1,000 steps, lr 1e-4, FP16, 34 min on T4 |
| Smoothed loss (noise-prediction MSE) | 0.157 (step 250) → 0.152 (step 500) → 0.134 (step 1,000) |

## 5. Semantic Consistency – Manual Rubric (seed 12345)

| Prompt | Config | Objects present | Colours correct | Spatial relation correct |
|---|---|---|---|---|
| P1 car + bicycle + tree + mountain | LOW | 3/4 (no bicycle) | yes | partial |
| | FAST | 3/4 (no bicycle) | yes | partial |
| | BALANCED | 3/4 (no bicycle) | partial | partial |
| | QUALITY | 4/4 | **no – car yellow** | partial |
| P5 woman in red jacket, left of blue motorcycle | LOW | 2/3 (no motorcycle) | yes | no |
| | FAST | 3/3 | no (motorcycle red) | no |
| | BALANCED / QUALITY | 3/3 | partial | no |
| P11 red umbrella against blue chair | LOW | 4/4 | **yes** | partial |
| | FAST | 4/4 | no (chair red) | partial |
| | BALANCED / QUALITY | 4/4 | **no – colours swapped** | partial |
| P12 bottle behind plate, 2 green pears | LOW | 3/4 (no pears) | yes | yes |
| | FAST | 2/4 (glass, not bottle) | partial | no |
| | BALANCED | 4/4 (1 pear, not 2) | yes | yes |
| | QUALITY | 4/4 (1 pear, not 2) | yes | partial |

Every configuration captures the main objects, but attribute binding, counts and spatial relations fail at every compute level, and CLIP does not detect these errors.

## 6. How to Reproduce

Requires an NVIDIA GPU (e.g. Google Colab → Runtime → Change runtime type → **T4 GPU**).

```bash
git clone https://github.com/Smrithifs/litegen-ai.git
cd litegen-ai
pip install -r requirements.txt
python reproduce_metrics.py            # all configurations, ~30–40 min on a T4
python reproduce_metrics.py --quick    # skips FP32 baseline and diversity, ~10 min
```

In Colab, prefix the commands with `!` (e.g. `!python reproduce_metrics.py`) and run `%pip uninstall -y -q torchao` before installing.

**Outputs** (in `repro_results/`):
- `summary.csv` – one row per configuration (Sections 2–3)
- `per_image.csv` – latency, memory and CLIP for every generated image
- `images/<config>/` – every generated image

**Expected variation:** about ±0.05 s latency and ±0.002 CLIP between runs, due to GPU timing and library versions.

**Other evaluations**
- SD-Turbo benchmark: `M2_sdturbo.ipynb`
- SD 1.5 study, CLAHE tuning, artifact inpainting: `M1_SD15.ipynb`
- Controller, CV pipeline, best-of-4: `M4_DreamShaper_Adaptive_CV.ipynb`
- Diversity figures and visual comparison: `extra_eval_cell.py`
- LoRA training and before/after evaluation: `lora/train_lora_cells.py`
