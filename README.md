# LiteGen AI – Resource-Efficient Text-to-Image (Problem P3)

Adaptive text-to-image system: DreamShaper 8 (SD 1.5) + LCM-LoRA, SD-Turbo (distilled), a budget-aware controller and a CLIP-guarded classical CV post-processing step. All numbers measured on a Tesla T4 (12 prompts, seed 12345, warm-up, median latency, peak allocated GPU memory, CLIP ViT-B/32).

## Results
| Variant | CLIP ↑ | Latency (s) | Peak memory (MB) | Model size | Latency gain | Memory gain |
|---|---|---|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | 15.00 | 4870 | 4067 MB | – | – |
| FP16, 30 steps | 0.361 | 4.87 | 2676 | 2034 MB | 67.5% | 45.1% |
| BALANCED: FP16, 15 steps | 0.356 | 2.52 | 2676 | 2034 MB | 83.2% | 45.1% |
| FP16, 8 steps | 0.349 | 1.47 | 2676 | 2034 MB | 90.2% | 45.1% |
| LCM-LoRA, 4 steps | 0.334 | 0.88 | 2676 | 2034 MB | 94.2% | 45.1% |
| LOW: LCM-LoRA, 4 steps, 384px | 0.329 | 0.58 | 2409 | 2034 MB | 96.1% | 50.5% |
| FAST: SD-Turbo, 1 step | 0.337 | 0.34 | 3098 | 4.8 GB snapshot | 97.7% | 36.4% |
| QUALITY: best-of-4 × 15 steps | 0.373* | 10.20 | 2676 | 2034 MB | 32.0% | 45.1% |

*Optimistic: CLIP also selects the best of 4 seeds.

## Optimization strategies
FP16 reduced precision · fewer sampling steps · LCM-LoRA (consistency distillation) · reduced resolution (384px) · SD-Turbo (adversarial diffusion distillation) · CLIP-based best-of-N.

## Adaptive controller
Uses only measured values. If a profile's peak memory is above 90% of the memory budget, or it exceeds the latency limit, it steps down QUALITY → BALANCED → LOW. FAST falls back to LOW. If the GPU runs out of memory at run time, it retries with LOW.

## Classical CV (`cv_postprocess.py`)
Lanczos resize → LAB + CLAHE colour/contrast correction → bilateral edge-aware filter → Laplacian blur detection + conditional unsharp mask → artifact flags (blurry/dark/overexposed/low contrast). The result is kept only if CLIP does not drop by more than 0.002.

## Repository
- `notebooks/M2_sdturbo.ipynb` – SD-Turbo benchmark (1/2/4 steps)
- `notebooks/M3_Cv_CIE3.ipynb` – FP32/FP16, steps, resolution, scheduler, combined optimization experiments
- `notebooks/M4_DreamShaper_Adaptive_CV.ipynb` – full benchmark, LCM-LoRA, controller, CV pipeline
- `app.py` – Gradio demo (run `restore_cell.py` first in Colab)
- `results/` – CSV/JSON metrics

## Quickstart (Google Colab, T4 GPU)
```
pip install -r requirements.txt
# cell 1: restore_cell.py   cell 2: app.py  -> opens a public Gradio link
```

## Limitations
Measured on a T4, not on a phone. Even LOW needs ~2.4 GB, so phones would need INT8 or a smaller UNet. No training by us (published LCM-LoRA and SD-Turbo weights). CLIP misses count/spatial errors. FID not computed (12 images per configuration is too few).
