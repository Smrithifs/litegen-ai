# LiteGen AI – Resource-Efficient Text-to-Image Generation (Problem P3)

Adaptive text-to-image system for resource-constrained devices: DreamShaper 8 (SD 1.5) with FP16, fewer steps and LCM-LoRA; SD-Turbo (distilled) as a 1-step fast path; a budget-aware controller with fallback; and a CLIP-guarded classical CV post-processing step.

**Team:** [Member 1 – ID], [Member 2 – ID], [Member 3 – ID], [Member 4 – ID]
**Live demo:** [Gradio link] · **Video:** [link]

All numbers measured on an NVIDIA Tesla T4 (Google Colab): 12 complex multi-object prompts, seed 12345, warm-up, median latency, peak allocated GPU memory, CLIP ViT-B/32 similarity.

## Results
| Variant | CLIP ↑ | Diversity ↑ (4 seeds) | Latency | Peak memory | Model size | Latency gain | Memory gain |
|---|---|---|---|---|---|---|---|
| Baseline FP32, 30 steps | 0.361 | 0.095 | 15.00 s | 4870 MB | 4067 MB | 97.9%| 45.1% |
| FP16, 30 steps | 0.361 | 0.112 | 4.87 s | 2676 MB | 2034 MB | 67.5% | 45.1% |
| **BALANCED** FP16, 15 steps | 0.356 | 0.091 | 2.54 s | 2693 MB | 2034 MB | 83.1% | 44.7% |
| FP16, 8 steps | 0.349 | – | 1.47 s | 2676 MB | 2034 MB | 90.2% | 45.1% |
| LCM-LoRA, 4 steps | 0.334 | – | 0.88 s | 2676 MB | 2034 MB | 94.2% | 45.1% |
| **LOW** LCM-LoRA, 4 steps, 384 px | 0.329 | 0.104 | 0.56 s | 2434 MB | 2034 MB | 96.3% | 50.0% |
| **FAST** SD-Turbo, 1 step | 0.337 | 0.088 | 0.32 s | 3106 MB | 4.8 GB snapshot | 97.9% | 36.2% |
| **QUALITY** best-of-4 × 15 steps | 0.373* | – | 10.15 s | 2693 MB | 2034 MB | 32.3% | 44.7% |

\*Optimistic: CLIP also selects the best of the 4 seeds.

**Diversity** (distribution-level metric in place of FID, which needs thousands of images): mean pairwise cosine distance between CLIP image embeddings of 4 seeds per prompt.

## What we built
### Model optimization (5 strategies)
1. FP16 reduced-precision inference
2. Reduced sampling steps (30 → 15 → 8 → 4 → 1)
3. LCM-LoRA (latent consistency distillation) fused into the UNet + LCM scheduler
4. Lower latent resolution (512 → 384 px)
5. SD-Turbo (adversarial diffusion distillation), 1 step

### Prompt encoding and conditional generation
CLIP tokenizer (77 tokens) → CLIP text encoder → cross-attention conditioning of the UNet; classifier-free guidance (7.5 for SD 1.5, 1.5 for LCM, 0 for SD-Turbo).

### Adaptive controller and fallback
Profiles: QUALITY, BALANCED, LOW, FAST, and AUTO (best quality that fits). A profile is allowed only if its measured peak memory is ≤ 90% of the memory budget and its latency is under the limit; otherwise it steps down QUALITY → BALANCED → LOW. FAST falls back to LOW. If even LOW does not fit, it warns and uses LOW. A run-time GPU out-of-memory error triggers a retry with LOW.

### Classical computer vision (`cv_postprocess.py`)
Lanczos resize → LAB + CLAHE colour/contrast correction → bilateral edge-aware filter → Laplacian blur detection + conditional unsharp mask → quality metrics and artifact flags (blurry, dark, overexposed, low contrast). The enhanced image is kept only if CLIP does not drop by more than 0.002 (CLIP guard). Result: delivered CLIP 0.3289 → 0.3309; enhanced image kept in 6 of 12 cases.

## Additional experiments
### LoRA fine-tuning ([member name])
DreamShaper 8 + LCM-LoRA, LoRA fine-tune on a T4: FP16, AdamW, lr 1e-5, batch 4, 5 epochs, 2,500 steps, 2,000 captioned images. Loss 0.184 (step 100) → 0.048 (step 2,500).

| Metric (LOW profile, 12 prompts) | Original | Fine-tuned |
|---|---|---|
| CLIP similarity | 0.329 | 0.351 |
| Latency | 0.58 s | 0.60 s |
| Peak memory | 2,409 MB | 2,420 MB |

Evidence: `training/` (notebook with outputs, LoRA checkpoint). Single small run; the main results table uses the published weights.

### Mobile test: iPhone 15
Tested on an Apple iPhone 15 (screenshot in `results/`). Runtime: [Core ML / browser (server-side inference)]. On-device latency and memory were not benchmarked; all quantitative results are from the T4.

## Repository
| Path | Contents |
|---|---|
| `notebooks/M2_sdturbo.ipynb` | SD-Turbo benchmark (1/2/4 steps) |
| `notebooks/M3_Cv_CIE3.ipynb` | FP32/FP16, steps, resolution, scheduler and combined optimization experiments |
| `notebooks/M4_DreamShaper_Adaptive_CV.ipynb` | Full benchmark, LCM-LoRA, controller, CV pipeline |
| `restore_cell.py` | Restores measured stats, models and CV module in a fresh runtime |
| `app.py` | Gradio demo (Generate, Benchmarks, Gallery, About tabs) |
| `extra_eval_cell.py` | Diversity metric, gallery images, CV rerun, visual comparison figure |
| `results/` | CSV/JSON metrics, `visual_comparison.png`, `diversity_prompt11.png`, per-config images |
| `training/` | LoRA fine-tuning notebook and checkpoint |
| `REPORT.md` | Technical report |

## Quickstart (Google Colab, T4 GPU)
```bash
git clone [repo-url]
cd litegen-ai
pip install -r requirements.txt
```
In Colab: Runtime → T4 GPU, then run in order:
1. `%pip uninstall -y -q torchao` and `%pip install -q -r requirements.txt`
2. `restore_cell.py`
3. `app.py` → prints a public `gradio.live` link
4. (optional) `extra_eval_cell.py` → diversity metric, gallery images, figures

The first generation after switching profile downloads and loads the model (1–2 min); after that, FAST takes ~0.3 s and LOW ~0.6 s.

## Failure cases
- **Attribute binding:** for P1 ("red car beside a yellow bicycle") the CLIP-selected QUALITY image has a yellow car; for P11, BALANCED/QUALITY swapped the umbrella and chair colours. CLIP treats the prompt almost as a bag of words and does not catch this.
- **Counts and spatial relations:** "two green pears" → one pear; "woman to the left of the motorcycle" → right of it or on it.
- **Small objects at low compute:** LOW drops secondary objects (the two birds in P10).
- **Harmful enhancement:** CLAHE can lower CLIP (e.g. 0.362 → 0.344); the CLIP guard rejects it.

## Limitations
- Quantitative results measured on a T4 GPU, not on a phone. LOW still needs ~2.4 GB.
- No INT8 quantization yet (estimated ~1 GB weights).
- 12 prompts, 4 seeds; FID not computed.
- CLIP misses colour, count and spatial errors; QUALITY's CLIP is selection-biased.
- SD-Turbo numbers come from a separate run; model size is a download snapshot, not in-memory weights.

## Future work
INT8 conversion of the LOW profile via Core ML Stable Diffusion (iOS) or MediaPipe / Qualcomm AI Hub (Android); attention-guided sampling (Attend-and-Excite) for attribute binding; VQA-based scoring (e.g. TIFA) instead of CLIP for best-of-N selection.
