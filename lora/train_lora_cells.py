# ============================================================================
# LiteGen AI – REAL LoRA FINE-TUNE (new Colab notebook, Runtime -> T4 GPU)
# Goal: improve colour–object binding. Data: COCO captions that contain colour words.
# Run each "CELL" in its own Colab cell, in order.
# ============================================================================

# ---------------- CELL 1: install ----------------
%pip uninstall -y -q torchao
%pip install -q diffusers transformers accelerate peft datasets safetensors bitsandbytes
import diffusers, subprocess
v = diffusers.__version__.split(".dev")[0]
url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{v}/examples/text_to_image/train_text_to_image_lora.py"
print(subprocess.run(["wget", "-q", "-O", "train_lora.py", url]).returncode, v)   # 0 = ok

# ---------------- CELL 2: build a colour-binding dataset (≈5 min) ----------------
import os, json, re, itertools
from datasets import load_dataset
COLOURS = r"\b(red|blue|green|yellow|white|black|orange|purple|pink|brown|grey|gray)\b"
OBJ = r"\b(car|bus|truck|bicycle|motorcycle|chair|umbrella|bench|vase|cup|bottle|plate|sofa|couch|table|dog|cat|bird|boat|kite|jacket|shirt|train|apple|pear|orange)\b"
ds = load_dataset("sayakpaul/coco-30-val-2014", split="train", streaming=True)
cols = None; os.makedirs("data", exist_ok=True); meta = []; N = 800
for ex in ds:
    if cols is None:
        cols = list(ex.keys()); print("columns:", cols)
        IMG = next(c for c in cols if "image" in c.lower())
        CAP = next(c for c in cols if "caption" in c.lower() or "text" in c.lower())
    cap = ex[CAP] if isinstance(ex[CAP], str) else ex[CAP][0]
    if len(re.findall(COLOURS, cap.lower())) >= 1 and re.search(OBJ, cap.lower()):
        fn = f"{len(meta):05d}.jpg"
        ex[IMG].convert("RGB").resize((512, 512)).save(f"data/{fn}", quality=92)
        meta.append({"file_name": fn, "text": cap})
        if len(meta) >= N: break
with open("data/metadata.jsonl", "w") as f:
    for m in meta: f.write(json.dumps(m) + "\n")
print("images:", len(meta)); print(meta[:3])

# ---------------- CELL 3: train (≈45–75 min on T4) ----------------
!accelerate launch --mixed_precision=fp16 train_lora.py \
  --pretrained_model_name_or_path="Lykon/dreamshaper-8" \
  --train_data_dir="data" --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --max_train_steps=1000 --learning_rate=1e-4 --lr_scheduler="cosine" --lr_warmup_steps=50 \
  --rank=8 --seed=12345 --mixed_precision="fp16" \
  --gradient_checkpointing --use_8bit_adam \
  --checkpointing_steps=250 --report_to="tensorboard" \
  --output_dir="lora_colour" 2>&1 | tee train_log.txt

# ---------------- CELL 4: plot the REAL loss curve ----------------
import glob, pandas as pd, matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ev = EventAccumulator(glob.glob("lora_colour/logs/**/events*", recursive=True)[0]); ev.Reload()
tag = [t for t in ev.Tags()["scalars"] if "loss" in t][0]
loss = pd.DataFrame([(e.step, e.value) for e in ev.Scalars(tag)], columns=["step", "loss"])
loss["smoothed"] = loss.loss.rolling(50, min_periods=1).mean()
loss.to_csv("lora_loss.csv", index=False)
plt.figure(figsize=(7, 3.5)); plt.plot(loss.step, loss.loss, alpha=.25, label="raw"); plt.plot(loss.step, loss.smoothed, label="smoothed (50)")
plt.xlabel("step"); plt.ylabel("noise-prediction MSE"); plt.legend(); plt.tight_layout(); plt.savefig("lora_loss.png", dpi=120); plt.show()
print(loss.iloc[[0, len(loss)//4, len(loss)//2, -1]])

# ---------------- CELL 5: before / after on the 12 benchmark prompts ----------------
import torch, time, torch.nn.functional as F
from diffusers import StableDiffusionPipeline
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
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
@torch.inference_mode()
def clip(p, im):
    x = cp(text=[p], images=[im], return_tensors="pt", padding=True, truncation=True).to("cuda"); o = cm(**x)
    return float((F.normalize(o.text_embeds, dim=-1) * F.normalize(o.image_embeds, dim=-1)).sum())
pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.set_progress_bar_config(disable=True); os.makedirs("eval", exist_ok=True)
def run(tag):
    rows = []
    for i, p in enumerate(PROMPTS, 1):
        for s in [12345, 101, 202, 303]:
            g = torch.Generator("cuda").manual_seed(s); torch.cuda.synchronize(); t = time.perf_counter()
            im = pipe(p, num_inference_steps=15, guidance_scale=7.5, generator=g).images[0]; torch.cuda.synchronize()
            im.save(f"eval/{tag}_p{i:02d}_s{s}.png"); rows.append(dict(model=tag, prompt_id=i, seed=s, clip=clip(p, im), latency=time.perf_counter() - t))
    return rows
R = run("base")
pipe.load_lora_weights("lora_colour"); R += run("lora")
R = pd.DataFrame(R); R.to_csv("lora_eval.csv", index=False)
print(R.groupby("model")[["clip", "latency"]].agg(["mean", "median"]).round(4))
print("Per-prompt CLIP gain:\n", (R[R.model == "lora"].groupby("prompt_id").clip.mean() - R[R.model == "base"].groupby("prompt_id").clip.mean()).round(4))

# ---------------- CELL 6: download everything ----------------
!zip -qr lora_results.zip lora_colour/pytorch_lora_weights.safetensors lora_loss.csv lora_loss.png lora_eval.csv train_log.txt eval data/metadata.jsonl
from google.colab import files; files.download("lora_results.zip")
