# Akshar — Project Status & Reference

## Goal
Fine-tune Gemma 4 E4B (multimodal) to transcribe handwritten Indic text AND translate it to English in one prompt. Deploy fully offline via Cactus SDK in a React Native app.

## Hardware
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CUDA: 13.0 (Blackwell / cu130)
- OS: Windows 11

## Pipeline

```
1. Dataset prep     ← IN PROGRESS (running dataset_prep.py, ~10 scripts × 20k samples)
2. Fine-tuning      ← NEXT (finetune.py — QLoRA + Unsloth/HF PEFT)
3. Export to GGUF   ← After training (llama.cpp convert_hf_to_gguf.py)
4. Cactus inference ← Load GGUF on-device
5. React Native app ← Camera → OCR + translation, fully offline
```

## Current Status

### ✅ Done
- Zero-shot benchmark: avg ~17% WRR across 10 scripts (Hindi 49%, Kannada 3.6%)
- `dataset_prep.py` updated with local IHTR loader, image quality filter, streaming
- `finetune.py` written: Unsloth probe → HF TRL+PEFT fallback, loss masking, weighted sampler
- `finetune_config.yaml` with all QLoRA hyperparameters
- `pyproject.toml` updated to cu130 index

### 🔄 In Progress
- `python dataset_prep.py --source local --max_per_script 20000 --output_dir ./data`
  - ~2/10 scripts done when last checked (Kannada + Telugu)
  - ~95 min total, running in background
  - Output: `data/train.jsonl` + `data/val.jsonl` (~180k + 20k records)

### ⚠️ Blocked — Fix torch CUDA after dataset_prep finishes
```bash
# torch 2.11.0+cpu is installed — need CUDA build for RTX 5090
# Run AFTER dataset_prep.py completes (PIL lock released):
uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130 --reinstall
```

### ⏳ Pending
1. Verify dataset: `python verify_dataset.py --data_dir ./data`
2. Smoke test (50 steps): `python finetune.py --max_steps 50`
3. Full training (~1-2 days): `python finetune.py`
4. Post-training eval: `python benchmark_gemma4_indic_hw.py --run --model ./merged_model --samples 100`

---

## Dataset

- Source: IIIT-INDIC-HW-WORDS (IHTR 2023), locally downloaded
- Local path: `data/indic_hw/train/<script>/train.txt` + images
- 10 scripts, ~73k samples each, using 20k/script for training

| Script     | Folder        | Zero-shot WRR | Sample weight |
|------------|---------------|---------------|---------------|
| Kannada    | kannada       | 3.6%          | 2.0×          |
| Telugu     | telugu        | 4.8%          | 2.0×          |
| Hindi      | devanagari    | 49.0%         | 1.0×          |
| Bengali    | bengali       | 4.0%          | 2.0×          |
| Tamil      | tamil         | 4.4%          | 2.0×          |
| Malayalam  | malayalam     | 2.2%          | 2.0×          |
| Gujarati   | gujarati      | 15.8%         | 1.5×          |
| Urdu       | urdu          | 17.6%         | 1.5×          |
| Odia       | odia          | 2.2%          | 2.0×          |
| Gurumukhi  | gurumukhi     | 3.6%          | 2.0×          |

## Model

- Base: Gemma 4 E4B (`./gemma-4-E4B-it`, 4B params)
- Architecture: 42 LM layers (hidden 2560), 16 vision layers (hidden 768)
- Vision-LM bridge: `embed_vision.embedding_projection` [2560×768]
- ⚠️ Vision tower uses custom int8 format (`.linear.weight` sub-keys) — NOT compatible with PEFT LoRA targeting

## Training Config (finetune_config.yaml)

| Param | Value |
|-------|-------|
| LoRA r / alpha | 16 / 32 |
| Quantization | 4-bit NF4 |
| Batch × grad_accum | 4 × 8 = 32 effective |
| Learning rate | 2e-4 cosine |
| Epochs | 3 |
| Max seq length | 512 |
| Train vision tower | False |
| Script oversampling | 2× for WRR < 10% |

## JSONL Schema (train.jsonl / val.jsonl)

```json
{
  "script": "Kannada",
  "label": "ಕರ್ನಾಟಕ",
  "image_b64": "<base64 PNG>",
  "messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Look at this..."}]},
    {"role": "model", "content": "Transcription: ಕರ್ನಾಟಕ\nTranslation: [TRANSLATE]"}
  ]
}
```

**Loss masking:** user turn + `[TRANSLATE]` span both set to -100. Only "Transcription: X" is trained.

## Key Files

| File | Purpose |
|------|---------|
| `dataset_prep.py` | Loads local IHTR → JSONL with base64 images |
| `finetune.py` | QLoRA training, loss masking, eval, checkpointing |
| `finetune_config.yaml` | All hyperparameters |
| `verify_dataset.py` | Sanity check for JSONL output |
| `benchmark_gemma4_indic_hw.py` | Zero-shot eval harness (reuse `compute_metrics`) |
| `data/indic_hw/train/<script>/` | Raw IHTR images + labels |
| `data/train.jsonl` / `data/val.jsonl` | Prepared training data |
| `checkpoints/` | LoRA adapters (every 500 steps) |
| `merged_model/` | Merged bf16 model (for GGUF export) |

## GGUF Export (Post-Training)

```bash
# 1. Merge adapter
python finetune.py --merge_only \
    --adapter_path ./checkpoints/checkpoint-best \
    --output_dir ./merged_model

# 2. Convert to GGUF (llama.cpp)
python convert_hf_to_gguf.py ./merged_model \
    --outtype bf16 --outfile akshar_gemma4_e4b.gguf

# 3. Quantize for mobile
./quantize akshar_gemma4_e4b.gguf akshar_gemma4_e4b_q4km.gguf Q4_K_M
```
⚠️ Risk: llama.cpp may not have `Gemma4ForConditionalGeneration` registered yet.
Fallback: Ollama Modelfile or export text backbone only.

## Known Issues / Risks

1. **torch 2.11.0+cpu** — CPU build in venv. Must reinstall cu130 after dataset prep finishes.
2. **Unsloth + Gemma 4** — Unsloth may not support Gemma 4 yet (new architecture). `finetune.py` auto-falls back to HF TRL+PEFT.
3. **GGUF conversion** — `llama.cpp` Gemma 4 multimodal support unconfirmed.
4. **Vision tower LoRA** — Vision tower is custom int8 format, skip LoRA on it. Only bridge is trained as plain bf16.

## Repo

GitHub: https://github.com/shashanka300/Akshar_gema_kagle.git
Branch: master
