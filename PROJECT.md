# Akshar — Project Status & Reference

## Goal
Fine-tune Gemma 4 E4B (multimodal) to transcribe handwritten Indic text AND translate it to English in one prompt. Deploy fully offline via Cactus SDK in a React Native app.

## Hardware
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CUDA: 13.0 (Blackwell / cu130)
- OS: Windows 11

## Pipeline

```
1. Dataset prep     ✅ COMPLETE — 179,993 train / 19,995 val across 10 scripts
2. Fine-tuning      🔄 IN PROGRESS — finetune.py debugging/smoke test
3. Export to GGUF   ⏳ After training (llama.cpp convert_hf_to_gguf.py)
4. Cactus inference ⏳ Load GGUF on-device
5. React Native app ⏳ Camera → OCR + translation, fully offline
```

---

## Current Status

### ✅ Completed
- Zero-shot benchmark: avg ~17% WRR (Hindi 49%, Kannada 3.6%)
- `dataset_prep.py`: local IHTR loader, image quality filter, streaming fix
- `data/train.jsonl`: 179,993 records | `data/val.jsonl`: 19,995 records
- Unsloth 2026.4.2 installed and confirmed working on RTX 5090
- `finetune.py` fully rewritten with `AksharDataset` (streaming, zero Arrow cache)
- Loss masking implemented (user turn masked to -100)
- All processor outputs passed through (fixes `pixel_position_ids` AttributeError)
- Images resized to 224×224 to prevent Gemma4 anyres tiling OOM

### 🔄 In Progress — Smoke test (50 steps)
Run:
```bash
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"
python -X utf8 -B finetune.py --max_steps 50
```
Last known state: **OOM fixed (batch 2, grad_accum 16, seq 1024, fixed image size)** — not yet confirmed passing.

### ⏳ Pending
1. Confirm smoke test passes (loss decreases, no crash, VRAM < 28 GB)
2. Full training (3 epochs, ~4-8 hrs): `python -X utf8 finetune.py`
3. Post-training eval: `python benchmark_gemma4_indic_hw.py --run --model ./checkpoints/final --samples 100`
4. GGUF export + Cactus SDK integration

---

## Dataset

| Script     | Folder        | WRR baseline | Train  | Val  |
|------------|---------------|--------------|--------|------|
| Kannada    | kannada       | 3.6%         | 18,000 | 2,000|
| Telugu     | telugu        | 4.8%         | 18,000 | 2,000|
| Hindi      | devanagari    | 49.0%        | 18,000 | 2,000|
| Bengali    | bengali       | 4.0%         | 18,000 | 2,000|
| Tamil      | tamil         | 4.4%         | 18,000 | 2,000|
| Malayalam  | malayalam     | 2.2%         | 17,993 | 1,995|
| Gujarati   | gujarati      | 15.8%        | 18,000 | 2,000|
| Urdu       | urdu          | 17.6%        | 18,000 | 2,000|
| Odia       | odia          | 2.2%         | 18,000 | 2,000|
| Gurumukhi  | gurumukhi     | 3.6%         | 18,000 | 2,000|

JSONL schema:
```json
{
  "script": "Kannada",
  "label": "ಕರ್ನಾಟಕ",
  "image_b64": "<base64 PNG>",
  "messages": [
    {"role": "user",  "content": [{"type": "image"}, {"type": "text", "text": "Look at this..."}]},
    {"role": "model", "content": "Transcription: ಕರ್ನಾಟಕ\nTranslation: [TRANSLATE]"}
  ]
}
```

---

## Training Config (finetune_config.yaml)

| Param | Value | Note |
|-------|-------|------|
| LoRA r / alpha | 32 / 64 | Increased from 16/32 for RTX 5090 |
| LoRA dropout | 0.0 | Unsloth recommendation |
| Quantization | 4-bit NF4 | ~8GB model footprint |
| Batch × grad_accum | 16 × 2 = 32 effective | Unsloth VRAM savings |
| Learning rate | 2e-4 cosine | AdamW 8-bit |
| Epochs | 3 | ~4-8 hrs on RTX 5090 |
| Max seq length | 1024 | Increased from 512 |
| target_modules | q,k,v,o,gate,up,down,embed_tokens,lm_head | |
| gradient_checkpointing | "unsloth" | |

---

## Key Files

| File | Purpose |
|------|---------|
| `dataset_prep.py` | Local IHTR loader → JSONL with base64 images |
| `finetune.py` | Unsloth + SFTTrainer training script |
| `finetune_config.yaml` | All hyperparameters |
| `verify_dataset.py` | JSONL sanity check (run with `-X utf8` on Windows) |
| `benchmark_gemma4_indic_hw.py` | Zero-shot eval harness |
| `data/train.jsonl` / `data/val.jsonl` | Training data (gitignored, ~8GB) |
| `checkpoints/` | LoRA adapters (gitignored) |
| `PROJECT.md` | This file — full project log |

---

## Challenges & Resolutions Log

### Challenge 1: torch CPU build instead of CUDA
- **Symptom**: `torch 2.11.0+cpu` installed despite RTX 5090 being present
- **Cause**: uv resolved from wrong index; initial install used cu124 index but machine has cu130
- **Fix**: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
- **Config updated**: `pyproject.toml` now points to `pytorch-cu130` index

### Challenge 2: PIL file handle exhaustion ("Too many open files")
- **Symptom**: `OSError: [Errno 24] Too many open files` around Telugu in dataset_prep
- **Cause**: PIL lazy-loads images (keeps fd open); collecting 20k PIL objects before processing hits OS fd limit
- **Fix**: Added `img.load()` after `Image.open()` to force pixel data into RAM and release the fd. Also switched from collecting all samples to streaming directly to JSONL.
- **File**: `dataset_prep.py` — `load_script_local()`

### Challenge 3: Windows cp1252 console rejecting Unicode
- **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'`
- **Cause**: Windows terminal defaults to cp1252; Indic script labels and Unicode arrows crash stdout
- **Fix**: Always run Python scripts with `-X utf8` flag: `python -X utf8 finetune.py`
- **Also affected**: `verify_dataset.py` (box-drawing chars in print statements)

### Challenge 4: Arrow schema error in HF `load_dataset("json", ...)`
- **Symptom**: `pyarrow.lib.ArrowInvalid: JSON parse error: Column(/messages/[]/content) changed from array to string in row 0`
- **Cause**: `messages[0].content` is a list (image + text objects) but `messages[1].content` is a plain string. Arrow infers a fixed schema from row 0 and rejects the inconsistency.
- **Fix**: Replaced `load_dataset("json", ...)` with `HFDataset.from_generator(flat_generator)` where the generator pre-flattens messages into two scalar string fields (`user_text`, `model_text`). `process_batch` reconstructs full message dicts on the fly.
- **File**: `finetune.py` — `train()` and `process_batch()`

### Challenge 5: MemoryError loading 180k base64 records
- **Symptom**: `MemoryError` during `load_dataset("json", ...)` — pandas tried to read entire 8GB JSONL into RAM
- **Cause**: HF `load_dataset` with pandas backend reads the full file before streaming
- **Fix**: Same as Challenge 4 — `from_generator()` streams records one-by-one, never loading the full file

### Challenge 6: `cfg.max_steps` AttributeError
- **Symptom**: `AttributeError: 'FinetuneConfig' object has no attribute 'max_steps'`
- **Cause**: `max_steps` referenced in `train()` but not declared in the `FinetuneConfig` dataclass
- **Fix**: Added `max_steps: Optional[int] = None` to `FinetuneConfig`

### Challenge 7: Gemma4 processor rejects batched images
- **Symptom**: `ValueError: Received inconsistently sized batches of images (1) and text (8)`
- **Cause**: Gemma4's `AutoProcessor.__call__` requires exactly one image per text input; passing `images=[img1, img2, ...]` with `text=[t1, t2, ...]` fails
- **Fix**: Loop over each sample individually in `process_batch`, calling `processor(text=t, images=[img])` once per sample, then concatenate results
- **File**: `finetune.py` — `process_batch()`

### Challenge 8: HF Arrow cache filled 97 GB of disk (OSError: No space left on device)
- **Symptom**: `OSError: [Errno 28] No space left on device` during `dataset.map()` tokenisation step
- **Cause**: HF `Dataset.map()` writes a full Arrow cache of tokenised data to `~/.cache/huggingface/datasets/`. With 180k samples × 1024 tokens each, the cache grew to 97 GB and exhausted the disk.
- **Fix**: Replaced `HFDataset.from_generator() + .map()` with a custom `AksharDataset(TorchDataset)` that builds a byte-offset index (~1.4 MB) at init and tokenises each sample on-the-fly in `__getitem__`. Zero disk writes, zero pre-tokenisation wait.
- **File**: `finetune.py` — replaced `process_batch` + `raw_dataset.map()` with `AksharDataset` class
- **Cleanup**: Delete stale cache manually: `Remove-Item -Recurse -Force "C:\Users\shash\.cache\huggingface\datasets\"`

### Challenge 9: AttributeError — `pixel_position_ids` missing from batch
- **Symptom**: `AttributeError: 'bool' object has no attribute 'all'` inside `modeling_gemma4.py` line 1903: `padding_positions = (pixel_position_ids == -1).all(dim=-1)`
- **Cause**: Gemma4 processor outputs several tensors beyond `input_ids / attention_mask / pixel_values`, including `pixel_position_ids` and possibly `token_type_ids`. The old `__getitem__` only returned 4 hardcoded keys; missing fields arrive as `None` in the model, `None == -1` returns Python `False`, and `.all()` fails on a bool.
- **Fix**: Return all processor outputs dynamically: `result = {k: v[0] for k, v in inp.items()}` then add `labels`. This passes every key the processor emits through to the model.
- **File**: `finetune.py` — `AksharDataset.__getitem__`

### Challenge 10: CUDA OOM — 58.89 GiB allocated on 31.84 GiB card
- **Symptom**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 GiB. GPU 0 has a total capacity of 31.84 GiB of which 0 bytes is free. Of the allocated memory 58.89 GiB is allocated by PyTorch`
- **Cause (1 — variable pixel_values)**: Gemma4 uses anyres/dynamic tiling — different images produce different numbers of tiles, giving `pixel_values` shapes like `[1, 3, 224, 224]` vs `[4, 3, 224, 224]`. When the DataLoader collates a batch of 16 such tensors it either pads to `[16, max_tiles, 3, 224, 224]` (massive) or errors. Memory accumulates until the card overflows.
- **Cause (2 — batch/seq too large)**: `per_device_train_batch_size=16` with `max_seq_length=2024` was too aggressive for a multimodal model even with 4-bit quant and gradient checkpointing.
- **Fix**:
  1. Resize all images to 224×224 before the processor call (`img.resize((224, 224), Image.LANCZOS)`) — forces `pixel_values` to always be `[3, 224, 224]` (consistent shape, no tiling)
  2. `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 16` (effective batch stays 32)
  3. `max_seq_length: 1024` (OCR labels are short; 2024 was wasteful)
  4. `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation
- **File**: `finetune.py` — `AksharDataset.__getitem__`; `finetune_config.yaml`

---

## GGUF Export (Post-Training)

```bash
# 1. Merge LoRA adapter into base model
python finetune.py --merge_only \
    --adapter_path ./checkpoints/final \
    --output_dir ./merged_model

# 2. Convert to GGUF (llama.cpp)
python convert_hf_to_gguf.py ./merged_model \
    --outtype bf16 --outfile akshar_gemma4_e4b.gguf

# 3. Quantize for mobile (Q4_K_M)
./quantize akshar_gemma4_e4b.gguf akshar_gemma4_e4b_q4km.gguf Q4_K_M
```
**Risk**: llama.cpp may not support `Gemma4ForConditionalGeneration` yet.
**Fallback**: Ollama Modelfile, or export text backbone only.

---

## Repo

GitHub: https://github.com/shashanka300/Akshar_gema_kagle.git  
Branch: master
