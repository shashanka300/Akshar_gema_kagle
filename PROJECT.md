# Akshar — Project Status & Reference

## Goal
Fine-tune Gemma 4 E4B (multimodal) to transcribe handwritten Indic text AND translate it to English in one prompt. Deploy fully offline via Cactus SDK in a React Native app.

## Hardware
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CUDA: 13.0 (Blackwell / cu130)
- OS: Windows 11

## Pipeline

```
1. Dataset prep     ✅ COMPLETE — path-reference JSONL index (~10 MB)
2. Fine-tuning      🔄 READY — finetune.py + finetune_notebook.ipynb fully fixed
3. Export to GGUF   ⏳ After training
4. Cactus inference ⏳ Load GGUF on-device
5. React Native app ⏳ Camera → OCR + translation, fully offline
```

---

## Current Status

### ✅ Completed
- Zero-shot benchmark: avg ~17% WRR (Hindi 49%, Kannada 3.6%)
- `dataset_prep.py` rewritten: path-reference JSONL (10 MB vs old 72 GB base64)
- Real curated validation set used (10k images from `validation/validationset/`)
- `finetune.py` fully fixed against Unsloth reference notebook + docs
- `get_chat_template(processor, "gemma-4")` applied (was missing — #1 bug)
- `processing_class = processor.tokenizer` (was passing full `processor`)
- `train_on_responses_only(trainer, ...)` applied correctly after trainer creation
- `UnslothVisionDataCollator(model, processor)` — simple form, no extra kwargs
- `get_peft_model` cleaned: removed invalid `use_gradient_checkpointing`/`max_seq_length`, added `target_modules="all-linear"`, `bias="none"`, `random_state=3407`
- LoRA r/alpha bumped to 32/32 (was 16/16)
- `max_grad_norm=0.3`, `seed=3407`, `weight_decay=0.001` added to SFTConfig
- Image-before-text order enforced in all data + inference code
- `AksharDataset` supports both path-reference (new) and base64 (legacy) JSONL
- Step-by-step Jupyter notebooks created for both data prep and training

### 🔄 Next — Run the pipeline
```bash
# 1. Generate index files (minutes, not hours)
jupyter notebook data_prep_notebook.ipynb
# Or: python -X utf8 dataset_prep.py --source local --output_dir ./data

# 2. Smoke test (50 steps)
jupyter notebook finetune_notebook.ipynb
# Or: $env:PYTORCH_ALLOC_CONF="expandable_segments:True"
#     python -X utf8 -B finetune.py --max_steps 50

# 3. Full training
# Set max_steps = None in notebook cell 1, or:
python -X utf8 -B finetune.py
```

### ⏳ Pending
1. Confirm smoke test passes (loss starts ~13-15 then decreases, VRAM < 28 GB)
2. Full training (3 epochs, ~4-8 hrs): `python -X utf8 finetune.py`
3. Post-training eval: `python benchmark_gemma4_indic_hw.py --run --model ./checkpoints/final --samples 100`
4. GGUF export + Cactus SDK integration

---

## Dataset

### On-disk structure
```
data/indic_hw/
├── train/                              (783k images across 10 scripts)
│   ├── devanagari/
│   │   ├── train.txt                   "<relative_path> <label>" per line
│   │   └── train/                      nested image dirs
│   ├── kannada/
│   │   ├── train.txt
│   │   └── train/                      flat image files
│   └── ... (bengali, gujarati, gurumukhi, malayalam, odia, tamil, telugu, urdu)
│
└── validation/validationset/           (10k curated images, 1000/script)
    ├── devanagari/
    │   ├── val.txt                     uses "test/0.jpg" prefix (remapped to val/)
    │   └── val/                        1000 images
    └── ... (10 scripts)
```

### Generated index files
```
data/train_index.jsonl     ~10 MB     path-reference (NEW)
data/val_index.jsonl       ~1 MB      curated val set (NEW)
data/train.jsonl           ~65 GB     base64-embedded (LEGACY — can delete)
data/val.jsonl             ~6.5 GB    split from train (LEGACY — can delete)
data/stats.json
```

### Index JSONL schema (new — lightweight)
```json
{
  "script": "Hindi",
  "label": "केंद्रों",
  "image_path": "E:/akshar_benchmark/data/indic_hw/train/devanagari/train/8/251/21.jpg"
}
```

### Baseline per script

| Script     | Folder        | WRR baseline | On-disk images | Curated val |
|------------|---------------|--------------|----------------|-------------|
| Hindi      | devanagari    | 49.0%        | 69,853         | 1,000       |
| Kannada    | kannada       | 3.6%         | 73,517         | 1,000       |
| Telugu     | telugu        | 4.8%         | 88,534         | 1,000       |
| Bengali    | bengali       | 4.0%         | 82,554         | 1,000       |
| Tamil      | tamil         | 4.4%         | 75,736         | 1,000       |
| Malayalam  | malayalam     | 2.2%         | 85,270         | 1,000       |
| Gujarati   | gujarati      | 15.8%        | 82,563         | 1,000       |
| Urdu       | urdu          | 17.6%        | 71,207         | 1,000       |
| Odia       | odia          | 2.2%         | 73,400         | 1,000       |
| Gurumukhi  | gurumukhi     | 3.6%         | 81,042         | 1,000       |
| **Total**  |               |              | **783,676**    | **10,000**  |

---

## Training Config (finetune_config.yaml)

| Param | Value | Note |
|-------|-------|------|
| LoRA r / alpha | 32 / 32 | Recommended alpha == r (Unsloth docs) |
| LoRA dropout | 0.0 | Unsloth recommendation |
| target_modules | `"all-linear"` | Hits both vision + language towers |
| bias | `"none"` | Per reference notebook |
| Quantization | 4-bit NF4 | ~6 GB model footprint |
| Batch x grad_accum | 2 x 16 = 32 effective | Tuned for 32 GB VRAM |
| Learning rate | 2e-4 cosine | AdamW 8-bit |
| max_grad_norm | 0.3 | Prevents unstable large gradient steps |
| weight_decay | 0.001 | Per Unsloth docs |
| seed | 3407 | Reproducibility |
| Epochs | 3 | ~4-8 hrs on RTX 5090 |
| Max seq length | 1024 | prompt ~30 + image 280 soft tokens + label |
| gradient_checkpointing | `"unsloth"` | Set in `from_pretrained` |
| Chat template | `"gemma-4"` | Applied via `get_chat_template(processor, "gemma-4")` |
| Loss masking | `train_on_responses_only(trainer, ...)` | `instruction_part="<\|turn>user\n"`, `response_part="<\|turn>model\n"` |

---

## Key Files

| File | Purpose |
|------|---------|
| `data_prep_notebook.ipynb` | Step-by-step data prep with visual inspection (11 code cells) |
| `finetune_notebook.ipynb` | Step-by-step training with before/after inference (15 code cells) |
| `dataset_prep.py` | CLI: IHTR images → path-reference JSONL index |
| `finetune.py` | CLI: Unsloth + SFTTrainer training script |
| `finetune_config.yaml` | All hyperparameters |
| `gemma4_(e4b)_vision.py` | Unsloth reference notebook (LaTeX OCR example) |
| `verify_dataset.py` | JSONL sanity check (run with `-X utf8` on Windows) |
| `benchmark_gemma4_indic_hw.py` | Zero-shot eval harness |
| `prepare_cvit_data.py` | Helper: normalize CVIT downloads |
| `data/train_index.jsonl` | Training index (path references, ~10 MB) |
| `data/val_index.jsonl` | Validation index (curated set, ~1 MB) |
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
- **Cause**: HF `Dataset.map()` writes a full Arrow cache of tokenised data to `~/.cache/huggingface/datasets/`. With 180k samples x 1024 tokens each, the cache grew to 97 GB and exhausted the disk.
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
  1. Resize all images to 224x224 before the processor call (`img.resize((224, 224), Image.LANCZOS)`) — forces `pixel_values` to always be `[3, 224, 224]` (consistent shape, no tiling)
  2. `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 16` (effective batch stays 32)
  3. `max_seq_length: 1024` (OCR labels are short; 2024 was wasteful)
  4. `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation
- **File**: `finetune.py` — `AksharDataset.__getitem__`; `finetune_config.yaml`

### Challenge 11: Vision training data had image/text order reversed + wrong inference prompt
- **Symptom**: Model trains on `[text, image]` order but infers with `[image, text]` — train/inference mismatch degrades quality
- **Cause (1 — content order)**: `convert_to_conversation()` in `gemma4_(e4b)_vision.py` placed `{"type": "text"}` before `{"type": "image"}` in the user content list. Unsloth docs explicitly require: *"Put image and/or audio before text."*
- **Cause (2 — wrong prompt)**: The second inference block (post-LoRA load) used `sample["text"]` (the LaTeX ground-truth answer) as the user prompt instead of the instruction string, effectively feeding the answer as the question.
- **Cause (3 — docstring)**: The format example in the docstring showed two `"role": "user"` entries instead of `user` + `assistant`.
- **Fix**: Swapped to `[image, text]` order in `convert_to_conversation`, fixed the inference prompt to use `instruction`, corrected the docstring.
- **File**: `gemma4_(e4b)_vision.py`

### Challenge 12: finetune.py had 9 bugs vs Unsloth reference notebook / docs
- **Symptom**: Training would either produce garbage models or fail silently
- **Root causes & fixes** (all in `finetune.py` and `finetune_config.yaml`):

| Bug | Was | Fixed to |
|-----|-----|----------|
| No `get_chat_template` call | Tokenizer didn't know `<\|turn>user`/`<\|turn>model` markers | `processor = get_chat_template(processor, "gemma-4")` |
| `processing_class=processor` | Full processor object | `processor.tokenizer` |
| Loss masking kwargs in collator | `UnslothVisionDataCollator(..., train_on_responses_only=True, ...)` — undocumented, likely ignored | `UnslothVisionDataCollator(model, processor)` + separate `train_on_responses_only(trainer, ...)` |
| Invalid `get_peft_model` kwargs | `use_gradient_checkpointing`, `max_seq_length` passed | Removed; added `target_modules="all-linear"`, `bias="none"`, `random_state=3407` |
| LoRA r=16, alpha=16 | Too small for 32 GB VRAM | r=32, alpha=32 |
| No `max_grad_norm` | Defaulted to 1.0 | 0.3 |
| No `seed` | Non-reproducible | 3407 |
| `weight_decay=0.0` | Docs recommend 0.001 | 0.001 |
| Import path wrong | `from unsloth import UnslothVisionDataCollator` | `from unsloth.trainer import UnslothVisionDataCollator` |

### Challenge 13: 72 GB base64 JSONL + wrong validation split
- **Symptom**: `dataset_prep.py` took hours, produced 65 GB train.jsonl + 6.5 GB val.jsonl; 10k curated validation images on disk were never used; 20k training samples wasted as val split
- **Cause (1 — base64 embedding)**: Every image was PNG-encoded to base64 and embedded in the JSONL. With 180k images this created a 65 GB file.
- **Cause (2 — no val loader)**: `dataset_prep.py` only loaded from `train/` and carved off the last 10% as validation. The actual curated validation set at `validation/validationset/` (1000 images/script) was completely ignored.
- **Cause (3 — val.txt path mismatch)**: The validation `val.txt` files reference images as `test/0.jpg` but the actual images live in `val/0.jpg`.
- **Fix**:
  1. Switched to path-reference JSONL: `{"script", "label", "image_path"}` — ~10 MB total
  2. Added `load_val_local()` that reads from `validation/validationset/<script>/`
  3. Remaps `test/` prefix to `val/` when resolving image paths
  4. `AksharDataset` updated to support both `image_path` (new) and `image_b64` (legacy)
  5. `finetune.py` auto-detects: prefers `train_index.jsonl`/`val_index.jsonl`, falls back to legacy
- **Files**: `dataset_prep.py` (rewritten), `finetune.py` (AksharDataset + path selection), `data_prep_notebook.ipynb` (new)

---

## Gemma 4 / Unsloth — Key Learnings

Reference: https://unsloth.ai/docs/models/gemma-4 and https://unsloth.ai/docs/models/gemma-4/train

### Data Format Rules
- **Image/audio before text**: Always place `{"type": "image"}` before `{"type": "text"}` in the user content list. This applies to both training data and inference prompts.
- **`<bos>` token removal**: When using `apply_chat_template` for text fine-tuning, call `.removeprefix('<bos>')` — the processor adds it automatically during training.
- **Vision data collator**: Must use `UnslothVisionDataCollator(model, processor)` and set `remove_unused_columns = False`, `dataset_text_field = ""`, `dataset_kwargs = {"skip_prepare_dataset": True}` in `SFTConfig`.

### Chat Templates
- `"gemma-4"` — use for E2B, E4B (smaller models)
- `"gemma-4-thinking"` — use for 26B, 31B (preserves reasoning/thinking ability)

### Critical: `get_chat_template` is mandatory
```python
from unsloth import get_chat_template
processor = get_chat_template(processor, "gemma-4")
```
Without this, the tokenizer doesn't know the `<|turn>user` / `<|turn>model` markers and all tokenisation is garbage. This must be called AFTER `from_pretrained` and `get_peft_model`, BEFORE creating the trainer.

### Loss Masking (response-only training)
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part = "<|turn>model\n",
)
```
Must be called AFTER creating the `SFTTrainer`, not passed as kwargs to the collator.

### SFTTrainer setup for vision
```python
from unsloth.trainer import UnslothVisionDataCollator  # not from unsloth directly

trainer = SFTTrainer(
    model=model,
    processing_class=processor.tokenizer,               # .tokenizer, NOT processor
    data_collator=UnslothVisionDataCollator(model, processor),  # simple form
    train_dataset=train_dataset,
    args=SFTConfig(
        remove_unused_columns=False,                     # REQUIRED
        dataset_text_field="",                           # REQUIRED
        dataset_kwargs={"skip_prepare_dataset": True},   # REQUIRED
        max_length=1024,                                 # REQUIRED
        ...
    ),
)
```

### `get_peft_model` — valid vs invalid params
```python
# VALID:
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32, lora_alpha=32, lora_dropout=0,
    bias="none", random_state=3407,
    use_rslora=False, loftq_config=None,
    target_modules="all-linear",
)

# INVALID (these belong in from_pretrained, not get_peft_model):
# use_gradient_checkpointing, max_seq_length
```

### Expected Loss Values
- E2B and E4B multimodal models typically show **13-15 initial loss** — this is normal for these architectures, not a bug.

### Inference Parameters (Google recommended)
- `temperature = 1.0`, `top_p = 0.95`, `top_k = 64`
- End-of-sentence token: `<turn|>`
- Start with 32K context for responsiveness, increase as needed.
- Multi-turn: only keep the final visible answer in chat history — do not feed prior thought blocks back into the next turn.

### Visual Token Budgets
Allowed values: 70, 140, 280, 560, 1120

### Audio
- E2B/E4B only, max 30 seconds
- Format: `{"type": "audio", "audio": "/path/to/audio.wav"}`

### VRAM Requirements (LoRA, 4-bit)
| Model | 4-bit | 8-bit | BF16 |
|-------|-------|-------|------|
| E2B | 4 GB | 5-8 GB | 10 GB |
| E4B | 5.5-6 GB | 9-12 GB | 16 GB |
| 26B-A4B | 16-18 GB | 28-30 GB | 52 GB |
| 31B | 17-20 GB | 34-38 GB | 62 GB |

### GGUF Export (Unsloth native)
```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")
# Supported: "q4_k_m", "q8_0", "f16"
```

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
