# Akshar Benchmark — Code Audit vs. Gemma 4 E4B

> Strict review of this repo against the local `./gemma-4-E4B-it` weights and
> the Gemma 4 model card. Evidence is taken from the local `config.json`,
> `chat_template.jinja`, `tokenizer_config.json`, `processor_config.json` and
> the project source — not from third-party claims.

## Ground truth about the model (from local files)

- `gemma-4-E4B-it/config.json:3` — `architectures: ["Gemma4ForConditionalGeneration"]`.
  This is a **multimodal conditional-generation model**, not a CausalLM.
- `gemma-4-E4B-it/config.json:58,121,177` — `model_type: gemma4`, with
  `gemma4_text`, `gemma4_vision`, `gemma4_audio` sub-configs.
- `gemma-4-E4B-it/processor_config.json:35,49` — processor class is
  `Gemma4Processor`, image processor is `Gemma4ImageProcessor`;
  `image_seq_length: 280` (every image = exactly 280 soft tokens).
- `gemma-4-E4B-it/tokenizer_config.json` — **real** turn delimiters:
  - `sot_token = "<|turn>"`  (start of turn)
  - `eot_token = "<turn|>"`  (end of turn)
  - `image_token = "<|image|>"`
  - `padding_side = "left"`

  There is **no** `<start_of_turn>` / `<end_of_turn>` token anywhere in
  Gemma 4. Those are **Gemma 2/3** markers.
- `gemma-4-E4B-it/chat_template.jinja:180,187,261` — the template emits
  `<|turn>role\n … <turn|>\n`, and the image placeholder is
  `\n\n<|image|>\n\n`.

## Severity summary

**Must-fix for training to produce meaningful updates:**
#1 loss-mask boundary, #2 FastLanguageModel on multimodal, #4 dataset schema
mismatch, #5 `[TRANSLATE]` literal, #11 224×224 resize, #6 YAML silently
dropped.

**Must-fix for benchmark to run at all:** #3 `AutoModelForMultimodalLM`
import error, #29 benchmark prompt vs training target mismatch.

**Strongly recommended:** #8 no eval set, #12 `padding="max_length"`,
#21 LoRA on `embed_tokens`/`lm_head` with tied embeddings, #7 script
balancing advertised but not implemented, #24 `dataloader_num_workers=0`,
#22 no LR warmup/scheduler.

**Housekeeping:** #9 `--max_steps` dead flag, #23 no `save_total_limit`,
#33/34/35 dependency hygiene, #26/40 config drift.

---

## Critical correctness bugs

### 1. Loss-mask boundary string is wrong — labels are never masked
`finetune.py:142-143`:
```python
model_header_ids = self.processor.tokenizer.encode(
    "<start_of_turn>model\n", add_special_tokens=False
)
```
That string does not exist in the Gemma 4 tokenizer. With
`add_special_tokens=False`, it gets split into ordinary BPE pieces
(`<`, `start`, `_`, `of`, `turn`, `>`, `model`, …) that can never match the
actual sequence emitted by the chat template (which contains the `<|turn>`
*special* token). The backwards search at `finetune.py:147` therefore never
finds the boundary, `span_start` stays `-1`, and the fallback at `:156`
prints a warning and **skips masking entirely**.

Consequence: every sample teaches the model to reproduce the system/user
prompt, the 280 image-soft-tokens, *and* the target. The loss signal is
dominated by text the model already knows; only a tiny fraction comes from
the transcription you care about.

Correct boundary is `"<|turn>model\n"`, and it needs to be tokenised in a
way that emits the special token (or label-build the two halves separately
and concat).

### 2. `finetune.py` loads a multimodal model through `FastLanguageModel`
`finetune.py:189-193` uses `unsloth.FastLanguageModel.from_pretrained(...)`.
`FastLanguageModel` is Unsloth's text-only entry point (`*ForCausalLM`). The
local weights are `Gemma4ForConditionalGeneration` with a 150M-param vision
tower and 300M-param audio tower. Either this raises, or it silently loads
only the text tower, in which case training "works" but the model isn't
seeing any pixels.

Correct path: `unsloth.FastVisionModel` (or the unified `FastModel`).

### 3. `benchmark_gemma4_indic_hw.py` imports a class that doesn't exist
`benchmark_gemma4_indic_hw.py:319`:
```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
```
There is no `AutoModelForMultimodalLM` in transformers. The moment anyone
runs `--run`, this import fails. Correct class: `AutoModelForImageTextToText`
(or load `Gemma4ForConditionalGeneration` directly).

### 4. Training data schema is inconsistent with the reader
`dataset_prep.py:140-152` writes:
```python
"messages": [
    {"role": "user",  "content": f"<image>\n{USER_PROMPT}"},   # STRING
    {"role": "model", "content": build_target(label)},
]
```
but `finetune.py:111` and `verify_dataset.py:80` iterate
`messages[0]["content"]` expecting a **list of dicts** with `type`/`text`
keys:
```python
user_text = next(c["text"] for c in msgs[0]["content"] if c["type"] == "text")
```
Iterating a string yields characters; `c["type"]` on a character raises.
The existing JSONL on disk must have been produced by an earlier version
with list format. Regenerating now would crash training immediately. Pick
one schema and align all three files.

### 5. `[TRANSLATE]` placeholder is trained as a literal label
`dataset_prep.py:104-110` builds targets as:
```python
f"Transcription: {label}\nTranslation: [TRANSLATE]"
```
with a comment claiming `[TRANSLATE]` is "masked from loss". Nothing in
`finetune.py` masks sub-spans inside the model turn — and because of #1,
nothing is masked at all. The model is therefore trained to output the
literal six-character string `[TRANSLATE]` after every transcription. After
fine-tuning, every prediction will contain `[TRANSLATE]` verbatim, and the
benchmark's `build_ocr_prompt` (which asks for "ONLY the word") will be
fighting a model that learned the opposite.

Fix: drop the `Translation:` line entirely at training time, or supply real
English translations, or actually replace `[TRANSLATE]`'s token IDs with -100.

### 6. YAML config is silently ignored for every key not on the dataclass
`finetune.py:53-63` `from_yaml` only copies keys that `hasattr(cfg, k)`.
Everything the YAML claims to configure that isn't a dataclass field is
**dropped without warning**:

- `target_modules`, `train_vision_encoder`
- `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant`
- `dataloader_num_workers`, `eval_steps`
- `training_strategy`, `script_weights`
- `wandb_project`, `wandb_run_name`

Several dataclass fields also disagree with the YAML, so what actually runs
depends on which wins:

| Key | YAML | Dataclass default |
|---|---|---|
| `lora_r` | 16 | 32 |
| `lora_alpha` | 16 | 64 |
| `per_device_train_batch_size` | 1 | 16 |
| `gradient_accumulation_steps` | 32 | 2 |
| `max_seq_length` | 512 | 1024 |
| `dataloader_num_workers` | 4 | hardcoded 0 in `TrainingArguments` |

PROJECT.md describes a third combination (`batch=2`, `grad_accum=16`,
`seq=1024`) that matches neither.

### 7. Script-balanced sampling exists only in YAML
`finetune_config.yaml:47-58` declares `training_strategy: balanced` and
per-script oversampling weights. No code anywhere reads or applies them.
`train.jsonl` is written script-by-script in `dataset_prep.py:345`.
HF `Trainer` shuffles per epoch so this is "merely unbalanced", but the
weighted sampler the YAML advertises does not exist.

### 8. No validation set is used
`finetune.py:216-240` constructs `SFTTrainer` with only `train_dataset`.
`val.jsonl` is never loaded; there's no `eval_dataset`, no
`evaluation_strategy`, no `eval_steps` in `TrainingArguments`. The YAML's
`eval_steps: 250` is dead config. Training is blind.

### 9. `--max_steps` flag is wired to nothing
`finetune.py:252,259-260` parses `--max_steps` and sets `cfg.max_steps`, but
the `TrainingArguments(...)` call at `:225` never passes it. Smoke-test
override has been dead since it was added.

### 10. `SFTTrainer` should get `processing_class=processor`, not `tokenizer`
`finetune.py:218` passes `tokenizer=tokenizer`. Recent TRL uses
`processing_class=` for vision trainers; passing just a `tokenizer` hides
the image processor from the trainer — which is why `skip_prepare_dataset=True`
is needed as a workaround.

---

## Data-pipeline issues

### 11. Hard-resize to 224×224 destroys OCR signal
`finetune.py:119` forces every image to 224×224 via LANCZOS. But:

- `processor_config.json:36,42,48` says Gemma 4 emits **exactly 280 soft
  tokens per image**, fixed. `image_seq_length = 280`. There's no variable
  tiling in the default E4B image path.
- Gemma 4's vision encoder was pretrained at higher native resolution
  (`patch_size 16`, `default_output_length 280`,
  `position_embedding_size 10240`, `config.json:183-185`). Downscaling
  Bengali/Kannada handwriting to 224×224 throws away diacritics and stroke
  features that are the entire point of this benchmark.
- The real "58 GB spike" was almost certainly caused by
  `padding="max_length"` × `max_length=1024` × `batch_size=16`, not tiling.

Let the processor do its native resize; solve variance at the collator.

### 12. Padding to `max_length` on every sample wastes ~95% of compute
`finetune.py:130-132` sets `padding="max_length"`, `max_length=1024`. OCR
targets are ~5–10 tokens, prompt ~30, image 280 (fixed). Every sample is
~320 real tokens padded out to 1024. ~3× the FLOPs necessary.

### 13. Tokenizer pads left, but training expects right-padded inputs
`tokenizer_config.json` sets `padding_side: "left"`. Left padding is correct
for generation, not training. Combined with `padding="max_length"` and
`labels[input_ids == pad_id] = -100`, it's technically correct but fragile.
Set `processor.tokenizer.padding_side = "right"` in the dataset.

### 14. BOS token may be double-inserted
`chat_template.jinja:155` emits `{{ bos_token }}` as the first token. Then
`finetune.py:125-133` calls `apply_chat_template(..., tokenize=False)` and
feeds the resulting string back through `processor(text=text, ...)`. Whether
BOS is duplicated depends on Gemma4Processor's default
`add_special_tokens`. `dataset_kwargs={"add_special_tokens": False}` is set
for SFTTrainer's pretokenization, but `skip_prepare_dataset=True` means that
kwarg is never applied. Pass `add_special_tokens=False` to the processor
call at `:126`.

### 15. Dataset shipped as base64-in-JSONL
`dataset_prep.py:113-116,139` inflates every image to PNG-base64 inside the
JSONL. 65 GB for 180k images — ~3× what raw PNGs on disk would be.
`AksharDataset` opens and base64-decodes per sample anyway; image-paths in
JSONL + images on disk would be faster, smaller, worker-friendlier.

### 16. `SYSTEM_PROMPT` defined and never used
`dataset_prep.py:88-95` defines a system prompt that is never attached to
any message. Gemma 4's template *does* support `system` turns. Using one is
where you'd pin output format cleanly, removing the need for `[TRANSLATE]`
hacks.

### 17. `_count_label_lines` over-estimates for HF source
`dataset_prep.py:284-287` returns `max_per_script` as the total when
`source="hf"`. If HF returns fewer samples than `max_per_script`, val split
is empty. Minor while you use `local`, but flag it.

### 18. `load_script_local` swallows every exception
`dataset_prep.py:226-227` catches every exception with `pass`. Bad decode,
wrong-mode image, permission error, disk failure all look identical. At
minimum log exception type + count.

### 19. Train/val split is sequential
`dataset_prep.py:374-379` writes the first 90% of the label file to train
and the last 10% to val. If the source has any ordering (time, writer,
region) the val set is systematically biased. Use a hash of the image path.

### 20. IHTR's real held-out val set isn't used
`benchmark_gemma4_indic_hw.py:172-186` already downloads the official IHTR
validation set into `data/indic_hw/validation/…`, but `dataset_prep.py`
only reads from `train/` and carves out its own 10%. Real evaluation should
fine-tune on `train/` and eval on `val/`.

---

## Training-config issues

### 21. LoRA on `embed_tokens` + `lm_head` is wasteful here
Per PROJECT.md and `finetune.py:199-204`, LoRA adapts `embed_tokens` and
`lm_head`. Gemma 4's vocab is 262 144 × 2 560 — each matrix is ~670 M
parameters, so LoRA-adapting them blows the adapter to the 2.8 GB seen in
`checkpoints/checkpoint-50/adapter_model.safetensors`. For handwriting OCR
on a tokenizer that already covers all 10 Indic scripts natively, this buys
nothing. Also: `tie_word_embeddings: true` in `config.json:150` — adapting
them independently is mathematically redundant.

### 22. No LR scheduler or warmup
`TrainingArguments` in `finetune.py:225-239` doesn't set `lr_scheduler_type`
or `warmup_*`. Default is linear decay, 0 warmup — aggressive at 2e-4 on a
4-bit base. PROJECT.md claims cosine; code doesn't make it cosine.

### 23. `save_steps=250` with no `save_total_limit`
3 epochs × 180 k / effective batch 32 ≈ 17 000 steps ÷ 250 = **68
checkpoints**. Each LoRA checkpoint is ~2.8 GB → **~190 GB** of adapter
files. Set `save_total_limit=3` and raise `save_steps`.

### 24. `dataloader_num_workers=0` hardcoded
`finetune.py:238` hardcodes `0`, ignoring the YAML's `4`. Per-sample base64
decode + PIL decode + resize on the main process is a serialised bottleneck
in front of a 5090.

### 25. `max_steps` never forwarded to `TrainingArguments`
(See #9.)

### 26. W&B entity/project hardcoded
`finetune.py:176-186` hardcodes `entity="silverhack300"`, `project="gemma"`,
ignoring `wandb_project` / `wandb_run_name` from the YAML.

### 27. `.env` contains an exposed W&B API key
Gitignored, so it won't end up on GitHub, but prefer `wandb login` (stores
token in `~/.netrc` with proper permissions).

---

## Benchmark-script issues

### 28. `AutoModelForMultimodalLM` import
(See #3.)

### 29. Benchmark prompt vs training target format disagree
Training teaches the model to emit
`Transcription: <word>\nTranslation: [TRANSLATE]`.
Benchmark prompt at `benchmark_gemma4_indic_hw.py:358-365` says:

> output ONLY the word in its original script.

After fine-tuning the model ignores this and emits the training format.
`compute_metrics` at `:125-132` strips whitespace but doesn't parse out the
`Transcription: …` prefix, so every metric is penalised by prefix length.
Align the target, or post-parse `Transcription: (.+)` in the benchmark.

### 30. `wer = 1.0 if not exact match` is not a true WER
`benchmark_gemma4_indic_hw.py:130` computes wer as 0/1, then `print_summary`
labels it "WRR". Defensible for single-word OCR, but don't call it WER in
logs; call it `EM` / `accuracy`.

### 31. `enable_thinking=False` with `do_sample=False` is fine, but worth
trying thinking mode as an ablation — Gemma 4's chain-of-thought can
resolve ambiguous glyphs before committing.

### 32. `processor.parse_response` is undocumented for pre-5.5 transformers
`benchmark_gemma4_indic_hw.py:398` calls a Gemma-4-specific method. If
`transformers` is downgraded it silently crashes. Pin transformers.

---

## Dependency / environment issues

### 33. `unsloth` listed twice in `pyproject.toml`
```toml
"unsloth>=2024.1; sys_platform != 'win32'",
"unsloth>=2026.4.2",
```
Use a single entry.

### 34. `transformers>=4.50.0` is too low
Local model reports `"transformers_version": "5.5.0.dev0"` and uses modules
(`Gemma4ForConditionalGeneration`, `Gemma4Processor`, `parse_response`)
that only exist on main. Pin to the dev version.

### 35. `torch` is installed from cu130 index but not version-pinned
Combined with the hard dependency on a cu130-specific PyTorch build, anybody
re-running `uv sync` later will get a different torch. Pin exactly.

### 36. `unsloth_compiled_cache/` holds stale compiled kernels across
`transformers` upgrades. Add to cleanup checklist.

---

## Minor / cosmetic

### 37. `levenshtein_distance` is **not** recursive
On inspection, `benchmark_gemma4_indic_hw.py:111-122` is a proper iterative
two-row O(n·m) implementation. No action.

### 38. `verify_dataset.py:73` counts blank lines
Uses `i < n_samples` where `i` is the raw line index including empty lines.
Use a separate counter.

### 39. `dataset_prep.py` `format_sample` docstring is copy-paste folklore
References "Some versions of SFTTrainer prefer a single string with an
`<image>` placeholder". This is Gemma 2/3 advice; Gemma 4's own template
(`chat_template.jinja:233-243`) walks a **list** of content parts. The
misleading comment directly caused #4.

### 40. PROJECT.md drift
PROJECT.md describes "batch=2, grad_accum=16, seq=1024" which matches
neither the YAML nor the dataclass. Pick one source of truth.
