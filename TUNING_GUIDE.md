# Akshar — Gemma 4 E4B Fine-tuning Diagnostic & Tuning Guide

**Date**: 2026-04-08
**Model**: Gemma 4 E4B (4-bit QLoRA via Unsloth)
**Task**: Indic Handwritten OCR (~775K images, ~10 scripts)
**Hardware**: RTX 5090 (32 GB VRAM), Windows 11
**Notebook**: `finetune_notebook.ipynb`
**Reference**: [Unsloth Gemma 4 Training Docs](https://unsloth.ai/docs/models/gemma-4/train) | [Reference Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb)

---

## Why the Model Didn't Learn (Root Cause Analysis)

### 1. PRIMARY: `max_steps=50` — catastrophically too few steps

The dataset has **775,650 training samples** with effective batch size 32. One full epoch = ~24,239 steps.
Training ran for only **50 steps** = **0.2% of one epoch**. The model barely saw any data.

The reference notebook's `max_steps=60` is a tiny demo value on a small LaTeX OCR dataset. It explicitly
states: *"We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run."*

The model **did show signs of learning**: train loss fell to 2.42, val loss to 0.36. It just hadn't seen enough
data to produce correct words.

**Fix applied**: Set `max_steps = None`, let `num_epochs=3` control training.

### 2. `max_seq_length=1024` vs reference's `2048`

The reference uses `max_length=2048`. With vision tokens + prompt tokens, 1024 may truncate sequences,
meaning the model never sees the label during training for longer examples.

**Fix applied**: Changed to `max_seq_length = 2048`.

### 3. `per_device_train_batch_size=2` vs reference's `1`

Batch size 2 per device with large variable-size images causes more VRAM pressure. The reference uses 1.

**Fix applied**: Changed to `per_device_train_batch_size = 1` (effective batch = 16 with grad_accum=16).

### 4. Inference uses `temperature=1.0` — bad for OCR

For OCR evaluation, you want deterministic output. `temperature=1.0` introduces randomness, making
evaluation unreliable.

**Fix applied**: Changed to `do_sample=False` (greedy decoding).

### 5. Things that were already correct

| Aspect                     | Value                                       | Status |
|----------------------------|---------------------------------------------|--------|
| Chat template              | `"gemma-4"`                                 | OK     |
| LoRA r/alpha               | 32/32                                       | OK     |
| `processing_class`         | `processor.tokenizer`                       | OK     |
| Data collator              | `UnslothVisionDataCollator(model, processor)` | OK   |
| Required SFTConfig flags   | All 4 set                                   | OK     |
| Optimizer                  | `adamw_8bit`                                | OK     |
| Learning rate              | `2e-4`                                      | OK     |
| `max_grad_norm`            | `0.3`                                       | OK     |
| Image-before-text order    | Yes                                         | OK     |

---

## If Things Still Don't Improve — Tuning Playbook

### Tier 1 — Quick wins (change a number, re-run)

**Cell `f45dfb55` (Config defaults)**

| Parameter                      | Current  | Try              | Why |
|--------------------------------|----------|------------------|-----|
| `learning_rate`                | `2e-4`   | `5e-5` or `1e-4` | 2e-4 may be too aggressive for a 4.5B model. If loss oscillates/spikes, lower it |
| `num_epochs`                   | `3`      | `1` first        | Start with 1 epoch to see if loss decreases at all. Saves hours |
| `warmup_ratio`                 | `0.03`   | `0.10`           | Longer warmup stabilises early training |
| `lora_r` / `lora_alpha`       | `32/32`  | `64/64`          | Higher rank = more capacity to adapt. VRAM headroom exists (only 41% used) |
| `gradient_accumulation_steps`  | `16`     | `8`              | Effective batch 8 = noisier but more frequent weight updates |
| `weight_decay`                 | `0.001`  | `0.01`           | If overfitting later (train loss low, val loss high) |

**Cell `b45cad69` (Trainer / SFTConfig)**

| Parameter      | Current | Try                                     | Why |
|----------------|---------|-----------------------------------------|-----|
| `warmup_steps` | `5`     | **Remove**, use `warmup_ratio` instead  | Currently hardcoded `warmup_steps=5` overrides `warmup_ratio`. With 145K total steps, `warmup_ratio=0.03` = ~4,350 warmup steps — much better than 5. **This is a bug.** |
| `save_steps`   | `500`   | `1000` or `2000`                        | Fewer checkpoint writes = faster training |
| `eval_steps`   | `500`   | `2000`                                  | Eval on 512 samples every 500 steps is expensive |

---

### Tier 2 — LoRA architecture changes

**Cell `e05b988e` (LoRA adapters)**

| Parameter                | Current        | Try          | Why |
|--------------------------|----------------|--------------|-----|
| `finetune_vision_layers` | `True`         | `False`      | Unsloth docs recommend starting with vision OFF. Language model may learn OCR output faster with frozen vision encoder |
| `lora_dropout`           | `0.0`          | `0.05`       | Light regularisation if overfitting |
| `target_modules`         | `"all-linear"` | Remove (default) | Adapting fewer layers = faster, less instability risk |

---

### Tier 3 — Data pipeline changes

**Cell `335d18c2` (AksharDataset)**

| Parameter       | Current                      | Try                              | Why |
|-----------------|------------------------------|----------------------------------|-----|
| Image `max_side`| `512`                        | `256` or `384`                   | Smaller images = fewer vision tokens = faster training, fits in `max_length=2048` more safely |
| `USER_PROMPT`   | Long sentence (45 words)     | `"OCR:"`                         | Shorter prompt = fewer tokens wasted on instruction, more room for label |
| Add subset mode | Full 775K                    | `max_samples=50000` on train set | Quick iteration: if model learns on 50K, it'll learn on 775K. Saves hours per experiment |

---

### Tier 4 — Deeper / higher-risk changes

**Cell `fb20b8a0` (Chat template)**

| Try                                | Why |
|------------------------------------|-----|
| `"gemma-4-thinking"` instead of `"gemma-4"` | Different template format. Unlikely to help but worth trying if everything else fails |

**Cell `533203b9` (Model loading)**

| Parameter      | Current | Try                    | Why |
|----------------|---------|------------------------|-----|
| `load_in_4bit` | `True`  | `False` (16-bit LoRA)  | 4-bit quantisation can cause gradient instabilities. You have 32GB VRAM — 16-bit LoRA may fit |

**New diagnostic cell (add after cell `b45cad69`)**

Gradient norm check to diagnose vanishing/exploding gradients:

```python
# Quick gradient check — run after 1 training step
FastVisionModel.for_training(model)
batch = collator([train_dataset[0]])
batch = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
loss = model(**batch).loss
loss.backward()
total_norm = sum(
    p.grad.norm().item()**2
    for p in model.parameters() if p.grad is not None
)**0.5
print(f"Loss: {loss.item():.4f}  Grad norm: {total_norm:.2e}")
model.zero_grad()
```

- If grad norm is **< 1e-6**: vanishing gradients. Try unfreezing more layers or higher LR.
- If grad norm is **> 1e+3**: exploding gradients. Lower LR, reduce `max_grad_norm`.
- **Healthy range**: 0.01 — 10.0

---

## Recommended Experiment Order

```
Step 1: Fix warmup_steps=5 bug in cell b45cad69
        (replace with warmup_ratio from config)

Step 2: Run 1 epoch, check if loss drops steadily
        - If YES -> continue to full 3 epochs
        - If NO  -> go to Step 3

Step 3: Try learning_rate=5e-5
        - If loss now drops -> use this LR
        - If still flat     -> go to Step 4

Step 4: Try learning_rate=1e-5 + max_grad_norm=0.1
        - If loss now drops -> use these
        - If still flat     -> go to Step 5

Step 5: Set finetune_vision_layers=False in cell e05b988e
        - Freeze vision encoder, only train language LoRA
        - If loss now drops -> vision layers were destabilising
        - If still flat     -> go to Step 6

Step 6: Switch to load_in_4bit=False (16-bit) in cell 533203b9
        - Rules out quantisation instability
        - If loss now drops -> 4-bit was the problem
        - If still flat     -> go to Step 7

Step 7: NUCLEAR OPTION — overfit test
        - Set max_samples=100 on train dataset
        - Train for 500 steps
        - If model CAN overfit 100 samples -> data format is fine,
          problem is capacity/hyperparams at scale
        - If model CANNOT overfit 100 samples -> data format or
          pipeline is broken. Audit the collator output manually.
```

---

## Key Metrics to Watch

| Metric | Healthy | Problem |
|--------|---------|---------|
| Train loss | Steadily decreasing | Flat = underfitting; oscillating = LR too high; NaN = numerical issue |
| Val loss | Decreasing, slightly above train | Rising while train drops = overfitting |
| Grad norm | 0.01 — 10.0 | < 1e-6 = vanishing; > 1e+3 = exploding |
| VRAM usage | < 90% of 32GB | OOM = reduce batch size or image size |
| Predictions | Progressively closer to labels | All identical outputs = mode collapse |

---

## Cell Reference Map

| Cell ID      | Step | Contents |
|--------------|------|----------|
| `5f68bc44`   | 0    | Environment check |
| `f45dfb55`   | 1    | **Config** — LR, epochs, batch size, LoRA rank, max_steps |
| `533203b9`   | 2    | **Model loading** — 4-bit vs 16-bit |
| `e05b988e`   | 3    | **LoRA adapters** — vision/language layers, rank, dropout |
| `fb20b8a0`   | 4    | **Chat template** — `"gemma-4"` |
| `335d18c2`   | 5    | **Dataset** — image loading, prompt, max_side |
| `ce4e107e`   | 7    | **Pre-training inference** — baseline check |
| `b45cad69`   | 8    | **Trainer/SFTConfig** — optimizer, scheduler, warmup, grad norm |
| `c56c5c34`   | 10   | **Training** — `trainer.train()` |
| `3c158d77`   | 12   | **Post-training inference** — evaluation |

---

## Pause & Resume Training

### How to Pause

Just **stop/interrupt the notebook cell** (click the stop button or press `Ctrl+C`).
The last saved checkpoint will be in `./checkpoints/checkpoint-XXXX/`.

Checkpoints are saved every `save_steps` steps (default 500). Each checkpoint contains everything
needed to resume:

```
checkpoints/
  checkpoint-500/
  checkpoint-1000/
  checkpoint-1500/           <-- example: latest
    adapter_model.safetensors   # LoRA weights
    optimizer.pt                # optimizer state (Adam momenta)
    scheduler.pt                # LR scheduler position
    rng_state.pth               # random number generator state
    trainer_state.json          # step count, full loss history
    training_args.bin           # training config snapshot
```

### How to Resume

Change **cell `c56c5c34`** (the training cell) from:

```python
trainer_stats = trainer.train()
```

to:

```python
trainer_stats = trainer.train(resume_from_checkpoint=True)
```

The Trainer will automatically:
1. Find the latest checkpoint in `output_dir` (`./checkpoints/`)
2. Restore LoRA weights, optimizer state, LR scheduler position, and step count
3. Continue training from exactly where it stopped

### Resume from a specific checkpoint

```python
trainer_stats = trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-1500")
```

### Important notes

- **Re-run cells 0-9 first** before resuming — the model, processor, dataset, and trainer must be
  recreated in memory. Only the training cell (cell 10) changes.
- **Checkpoints accumulate** — `save_total_limit=3` keeps only the 3 most recent. Increase this
  if you want to keep more rollback points.
- **Changing hyperparameters mid-run**: You can change LR, batch size, etc. in the config/trainer
  cells before resuming. The optimizer state will be restored but the new config will apply going
  forward. Note: changing batch size or grad_accum changes the total step count, which may affect
  the LR scheduler curve.

---

## Speed Optimisation

### Training time estimates (775K samples, RTX 5090)

| Config | Steps (1 epoch) | Est. time (1 epoch) | Est. time (3 epochs) |
|--------|-----------------|---------------------|----------------------|
| batch=1, accum=16, eff=16, img=512 (original) | ~48,478 | ~55 hrs | ~164 hrs (6.8 days) |
| batch=3, accum=8, eff=24, img=384 | ~32,318 | ~24 hrs | ~71 hrs (3.0 days) |
| batch=4, accum=8, eff=32, img=384 | ~24,239 | ~24 hrs | ~71 hrs (3.0 days) |
| **batch=4, accum=8, eff=32, img=256** | **~24,239** | **~13 hrs** | **~39 hrs (1.6 days)** |
| batch=6, accum=8, eff=48, img=256 | ~16,159 | ~13 hrs | ~39 hrs (1.6 days) |

### Recommended fast config

| Lever | Current | Recommended | Cell to edit | Impact |
|-------|---------|-------------|-------------|--------|
| `per_device_train_batch_size` | `1` | `4` | `f45dfb55` | Better GPU utilization (41% -> ~65%) |
| `gradient_accumulation_steps` | `16` | `8` | `f45dfb55` | Effective batch = 32 |
| Image `max_side` | `512` | `256` | `335d18c2` | Fewer vision tokens (biggest speed win) |
| `num_epochs` | `3` | `1` first | `f45dfb55` | Validate learning before committing to 3 |
| `eval_steps` | `500` | `2000` | `f45dfb55` | Less time spent evaluating |
| `save_steps` | `500` | `2000` | `f45dfb55` | Less I/O for checkpoints |

> **Note on image size 256 vs 512**: For handwritten *words* (not full pages), 256px is usually
> enough detail. If accuracy suffers after training, bump back to 384.

### Strategy: fast iteration

1. Train 1 epoch with the fast config (~13 hrs)
2. Check if loss is dropping and predictions improve
3. If yes, resume for epochs 2-3
4. If no, tweak hyperparameters (see Tuning Playbook above) and re-run 1 epoch

---

## References

- [Unsloth Gemma 4 Training Docs](https://unsloth.ai/docs/models/gemma-4/train)
- [Reference Notebook (Colab)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [IIIT-INDIC-HW-WORDS Dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words)
- Executive Summary (diagnostic analysis): `C:\Users\shash\Downloads\Executive Summary.pdf`
