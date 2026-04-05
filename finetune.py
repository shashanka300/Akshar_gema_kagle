"""
finetune.py
============
Akshar — Fine-tune Gemma 4 E4B for Indic Handwriting OCR via QLoRA.

Framework: Unsloth (if available/compatible) → HF TRL + PEFT (fallback)
Task:      Transcription-only; [TRANSLATE] span masked from loss.

Usage:
  # Standard training run
  python finetune.py

  # Override config values
  python finetune.py --config finetune_config.yaml --max_steps 50  # smoke test

  # Merge LoRA adapter into full model for GGUF export
  python finetune.py --merge_only \\
      --adapter_path ./checkpoints/checkpoint-500 \\
      --output_dir ./merged_model
"""

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("akshar")

# ── Framework probe ───────────────────────────────────────────────────────────

USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel  # type: ignore
    USE_UNSLOTH = True
    log.info("Unsloth detected — will attempt to use FastLanguageModel")
except ImportError:
    log.info("Unsloth not installed — using HF TRL + PEFT")

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Paths
    model_path: str = "./gemma-4-E4B-it"
    data_dir: str = "./data"
    output_dir: str = "./checkpoints"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    train_vision_encoder: bool = False

    # 4-bit quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512
    dataloader_num_workers: int = 4

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    merged_save_every_n: int = 5

    # Sampling
    training_strategy: str = "balanced"
    script_weights: dict = field(default_factory=lambda: {
        "Kannada": 2.0, "Telugu": 2.0, "Bengali": 2.0,
        "Tamil": 2.0, "Malayalam": 2.0, "Gurumukhi": 2.0,
        "Odia": 2.0, "Gujarati": 1.5, "Urdu": 1.5, "Hindi": 1.0,
    })

    # Logging
    wandb_project: str = "akshar-indic-ocr"
    wandb_run_name: str = "gemma4-e4b-qlora"
    report_to: str = "wandb"

    # CLI-only overrides
    max_steps: Optional[int] = None  # stop early (smoke test)

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ── Metrics (mirrors benchmark_gemma4_indic_hw.py) ───────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def compute_metrics(gt: str, pred: str) -> dict:
    gt, pred = gt.strip(), pred.strip()
    d = _levenshtein(gt, pred)
    return {
        "cer": d / max(len(gt), 1),
        "wrr": 1.0 if gt == pred else 0.0,
        "ned": d / max(len(gt), len(pred), 1),
    }


def extract_transcription(raw: str) -> str:
    """Pull the label from 'Transcription: X\\nTranslation: ...' output."""
    for line in raw.splitlines():
        if line.startswith("Transcription:"):
            return line[len("Transcription:"):].strip()
    return raw.strip()


# ── Dataset ───────────────────────────────────────────────────────────────────

class IndicOCRDataset(Dataset):
    """
    Reads train.jsonl / val.jsonl.

    Each __getitem__ returns a dict ready for model forward():
      pixel_values  — processed image tensor  [C, H, W]
      input_ids     — full conversation token IDs
      attention_mask
      labels        — input_ids with -100 on user turn + [TRANSLATE] span
      script        — string (for per-script eval)
      label         — ground truth string
    """

    LOSS_IGNORE = -100

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 512,
        script_weights: Optional[dict] = None,
    ):
        self.processor    = processor
        self.max_length   = max_length

        # Find [TRANSLATE] token IDs once (it tokenizes to multiple tokens)
        self._translate_ids = processor.tokenizer.encode(
            "[TRANSLATE]", add_special_tokens=False
        )

        self.records: list[dict] = []
        self.sample_weights: list[float] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.records.append(rec)
                w = (script_weights or {}).get(rec.get("script", ""), 1.0)
                self.sample_weights.append(w)

        log.info(f"Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Decode image
        img_bytes = base64.b64decode(rec["image_b64"])
        image     = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Build inputs via processor
        # apply_chat_template expects the messages list; image passed separately
        text = self.processor.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids      = inputs["input_ids"].squeeze(0)          # [seq]
        attention_mask = inputs["attention_mask"].squeeze(0)     # [seq]
        pixel_values   = inputs["pixel_values"].squeeze(0)       # [C, H, W]

        labels = self._build_labels(input_ids, rec["messages"])

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "script":         rec.get("script", ""),
            "label":          rec.get("label", ""),
        }

    def _build_labels(self, input_ids: torch.Tensor, messages: list) -> torch.Tensor:
        """
        Mask everything except the model's transcription target.

        Strategy:
        1. Encode only the model turn target to find its token span.
        2. Locate the span in input_ids by scanning right-to-left.
        3. Set everything outside that span to -100.
        4. Also mask the [TRANSLATE] sub-sequence within the span.
        """
        labels = input_ids.clone()

        # Build the model-turn-only text so we can locate it
        model_content = messages[-1]["content"]  # "Transcription: X\nTranslation: [TRANSLATE]"
        model_ids = self.processor.tokenizer.encode(
            model_content, add_special_tokens=False
        )

        # Find the model turn span in input_ids (search from the right)
        span_start = self._find_subseq(input_ids.tolist(), model_ids)
        if span_start == -1:
            # Fallback: mask everything (safer than wrong labels)
            labels[:] = self.LOSS_IGNORE
            return labels

        # Mask everything before the model turn
        labels[:span_start] = self.LOSS_IGNORE

        # Mask [TRANSLATE] span within the model turn
        if self._translate_ids:
            t_start = self._find_subseq(
                input_ids[span_start:].tolist(), self._translate_ids
            )
            if t_start != -1:
                abs_start = span_start + t_start
                abs_end   = abs_start + len(self._translate_ids)
                labels[abs_start:abs_end] = self.LOSS_IGNORE

        # Mask padding positions
        labels[input_ids == self.processor.tokenizer.pad_token_id] = self.LOSS_IGNORE

        return labels

    @staticmethod
    def _find_subseq(seq: list[int], subseq: list[int]) -> int:
        """Return index of first occurrence of subseq in seq, or -1."""
        n, m = len(seq), len(subseq)
        if m == 0 or m > n:
            return -1
        for i in range(n - m + 1):
            if seq[i : i + m] == subseq:
                return i
        return -1


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Stack tensors; keep script/label lists for eval."""
    keys = ["pixel_values", "input_ids", "attention_mask", "labels"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["scripts"] = [b["script"] for b in batch]
    out["labels_text"] = [b["label"] for b in batch]
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(cfg: FinetuneConfig):
    """
    Load Gemma 4 E4B with 4-bit QLoRA.

    Vision tower is frozen (custom int8 storage format incompatible with PEFT).
    Vision-language bridge (embed_vision.embedding_projection) is kept trainable.
    LoRA targets language model attention + MLP layers only.
    """
    from transformers import AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    # ── Attempt Unsloth path ──────────────────────────────────────────────────
    if USE_UNSLOTH:
        try:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=cfg.model_path,
                max_seq_length=cfg.max_seq_length,
                load_in_4bit=cfg.load_in_4bit,
                dtype=compute_dtype,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules,
                bias="none",
            )
            log.info("Unsloth model loaded successfully.")
            _freeze_vision_tower(model, cfg)
            model.print_trainable_parameters()
            return model, processor
        except Exception as e:
            log.warning(f"Unsloth failed ({e}), falling back to HF TRL + PEFT.")

    # ── HF TRL + PEFT path ────────────────────────────────────────────────────
    from transformers import Gemma4ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    model = Gemma4ForConditionalGeneration.from_pretrained(
        cfg.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",  # flash_attn_2 may not support Gemma4 yet
    )

    model = prepare_model_for_kbit_training(model)

    _freeze_vision_tower(model, cfg)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor


def _freeze_vision_tower(model, cfg: FinetuneConfig):
    """Freeze vision encoder; keep bridge trainable."""
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            if not cfg.train_vision_encoder:
                param.requires_grad = False
        elif "embed_vision" in name:
            # Vision-language bridge — always train this
            param.requires_grad = True


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, processor, cfg: FinetuneConfig, step: int):
    ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    log.info(f"Saved adapter checkpoint → {ckpt_dir}")

    # Also save a merged (full bf16) model periodically for GGUF export
    n = cfg.save_steps * cfg.merged_save_every_n
    if step % n == 0:
        merged_dir = Path(cfg.output_dir) / f"merged-{step}"
        log.info(f"Saving merged model → {merged_dir} (this takes a few minutes)")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        del merged
        torch.cuda.empty_cache()
        log.info(f"Merged model saved → {merged_dir}")

    # Prune old checkpoints
    _prune_checkpoints(cfg)


def _prune_checkpoints(cfg: FinetuneConfig):
    ckpts = sorted(
        Path(cfg.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    while len(ckpts) > cfg.save_total_limit:
        oldest = ckpts.pop(0)
        import shutil
        shutil.rmtree(oldest, ignore_errors=True)
        log.info(f"Pruned old checkpoint: {oldest}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, val_dataset: IndicOCRDataset, device, n_samples: int = 200) -> dict:
    """
    Run greedy decoding on a sample of val set, compute WRR/CER per script.
    Returns overall and per-script metrics.
    """
    model.eval()

    indices = list(range(min(n_samples, len(val_dataset))))
    per_script: dict[str, list[float]] = {}
    all_wrr, all_cer = [], []

    for idx in indices:
        item = val_dataset[idx]
        gt_label = item["label"]
        script   = item["script"]

        pixel_values   = item["pixel_values"].unsqueeze(0).to(device)
        input_ids      = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)

        # Generate only the model turn (mask input, generate continuation)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=64,
            do_sample=False,
        )
        # Decode only the new tokens
        new_tokens = generated[0][input_ids.shape[1]:]
        raw_pred   = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_label = extract_transcription(raw_pred)

        m = compute_metrics(gt_label, pred_label)
        all_wrr.append(m["wrr"])
        all_cer.append(m["cer"])

        if script not in per_script:
            per_script[script] = {"wrr": [], "cer": []}
        per_script[script]["wrr"].append(m["wrr"])
        per_script[script]["cer"].append(m["cer"])

    results = {
        "overall_wrr": sum(all_wrr) / max(len(all_wrr), 1),
        "overall_cer": sum(all_cer) / max(len(all_cer), 1),
        "per_script":  {
            s: {
                "wrr": sum(v["wrr"]) / max(len(v["wrr"]), 1),
                "cer": sum(v["cer"]) / max(len(v["cer"]), 1),
            }
            for s, v in per_script.items()
        },
    }
    model.train()
    return results


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: FinetuneConfig):
    # Setup logging
    if cfg.report_to == "wandb":
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.__dict__)
        except ImportError:
            log.warning("wandb not installed — logging to stdout only")
            cfg.report_to = "none"

    model, processor = load_model_and_processor(cfg)
    device = next(p for p in model.parameters() if p.requires_grad).device

    # Datasets
    train_dataset = IndicOCRDataset(
        Path(cfg.data_dir) / "train.jsonl",
        processor,
        max_length=cfg.max_seq_length,
        script_weights=cfg.script_weights,
    )
    val_dataset = IndicOCRDataset(
        Path(cfg.data_dir) / "val.jsonl",
        processor,
        max_length=cfg.max_seq_length,
    )

    pad_id = processor.tokenizer.pad_token_id

    # Weighted sampler for script oversampling
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    from transformers import get_cosine_schedule_with_warmup
    total_steps   = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps  = int(total_steps * cfg.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler()

    log.info(f"Training: {total_steps} gradient steps, {warmup_steps} warmup")

    global_step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        log.info(f"=== Epoch {epoch + 1}/{cfg.num_epochs} ===")

        for batch_idx, batch in enumerate(train_loader):
            if cfg.max_steps and global_step >= cfg.max_steps:
                log.info(f"Reached max_steps={cfg.max_steps}, stopping.")
                break

            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    log.info(
                        f"step={global_step}  loss={loss.item() * cfg.gradient_accumulation_steps:.4f}"
                        f"  lr={lr:.2e}"
                    )
                    if cfg.report_to == "wandb":
                        import wandb
                        wandb.log({"train/loss": loss.item() * cfg.gradient_accumulation_steps,
                                   "train/lr": lr}, step=global_step)

                if global_step % cfg.eval_steps == 0:
                    log.info("Running evaluation...")
                    metrics = evaluate(model, processor, val_dataset, device)
                    log.info(
                        f"Val WRR={metrics['overall_wrr']:.3f}  CER={metrics['overall_cer']:.3f}"
                    )
                    for s, m in metrics["per_script"].items():
                        log.info(f"  {s}: WRR={m['wrr']:.3f}  CER={m['cer']:.3f}")
                    if cfg.report_to == "wandb":
                        import wandb
                        wandb.log({"val/wrr": metrics["overall_wrr"],
                                   "val/cer": metrics["overall_cer"]}, step=global_step)

                if global_step % cfg.save_steps == 0:
                    save_checkpoint(model, processor, cfg, global_step)

        if cfg.max_steps and global_step >= cfg.max_steps:
            break

    # Final checkpoint
    save_checkpoint(model, processor, cfg, global_step)
    log.info("Training complete.")

    if cfg.report_to == "wandb":
        import wandb
        wandb.finish()


# ── Merge-only mode (for GGUF export) ────────────────────────────────────────

def merge_adapter(adapter_path: str, output_dir: str):
    """
    Load a LoRA adapter checkpoint and merge weights into the base model.
    The merged safetensors output is ready for llama.cpp GGUF conversion.
    """
    from transformers import AutoProcessor
    from peft import PeftModel, PeftConfig

    log.info(f"Loading adapter from {adapter_path}")
    peft_cfg = PeftConfig.from_pretrained(adapter_path)

    from transformers import Gemma4ForConditionalGeneration
    base = Gemma4ForConditionalGeneration.from_pretrained(
        peft_cfg.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # CPU merge to avoid VRAM spike
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    log.info("Merging LoRA weights...")
    merged = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(out)

    log.info(f"Merged model saved → {out}")
    log.info("Next: convert to GGUF with llama.cpp/convert_hf_to_gguf.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Akshar fine-tuning script")
    parser.add_argument("--config", default="finetune_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N gradient steps (smoke test)")
    parser.add_argument("--merge_only", action="store_true",
                        help="Merge LoRA adapter and exit (no training)")
    parser.add_argument("--adapter_path", type=str,
                        help="Adapter checkpoint path (required with --merge_only)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output_dir from config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge_only:
        if not args.adapter_path:
            print("ERROR: --adapter_path is required with --merge_only", file=sys.stderr)
            sys.exit(1)
        out = args.output_dir or "./merged_model"
        merge_adapter(args.adapter_path, out)
        sys.exit(0)

    cfg = FinetuneConfig.from_yaml(args.config) if Path(args.config).exists() else FinetuneConfig()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.output_dir:
        cfg.output_dir = args.output_dir

    train(cfg)
