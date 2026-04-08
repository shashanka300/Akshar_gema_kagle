"""
Akshar — fine-tuning for Gemma 4 E4B on Indic handwriting OCR.

Uses Unsloth's FastVisionModel (multimodal) + UnslothVisionDataCollator so that:
  - The real vision tower is loaded and LoRA-adapted (not just the text tower).
  - Loss masking is driven by the actual Gemma 4 chat template tokens
    (<|turn>user\\n / <|turn>model\\n), not the Gemma 2/3 markers.
  - Images are processed at the model's native resolution (fixed 280 soft tokens).

Dataset is streamed from JSONL via byte-offset index — no Arrow cache.
"""

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import yaml
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("akshar")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Paths
    model_path: str = "./gemma-4-E4B-it"
    data_dir: str = "./data"
    output_dir: str = "./checkpoints"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    # Quantisation (Unsloth handles the BitsAndBytesConfig internally)
    load_in_4bit: bool = True

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.001

    # Checkpointing / logging
    logging_steps: int = 1
    save_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 0  # >0 deadlocks on Windows with file-seeking datasets
    max_steps: Optional[int] = None   # smoke-test / CLI override

    # Eval
    eval_steps: int = 500
    eval_batch_size: int = 2
    max_eval_samples: Optional[int] = 512

    # Logging backend
    report_to: str = "wandb"
    wandb_entity: Optional[str] = "silverhack300"
    wandb_project: str = "akshar-indic-ocr"
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        if not os.path.exists(path):
            log.warning(f"No YAML at {path}; using dataclass defaults.")
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = []
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
            else:
                unknown.append(k)
        if unknown:
            log.warning(
                f"Ignoring unknown YAML keys (not on FinetuneConfig): {unknown}"
            )
        return cfg


# ── Dataset ───────────────────────────────────────────────────────────────────

class AksharDataset(TorchDataset):
    """
    Streaming dataset backed by a JSONL index file.

    Supports two JSONL formats:
      - Path-reference (new, ~10 MB): {"script", "label", "image_path"}
      - Base64-embedded (legacy, ~65 GB): {"script", "label", "image_b64", "messages"}

    At __init__ we scan the file once and record the byte offset of each line
    (~8 bytes per record). At __getitem__ we seek, read the record, load the
    image from disk (or decode base64), and return a plain dict for the
    UnslothVisionDataCollator.
    """

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.jsonl_path = jsonl_path

        log.info(f"Indexing {jsonl_path} ...")
        self.offsets: list[int] = []
        with open(jsonl_path, "rb") as f:
            pos = 0
            for raw_line in f:
                if raw_line.strip():
                    self.offsets.append(pos)
                    if max_samples and len(self.offsets) >= max_samples:
                        break
                pos += len(raw_line)
        log.info(f"Indexed {len(self.offsets):,} records.")

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_record(self, idx: int) -> dict:
        with open(self.jsonl_path, "rb") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline())

    USER_PROMPT = (
        "Read the handwritten word in this image and output ONLY the word "
        "in its original script. Do not add any explanation or translation."
    )

    def __getitem__(self, idx: int) -> dict:
        rec = self._read_record(idx)

        # ── Load image ──
        if "image_path" in rec:
            # New path-reference format: load directly from disk
            img = Image.open(rec["image_path"]).convert("RGB")
            img.load()
        elif "image_b64" in rec:
            # Legacy base64 format: decode from JSONL
            img = Image.open(BytesIO(base64.b64decode(rec["image_b64"]))).convert("RGB")
            img.load()
        else:
            raise ValueError(f"Record {idx} has neither image_path nor image_b64")

        # Cap image size to prevent Gemma4 anyres tiling from creating
        # massive tensors (Challenge 10). Max 512px on longest side.
        max_side = 512
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # ── Extract label ──
        label = rec.get("label", "")
        if not label:
            asst = rec.get("messages", [{}])[1].get("content", "")
            if isinstance(asst, list):
                label = next(
                    (c["text"] for c in asst if isinstance(c, dict) and c.get("type") == "text"),
                    "",
                )
            else:
                label = str(asst)

        # Image BEFORE text — required by Gemma 4 / Unsloth docs
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": self.USER_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label},
                    ],
                },
            ],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def _init_wandb(cfg: FinetuneConfig) -> None:
    if cfg.report_to != "wandb":
        return
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; setting report_to='none'.")
        cfg.report_to = "none"
        return
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config={
            "model_path": cfg.model_path,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_seq_length": cfg.max_seq_length,
        },
    )


def train(cfg: FinetuneConfig) -> None:
    _init_wandb(cfg)

    # 1. Load the multimodal model + processor via Unsloth's vision entry point.
    log.info(f"Loading Gemma 4 from {cfg.model_path} (4bit={cfg.load_in_4bit})")
    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.model_path,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # 2. Attach LoRA adapters for parameter efficient fine-tuning.
    #    Matches reference notebook: target_modules="all-linear" hits both
    #    vision and language towers; gradient checkpointing is set above.
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    # 3. Apply the Gemma 4 chat template so the processor/tokenizer knows
    #    the correct turn markers (<|turn>user / <|turn>model). Without
    #    this, tokenisation produces garbage.
    processor = get_chat_template(processor, "gemma-4")

    # 4. Datasets — prefer new path-reference index files, fall back to legacy base64.
    train_index = os.path.join(cfg.data_dir, "train_index.jsonl")
    val_index   = os.path.join(cfg.data_dir, "val_index.jsonl")
    train_legacy = os.path.join(cfg.data_dir, "train.jsonl")
    val_legacy   = os.path.join(cfg.data_dir, "val.jsonl")

    train_path = train_index if os.path.exists(train_index) else train_legacy
    val_path   = val_index   if os.path.exists(val_index)   else val_legacy

    log.info(f"Train data: {train_path}")
    train_dataset = AksharDataset(train_path)

    eval_dataset = None
    if os.path.exists(val_path):
        eval_dataset = AksharDataset(val_path, max_samples=cfg.max_eval_samples)
        log.info(f"Eval dataset: {len(eval_dataset):,} samples (capped)")
    else:
        log.warning("No validation JSONL found; training without eval.")

    # 5. Vision data collator — matches reference notebook: just (model, processor).
    #    Loss masking is applied separately via train_on_responses_only on the trainer.
    collator = UnslothVisionDataCollator(model, processor)

    # 6. Trainer setup. Use SFTConfig (the TrainingArguments superset TRL wants).
    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=5,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=0.3,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        optim="adamw_8bit",
        seed=3407,
        dataloader_num_workers=cfg.dataloader_num_workers,

        # Required for vision fine-tuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg.max_seq_length,
    )
    if cfg.max_steps is not None and cfg.max_steps > 0:
        sft_kwargs["max_steps"] = cfg.max_steps
    if eval_dataset is not None:
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = cfg.eval_steps
        sft_kwargs["per_device_eval_batch_size"] = cfg.eval_batch_size

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_kwargs),
    )

    # Note: train_on_responses_only is for TEXT fine-tuning only (needs .map()).
    # For VISION fine-tuning, UnslothVisionDataCollator handles loss masking internally.

    log.info("Handing over to RTX 5090. Happy training!")
    trainer.train()

    if cfg.report_to == "wandb":
        import wandb
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akshar fine-tuning script")
    parser.add_argument("--config", default="finetune_config.yaml", help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (smoke test)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name override")

    args = parser.parse_args()

    cfg = FinetuneConfig.from_yaml(args.config)

    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.run_name is not None:
        cfg.wandb_run_name = args.run_name

    train(cfg)
