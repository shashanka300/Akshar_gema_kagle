"""
Akshar — Optimized for RTX 5090 & Unsloth
Fixed: MemoryError by using Hugging Face Datasets (Memory Mapping)
"""

import argparse
import base64
import logging
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict

import torch
import yaml
from PIL import Image
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("akshar")

@dataclass
class FinetuneConfig:
    model_path: str = "./gemma-4-E4B-it"
    data_dir: str = "./data"
    output_dir: str = "./checkpoints"
    lora_r: int = 32 # Increased for 5090
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    train_vision_encoder: bool = False
    num_epochs: int = 3
    per_device_train_batch_size: int = 16 # Optimized for 32GB VRAM
    gradient_accumulation_steps: int = 2  # Effective batch 32
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 250
    report_to: str = "none" # Use "wandb" if installed
    max_steps: Optional[int] = None  # for smoke test / CLI override

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        if not os.path.exists(path): return cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        return cfg

def formatting_prompts_func(examples, processor):
    """
    Decodes base64 images and applies templates on-the-fly.
    This prevents the MemoryError you saw earlier.
    """
    instructions = examples["messages"]
    images_b64 = examples["image_b64"]
    processed_images = []
    texts = []

    for msg, b64 in zip(instructions, images_b64):
        # Decode only what is needed for the current batch
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        processed_images.append(img)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    return {"text": texts, "image": processed_images}

# ── Main Training Loop ────────────────────────────────────────────────────────
def process_batch(batch, processor, max_seq_length):
    """
    Decodes base64 strings to images and tokenises text.
    `batch["user_text"]`  — the user prompt string (pre-extracted)
    `batch["model_text"]` — the model target string (pre-extracted)
    `batch["image_b64"]`  — base64 PNG
    """
    images = []
    for b64 in batch["image_b64"]:
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        img.load()
        images.append(img)

    # Re-construct messages from the pre-extracted flat strings
    message_lists = []
    for user_text, model_text in zip(batch["user_text"], batch["model_text"]):
        message_lists.append([
            {"role": "user",  "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "model", "content": model_text},
        ])

    # Process each sample individually — Gemma4 processor requires exactly
    # one image per call; batched multi-image calls raise ValueError
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for msg, img in zip(message_lists, images):
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        inp = processor(
            text=text,
            images=[img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        all_input_ids.append(inp["input_ids"][0].tolist())
        all_attention_mask.append(inp["attention_mask"][0].tolist())
        all_pixel_values.append(inp["pixel_values"][0].tolist())
        all_labels.append(inp["input_ids"][0].tolist())  # labels = input_ids (no masking yet)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "pixel_values":   all_pixel_values,
        "labels":         all_labels,
    }

def train(cfg: FinetuneConfig):
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model_path,
        max_seq_length = cfg.max_seq_length,
        load_in_4bit = True,
    )

    # 2. Add LoRA (Targeting 5090 Blackwell optimization)
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "embed_tokens", "lm_head"],
        lora_alpha = cfg.lora_alpha,
        use_gradient_checkpointing = "unsloth",
    )

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # 3. Load dataset — flatten nested messages to Arrow-safe scalar strings
    # Arrow can't handle mixed types in a list column (content is array in
    # user turn, string in model turn), so we pre-extract to flat fields.
    log.info(f"Loading {cfg.data_dir}/train.jsonl ...")

    import json as _json

    def flat_generator():
        """Yields Arrow-safe flat dicts: image_b64, user_text, model_text."""
        with open(os.path.join(cfg.data_dir, "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                msgs = rec["messages"]
                # user content is a list: [{type:image}, {type:text, text:...}]
                user_text  = next(c["text"] for c in msgs[0]["content"] if c["type"] == "text")
                model_text = msgs[1]["content"]
                yield {
                    "image_b64":  rec["image_b64"],
                    "user_text":  user_text,
                    "model_text": model_text,
                }

    from datasets import Dataset as HFDataset
    raw_dataset = HFDataset.from_generator(flat_generator)

    log.info(f"Loaded {len(raw_dataset)} records. Tokenising (map to disk cache)...")
    train_dataset = raw_dataset.map(
        lambda x: process_batch(x, processor, cfg.max_seq_length),
        batched=True,
        batch_size=8,
        remove_columns=raw_dataset.column_names,
        desc="Tokenising Akshar Data",
        writer_batch_size=200,
    )

    # 4. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        max_seq_length = cfg.max_seq_length,
        dataset_kwargs = {"skip_prepare_dataset": True},  # already tokenised
        args = TrainingArguments(
            per_device_train_batch_size = cfg.per_device_train_batch_size,
            gradient_accumulation_steps = cfg.gradient_accumulation_steps,
            max_steps = cfg.max_steps if cfg.max_steps is not None else -1,
            num_train_epochs = cfg.num_epochs if cfg.max_steps is None else 1,
            learning_rate = cfg.learning_rate,
            bf16 = True,
            logging_steps = cfg.logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            save_steps = cfg.save_steps,
            output_dir = cfg.output_dir,
            remove_unused_columns = False,
        ),
    )

    log.info("Handing over to RTX 5090. Happy training!")
    trainer.train()

# ── Updated CLI Logic ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akshar fine-tuning script")
    parser.add_argument("--config", default="finetune_config.yaml", help="Path to YAML config")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (smoke test)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()

    # 1. Start with defaults or YAML
    config = FinetuneConfig.from_yaml(args.config)

    # 2. Apply CLI Overrides
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size

    train(config)
