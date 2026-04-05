"""
dataset_prep.py
================
Akshar — Dataset preparation for Gemma 4 E4B fine-tuning.

Loads IIIT-INDIC-HW-WORDS per script from HuggingFace,
formats each sample into a Gemma 4 vision-language conversation,
and saves a unified train/val split as JSONL.

Dataset sources (c3rl org on HuggingFace):
  c3rl/IIIT-INDIC-HW-WORDS-Hindi       ~70k train
  c3rl/IIIT-INDIC-HW-WORDS-Kannada     ~70k train  ← priority (0% baseline)
  c3rl/IIIT-INDIC-HW-WORDS-Bengali     ~70k train
  c3rl/IIIT-INDIC-HW-WORDS-Tamil       ~70k train
  ... (add more scripts as needed)

Each sample schema after formatting:
  {
    "image": <PIL Image>,          # handwritten word crop
    "label": "ಕರ್ನಾಟಕ",           # ground truth Indic text
    "script": "Kannada",
    "messages": [                  # Gemma 4 chat format
      { "role": "user",    "content": [ {type: image}, {type: text} ] },
      { "role": "model",   "content": "Transcription: ಕರ್ನಾಟಕ\nTranslation: Karnataka" }
    ]
  }

Usage:
  python dataset_prep.py --scripts Kannada Bengali Tamil --output_dir ./data
"""

import os
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from datasets import load_dataset, concatenate_datasets, Dataset
from PIL import Image
from tqdm import tqdm

# ── Script config ────────────────────────────────────────────────────────────

# HuggingFace dataset IDs — ordered by training priority
# Priority: Kannada (0% baseline) → Telugu → Hindi → Bengali → Tamil → rest
# All 10 scripts are included; order controls which script's samples appear
# first in the JSONL (matters for curriculum-style training).
SCRIPT_DATASETS = {
    "Kannada":    "c3rl/IIIT-INDIC-HW-WORDS-Kannada",    # 0% WRR baseline — top priority
    "Telugu":     "c3rl/IIIT-INDIC-HW-WORDS-Telugu",     # 4% WRR baseline
    "Hindi":      "c3rl/IIIT-INDIC-HW-WORDS-Hindi",      # 52% WRR — anchor script
    "Bengali":    "c3rl/IIIT-INDIC-HW-WORDS-Bengali",    # 4% WRR baseline
    "Tamil":      "c3rl/IIIT-INDIC-HW-WORDS-Tamil",      # 4% WRR baseline
    "Malayalam":  "c3rl/IIIT-INDIC-HW-WORDS-Malayalam",  # 4% WRR baseline
    "Gujarati":   "c3rl/IIIT-INDIC-HW-WORDS-Gujarati",   # 14% WRR baseline
    "Urdu":       "c3rl/IIIT-INDIC-HW-WORDS-Urdu",       # 10% WRR baseline
    "Odia":       "c3rl/IIIT-INDIC-HW-WORDS-Odia",       # 4% WRR baseline
    "Gurumukhi":  "c3rl/IIIT-INDIC-HW-WORDS-Gurumukhi",  # 2% WRR baseline (Punjabi)
}

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Indic script OCR and translation assistant. "
    "When given an image of handwritten text, you first transcribe it exactly "
    "as written, then translate it to English. "
    "Always respond in this exact format:\n"
    "Transcription: <exact text in original script>\n"
    "Translation: <English translation>"
)

USER_PROMPT = (
    "Look at this handwritten word image carefully. "
    "Transcribe the handwritten text exactly as written in its original script, "
    "then provide the English translation."
)

def build_target(label: str, script: str) -> str:
    """
    For fine-tuning we need a ground-truth target string.
    Translation is approximated using the label itself for now —
    in production, pre-compute translations using IndicTrans2.
    
    For the hackathon, we teach the model the transcription task;
    translation emerges from Gemma 4's multilingual pretraining.
    We mark translation as [TRANSLATE] so it's not penalized during training.
    """
    return f"Transcription: {label}\nTranslation: [TRANSLATE]"


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for storage in JSONL."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def format_sample(
    image: Image.Image,
    label: str,
    script: str,
) -> dict:
    """
    Format a single dataset sample into Gemma 4 chat structure.
    
    Gemma 4 multimodal chat format:
      - user turn: [image_token, text_prompt]
      - model turn: target string
    
    Image is stored as base64 in JSONL for portability.
    During training, this is decoded back to PIL and passed to the processor.
    """
    target = build_target(label, script)

    return {
        "script": script,
        "label": label,
        "image_b64": image_to_base64(image),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},           # placeholder; image passed separately
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {
                "role": "model",
                "content": target,
            },
        ],
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_script_dataset(script: str, split: str = "train") -> Optional[Dataset]:
    """Load a single script's dataset from HuggingFace."""
    hf_id = SCRIPT_DATASETS.get(script)
    if not hf_id:
        print(f"[WARN] No dataset registered for script: {script}")
        return None

    print(f"[INFO] Loading {script} ({split}) from {hf_id} ...")
    try:
        ds = load_dataset(hf_id, split=split)
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load {script}: {e}")
        return None


def process_and_save(
    scripts: list[str],
    output_dir: str,
    val_ratio: float = 0.1,
    max_per_script: Optional[int] = None,
):
    """
    Load, format, and save all scripts as JSONL files.
    
    Output files:
      {output_dir}/train.jsonl
      {output_dir}/val.jsonl
      {output_dir}/stats.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"
    stats      = {}

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path,   "w", encoding="utf-8") as f_val:

        for script in scripts:
            print(f"\n{'='*50}")
            print(f"Processing: {script}")
            print(f"{'='*50}")

            ds = load_script_dataset(script, split="train")
            if ds is None:
                continue

            # Optionally cap samples per script (useful for fast iteration)
            if max_per_script and len(ds) > max_per_script:
                ds = ds.select(range(max_per_script))
                print(f"[INFO] Capped at {max_per_script} samples")

            n_total = len(ds)
            n_val   = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val

            print(f"[INFO] {n_train} train / {n_val} val samples")

            train_count = 0
            val_count   = 0
            error_count = 0

            for idx, sample in enumerate(tqdm(ds, desc=f"{script}")):
                try:
                    image = sample["image"]
                    label = sample["text"].strip()

                    # Skip empty labels
                    if not label:
                        continue

                    # Ensure image is RGB
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    formatted = format_sample(image, label, script)
                    line = json.dumps(formatted, ensure_ascii=False)

                    if idx < n_train:
                        f_train.write(line + "\n")
                        train_count += 1
                    else:
                        f_val.write(line + "\n")
                        val_count += 1

                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # only log first 5 errors
                        print(f"[WARN] Error at index {idx}: {e}")

            stats[script] = {
                "train": train_count,
                "val":   val_count,
                "errors": error_count,
            }
            print(f"[OK] {script}: {train_count} train, {val_count} val, {error_count} errors")

    # Save stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset prep complete!")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Stats: {stats_path}")

    total_train = sum(v["train"] for v in stats.values())
    total_val   = sum(v["val"]   for v in stats.values())
    print(f"  Total: {total_train} train / {total_val} val")
    print(f"{'='*50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Akshar dataset preparation")
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=list(SCRIPT_DATASETS.keys()),  # all 10 by default, in priority order
        choices=list(SCRIPT_DATASETS.keys()),
        help="Which scripts to include (default: all 10 in priority order)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Where to save train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of each script to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--max_per_script",
        type=int,
        default=None,
        help="Cap samples per script (useful for quick smoke tests, e.g. 1000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_save(
        scripts=args.scripts,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_per_script=args.max_per_script,
    )