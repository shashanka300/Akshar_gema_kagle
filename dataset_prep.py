"""
dataset_prep.py
================
Akshar — Dataset preparation for Gemma 4 E4B fine-tuning.

Loads IIIT-INDIC-HW-WORDS per script from local IHTR data or HuggingFace,
formats each sample into a Gemma 4 vision-language conversation,
and saves a unified train/val split as JSONL.

Local data layout (default, --source local):
  data/indic_hw/train/<folder>/train.txt   — "<relative_path> <label>" per line
  data/indic_hw/train/<folder>/train/      — image files

HuggingFace datasets (--source hf):
  c3rl/IIIT-INDIC-HW-WORDS-{Script}

Each output JSONL record:
  {
    "script": "Kannada",
    "label": "ಕರ್ನಾಟಕ",
    "image_b64": "<base64 PNG>",
    "messages": [
      { "role": "user",  "content": [ {type: image}, {type: text} ] },
      { "role": "model", "content": "Transcription: ಕರ್ನಾಟಕ\\nTranslation: [TRANSLATE]" }
    ]
  }

Usage:
  # Medium run: 20k/script → ~180k train + 20k val
  python dataset_prep.py --source local --max_per_script 20000 --output_dir ./data

  # Smoke test: 100/script
  python dataset_prep.py --source local --max_per_script 100 --output_dir ./data

  # Specific scripts only
  python dataset_prep.py --source local --scripts Kannada Telugu --max_per_script 5000
"""

import os
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Iterator

from PIL import Image
from tqdm import tqdm

# ── Script config ────────────────────────────────────────────────────────────

# HuggingFace dataset IDs — ordered by training priority
SCRIPT_DATASETS = {
    "Kannada":    "c3rl/IIIT-INDIC-HW-WORDS-Kannada",
    "Telugu":     "c3rl/IIIT-INDIC-HW-WORDS-Telugu",
    "Hindi":      "c3rl/IIIT-INDIC-HW-WORDS-Hindi",
    "Bengali":    "c3rl/IIIT-INDIC-HW-WORDS-Bengali",
    "Tamil":      "c3rl/IIIT-INDIC-HW-WORDS-Tamil",
    "Malayalam":  "c3rl/IIIT-INDIC-HW-WORDS-Malayalam",
    "Gujarati":   "c3rl/IIIT-INDIC-HW-WORDS-Gujarati",
    "Urdu":       "c3rl/IIIT-INDIC-HW-WORDS-Urdu",
    "Odia":       "c3rl/IIIT-INDIC-HW-WORDS-Odia",
    "Gurumukhi":  "c3rl/IIIT-INDIC-HW-WORDS-Gurumukhi",
}

# Mapping from script name → on-disk IHTR folder name
IHTR_LOCAL_MAP = {
    "Hindi":      "devanagari",
    "Bengali":    "bengali",
    "Gujarati":   "gujarati",
    "Gurumukhi":  "gurumukhi",
    "Kannada":    "kannada",
    "Malayalam":  "malayalam",
    "Odia":       "odia",
    "Tamil":      "tamil",
    "Telugu":     "telugu",
    "Urdu":       "urdu",
}

# Image quality thresholds — filter out noise/degenerate crops
MIN_WIDTH  = 20    # pixels
MIN_HEIGHT = 20    # pixels
MIN_AREA   = 800   # width × height pixels²
MAX_ASPECT = 15.0  # width / height ratio

# ── Prompt templates ──────────────────────────────────────────────────────────

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


def build_target(label: str) -> str:
    """
    Training target for the model turn.
    Translation is marked [TRANSLATE] — masked from loss during fine-tuning.
    Gemma 4's multilingual pretraining handles translation at inference.
    """
    return f"Transcription: {label}\nTranslation: [TRANSLATE]"


def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def passes_quality_filter(img: Image.Image) -> bool:
    w, h = img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    if w * h < MIN_AREA:
        return False
    if h > 0 and (w / h) > MAX_ASPECT:
        return False
    return True


def format_sample(image: Image.Image, label: str, script: str) -> dict:
    """Format a single sample into the Gemma 4 chat JSONL schema."""
    return {
        "script": script,
        "label": label,
        "image_b64": image_to_base64(image),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {
                "role": "model",
                "content": build_target(label),
            },
        ],
    }


# ── Local IHTR loader ─────────────────────────────────────────────────────────

def load_script_local(
    script: str,
    data_dir: str,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[Image.Image, str]]:
    """
    Load samples from locally extracted IHTR training data.

    Layout:
      <data_dir>/train/<folder>/train.txt   — "<img_relative_path> <label>"
      <data_dir>/train/<folder>/<img_relative_path>

    Yields (PIL.Image RGB, label) tuples that pass the quality filter.
    """
    folder = IHTR_LOCAL_MAP.get(script)
    if not folder:
        print(f"[WARN] No local folder mapping for script: {script}")
        return

    script_dir = Path(data_dir) / "train" / folder
    label_file = script_dir / "train.txt"

    if not label_file.exists():
        print(f"[ERROR] Label file not found: {label_file}")
        return

    print(f"[INFO] Loading {script} from {script_dir} ...")

    filtered = 0
    yielded  = 0

    with open(label_file, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if max_samples and yielded >= max_samples:
            break

        line = line.strip()
        if not line:
            continue

        # Format: "<relative_img_path> <label>"  (split on first space)
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        rel_path, label = parts
        label = label.strip()
        if not label:
            continue

        img_path = script_dir / rel_path
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path)
            img.load()  # force pixel data into RAM — releases the file handle
            if img.mode != "RGB":
                img = img.convert("RGB")

            if not passes_quality_filter(img):
                filtered += 1
                continue

            yield img, label
            yielded += 1

        except Exception:
            pass  # skip corrupt images silently

    if filtered:
        print(f"[INFO] {script}: filtered {filtered} low-quality images")


# ── HuggingFace loader ────────────────────────────────────────────────────────

def load_script_hf(
    script: str,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[Image.Image, str]]:
    """Load samples from HuggingFace c3rl datasets."""
    from datasets import load_dataset

    hf_id = SCRIPT_DATASETS.get(script)
    if not hf_id:
        print(f"[WARN] No HF dataset registered for script: {script}")
        return

    print(f"[INFO] Loading {script} from {hf_id} ...")
    try:
        ds = load_dataset(hf_id, split="train")
    except Exception as e:
        print(f"[ERROR] Failed to load {script} from HF: {e}")
        return

    for idx, sample in enumerate(ds):
        if max_samples and idx >= max_samples:
            break
        try:
            img   = sample["image"]
            label = sample["text"].strip()
            if not label:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            if not passes_quality_filter(img):
                continue
            yield img, label
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_label_lines(
    script: str,
    source: str,
    data_dir: str,
    max_per_script: Optional[int],
) -> int:
    """
    Quickly count how many valid (parseable, image-exists) lines are in the
    label file — without opening any images. Used to compute the train/val
    split boundary before streaming.
    """
    if source != "local":
        # For HF we can't pre-count without downloading; use max_per_script as
        # a conservative estimate and let the actual split fall where it may.
        return max_per_script or 0

    folder = IHTR_LOCAL_MAP.get(script)
    if not folder:
        return 0

    label_file = Path(data_dir) / "train" / folder / "train.txt"
    if not label_file.exists():
        return 0

    script_dir = Path(data_dir) / "train" / folder
    count = 0
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if max_per_script and count >= max_per_script:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            rel_path, label = parts
            if not label.strip():
                continue
            if (script_dir / rel_path).exists():
                count += 1
    return count


# ── Main prep pipeline ────────────────────────────────────────────────────────

def process_and_save(
    scripts: list[str],
    output_dir: str,
    source: str = "local",
    data_dir: str = "./data/indic_hw",
    val_ratio: float = 0.1,
    max_per_script: Optional[int] = None,
):
    """
    Load, format, and save all scripts as JSONL files.

    Outputs:
      {output_dir}/train.jsonl
      {output_dir}/val.jsonl
      {output_dir}/stats.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.jsonl"
    val_path   = out / "val.jsonl"
    stats      = {}

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path,   "w", encoding="utf-8") as f_val:

        for script in scripts:
            print(f"\n{'='*55}")
            print(f"  {script}")
            print(f"{'='*55}")

            # Pre-count lines to determine the split boundary without loading images
            n_total = _count_label_lines(script, source, data_dir, max_per_script)
            if n_total == 0:
                print(f"[WARN] No samples found for {script} — skipping")
                continue

            n_val   = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val
            print(f"[INFO] ~{n_total} samples -> {n_train} train / {n_val} val")

            if source == "local":
                loader = load_script_local(script, data_dir, max_per_script)
            else:
                loader = load_script_hf(script, max_per_script)

            train_count = val_count = error_count = 0

            # Stream directly — never hold more than one image in memory at a time
            for idx, (img, label) in enumerate(tqdm(loader, total=n_total, desc=script, unit="img")):
                try:
                    record = format_sample(img, label, script)
                    del img  # release immediately after base64 encoding
                    line   = json.dumps(record, ensure_ascii=False)

                    if idx < n_train:
                        f_train.write(line + "\n")
                        train_count += 1
                    else:
                        f_val.write(line + "\n")
                        val_count += 1

                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"[WARN] Error at index {idx}: {e}")

            stats[script] = {
                "train":  train_count,
                "val":    val_count,
                "errors": error_count,
            }
            print(f"[OK] {script}: {train_count} train, {val_count} val, {error_count} errors")

    # Write stats
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_train = sum(v["train"] for v in stats.values())
    total_val   = sum(v["val"]   for v in stats.values())

    print(f"\n{'='*55}")
    print(f"Dataset prep complete!")
    print(f"  Train : {train_path}  ({total_train} records)")
    print(f"  Val   : {val_path}  ({total_val} records)")
    print(f"  Stats : {out / 'stats.json'}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Akshar dataset preparation")
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="Data source: 'local' (IHTR on disk) or 'hf' (HuggingFace). Default: local",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/indic_hw",
        help="Root of local IHTR data (only used when --source local). Default: ./data/indic_hw",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=list(SCRIPT_DATASETS.keys()),
        choices=list(SCRIPT_DATASETS.keys()),
        help="Which scripts to include (default: all 10 in priority order)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Where to save train.jsonl / val.jsonl / stats.json. Default: ./data",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of each script reserved for validation. Default: 0.1",
    )
    parser.add_argument(
        "--max_per_script",
        type=int,
        default=None,
        help="Cap samples per script before quality filtering (e.g. 20000 for medium run)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_save(
        scripts=args.scripts,
        output_dir=args.output_dir,
        source=args.source,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        max_per_script=args.max_per_script,
    )
