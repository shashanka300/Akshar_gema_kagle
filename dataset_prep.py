"""
dataset_prep.py
================
Akshar — Dataset preparation for Gemma 4 E4B fine-tuning.

Creates lightweight path-reference JSONL index files (~10 MB) instead of
embedding 65 GB of base64 images. Images are loaded from disk at training
time by AksharDataset.

Local data layout:
  Training:
    data/indic_hw/train/<folder>/train.txt   — "<relative_path> <label>" per line
    data/indic_hw/train/<folder>/train/      — image files (nested or flat)

  Validation (curated, 1000/script):
    data/indic_hw/validation/validationset/<folder>/val.txt  — "<relative_path> <label>"
    data/indic_hw/validation/validationset/<folder>/val/     — image files

Each output JSONL record (lightweight — no base64):
  {
    "script":     "Hindi",
    "label":      "केंद्रों",
    "image_path": "E:/akshar_benchmark/data/indic_hw/train/devanagari/train/8/251/21.jpg"
  }

Usage:
  # Full run (all scripts, all samples)
  python -X utf8 dataset_prep.py --source local --output_dir ./data

  # Capped run: 20k/script
  python -X utf8 dataset_prep.py --source local --max_per_script 20000 --output_dir ./data

  # Smoke test: 100/script
  python -X utf8 dataset_prep.py --source local --max_per_script 100 --output_dir ./data

  # Specific scripts only
  python -X utf8 dataset_prep.py --source local --scripts Kannada Telugu --max_per_script 5000
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Iterator

from PIL import Image
from tqdm import tqdm

# ── Script config ────────────────────────────────────────────────────────────

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

# Mapping from script name -> on-disk IHTR folder name
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

# Image quality thresholds
MIN_WIDTH  = 20
MIN_HEIGHT = 20
MIN_AREA   = 800
MAX_ASPECT = 15.0


def passes_quality_filter(img: Image.Image) -> bool:
    w, h = img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    if w * h < MIN_AREA:
        return False
    if h > 0 and (w / h) > MAX_ASPECT:
        return False
    return True


# ── Local IHTR training loader ───────────────────────────────────────────────

def load_train_local(
    script: str,
    data_dir: str,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, str]]:
    """
    Yield (absolute_image_path, label) from the training split.

    Layout:
      <data_dir>/train/<folder>/train.txt  — "<img_relative_path> <label>"
      <data_dir>/train/<folder>/<img_relative_path>
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

    print(f"[INFO] Loading {script} train from {script_dir} ...")

    filtered = 0
    yielded  = 0

    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if max_samples and yielded >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            # Format: "<relative_img_path> <label>" (split on first space)
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

            # Quality filter (only opens image briefly)
            try:
                img = Image.open(img_path)
                img.load()
                if not passes_quality_filter(img):
                    filtered += 1
                    continue
                del img  # release immediately
            except Exception:
                continue

            yielded += 1
            yield str(img_path.resolve()), label

    if filtered:
        print(f"[INFO] {script}: filtered {filtered} low-quality training images")


# ── Local IHTR validation loader ─────────────────────────────────────────────

def load_val_local(
    script: str,
    data_dir: str,
    max_samples: Optional[int] = None,
) -> Iterator[tuple[str, str]]:
    """
    Yield (absolute_image_path, label) from the curated validation split.

    Layout:
      <data_dir>/validation/validationset/<folder>/val.txt — "<path> <label>"
      <data_dir>/validation/validationset/<folder>/val/    — image files

    Note: val.txt references images as "test/0.jpg" but they live in "val/".
    We remap test/ -> val/ when resolving paths.
    """
    folder = IHTR_LOCAL_MAP.get(script)
    if not folder:
        return

    val_dir    = Path(data_dir) / "validation" / "validationset" / folder
    label_file = val_dir / "val.txt"

    if not label_file.exists():
        print(f"[WARN] No validation label file: {label_file}")
        return

    print(f"[INFO] Loading {script} val from {val_dir} ...")
    yielded = 0

    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if max_samples and yielded >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            rel_path, label = parts
            label = label.strip()
            if not label:
                continue

            # Fix path: val.txt says "test/0.jpg" but images are in "val/"
            if rel_path.startswith("test/"):
                rel_path = "val/" + rel_path[len("test/"):]

            img_path = val_dir / rel_path
            if not img_path.exists():
                # Also try without subfolder prefix
                img_path = val_dir / "val" / Path(rel_path).name
            if not img_path.exists():
                continue

            yielded += 1
            yield str(img_path.resolve()), label


# ── HuggingFace loader ───────────────────────────────────────────────────────

def load_hf(
    script: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Iterator[tuple[str, str]]:
    """Load from HuggingFace, save images locally, yield (path, label)."""
    from datasets import load_dataset

    hf_id = SCRIPT_DATASETS.get(script)
    if not hf_id:
        print(f"[WARN] No HF dataset for script: {script}")
        return

    print(f"[INFO] Loading {script} from {hf_id} ({split}) ...")
    try:
        ds = load_dataset(hf_id, split=split)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return

    out_dir = Path(cache_dir or f"./data/hf_images/{script}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

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

            img_path = out_dir / f"{idx:06d}.png"
            if not img_path.exists():
                img.save(img_path, format="PNG")

            yield str(img_path.resolve()), label
        except Exception:
            pass


# ── Counting helpers ─────────────────────────────────────────────────────────

def _count_lines(label_file: Path, max_count: Optional[int] = None) -> int:
    """Quickly count valid label lines without opening images."""
    if not label_file.exists():
        return 0
    count = 0
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if max_count and count >= max_count:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1].strip():
                count += 1
    return count


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_and_save(
    scripts: list[str],
    output_dir: str,
    source: str = "local",
    data_dir: str = "./data/indic_hw",
    max_per_script: Optional[int] = None,
    max_val_per_script: Optional[int] = None,
):
    """
    Build lightweight path-reference JSONL index files.

    Outputs:
      {output_dir}/train_index.jsonl   — training records (image_path, not base64)
      {output_dir}/val_index.jsonl     — validation records (from curated val set)
      {output_dir}/stats.json          — per-script counts
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train_index.jsonl"
    val_path   = out / "val_index.jsonl"
    stats      = {}

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path,   "w", encoding="utf-8") as f_val:

        for script in scripts:
            print(f"\n{'='*60}")
            print(f"  {script}")
            print(f"{'='*60}")

            # ── Training data ──
            if source == "local":
                train_loader = load_train_local(script, data_dir, max_per_script)
            else:
                train_loader = load_hf(script, "train", max_per_script)

            folder = IHTR_LOCAL_MAP.get(script, script.lower())
            label_file = Path(data_dir) / "train" / folder / "train.txt"
            est_total = _count_lines(label_file, max_per_script) if source == "local" else (max_per_script or 0)

            train_count = 0
            for img_path, label in tqdm(train_loader, total=est_total, desc=f"{script} train", unit="img"):
                record = {"script": script, "label": label, "image_path": img_path}
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_count += 1

            # ── Validation data (curated on-disk set) ──
            if source == "local":
                val_loader = load_val_local(script, data_dir, max_val_per_script)
            else:
                val_loader = load_hf(script, "test", max_val_per_script)

            val_count = 0
            for img_path, label in tqdm(val_loader, desc=f"{script} val", unit="img"):
                record = {"script": script, "label": label, "image_path": img_path}
                f_val.write(json.dumps(record, ensure_ascii=False) + "\n")
                val_count += 1

            stats[script] = {"train": train_count, "val": val_count}
            print(f"[OK] {script}: {train_count} train, {val_count} val")

    # Write stats
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_train = sum(v["train"] for v in stats.values())
    total_val   = sum(v["val"]   for v in stats.values())

    train_size = train_path.stat().st_size / 1024 / 1024
    val_size   = val_path.stat().st_size / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Dataset prep complete!")
    print(f"  Train : {train_path}  ({total_train:,} records, {train_size:.1f} MB)")
    print(f"  Val   : {val_path}  ({total_val:,} records, {val_size:.1f} MB)")
    print(f"  Stats : {out / 'stats.json'}")
    print(f"{'='*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Akshar dataset preparation")
    parser.add_argument(
        "--source", choices=["local", "hf"], default="local",
        help="Data source: 'local' (IHTR on disk) or 'hf' (HuggingFace). Default: local",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/indic_hw",
        help="Root of local IHTR data (default: ./data/indic_hw)",
    )
    parser.add_argument(
        "--scripts", nargs="+", default=list(SCRIPT_DATASETS.keys()),
        choices=list(SCRIPT_DATASETS.keys()),
        help="Which scripts to include (default: all 10)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Where to save index JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--max_per_script", type=int, default=None,
        help="Cap training samples per script (e.g. 20000)",
    )
    parser.add_argument(
        "--max_val_per_script", type=int, default=None,
        help="Cap validation samples per script (default: use all ~1000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_save(
        scripts=args.scripts,
        output_dir=args.output_dir,
        source=args.source,
        data_dir=args.data_dir,
        max_per_script=args.max_per_script,
        max_val_per_script=args.max_val_per_script,
    )
