"""
verify_dataset.py
==================
Quick sanity check on the prepared JSONL dataset.
Prints sample prompts, label distribution, and image stats.

Usage:
  python verify_dataset.py --data_dir ./data
"""

import json
import base64
import argparse
from io import BytesIO
from collections import Counter
from pathlib import Path

from PIL import Image


def decode_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes))


def verify(data_dir: str, n_samples: int = 5):
    data_dir = Path(data_dir)

    for split in ["train", "val"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Split: {split.upper()} — {path}")
        print(f"{'='*60}")

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        print(f"Total samples: {len(lines)}")

        script_counter = Counter()
        label_lengths  = []
        img_widths     = []

        for i, line in enumerate(lines):
            sample = json.loads(line)
            script_counter[sample["script"]] += 1
            label_lengths.append(len(sample["label"]))

            # Decode and check image dimensions
            img = decode_image(sample["image_b64"])
            img_widths.append(img.size[0])

            # Print first N samples in detail
            if i < n_samples:
                print(f"\n── Sample {i+1} ──────────────────────")
                print(f"  Script : {sample['script']}")
                print(f"  Label  : {sample['label']}")
                print(f"  ImgSize: {img.size}")
                print(f"  User   : {sample['messages'][0]['content'][1]['text'][:60]}...")
                print(f"  Target : {sample['messages'][1]['content']}")

        print(f"\n── Script distribution ──────────────")
        for script, count in script_counter.most_common():
            print(f"  {script:<12}: {count:>6} samples")

        print(f"\n── Label length stats ──────────────")
        print(f"  Min  : {min(label_lengths)}")
        print(f"  Max  : {max(label_lengths)}")
        print(f"  Mean : {sum(label_lengths)/len(label_lengths):.1f}")

        print(f"\n── Image width stats ───────────────")
        print(f"  Min  : {min(img_widths)}px")
        print(f"  Max  : {max(img_widths)}px")
        print(f"  Mean : {sum(img_widths)/len(img_widths):.1f}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()
    verify(args.data_dir, args.n_samples)
