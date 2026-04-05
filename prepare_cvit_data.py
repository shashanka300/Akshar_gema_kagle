"""
Helper: Prepare CVIT dataset downloads into benchmark-ready format.

The IIIT-INDIC-HW-WORDS dataset comes in slightly different formats
depending on the script. This helper normalizes them.

Usage:
  python prepare_cvit_data.py --input ~/Downloads/kannada.zip --script kannada
  python prepare_cvit_data.py --input ~/Downloads/bengali/ --script bengali
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path


DATA_DIR = Path("./data/indic_hw")


def extract_if_zip(input_path: Path, script: str) -> Path:
    """Extract zip file if needed, return directory path."""
    if input_path.suffix == ".zip":
        extract_to = DATA_DIR / script
        extract_to.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {input_path} to {extract_to}...")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_to)
        return extract_to
    return input_path


def find_images(directory: Path) -> list[Path]:
    """Recursively find all image files."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = []
    for f in directory.rglob("*"):
        if f.suffix.lower() in extensions and not f.name.startswith("."):
            images.append(f)
    return sorted(images)


def find_label_file(directory: Path) -> tuple[Path | None, str]:
    """Find and identify the label file format."""
    # Check common label file names
    candidates = [
        ("labels.json", "json"),
        ("labels.txt", "tsv"),
        ("ground_truth.txt", "lines"),
        ("annotation.txt", "tsv"),
        ("gt.txt", "lines"),
        ("transcription.txt", "lines"),
    ]

    for fname, fmt in candidates:
        for f in directory.rglob(fname):
            return f, fmt

    # Check for any .txt that might be labels
    for f in directory.rglob("*.txt"):
        if f.name.lower() not in ("readme.txt", "license.txt"):
            # Peek at format
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline().strip()
            if "\t" in first_line:
                return f, "tsv"
            elif " " in first_line and first_line.split(" ")[0].endswith(
                (".png", ".jpg")
            ):
                return f, "space_sep"
            else:
                return f, "lines"

    return None, "unknown"


def parse_labels(label_path: Path, fmt: str) -> dict[str, str]:
    """Parse label file into {filename: ground_truth} dict."""
    labels = {}

    if fmt == "json":
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    elif fmt == "tsv":
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    labels[parts[0].strip()] = parts[1].strip()

    elif fmt == "space_sep":
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    labels[parts[0].strip()] = parts[1].strip()

    elif fmt == "lines":
        with open(label_path, "r", encoding="utf-8") as f:
            labels = {
                f"line_{i:06d}": line.strip()
                for i, line in enumerate(f)
                if line.strip()
            }

    return labels


def normalize_dataset(input_path: Path, script: str):
    """Normalize dataset into benchmark-ready format."""
    target_dir = DATA_DIR / script
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = find_images(input_path)
    print(f"Found {len(images)} images")

    # Find label file
    label_path, fmt = find_label_file(input_path)
    if label_path is None:
        print("ERROR: No label file found!")
        print("Expected one of: labels.json, labels.txt, ground_truth.txt")
        return

    print(f"Found labels: {label_path} (format: {fmt})")
    labels = parse_labels(label_path, fmt)
    print(f"Parsed {len(labels)} labels")

    # Build normalized labels.json
    normalized = {}
    matched = 0

    for img_path in images:
        # Try matching by full name, stem, or relative path
        name = img_path.name
        stem = img_path.stem
        rel = str(img_path.relative_to(input_path))

        gt = labels.get(name) or labels.get(stem) or labels.get(rel)

        if gt:
            # Copy image to target directory
            dest = target_dir / name
            if img_path != dest:
                shutil.copy2(img_path, dest)
            normalized[name] = gt
            matched += 1

    # If no matches by filename, try positional matching
    if matched == 0 and fmt == "lines":
        print("No filename matches — trying positional matching...")
        gt_values = list(labels.values())
        for i, img_path in enumerate(images):
            if i < len(gt_values):
                dest = target_dir / img_path.name
                if img_path != dest:
                    shutil.copy2(img_path, dest)
                normalized[img_path.name] = gt_values[i]
                matched += 1

    # Save normalized labels
    out_path = target_dir / "labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"\nNormalized {matched} samples → {target_dir}")
    print(f"Labels saved to {out_path}")

    if matched < len(images):
        print(f"WARNING: {len(images) - matched} images had no matching label")
    if matched < len(labels):
        print(f"WARNING: {len(labels) - matched} labels had no matching image")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CVIT dataset for Akshar benchmark"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to downloaded zip or extracted directory"
    )
    parser.add_argument(
        "--script", type=str, required=True,
        choices=[
            "hindi", "bengali", "gujarati", "gurumukhi",
            "kannada", "malayalam", "odia", "tamil", "telugu", "urdu"
        ],
        help="Script name"
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist")
        return

    # Extract if zip
    if input_path.suffix == ".zip":
        input_path = extract_if_zip(input_path, args.script)

    normalize_dataset(input_path, args.script)


if __name__ == "__main__":
    main()
