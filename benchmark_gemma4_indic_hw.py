"""
Akshar Benchmark: Zero-shot Indic Handwriting OCR with Gemma 4
==============================================================
Usage:
  python benchmark_gemma4_indic_hw.py --download                          # download all 10 scripts
  python benchmark_gemma4_indic_hw.py --download --scripts hindi kannada  # download specific ones
  python benchmark_gemma4_indic_hw.py --run --model google/gemma-4-E4B-it --samples 100
  python benchmark_gemma4_indic_hw.py --run --scripts hindi kannada --samples 50
  python benchmark_gemma4_indic_hw.py --run --model google/gemma-4-26B-A4B-it --quantize 4bit
"""

import argparse
import json
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("./data/indic_hw")
RESULTS_DIR = Path("./results")

# IHTR 2023 competition — direct download links for all 10 scripts
# Training set: ~70-85K word images per script with labels
# Validation set: 1000 per script (we use this for benchmarking)
IHTR_TRAIN_URLS = {
    "bengali":    "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/bengali.zip",
    "hindi":      "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/devanagari.zip",
    "gujarati":   "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/gujarati.zip",
    "gurumukhi":  "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/gurumukhi.zip",
    "kannada":    "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/kannada.zip",
    "malayalam":  "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/malayalam.zip",
    "odia":       "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/odia.zip",
    "tamil":      "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/tamil.zip",
    "telugu":     "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/telugu.zip",
    "urdu":       "https://ilocr.iiit.ac.in/ihtr/assets/dataset/trainingset/urdu.zip",
}

# Validation set (all scripts in one zip)
IHTR_VAL_URL = "https://ilocr.iiit.ac.in/ihtr/assets/dataset/validationset.zip"

ALL_SCRIPTS = list(IHTR_TRAIN_URLS.keys())

# Script to language name mapping (for prompts)
SCRIPT_LANG_MAP = {
    "hindi":      "Hindi (Devanagari script)",
    "bengali":    "Bengali (Bangla script)",
    "gujarati":   "Gujarati",
    "gurumukhi":  "Punjabi (Gurmukhi script)",
    "kannada":    "Kannada",
    "malayalam":  "Malayalam",
    "odia":       "Odia (Oriya script)",
    "tamil":      "Tamil",
    "telugu":     "Telugu",
    "urdu":       "Urdu",
}

# IHTR uses "devanagari" as folder name for Hindi
IHTR_FOLDER_MAP = {
    "hindi": "devanagari",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    image_path: str
    ground_truth: str
    script: str


@dataclass
class PredictionResult:
    sample: Sample
    prediction: str
    cer: float
    wer: float
    ned: float
    inference_time: float


@dataclass
class ScriptResults:
    script: str
    num_samples: int
    avg_cer: float
    avg_wrr: float
    avg_ned: float
    median_cer: float
    avg_inference_time: float
    total_chars_gt: int
    total_chars_pred: int


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]


def compute_metrics(ground_truth: str, prediction: str) -> dict:
    gt = ground_truth.strip()
    pred = prediction.strip()
    edit_dist = levenshtein_distance(gt, pred)
    cer = edit_dist / max(len(gt), 1)
    wer = 0.0 if gt == pred else 1.0
    ned = edit_dist / max(len(gt), len(pred), 1)
    return {"cer": cer, "wer": wer, "ned": ned}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path):
    """Download a file with progress."""
    print(f"    Downloading {url}")
    print(f"    -> {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def download_and_extract(url: str, extract_to: Path, label: str):
    """Download zip and extract."""
    zip_path = extract_to / f"{label}.zip"

    if not zip_path.exists():
        download_file(url, zip_path)
    else:
        print(f"    [CACHED] {zip_path}")

    print(f"    Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"    Done: {extract_to}")


def download_all(scripts: list[str], data_dir: Path):
    """Download training and validation sets for specified scripts."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download validation set (all scripts in one zip — 1000 samples each)
    print("\n[1/2] Downloading validation set (all scripts, ~1000 samples each)...")
    val_dir = data_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    val_zip = val_dir / "validationset.zip"
    if not val_zip.exists():
        download_file(IHTR_VAL_URL, val_zip)
    print("    Extracting...")
    with zipfile.ZipFile(val_zip, "r") as zf:
        zf.extractall(val_dir)
    print("    Done")

    # Download training sets per script
    print(f"\n[2/2] Downloading training sets for {len(scripts)} scripts...")
    for script in scripts:
        url = IHTR_TRAIN_URLS.get(script)
        if not url:
            print(f"  [SKIP] {script}: no download URL")
            continue

        ihtr_name = IHTR_FOLDER_MAP.get(script, script)
        train_dir = data_dir / "train"
        train_dir.mkdir(exist_ok=True)

        # Check if already extracted
        if (train_dir / ihtr_name).exists():
            print(f"  {script.upper()}: [CACHED] already extracted")
            continue

        print(f"\n  {script.upper()}:")
        download_and_extract(url, train_dir, ihtr_name)

    print(f"\n{'='*60}")
    print(f"  Download complete!")
    print(f"  Validation data: {val_dir}")
    print(f"  Training data:   {data_dir / 'train'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Dataset loading — handles IHTR 2023 format
# ---------------------------------------------------------------------------

def parse_ihtr_labels(label_file: Path, base_dir: Path) -> list[tuple[str, str]]:
    """
    Parse IHTR label file.
    Format: image_relative_path ground_truth  (space-separated)
    Handles path mismatches: val.txt may say 'test/0.jpg' but images are in 'val/'
    """
    pairs = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                img_rel, gt = parts[0], parts[1]
                img_name = Path(img_rel).name  # e.g. "0.jpg"

                # Try paths in order of likelihood
                candidates = [
                    base_dir / img_rel,                         # exact: base/test/0.jpg
                    base_dir / "val" / img_name,                # fix: test/ -> val/
                    base_dir / "train" / img_name,              # fix: for train split
                    base_dir / img_name,                        # flat: base/0.jpg
                ]

                found = False
                for img_path in candidates:
                    if img_path.exists():
                        pairs.append((str(img_path), gt))
                        found = True
                        break

    return pairs


def find_label_file(base_dir: Path, ihtr_name: str, label_name: str) -> Optional[Path]:
    """Search for label file in various possible locations."""
    candidates = [
        base_dir / ihtr_name / label_name,
        base_dir / ihtr_name.capitalize() / label_name,
        base_dir / ihtr_name.upper() / label_name,
        # IHTR zips extract with extra subdirectory
        base_dir / "validationset" / ihtr_name / label_name,
        base_dir / "trainingset" / ihtr_name / label_name,
        base_dir / "validationset" / ihtr_name.capitalize() / label_name,
        base_dir / "trainingset" / ihtr_name.capitalize() / label_name,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Glob fallback
    if base_dir.exists():
        for f in base_dir.rglob(label_name):
            if ihtr_name.lower() in str(f).lower():
                return f
    return None


def load_samples(scripts: list[str], data_dir: Path, n_samples: int, split: str = "val") -> dict[str, list[Sample]]:
    """Load samples from IHTR 2023 format."""
    all_samples = {}

    for script in scripts:
        ihtr_name = IHTR_FOLDER_MAP.get(script, script)

        if split == "val":
            label_file = find_label_file(data_dir / "validation", ihtr_name, "val.txt")
        else:
            label_file = find_label_file(data_dir / "train", ihtr_name, "train.txt")

        if label_file is None:
            print(f"  [SKIP] {script}: {split}.txt not found")
            continue

        base_dir = label_file.parent
        pairs = parse_ihtr_labels(label_file, base_dir)
        samples = [Sample(img, gt, script) for img, gt in pairs]

        if not samples:
            print(f"  [SKIP] {script}: 0 samples loaded")
            continue

        if len(samples) > n_samples:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(samples), size=n_samples, replace=False)
            samples = [samples[i] for i in indices]

        all_samples[script] = samples
        print(f"  {script}: {len(samples)} samples loaded")

    return all_samples


# ---------------------------------------------------------------------------
# Model loading — official HF Gemma 4 pattern
# ---------------------------------------------------------------------------

def load_model(model_id: str, quantize: Optional[str] = None):
    import torch
    # Gemma 4 E4B is a multimodal image-text-to-text model; the correct auto
    # class in transformers >= 5.5 is AutoModelForImageTextToText.
    # (AutoModelForMultimodalLM does not exist in upstream transformers.)
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"\nLoading model: {model_id}")
    if quantize:
        print(f"  Quantization: {quantize}")

    processor = AutoProcessor.from_pretrained(model_id)

    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto",
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model.eval()

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.1f} GB")

    return model, processor


# ---------------------------------------------------------------------------
# Inference — official HF Gemma 4 image pattern
# ---------------------------------------------------------------------------

def build_ocr_prompt(script: str) -> str:
    lang = SCRIPT_LANG_MAP.get(script, script)
    return (
        f"This image contains a single handwritten word in {lang}. "
        f"Read the handwritten text and output ONLY the word in its original script. "
        f"Do not add any explanation, transliteration, or translation. "
        f"Output only the exact handwritten word."
    )


def run_inference(model, processor, image_path: str, prompt: str, max_new_tokens: int = 64) -> tuple[str, float]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Try the Gemma 4 parse_response helper if present; fall back to raw decode.
    text: str
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(response)
            if isinstance(parsed, str):
                text = parsed
            elif isinstance(parsed, dict):
                text = parsed.get("content") or parsed.get("text") or str(parsed)
            else:
                text = str(parsed)
        except Exception:
            text = response
    else:
        text = response

    prediction = _extract_word(text)
    return prediction, elapsed


def _extract_word(text: str) -> str:
    """
    Normalise a raw model decode into a single word.

    Handles three common cases:
      1) Clean single-word output ("ಕರ್ನಾಟಕ")
      2) Legacy dual-format ("Transcription: ಕರ್ನಾಟಕ\\nTranslation: ...")
      3) Multi-line noise — we take the first non-empty line.
    """
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    # Legacy "Transcription: ..." prefix from older fine-tuning runs.
    for prefix in ("Transcription:", "transcription:", "Transcription -", "Answer:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    # First non-empty line only.
    for line in cleaned.splitlines():
        line = line.strip()
        if line:
            return line
    return cleaned


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model, processor, samples_by_script: dict[str, list[Sample]]) -> dict[str, ScriptResults]:
    results = {}

    for script, samples in samples_by_script.items():
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {script.upper()} ({len(samples)} samples)")
        print(f"{'='*60}")

        prompt = build_ocr_prompt(script)
        predictions = []

        for i, sample in enumerate(samples):
            try:
                pred_text, elapsed = run_inference(model, processor, sample.image_path, prompt)
            except Exception as e:
                print(f"  [ERROR] sample {i}: {e}")
                pred_text, elapsed = "", 0.0

            metrics = compute_metrics(sample.ground_truth, pred_text)
            result = PredictionResult(
                sample=sample, prediction=pred_text,
                cer=metrics["cer"], wer=metrics["wer"], ned=metrics["ned"],
                inference_time=elapsed,
            )
            predictions.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                running_wrr = 1 - np.mean([p.wer for p in predictions])
                running_cer = np.mean([p.cer for p in predictions])
                print(
                    f"  [{i+1:4d}/{len(samples)}] "
                    f"WRR={running_wrr:.1%}  CER={running_cer:.3f}  "
                    f"last: '{sample.ground_truth}' -> '{pred_text}'"
                )

        cers = [p.cer for p in predictions]
        script_result = ScriptResults(
            script=script, num_samples=len(predictions),
            avg_cer=float(np.mean(cers)),
            avg_wrr=float(1 - np.mean([p.wer for p in predictions])),
            avg_ned=float(np.mean([p.ned for p in predictions])),
            median_cer=float(np.median(cers)),
            avg_inference_time=float(np.mean([p.inference_time for p in predictions])),
            total_chars_gt=sum(len(p.sample.ground_truth) for p in predictions),
            total_chars_pred=sum(len(p.prediction) for p in predictions),
        )
        results[script] = script_result

        pred_file = RESULTS_DIR / f"predictions_{script}.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps({
                    "image": p.sample.image_path,
                    "ground_truth": p.sample.ground_truth,
                    "prediction": p.prediction,
                    "cer": p.cer, "wer": p.wer, "ned": p.ned,
                    "time_s": p.inference_time,
                }, ensure_ascii=False) + "\n")

    return results


def print_summary(results: dict[str, ScriptResults], model_id: str):
    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS — {model_id}")
    print(f"{'='*80}")
    print(f"{'Script':<12} {'Samples':>8} {'WRR':>8} {'CER':>8} {'NED':>8} "
          f"{'Med CER':>8} {'Avg Time':>9}")
    print("-" * 80)

    all_wrr, all_cer = [], []

    for script in ALL_SCRIPTS:
        if script not in results:
            print(f"{script:<12} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>9}")
            continue
        r = results[script]
        print(
            f"{r.script:<12} {r.num_samples:>8} {r.avg_wrr:>7.1%} "
            f"{r.avg_cer:>7.3f} {r.avg_ned:>7.3f} {r.median_cer:>7.3f} "
            f"{r.avg_inference_time:>8.2f}s"
        )
        all_wrr.append(r.avg_wrr)
        all_cer.append(r.avg_cer)

    if all_wrr:
        print("-" * 80)
        print(f"{'AVERAGE':<12} {'':>8} {np.mean(all_wrr):>7.1%} {np.mean(all_cer):>7.3f}")
    print("=" * 80)

    print("\n  INTERPRETATION GUIDE:")
    print("  " + "-" * 40)
    if all_wrr:
        avg_wrr = np.mean(all_wrr)
        if avg_wrr > 0.7:
            print("  Zero-shot is already strong (>70% WRR).")
            print("  Fine-tuning may yield only marginal gains.")
            print("  -> Consider pivoting to Medical Triage.")
        elif avg_wrr > 0.4:
            print("  Moderate zero-shot performance (40-70% WRR).")
            print("  Fine-tuning has clear room for improvement.")
            print("  -> Akshar is viable with targeted fine-tuning.")
        else:
            print("  Weak zero-shot performance (<40% WRR).")
            print("  Large gap to close — strong fine-tuning story.")
            print("  -> Akshar has maximum impact potential.")

        sorted_scripts = sorted(results.values(), key=lambda r: r.avg_wrr)
        weakest = sorted_scripts[:3]
        print(f"\n  Weakest scripts (prioritize for fine-tuning):")
        for r in weakest:
            print(f"    {r.script}: {r.avg_wrr:.1%} WRR, {r.avg_cer:.3f} CER")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Akshar Benchmark: Zero-shot Indic Handwriting OCR with Gemma 4")

    parser.add_argument("--download", action="store_true", help="Download datasets (all 10 scripts by default)")
    parser.add_argument("--run", action="store_true", help="Run the benchmark")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B-it", help="Model ID")
    parser.add_argument("--quantize", type=str, choices=["4bit", "8bit"], default=None)
    parser.add_argument("--scripts", nargs="+", default=None, choices=ALL_SCRIPTS, help="Scripts to download/benchmark (default: all 10)")
    parser.add_argument("--samples", type=int, default=100, help="Samples per script (default: 100)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Which split to benchmark on (default: val)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scripts = args.scripts or ALL_SCRIPTS

    if args.download:
        download_all(scripts, data_dir)
        return

    if args.run:
        print(f"Loading {args.split} samples from {data_dir}...")
        samples = load_samples(scripts, data_dir, args.samples, split=args.split)

        if not samples:
            print("\n  ERROR: No data found! Run with --download first.")
            sys.exit(1)

        model, processor = load_model(args.model, args.quantize)
        results = run_benchmark(model, processor, samples)
        print_summary(results, args.model)

        summary_file = RESULTS_DIR / "benchmark_summary.json"
        summary = {
            "model": args.model, "quantize": args.quantize,
            "split": args.split, "samples_per_script": args.samples,
            "results": {k: asdict(v) for k, v in results.items()},
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {summary_file}")

    if not args.download and not args.run:
        parser.print_help()


if __name__ == "__main__":
    main()