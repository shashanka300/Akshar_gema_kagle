# Akshar Benchmark: Zero-Shot Indic Handwriting OCR with Gemma 4

## Purpose

Before committing to fine-tuning Gemma 4 for Indic handwriting (the "Akshar" 
hackathon idea), this benchmark answers one critical question:

**How well does Gemma 4 already handle handwritten Indic scripts zero-shot?**

- If WRR > 70%: Zero-shot is strong → pivot to Medical Triage idea
- If WRR 40-70%: Room for improvement → Akshar is viable
- If WRR < 40%: Large gap → Akshar has maximum impact potential

## Hardware Requirements

| Model | VRAM Required | Recommended Config |
|-------|--------------|-------------------|
| Gemma 4 E4B (bf16) | ~16 GB | `--model google/gemma-4-E4B-it` |
| Gemma 4 E4B (4-bit) | ~6 GB | `--model google/gemma-4-E4B-it --quantize 4bit` |
| Gemma 4 26B MoE (4-bit) | ~14 GB | `--model google/gemma-4-26B-A4B-it --quantize 4bit` |
| Gemma 4 31B (4-bit) | ~18 GB | `--model google/gemma-4-31B-it --quantize 4bit` |

Your RTX 5090 (32GB) can run any of these comfortably.

**Recommended first run:** E4B in bf16 — fast iteration, good baseline.  
**Recommended second run:** 26B MoE in 4-bit — establishes ceiling performance.

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate

# 2. Download Hindi data (HuggingFace)
python benchmark_gemma4_indic_hw.py --download

# 3. Run benchmark (start small — 50 samples, fast feedback)
python benchmark_gemma4_indic_hw.py --run --scripts hindi --samples 50

# 4. If that works, run on all available scripts
python benchmark_gemma4_indic_hw.py --run --samples 100
```

## Getting the Data

### Tier 1: HuggingFace (automatic)

Hindi is available directly:
```bash
python benchmark_gemma4_indic_hw.py --download
```

### Tier 2: CVIT (manual download — all 10 scripts)

1. Go to: https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words
2. Download zip files for each script
3. For Devanagari & Telugu, follow the separate link on that page
4. Extract into the expected structure:

```
data/indic_hw/
├── hindi/
│   ├── images/          # word-level .png files
│   └── labels.txt       # filename<TAB>ground_truth (one per line)
├── kannada/
│   ├── images/
│   └── labels.txt
├── bengali/
│   ├── ...
...
```

The script auto-detects three label formats:
- `labels.json` — `{"filename.png": "ground_truth_text"}`
- `labels.txt` — `filename.png\tground_truth_text` (tab-separated)
- `ground_truth.txt` — one label per line, matched to sorted image files

### Tier 3: IIIT-Indic-HW-UC (camera-captured, harder)

For a more realistic benchmark (phone camera photos, not flatbed scans):
- URL: https://cvit.iiit.ac.in/usodi/ucciohd.php
- 13 languages, 2.6M words, 1220 writers
- Same directory structure works

### Tier 4: ICDAR 2025 IHDR (competition data)

Validation sets with bounding boxes:
- URL: https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/dataset.html
- Per-language downloads for Bengali, Gujarati, Hindi, Kannada, etc.
- Page-level images (would need word cropping first)

## Output

Results are saved to `results/`:

```
results/
├── benchmark_summary.json          # Aggregated metrics per script
├── predictions_hindi.jsonl         # Per-sample: image, GT, prediction, CER
├── predictions_kannada.jsonl
└── ...
```

### Sample output table

```
================================================================================
  BENCHMARK RESULTS — google/gemma-4-E4B-it
================================================================================
Script       Samples      WRR      CER      NED  Med CER  Avg Time
--------------------------------------------------------------------------------
hindi            100    42.0%    0.387    0.352    0.333     0.45s
bengali          100    28.0%    0.512    0.478    0.500     0.43s
kannada          100    19.0%    0.623    0.589    0.600     0.47s
...
--------------------------------------------------------------------------------
AVERAGE                 29.7%    0.507
================================================================================

  INTERPRETATION GUIDE:
  ----------------------------------------
  Weak zero-shot performance (<40% WRR).
  Large gap to close — strong fine-tuning story.
  → Akshar has maximum impact potential.

  Weakest scripts (prioritize for fine-tuning):
    kannada: 19.0% WRR, 0.623 CER
    bengali: 28.0% WRR, 0.512 CER
    hindi: 42.0% WRR, 0.387 CER
```

(Numbers above are hypothetical — the whole point is to get real numbers.)

## What To Do With Results

### If Akshar is a GO:

1. Pick the 3 weakest scripts as fine-tuning targets
2. Use QLoRA on Gemma 4 E4B (most practical for RTX 5090)
3. Train on IIIT-INDIC-HW-WORDS training split
4. Re-run this benchmark to show before/after improvement
5. Wrap in agentic app for the hackathon submission

### If Akshar is a NO-GO:

1. You've lost 1 evening, not a weekend
2. Pivot to Medical Document Triage Agent
3. Reuse the Docling/LangGraph/Qdrant pipeline
4. The benchmark data + analysis still makes a good LinkedIn post
