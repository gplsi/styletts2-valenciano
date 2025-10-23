# Automatic Audio Dataset Maker — General Guide

This guide explains the purpose, options, workflow, outputs, and troubleshooting for building a speech dataset with this repository. It is intentionally general and does not depend on any specific user configuration.

## Purpose

Turn raw long-form audio into a clean, segmented speech dataset ready for model training and hosting on the Hugging Face Hub. The pipeline:
- Optionally normalizes, denoises, removes silence and non-speech.
- Transcribes with Whisper and creates SRTs.
- Segments audio using transcripts and time boundaries.
- Generates metadata and splits into train/eval.
- Computes optional audio quality metrics and normalized features.
- Saves Parquet shards locally and pushes the dataset to the Hub.

## Key Features

- Works with local audio (prefer .wav for the main pipeline).
- Transcription via Whisper (local) with automatic GPU→CPU fallback if CUDA JIT fails.
- Segment generation using SRT timing + heuristic duration targets.
- Dataset packaging to Hugging Face (train/eval split, Audio column).
- Optional quality filtering (e.g., SNR/C50, reverberation heuristics, speaking rate).
- Optional preprocessing tools: normalization, non-speech removal, denoising, desilencing.

## Folder Structure

- `RAW_AUDIO/` — Put your original (preferably WAV) audio files here.
- `SRTS/` — Transcription outputs (.srt) per input audio.
- `WAVS/` — Segmented clips produced from SRTs.
- `PARENT_CSV/metadata.csv` — Segment-level metadata (audio path | text | speaker_name).
- `METADATA/metadata_train.csv`, `METADATA/metadata_eval.csv` — Train/Eval splits.
- `UNFILTERED_PARQUET/` — The full dataset after processing (transcription → segmentation → enrichment/metadata-to-text), before any quality checks. It contains every segment the pipeline produced.
- `FILTERED_PARQUET/` — The curated subset that passed the quality thresholds (e.g., SNR/C50, reverberation heuristics, speaking rate bounds). Only “good” rows remain.Local Parquet dataset (post-filtering).

## Installation & Environment

- Python 3.10–3.12 recommended.
- Install requirements, then install a compatible PyTorch build for GPU (or CPU-only). If you see CUDA/Triton JIT issues, fallback to CPU is automatic; you can also explicitly run on CPU.
- Ensure FFmpeg is available on your system for some tools.

## Config Overview (`config.yaml`)

- `huggingface.username` — Your HF username
- `huggingface.repo_name` — Target dataset repo name on HF Hub
- `audio_processing.skip_transcription` — Skip Step 1 if SRTs already exist
- `audio_processing.transcription_method` — 1 = Whisper local
- `audio_processing.num_gpus` — Number of GPUs to use (null = auto)
- `audio_processing.sampling_rate` — Target sample rate (e.g., 24000)
- `audio_processing.refilter` — When true, skip to filtering to re-apply thresholds
- `dataset.eval_percentage` — Percentage for eval split (e.g., 10–15)
- `dataset.speaker_name_option` — "single" or "filename"
- `dataset.speaker_name` — Label when `speaker_name_option` == "single"

Tip: You can safely re-run the pipeline; it skips already-generated SRTs and will overwrite metadata and parquet shards as needed.

## Quick Start

- Put WAV files in `RAW_AUDIO/`.
- Configure `config.yaml` minimally (HF username/repo, sample rate, etc.).
- Run the pipeline:

```fish
python adm_main.py --config config.yaml
```

- To force CPU for Whisper (if GPU JIT is unstable):

```fish
env CUDA_VISIBLE_DEVICES= python adm_main.py --config config.yaml
```

## Optional Preprocessing Tools

These can improve input quality before the main pipeline. All accept `--input_dir` and `--output_dir`.

- Normalize loudness/peak:
```fish
python tools/normalize_folder.py --input_dir RAW_AUDIO --output_dir RAW_AUDIO_norm --target_lufs -23 --num_workers 1
```

- Remove non-speech/music:
```fish
python tools/remove_nonspeech.py --input_dir RAW_AUDIO_norm --output_dir RAW_AUDIO_speech --aggressiveness 2
```

- Denoise (Mel-Band-Roformer):
```fish
python tools/run_denoiser.py --input_dir RAW_AUDIO_speech --output_dir DENOISED
```

- Desilence (using `unsilence`):
```fish
python tools/desilence.py --input_dir DENOISED --output_dir RAW_AUDIO_final
```

If you process to a different folder (e.g., `RAW_AUDIO_final`), either move files back into `RAW_AUDIO/` or adjust the pipeline’s input discovery accordingly.

## End-to-End Pipeline Steps

1) Transcribe (Whisper)
- Generates `.srt` files in `SRTS/` for each input `.wav`.
- GPU preferred; if a CUDA/Triton JIT error occurs, automatically retries on CPU with fp16 disabled.

2) Segment + Metadata
- Uses SRT timing + Gaussian segment lengths (bounded) to create clips.
- Writes clips to `WAVS/` and metadata to `PARENT_CSV/metadata.csv`.

3) Train/Eval Split + First Push
- Splits to train/eval based on `eval_percentage`.
- Converts to `datasets.DatasetDict` and pushes to HF Hub (private by default).

4) Initial Processing
- DataSpeech-based enrichments: speaking rate and optional metrics.
- Resamples audio to target `sampling_rate` and pushes updated dataset to Hub.

5) Metadata→Text + Save Local Parquet
- Converts key metrics to text bins/labels (if enabled) and saves `UNFILTERED_PARQUET/`.

6) Filtering
- Applies quality thresholds (e.g., SNR, reverberation, speaking rate). Thresholds can be tweaked in filtering logic.
- Saves `FILTERED_PARQUET/`.

7) Push Filtered Dataset
- Filters the original Hub dataset by indices retained after quality checks.
- Pushes final filtered dataset to the Hub.

## Outputs & Artifacts

- Transcripts: `SRTS/*.srt`
- Segmented audio: `WAVS/*.wav`
- Metadata: `PARENT_CSV/metadata.csv` (pipe-delimited)
- Train/Eval CSV: `METADATA/metadata_train.csv`, `metadata_eval.csv`
- Local Parquet shards (pre- and post-filter): `UNFILTERED_PARQUET/`, `FILTERED_PARQUET/`
- Final dataset on Hugging Face Hub (private by default)

## Quality & Filtering

- Metrics (when enabled):
  - Speaking rate, phonemes
  - SNR/C50 (if you enable these computations)
  - Reverberation heuristics
- Adjust thresholds in the filtering script to match your data difficulty and target size.
- Refilter mode: set `audio_processing.refilter: true` to re-run filtering without re-processing/transcribing.

## Troubleshooting

- CUDA/Triton JIT errors with Whisper:
  - The pipeline auto-falls back to CPU.
  - To avoid GPU entirely: `env CUDA_VISIBLE_DEVICES= python adm_main.py --config config.yaml`
  - For full GPU usage, align your Torch/Triton/CUDA versions per PyTorch’s official wheels.

- Xet Storage warning on upload:
  - "Uploading files as a binary IO buffer is not supported..." — the pipeline automatically falls back to HTTP. No action required.

- pyannote/torchcodec warnings:
  - Optional; the pipeline still runs. If needed, install a compatible FFmpeg/TorchCodec combo.

- Empty/incorrect inputs:
  - Ensure `.wav` files exist in the input folder (default `RAW_AUDIO/`).
  - Confirm `config.yaml` has valid HF credentials and repo name.

## Performance Tips

- Use GPU where possible; reduce `--cpu_num_workers` if you hit memory limits.
- Keep `sampling_rate` consistent across steps (e.g., 24 kHz is a good default).
- Large corpora: expect multiple Parquet shards; the code handles sharding.

## Extending the Pipeline

- Alternate ASR backends (e.g., Faster-Whisper) can be added with a small integration layer.
- Additional metrics or filters can be added to `filter_parquet.py` and the DataSpeech steps.
- You can add a post-run hook to emit a dated report under `docs/` automatically.

---

With this guide, you can configure your dataset pipeline, choose optional preprocessing to fit your use case, run end-to-end, and iterate on quality filters until you reach your target dataset size and quality.
