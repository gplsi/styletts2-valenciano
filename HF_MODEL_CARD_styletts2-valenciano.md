---
license: apache-2.0
language:
- ca
datasets:
- gplsi/corts-valencianes-asr
tags:
- TTS
- StyleTTS2
- valenciano
- catalan
- styletts2
- gplsi
---


# StyleTTS2 Valenciano


## Overview

<details>
<summary>Click to expand</summary>

- [Model Description](#model-description)
- [Intended Uses and Limitations](#intended-uses-and-limitations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Additional Information](#additional-information)

</details>

---

## Model Description

**StyleTTS2 Valenciano** is an end-to-end text-to-speech model for the **Valencian variant of Catalan**, adapted from [StyleTTS2](https://github.com/yl4579/StyleTTS2) and trained on parliamentary recordings from the Corts Valencianes corpus ([gplsi/corts-valencianes-asr](https://huggingface.co/datasets/gplsi/corts-valencianes-asr)).

The model follows the original two-stage StyleTTS2 architecture:

1. **Acoustic backbone** (Stage 1): text aligner, acoustic text encoder, pitch extractor, acoustic style encoder, and iSTFTNet decoder. Trained for 100 epochs.
2. **Prosodic and diffusion modules** (Stage 2): prosodic text encoder (our valenciano PL-BERT, [gplsi/PL-BERT-va](https://huggingface.co/gplsi/PL-BERT-va)), duration and prosody predictor, prosodic style encoder, style diffusion denoiser, and SLM adversarial training using WavLM. Trained for 100 epochs in three progressive phases.

Features of this model:

- Trained on **89,330 utterances** from the Corts Valencianes corpus.
- Input is **phoneme strings** in IPA (178-symbol vocabulary, char-level).
- Uses our [gplsi/PL-BERT-va](https://huggingface.co/gplsi/PL-BERT-va) as the prosodic text encoder, which provides semantic context to the prosody predictor.
- Audio is generated at **24 kHz**.

### Modifications and bug fixes

Several bugs in the original StyleTTS2 codebase were found and fixed during the adaptation. These fixes are essential for reproducible training:

1. **Checkpoint key mismatch when loading Stage 1 into Stage 2** (GitHub issue [yl4579/StyleTTS2#254](https://github.com/yl4579/StyleTTS2/issues/254)): the original `load_checkpoint` used `strict=False` and could silently fail to load weights, producing NaN loss from step 0. Fixed by trying `strict=True` first and remapping the `module.` prefix on failure.
2. **JDC pitch extractor squeeze bug**: `squeeze()` without an argument removed the batch dimension when `batch_size == 1` per GPU. Replaced with `squeeze(-1)`.
3. **PL-BERT 512 position limit**: parliamentary sentences with > 512 phoneme characters crashed the BERT encoder. Affected batches are now skipped.
4. **Style encoder minimum input size**: the style encoder (4× downsample + 5×5 conv) requires at least 80 mel frames. Shorter samples crashed `predictor_encoder`. Affected batches are now skipped; `min_length` raised from 50 to 100 in the config.
5. **Misuse of `pretrained_model`**: pointing `pretrained_model` to `first_stage.pth` causes double-loading and optimizer corruption. Documented and protected against in the training guide.

---

## Intended Uses and Limitations

### Intended uses

- Research and educational use for Catalan/Valencian text-to-speech synthesis.
- As a baseline for further fine-tuning on better-curated multi-speaker Valencian corpora.
- For inference, use the LibriTTS-style notebook that accepts a reference audio (see [How to Get Started with the Model](#how-to-get-started-with-the-model)).

### Limitations

The most important limitation of this model is that it was **trained as single-speaker on multi-speaker data**. The Corts Valencianes corpus contains many different politicians' voices, but all training samples were labelled with `speaker_id = 0`. As a consequence:

- The style encoder learned an "averaged" style across all speakers.
- The style diffusion denoiser learned a high-variance, inconsistent style distribution.
- **Pure text-to-speech inference** (without reference audio, using `Inference_LJSpeech.ipynb`) produces noisy and inconsistent voices.
- **Reference-based inference** (using `StyleTTS2_Demo_LibriTTS.ipynb` with a chosen reference audio) produces noticeably cleaner audio and is the **recommended way to use this model**.

Other limitations:

- The model is specific to Valencian / Catalan phonology. Using it for other languages requires re-training.
- The training corpus is parliamentary speech, which means the model is biased towards a formal speaking style; conversational or expressive synthesis may be lower quality.

---

## How to Get Started with the Model

### Installation

Clone the training repository and install the dependencies:

```bash
git clone https://github.com/javimosa/styletts2-valenciano.git
cd styletts2-valenciano
python -m venv .venv && source .venv/bin/activate
pip install -r StyleTTS2/requirements.txt
```

You also need `espeak-ng` with support for the `ca-va` variant. See the project's [t3/README.md](https://github.com/javimosa/styletts2-valenciano/blob/main/t3/README.md) for instructions.

### Download the model and PL-BERT

```bash
# StyleTTS2 valenciano checkpoint
huggingface-cli download gplsi/StyleTTS2-va \
    Models/LJSpeech/epoch_2nd_00100.pth \
    Models/LJSpeech/config.yml \
    --local-dir StyleTTS2/

# PL-BERT (used as the prosodic text encoder)
huggingface-cli download gplsi/PL-BERT-va \
    step_50000.t7 config.yml util.py \
    --local-dir StyleTTS2/Utils/PLBERT/
```

### Inference

We recommend the reference-audio notebook:

```bash
jupyter notebook StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb
```

Provide a reference audio of the target voice and the Valencian text to synthesize. See the project repository for full inference details.

The pure text-to-speech notebook ([`StyleTTS2/Demo/Inference_LJSpeech.ipynb`](https://github.com/javimosa/styletts2-valenciano/blob/main/StyleTTS2/Demo/Inference_LJSpeech.ipynb)) is also available.

---

## Training Details

### Training data

The model was trained on the [`gplsi/corts-valencianes-asr`](https://huggingface.co/datasets/gplsi/corts-valencianes-asr) dataset (a parquet conversion of the public Corts Valencianes ASR corpus, originally released by Projecte Aina at the BSC).

| Split | Samples used |
|-------|--------------|
| Train | 89,330 (combined `clean_train_short` + `other_train_short`, length-filtered to 1–15 s) |
| Validation | 2,791 |

Audio was resampled from the original 16 kHz to 24 kHz to match the StyleTTS2 pipeline. Texts were phonemized at the sentence level (preserving inter-word phonetic effects) using `phonemizer` with `espeak-ng` and the `ca-va` variant.

### Training configuration

Both stages were trained using `accelerate` with mixed precision (fp16) on 4 × NVIDIA A100 64 GB GPUs.

**Stage 1 — acoustic backbone** (`train_first.py`):

- Epochs: 100
- Batch size: 8
- Max mel length: 512 frames
- TMA epoch (alignment refinement): 25
- Optimizer: AdamW, learning rate 1e-4

**Stage 2 — prosodic and diffusion modules** (`train_second.py`):

- Epochs: 100 (three phases)
  - Phase 1 (epochs 0 → 20): predictor, BERT, bert_encoder, predictor_encoder
  - Phase 2 (epochs 20 → 50): + diffusion
  - Phase 3 (epochs 50 → 100): + decoder and style_encoder fine-tuning + SLM adversarial (WavLM)
- BERT learning rate: 1e-5 (already pre-trained)
- Other learning rates: 1e-4
- Single-speaker mode (`multispeaker: false`)

The full configuration is in `Models/LJSpeech/config.yml` inside this repository.

### Model parameters

| Parameter | Value |
|---|---|
| Phoneme vocabulary (`n_token`) | 178 |
| Hidden dimension | 512 |
| Style dimension | 128 |
| Mel channels | 80 |
| Sample rate | 24 kHz |
| Decoder | iSTFTNet |
| Diffusion transformer | 3 layers, 8 heads |
| SLM | `microsoft/wavlm-base-plus` |

---

## Evaluation

The model was evaluated qualitatively by synthesizing a set of test sentences in Valencian and having native Valencian speakers assess the naturalness of the generated audio. Inference was performed with the reference-audio notebook (`StyleTTS2_Demo_LibriTTS.ipynb`), which is the recommended way to use this model (see [Limitations](#intended-uses-and-limitations)).

The native-speaker assessment confirms that:

- The pronunciation is recognizably Valencian, with the expected vowel realizations and word-level stress patterns.
- The main quality issue reported is background noise in the generated audio, particularly at the beginning and end of utterances. This noise is more pronounced when generating without a reference audio, and is attributable to the single-speaker / multi-speaker training mismatch described above. Pronunciation and naturalness themselves are not affected.

The model has not yet been benchmarked with formal subjective (MOS) or objective (WER, CER, ScoreQ) metrics. A future iteration with proper per-speaker labelling of the corpus could be valuable to substantially improve quality and stability.

---

## Citation

If this model contributes to your research, please cite the work:

```
@misc{gplsi2026styletts2valenciano,
      title={StyleTTS2 Valenciano},
      author={GPLSI Group, Universidad de Alicante},
      url={https://huggingface.co/gplsi/StyleTTS2-va},
      year={2026}
}
```

We also recommend citing the original StyleTTS2 and PL-BERT papers:

```
@inproceedings{li2023styletts2,
  title={StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models},
  author={Li, Yinghao Aaron and Han, Cong and Raghavan, Vinay S and Mischler, Gavin and Mesgarani, Nima},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@misc{li2023plbert,
      title={Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme Predictions},
      author={Yinghao Aaron Li and Cong Han and Xilin Jiang and Nima Mesgarani},
      year={2023},
      eprint={2301.08810},
      archivePrefix={arXiv}
}
```

---

## Additional Information

### Author

The [GPLSI](https://gplsi.dlsi.ua.es/) research group at the University of Alicante.

### Contact

For further information, please open an issue in the [project repository](https://github.com/javimosa/styletts2-valenciano).

### Copyright

Copyright (c) 2026 GPLSI, Universidad de Alicante.

### License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
