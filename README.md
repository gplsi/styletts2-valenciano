# StyleTTS2 para el Valenciano

Adaptación del modelo [StyleTTS2](https://github.com/yl4579/StyleTTS2) para generar voz sintética en valenciano, entrenada sobre el corpus de les Corts Valencianes. Proyecto desarrollado para la Universidad de Alicante.

## Recursos publicados

| Recurso | Enlace |
|---------|--------|
| Modelo final | [gplsi/styletts2-valenciano](https://huggingface.co/gplsi/styletts2-valenciano) |
| PL-BERT valenciano | [gplsi/PL-BERT](https://huggingface.co/gplsi/PL-BERT) |
| Dataset (parquet) | [javiimts/corts-valencianes-asr-parquet](https://huggingface.co/datasets/javiimts/corts-valencianes-asr-parquet) |
| Paper StyleTTS2 | [arXiv 2306.07691](https://arxiv.org/abs/2306.07691) |
| Repositorio | [javimosa/styletts2-valenciano](https://github.com/javimosa/styletts2-valenciano) |

## Tareas del proyecto

El proyecto se dividió en seis tareas. Cada una tiene su propio README con la documentación detallada.

| Tarea | Carpeta | Descripción |
|-------|---------|-------------|
| T01 | [t1/](t1/) | Exploración inicial sobre LJSpeech y LibriTTS (no completada) |
| T02 | [t2/](t2/) | Preparación del corpus: audio + texto al formato de StyleTTS2 |
| T03 | [t3/](t3/) | Transcripción fonética con espeak-ng `ca-va` |
| T04 | [t4/](t4/) | Selección del tokenizador (char-level IPA + RoBERTa-ca como etiquetas) |
| T05 | [t5/](t5/) | Entrenamiento del PL-BERT valenciano |
| T06 | [StyleTTS2/](StyleTTS2/) | Entrenamiento de StyleTTS2 (Stage 1 + Stage 2) e inferencia |

Para una guía paso a paso del entrenamiento completo, ver [TRAINING.md](TRAINING.md).

## Estructura del repositorio

Carpeta raíz:

- `README.md` (este archivo)
- `TRAINING.md`: guía de entrenamiento paso a paso

`StyleTTS2/`: fork de StyleTTS2 (T06).

- `README.md`: documentación original de upstream
- `README_VALENCIANO.md`: nuestra adaptación
- `Configs/config_valenciano.yml`
- `train_first.py`, `train_second.py`, `test_stage1.py`
- `models.py`, `meldataset.py`
- `Utils/PLBERT/`: PL-BERT para inferencia
- `Utils/ASR/`, `Utils/JDC/`: modelos auxiliares pre-entrenados
- `Demo/`: notebooks de inferencia
- `Models/LJSpeech/`: checkpoints locales

`t1/`: exploración (T01, incompleto).

`t2/`: preparación de datos (T02).

- `dataset_conversion.ipynb`: script principal
- `dataset_exp.ipynb`: análisis exploratorio
- `inspect_hf_dataset.py`
- Experimentos descartados: whisper, ClearVoice, Mel-Band-Roformer

`t3/`: fonetización (T03).

- `phonemizer_test.py`

`t4/`: tokenizador (T04, sólo documentación).

`t5/`: PL-BERT (T05).

- `T05a_prepare_dataset.py`
- `T05b_train_plbert.py`
- `text_utils.py`
- `dataset_valenciano/`, `token_maps_valenciano.pkl`, `dataset_config.json`
- `checkpoints_valenciano/`

## Requisitos

- Python 3.12.
- espeak-ng 1.51 o superior, con la variante `ca-va`. Recomendado compilar desde fuente; ver [t3/README.md](t3/README.md).
- CUDA y GPU para entrenar. Para inferencia basta una GPU de gama media.
- ffmpeg (lo necesita `torchcodec` al cargar audio).

Instalación completa en [TRAINING.md](TRAINING.md#0-pre-requisitos).

## Inferencia con el modelo entrenado

```bash
# Descargar modelo y PL-BERT desde HuggingFace
mkdir -p StyleTTS2/Models/LJSpeech
huggingface-cli download gplsi/styletts2-valenciano \
    Models/LJSpeech/epoch_2nd_00100.pth \
    Models/LJSpeech/config.yml \
    --local-dir StyleTTS2/

huggingface-cli download gplsi/PL-BERT \
    step_50000.t7 config.yml util.py \
    --local-dir StyleTTS2/Utils/PLBERT/

# Abrir el notebook
jupyter notebook StyleTTS2/Demo/Inference_LJSpeech.ipynb
```

Para mejor calidad usar el notebook con audio de referencia: [StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb](StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb). Ver más abajo el porqué.

## Decisiones de diseño

- Tokenizador char-level IPA (178 símbolos) en lugar de WordPiece/BPE. Detalle en [t4/README.md](t4/README.md).
- PL-BERT entrenado con supervisión semántica de RoBERTa-ca, para que el encoder capture significado además de patrones fonéticos. Detalle en [t5/README.md](t5/README.md).
- Fonemización a nivel de frase (no palabra por palabra), para preservar efectos fonéticos entre palabras.

## Limitaciones conocidas

El modelo se entrenó como single-speaker, pero el corpus de Corts Valencianes contiene voces de muchos políticos distintos, todos etiquetados como `speaker_id = 0`. Como consecuencia, la inferencia "texto puro" con `Inference_LJSpeech.ipynb` genera estilos inconsistentes y a veces ruidosos. La inferencia con audio de referencia (`StyleTTS2_Demo_LibriTTS.ipynb`) funciona claramente mejor. Más detalle en [StyleTTS2/README_VALENCIANO.md](StyleTTS2/README_VALENCIANO.md).

Durante el desarrollo aparecieron varios bugs en el código original (NaN por mismatch de claves al cargar checkpoints, crash de JDC con batch=1, crash del style encoder con mels cortos, BERT con secuencias > 512). Todos están corregidos en este fork y documentados en `StyleTTS2/README_VALENCIANO.md`.

## Seguimiento del proyecto

Las tareas y el seguimiento se llevaron en [Linear](https://linear.app/tts-valenciano).

## Referencias

- Paper StyleTTS2: [Li et al., NeurIPS 2023](https://arxiv.org/abs/2306.07691)
- Paper PL-BERT: [Li et al., 2023](https://arxiv.org/abs/2301.08810)
- Repositorio upstream StyleTTS2: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- Repositorio upstream PL-BERT: [yl4579/PL-BERT](https://github.com/yl4579/PL-BERT)
- Modelo de catalán: [projecte-aina/roberta-base-ca-v2](https://huggingface.co/projecte-aina/roberta-base-ca-v2)
- Dataset original: [projecte-aina/corts_valencianes_asr_a](https://huggingface.co/datasets/projecte-aina/corts_valencianes_asr_a)
