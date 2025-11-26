---
language:
- ca
- val
license: mit
tags:
- text-to-speech
- phoneme-bert
- styletts2
- valencian
- catalan
pipeline_tag: feature-extraction
---

# PL-BERT Valenciano

Modelo PL-BERT (Phoneme-Level BERT) entrenado en valenciano para uso con StyleTTS2.

## Descripción del Modelo

Este es un modelo BERT entrenado exclusivamente con fonemas valencianos/catalanes, diseñado para ser utilizado como encoder de texto en sistemas de síntesis de voz (TTS) como StyleTTS2.

### Características

- **Arquitectura**: BERT base
- **Vocabulario**: N/A tokens fonéticos
- **Hidden size**: 768
- **Num layers**: 12
- **Attention heads**: 12
- **Max position embeddings**: 512
- **Training steps**: 25000

### Tokenizador

Este modelo utiliza el tokenizador: `javiimts/bert-ca-va-tokenizer`

## Entrenamiento

El modelo fue entrenado usando:
- **Dataset**: Corts Valencianes (transcripciones parlamentarias)
- **Objetivo**: Masked Language Modeling (MLM) con 15% de masking
- **Optimizador**: AdamW
- **Precisión mixta**: FP16
- **Steps**: 25000

## Integración con StyleTTS2

Este modelo está diseñado para reemplazar el PL-BERT original en StyleTTS2. Para usarlo:

1. Descarga los archivos del modelo
2. Actualiza la ruta `PLBERT_dir` en tu configuración de StyleTTS2
3. El modelo se cargará automáticamente usando `util.py`
4. Redimensiona los embeddings de StyleTTS2 para que coincidan con el nuevo tamaño del vocabulario

## Archivos incluidos

- `step_25000.t7`: Checkpoint del modelo entrenado (PyTorch)
- `config.yml`: Configuración de hiperparámetros
- `util.py`: Utilidades para cargar el modelo
- `README.md`: Esta documentación

