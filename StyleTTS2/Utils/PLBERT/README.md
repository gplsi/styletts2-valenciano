---
license: apache-2.0
language:
- ca
tags:
- valenciano
- catalan
- tts
- styletts2
- plbert
---

# PL-BERT Valenciano

Modelo PL-BERT entrenado para síntesis de voz en valenciano/catalán, diseñado para uso con StyleTTS2.

## Descripción

Este modelo es un AlbertModel entrenado con arquitectura dual:
- **Encoder**: AlbertModel (este modelo)
- **mask_predictor**: Predicción de fonemas enmascarados (descartado tras entrenamiento)
- **word_predictor**: Predicción de palabras con RoBERTa-ca (descartado tras entrenamiento)

## Configuración

| Parámetro | Valor |
|-----------|-------|
| vocab_size | 178 |
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| intermediate_size | 2048 |
| embedding_size | 128 (default AlbertModel) |

## Entrenamiento

- **Dataset**: Corts Valencianes (~89,331 muestras)
- **Steps**: 50000
- **Batch size**: 32
- **Supervisión semántica**: RoBERTa-ca (projecte-aina/roberta-base-ca-v2)

## Métricas

| Métrica | Valor |
|---------|-------|
| Perplexity | 5.93 |
| Word Accuracy Top-1 | 97.23% |
| Word Accuracy Top-5 | 99.18% |

## Uso con StyleTTS2

```python
from transformers import AlbertModel, AlbertConfig

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state

# Cargar modelo
model = CustomAlbert.from_pretrained("javiimts/plbert-valenciano")
model.eval()
```

## Licencia

Apache 2.0
