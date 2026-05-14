# T05 — Entrenamiento del Encoder Fonético (PL-BERT)

## Objetivo

Entrenar un encoder Transformer (Albert) que produzca representaciones contextuales de los fonemas. Estas representaciones son la entrada del **Prosodic Text Encoder** de StyleTTS2 en su segunda fase de entrenamiento (T06), lo que permite predecir prosodia (duración, tono, energía) consciente del significado de la frase, no sólo de la cadena fonética.

El modelo resultante está publicado en HuggingFace: [`gplsi/PL-BERT`](https://huggingface.co/gplsi/PL-BERT) 

## Idea base: PL-BERT

PL-BERT (Phoneme-Level BERT, [Yinghao Aaron Li et al., 2023](https://arxiv.org/abs/2301.08810)) es un BERT que opera a nivel de carácter fonético y se entrena con dos objetivos simultáneos:

1. **Predicción de fonemas enmascarados** (mask_predictor): qué carácter IPA falta en una posición enmascarada — fuerza al modelo a aprender patrones fonotácticos.
2. **Predicción de palabra** (word_predictor): a qué palabra pertenece cada posición — supervisión semántica que enseña al modelo a relacionar secuencias fonéticas con significado.

Tras entrenar, sólo se conserva el encoder Albert; las dos cabezas de predicción se descartan.

## Adaptación al valenciano

El PL-BERT original (en inglés) usa el tokenizador `transfo-xl-wt103` para generar las etiquetas del `word_predictor`. Para el valenciano hemos sustituido eso por **RoBERTa-ca** (`projecte-aina/roberta-base-ca-v2`), un modelo entrenado para catalán.

El uso es **offline**: RoBERTa-ca no es parte del modelo que se entrena. Sólo se usa una vez, durante T05a, para asignarle a cada palabra del corpus un identificador de token. Esos identificadores se mapean a un vocabulario reducido (19 022 palabras únicas en el corpus) y se guardan en `token_maps_valenciano.pkl`.

| Arquitectura PL-BERT original | Versión valenciana (este proyecto) |
|--------------------------------|------------------------------------|
| Encoder: AlbertModel | igual |
| Tokenizador de input: char-level IPA | igual (`TextCleaner`, 178 símbolos) |
| `mask_predictor` → 178 clases | igual |
| `word_predictor` → ~85 k clases (transfo-xl) | **19 022 clases** (RoBERTa-ca filtrado al corpus) |

## Pipeline

T05 se divide en dos sub-tareas:

### T05a — Preparación del dataset

Script: [`T05a_prepare_dataset.py`](T05a_prepare_dataset.py)


#### Qué hace

1. Carga `javiimts/corts-valencianes-asr-parquet` y concatena los splits `clean_train_short` + `other_train_short` (89 331 muestras).
2. Para cada muestra:
   - **Fonemiza la frase completa** con espeak `ca-va` (preserva efectos entre palabras, ver [t3](../t3/README.md)).
   - **Separa** el resultado por espacios → lista de palabras fonemizadas.
   - **Tokeniza cada palabra ortográfica** del texto original con RoBERTa-ca (`projecte-aina/roberta-base-ca-v2`), con la convención de espacio inicial (`encode(" " + word)`), y guarda el ID del primer subword.
   - **Filtra** muestras donde el número de palabras fonemizadas no coincide con el número de palabras del texto (raro — pasa con números o abreviaturas).
3. Construye `token_maps_valenciano.pkl` asignando enteros consecutivos `0..N-1` a los IDs de RoBERTa-ca **en orden de primera aparición** en el corpus. Añade una entrada especial para el separador de palabras.
4. Genera `dataset_config.json` con los metadatos.

#### Outputs

##### 1. `dataset_valenciano/` — dataset de HuggingFace en disco
89 331 muestras con dos columnas alineadas 1:1:

```python
{
    "phonemes":  ["el", "seɲˈoɾ", "kˈams", ...],
    "input_ids": [336,  3878,     5459,    ...]
}
```

- `phonemes`: lista de palabras fonemizadas (la fonemización se hizo a nivel de frase y luego se separó por espacios).
- `input_ids`: ID de RoBERTa-ca del primer subword de cada palabra **ortográfica** (no fonemizada). Es lo que el `word_predictor` aprenderá a predecir.

##### 2. `token_maps_valenciano.pkl` — mapeo roberta_id → token interno
Diccionario `{roberta_id: {"word": "<orth_word>", "token": <int>}}` con 19 022 entradas. Convierte los IDs de RoBERTa-ca (que pueden ser hasta ~50 000 y dispersos) en enteros consecutivos `0..19021` que sirven como clases del `word_predictor`.

La entrada especial `token_maps[50255] = {"word": "<sep>", "token": 19021}` representa el separador entre palabras.

##### 3. `dataset_config.json` — metadatos
```json
{
  "word_separator": 50255,
  "num_vocab": 19022,
  "tokenizer": "projecte-aina/roberta-base-ca-v2"
}
```
- `word_separator`: ID de RoBERTa-ca elegido como separador (no entra en colisión con los IDs reales del corpus).
- `num_vocab`: número de clases del `word_predictor` (palabras únicas + 1 separador).
- `tokenizer`: tokenizador que generó los `input_ids` (sólo se usa offline, ver [t4](../t4/README.md)).

#### Cómo lanzar T05a

```bash
cd t5/
python T05a_prepare_dataset.py

# Subset pequeño para depurar
python T05a_prepare_dataset.py --limit 100

# Cambiar el dataset / splits
python T05a_prepare_dataset.py \
    --dataset usuario/mi-corpus-parquet \
    --splits train_clean train_other
```

Tiempo aproximado: ~30 min sobre las ~89k muestras (la fonemización con espeak es el cuello de botella).

### T05b — Entrenamiento del modelo

Script: [`T05b_train_plbert.py`](T05b_train_plbert.py)

#### Arquitectura

`MultiTaskModel` que envuelve un `AlbertModel`:

| Parámetro | Valor |
|-----------|-------|
| `vocab_size` | 178 (char-level IPA) |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `intermediate_size` | 2048 |
| `max_position_embeddings` | 512 |
| `dropout` | 0.1 |

Cabezas:
- `mask_predictor`: `nn.Linear(768, 178)` — predicción de fonemas
- `word_predictor`: `nn.Linear(768, 19022)` — predicción de palabras

#### Dataloader (`FilePathDataset`)

Para cada muestra:

1. Lee `phonemes` (lista de palabras fonemizadas) e `input_ids` (uno por palabra).
2. Para cada palabra, decide aleatoriamente si enmascararla (con probabilidad `word_mask_prob = 0.15`):
   - Si se enmascara, reemplaza todos los caracteres de la palabra por `M` (con un pequeño porcentaje de `replace_prob = 0.2` y `phoneme_mask_prob = 0.1` que reemplazan por fonemas aleatorios o dejan el original — añade ruido como en BERT clásico).
3. Concatena los caracteres en una cadena fonética separada por espacios.
4. Cada carácter de la cadena hereda el `input_id` de la palabra a la que pertenece. Entre palabras se inserta un `word_separator` para que el modelo aprenda los límites de palabra.
5. Si el resultado supera `max_mel_length = 512`, se recorta aleatoriamente a 512 caracteres.
6. Tokeniza la cadena fonética con `TextCleaner` (char-level IPA, ver [t4](../t4/README.md)).
7. Mapea los `input_ids` de palabra a tokens internos via `token_maps_valenciano.pkl`.
8. Devuelve `(phoneme_ids, mapped_words, labels, masked_indices)`.

#### Función de pérdida

```python
loss = loss_vocab + loss_token
```

- `loss_vocab` (cross-entropy): predecir el token de palabra en **todas** las posiciones.
- `loss_token` (cross-entropy): predecir el fonema original en **sólo** las posiciones enmascaradas.

#### Hiperparámetros usados

| Parámetro | Valor |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Mixed precision | fp16 (vía `accelerate`) |
| `num_steps` (configurado) | 500 000 |
| **Pasos efectivos del modelo final** | **50 000** (`step_50000.t7`) |
| `save_interval` | 10 000 |

El entrenamiento se podría haber dejado correr más, pero las métricas (perplexity ~5.93, accuracy de palabra ~97%) eran ya muy buenas a los 50 k pasos y se decidió cerrar el experimento ahí.

#### Cómo lanzar el entrenamiento

Desde la carpeta `t5/`:

```bash
# Sin distribución (1 GPU)
python T05b_train_plbert.py

# Con accelerate (multi-GPU)
accelerate launch T05b_train_plbert.py
```

El script:
- Crea (o reanuda) el directorio `checkpoints_valenciano/`.
- Si existe un checkpoint anterior (`step_<N>.t7`), lo carga y continúa.
- Guarda un checkpoint cada 10 000 pasos.

## Outputs canónicos

| Output | Ubicación |
|--------|-----------|
| Checkpoints (todos los pasos guardados) | `checkpoints_valenciano/step_*.t7` |
| Checkpoint final | `checkpoints_valenciano/step_50000.t7` |
| Modelo en HuggingFace | [`gplsi/PL-BERT`](https://huggingface.co/gplsi/PL-BERT) (formato `safetensors` para integrar con StyleTTS2) |
| Copia para StyleTTS2 | `../StyleTTS2/Utils/PLBERT/` |

## Cómo se usa el modelo en StyleTTS2

En T06 (Stage 2 de StyleTTS2) se carga el encoder Albert (sin las cabezas) como **Prosodic Text Encoder**:

```python
from Utils.PLBERT.util import load_plbert
plbert = load_plbert("Utils/PLBERT/")  # contiene step_50000.t7 + config
```

Las representaciones del encoder se proyectan (por `bert_encoder`) y se entregan al `predictor` para predecir duración y prosodia.

## Lo que se probó pero no se usó

| Experimento | Resultado |
|-------------|-----------|
| **Entrenamiento alternativo con `BertForMaskedLM`** (estándar de HuggingFace, sin `word_predictor`, sin RoBERTa-ca) | Descartado. Sin la supervisión semántica el modelo aprende patrones fonotácticos pero no captura significado, lo que se esperaba que perjudicase la prosodia en T06. |
| **Tokenizador WordPiece sobre fonemas** (`javiimts/bert-ca-va-tokenizer`) | Descartado — ver [t4](../t4/README.md). |
| **Modelos PL-BERT pre-entrenados de BSC** (`BSC-LT/PL-BERT-wp-ca`) | Sólo usados para comparar arquitecturas. El modelo de BSC tiene `word_predictor` con ~1.13 M clases (trabajan con `transfo-xl`), no compatible directamente con nuestro pipeline. |

## Archivos en esta carpeta

| Archivo | Descripción |
|---------|-------------|
| [`T05a_prepare_dataset.py`](T05a_prepare_dataset.py) | Script de preparación del dataset PL-BERT |
| [`T05b_train_plbert.py`](T05b_train_plbert.py) | Script de entrenamiento del modelo |
| [`text_utils.py`](text_utils.py) | `TextCleaner` (copia local; idéntico al de StyleTTS2) |
| `dataset_valenciano/` | Dataset HF preparado por T05a (89 331 muestras) |
| `token_maps_valenciano.pkl` | Mapeo word_id → token interno (19 022 entradas) |
| `dataset_config.json` | Metadatos del dataset |
| `checkpoints_valenciano/` | Checkpoints guardados durante el entrenamiento |

## Dependencias

- `torch`, `transformers` (AlbertConfig, AlbertModel)
- `accelerate` (entrenamiento distribuido / mixed precision)
- `datasets` (lectura del dataset preparado)
- `tqdm`
- (Sólo para T05a) `phonemizer` con espeak-ng `ca-va`, `transformers` con tokenizador RoBERTa-ca

## Referencias

- [PL-BERT (paper)](https://arxiv.org/abs/2301.08810)
- [PL-BERT (repo original)](https://github.com/yl4579/PL-BERT)
- [RoBERTa-ca v2](https://huggingface.co/projecte-aina/roberta-base-ca-v2)
- [`gplsi/PL-BERT`](https://huggingface.co/gplsi/PL-BERT)
