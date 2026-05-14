# T04 — Selección y adaptación del tokenizador

## Objetivo

Decidir qué tokenizadores usar dentro del pipeline. La elección afecta al tamaño del vocabulario, a la generalización del modelo y a la facilidad de manejar fonemas o palabras no vistas durante el entrenamiento.

## Resumen — los dos tokenizadores que usamos

En este proyecto intervienen **dos tokenizadores con roles distintos**:

| # | Tokenizador | Cuándo se usa | Sobre qué opera | Para qué |
|---|-------------|---------------|-----------------|----------|
| 1 | **`TextCleaner`** (char-level IPA, 178 símbolos) | En tiempo de entrenamiento e inferencia, dentro del modelo | Cadena fonética IPA | Convierte cada carácter fonético en un entero — es la **entrada** del encoder de PL-BERT y de StyleTTS2 |
| 2 | **RoBERTa-ca** (`projecte-aina/roberta-base-ca-v2`) | **Sólo offline**, durante la preparación de datos para PL-BERT (T05a) | Texto ortográfico original (catalán/valenciano) | Asigna a cada palabra del corpus un ID, que luego sirve como **etiqueta** para el `word_predictor` de PL-BERT |

Ningún tokenizador WordPiece propio se usó en el modelo final. Este capítulo explica por qué y qué se probó.

## 1. Tokenizador de entrada: `TextCleaner` (char-level IPA)

Implementado por la clase `TextCleaner`, que viene incluida en StyleTTS2 ([`StyleTTS2/meldataset.py`](../StyleTTS2/meldataset.py) y [`StyleTTS2/text_utils.py`](../StyleTTS2/text_utils.py)).

```python
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭ..."

# 178 símbolos en total
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
```

Cada carácter IPA, cada signo de puntuación y cada letra latina se mapea a un entero único. Una cadena fonética como `"bˈɔn dˈia"` se convierte en una secuencia de IDs (uno por carácter).

### Por qué char-level y no WordPiece/BPE para la entrada

| Criterio | char-level (elegido) | WordPiece / BPE |
|----------|----------------------|------------------|
| **Tamaño del vocabulario** | 178 (fijo) | miles (depende del entrenamiento del tokenizador) |
| **OOV (out-of-vocabulary)** | Imposible — todos los fonemas IPA están cubiertos | Posible si aparece un carácter raro |
| **Robustez ante pronunciaciones extrañas** | Alta — el modelo ve los fonemas tal cual | Media — el WordPiece puede partir mal una palabra rara |
| **Compatibilidad con StyleTTS2 original** | Inmediata — es el comportamiento por defecto | Requiere modificar el dataloader |
| **Tamaño del embedding** | Pequeño (178 × hidden) | Grande (vocab × hidden) |

Como el alfabeto IPA es fijo (espeak nunca produce símbolos fuera de un conjunto cerrado) y el corpus es relativamente pequeño, **char-level es la elección natural**: no hay que entrenar nada, no hay vocabulario que mantener y la red es más pequeña.

## 2. Tokenizador para etiquetas semánticas: RoBERTa-ca

PL-BERT se entrena con dos tareas (ver [t5](../t5/README.md)):
- **Predicción de fonema enmascarado** (`mask_predictor`) — etiquetas: el carácter IPA original
- **Predicción de palabra** (`word_predictor`) — etiquetas: el ID de palabra correspondiente a esa posición

Para la segunda tarea hace falta un tokenizador que asigne a cada palabra del corpus un ID. Para el valenciano se eligió **`projecte-aina/roberta-base-ca-v2`** (RoBERTa-ca v2 del proyecto Aina del BSC).

### Por qué RoBERTa-ca

| Alternativa | Resultado |
|-------------|-----------|
| **`projecte-aina/roberta-base-ca-v2`** (la elegida) | Tokenizador BPE entrenado sobre catalán; cubre el vocabulario valenciano sin OOV; mantenido por el BSC |
| `transfo-xl-wt103` (el del PL-BERT original en inglés) | Está entrenado sobre Wikipedia inglesa; en valenciano genera muchos tokens "raros" y no captura el significado de las palabras |
| Tokenizador propio entrenado | Innecesario: RoBERTa-ca ya cubre el idioma y un tokenizador de palabras simple no aportaría supervisión semántica |

### Cómo se usa (offline, en T05a)

> Importante: RoBERTa-ca **no forma parte del modelo final**. Sólo se usa una vez, durante la preparación de datos de PL-BERT, para construir las etiquetas del `word_predictor`. Después de T05a no se vuelve a usar.

El flujo simplificado es:

1. Para cada frase del corpus, tokenizar el **texto ortográfico** (no el fonemizado) con RoBERTa-ca → IDs de RoBERTa-ca por palabra/sub-palabra.
2. Recoger todas las palabras únicas del corpus → asignarles enteros consecutivos `0..19021` (19 022 entradas en total).
3. Guardar el mapa en `t5/token_maps_valenciano.pkl` y los IDs en `t5/dataset_valenciano/`.

Esos enteros consecutivos son los que el `word_predictor` aprende a predecir. El tamaño 19 022 corresponde a las palabras únicas del corpus de Corts Valencianes.

## Lo que se probó pero no se usó

En las primeras fases del proyecto **no teníamos claro a qué se refería PL-BERT con "nivel de fonema"**. Cabían varias interpretaciones:

- **A** — cada posición del modelo es un único carácter IPA (char-level estricto).
- **B** — cada posición es una unidad sub-léxica de fonemas, entrenada con WordPiece o BPE sobre las cadenas fonéticas (similar a cómo BERT/RoBERTa estándar tokenizan texto).

La opción **A** es lo que hace PL-BERT realmente (un carácter IPA = un token), pero no era obvio leyendo la documentación. Para descartar la **B** se hizo el siguiente experimento:

- Entrenar un `BertTokenizer` WordPiece sobre el corpus ya fonemizado → publicarlo como [`javiimts/bert-ca-va-tokenizer`](https://huggingface.co/javiimts/bert-ca-va-tokenizer) (29 850 tokens).
- Entrenar una versión alternativa de PL-BERT (`BertForMaskedLM` estándar, sin `word_predictor`) usando ese tokenizador como entrada.

Tras comparar con la implementación de referencia y entender mejor el paper, quedó claro que la opción correcta es **A** (lo que también hace el repo original de PL-BERT y los demás puertos a otros idiomas). El experimento sirvió principalmente para entender bien la arquitectura.

**Descartado** porque:

1. La interpretación correcta del paper es char-level (opción A), no sub-léxica.
2. El enfoque char-level es más robusto: sin OOV, sin tokenizador adicional que mantener.
3. La supervisión semántica vía RoBERTa-ca (punto 2 de la sección anterior) ya aporta información a nivel de palabra como tarea auxiliar — más limpio que mezclar ambos roles en la entrada.

El modelo PL-BERT publicado en [`gplsi/PL-BERT`](https://huggingface.co/gplsi/PL-BERT) usa **char-level + RoBERTa-ca como etiquetas**, no este WordPiece.

## Inputs / Outputs (resumen)

| Tokenizador | Input | Output |
|-------------|-------|--------|
| `TextCleaner` | `"bˈɔn dˈia"` (string IPA) | `[42, 17, 89, 53, 0, 91, 17, 23, 12]` (lista de IDs char-level) |
| RoBERTa-ca (offline) | `"bon dia"` (texto ortográfico) | IDs de palabra → mapeados a `0..19021` en `token_maps_valenciano.pkl` |

## Cómo replicar

### Tokenizador de entrada (`TextCleaner`)

No hay nada que entrenar:

```python
from text_utils import TextCleaner
cleaner = TextCleaner()
ids = cleaner("bˈɔn dˈia")
```

Si se replica el proyecto con otro idioma:
- Si el fonemizador genera símbolos IPA estándar, el `TextCleaner` ya los cubre.
- Si aparece algún símbolo nuevo, basta con añadirlo a `_letters_ipa` en `meldataset.py` y `text_utils.py`.

### Tokenizador para etiquetas semánticas

Si se replica el proyecto con otro idioma, hay que elegir un tokenizador BPE/WordPiece pre-entrenado en ese idioma (idealmente uno robusto, p. ej. de la familia RoBERTa o BERT del idioma objetivo). Se usa **una vez** dentro del script T05a (ver [t5](../t5/README.md)) y nunca más.

## Estado actual de esta carpeta

Este directorio está vacío porque los experimentos del WordPiece descartado (`tokenizer_test.py`, `tokenizer_conversion.py`, `tokenizer_conversion_to_hf.py`, `tokenizer_coverage.ipynb`) se eliminaron al cerrar el proyecto, una vez confirmado que la decisión final era la combinación **`TextCleaner` (entrada) + RoBERTa-ca (etiquetas offline)**.

El código de los tokenizadores que sí se usan vive fuera de este directorio:

- [`StyleTTS2/meldataset.py`](../StyleTTS2/meldataset.py) — `TextCleaner` (entrada)
- [`StyleTTS2/text_utils.py`](../StyleTTS2/text_utils.py) — copia separada usada por la inferencia
- RoBERTa-ca: se descarga directamente de HuggingFace (`projecte-aina/roberta-base-ca-v2`) en el script T05a; no se almacena copia local
