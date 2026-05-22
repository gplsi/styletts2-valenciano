# T06 — Entrenamiento de StyleTTS2 (adaptación a valenciano)

> Esta carpeta contiene el código de [StyleTTS2](https://github.com/yl4579/StyleTTS2) (ver [`README.md`](README.md) para la documentación original) con modificaciones aplicadas para entrenar el modelo en valenciano sobre el corpus de Corts Valencianes.
>
> El modelo final está publicado en [`gplsi/StyleTTS2-va`](https://huggingface.co/gplsi/StyleTTS2-va).

## Objetivo

Entrenar un modelo TTS de calidad humana en valenciano que, dada una secuencia fonética, sintetice audio inteligible y prosódicamente natural.

## Arquitectura general (resumen)

StyleTTS2 sintetiza audio en dos partes:

1. **Módulo acústico** — convierte fonemas en una representación intermedia (mel) y luego en audio:
   - **Text Aligner** (inicializado con un ASR pre-entrenado): alinea fonemas con frames de mel.
   - **Acoustic Text Encoder** (`text_encoder`): codifica fonemas con contexto local (CNN) y global (BiLSTM).
   - **Pitch Extractor** (inicializado con JDC): extrae la curva de F0 del audio.
   - **Acoustic Style Encoder** (`style_encoder`): comprime el mel en un vector de estilo `s_a`.
   - **Decoder** (iSTFTNet): genera la forma de onda final.
   - **Discriminadores** (MPD + MSD): evalúan la calidad del audio generado.

2. **Módulo prosódico** — predice la prosodia (duración, F0, energía) a partir del texto sin necesidad de audio de referencia:
   - **Prosodic Text Encoder** (PL-BERT, ver [t5](../t5/README.md)): representaciones contextuales de los fonemas con conocimiento semántico.
   - **BERT Encoder**: proyecta la salida de PL-BERT.
   - **Prosody Predictor** (`predictor`): predice duración + F0 + energía.
   - **Prosodic Style Encoder** (`predictor_encoder`): copia separada del style encoder, para estilo prosódico.
   - **Style Diffusion Denoiser** (`diffusion`): genera el vector de estilo a partir del texto (sin referencia).
   - **SLM Discriminator** (WavLM): juzga la naturalidad del audio en el espacio del modelo de lenguaje hablado.

Para más detalle, ver el paper original [`2306.07691v2.pdf`](2306.07691v2.pdf).

## Modelos auxiliares (`Utils/`)

Se inicializan los módulos del bloque acústico con pesos pre-entrenados que el repositorio original ofrece. Estos pesos son **agnósticos al idioma** (ASR para alineamiento, JDC para detección de pitch) y luego se afinan en T06.

| Carpeta | Modelo | Origen | Uso |
|---------|--------|--------|-----|
| `Utils/ASR/` | ASR pre-entrenado (inglés) | Repo original | Inicialización del Text Aligner |
| `Utils/JDC/` | Detector de pitch JDC | Repo original | Inicialización del Pitch Extractor |
| `Utils/PLBERT/` | PL-BERT valenciano | Entrenado en T05 | Prosodic Text Encoder |

## El entrenamiento se hace en dos fases

### Stage 1 — Pre-entrenamiento acústico (`train_first.py`)

Aprende a reconstruir el audio cuando se le entregan F0, energía y estilo extraídos del audio real. No aprende a predecir esos parámetros.

**Módulos que entrena**:
- Desde el inicio: `text_encoder`, `style_encoder`, `decoder`.
- Desde el epoch `TMA_epoch = 25`: además `text_aligner`, `pitch_extractor`, discriminadores `mpd` y `msd`.

**Módulos que NO entrena**: `bert`, `bert_encoder`, `predictor`, `predictor_encoder`, `diffusion`. Estos se entrenarán en Stage 2.

**Pérdidas**: `lambda_mel * L_mel + lambda_mono * L_mono + lambda_s2s * L_s2s + lambda_gen * L_adv + lambda_slm * L_slm`.

**Salida**: un único checkpoint `first_stage.pth` con los pesos de los módulos del bloque acústico.

#### Lanzar Stage 1

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path Configs/config_valenciano.yml
```

En nuestro entrenamiento se usaron 100 epochs (≈ varios días en 4× A100).

### Stage 2 — Entrenamiento conjunto (`train_second.py`)

Aprende a predecir prosodia y estilo a partir del texto y, finalmente, afina end-to-end con un discriminador basado en WavLM.

Tiene **tres fases progresivas** controladas por los parámetros `diff_epoch` y `joint_epoch` del config:

| Fase | Rango de epochs | Módulos que entrena |
|------|-----------------|---------------------|
| 1 | `0 → diff_epoch` (0 → 20) | `predictor`, `bert`, `bert_encoder`, `predictor_encoder` |
| 2 | `diff_epoch → joint_epoch` (20 → 50) | + `diffusion` |
| 3 | `joint_epoch → epochs_2nd` (50 → 100) | + `decoder`, `style_encoder` (afinado fino) + SLM adversarial (`wd`) |

**Importante**: `epochs_2nd` debe ser estrictamente mayor que `joint_epoch`, de lo contrario la Fase 3 nunca se ejecuta y el modelo no termina de afinar end-to-end. Por eso usamos `epochs_2nd: 100`.

**Salida**: checkpoints `epoch_2nd_<N>.pth` con todos los módulos. El modelo final usado es `epoch_2nd_00100.pth`.

#### Lanzar Stage 2

```bash
cd StyleTTS2
accelerate launch train_second.py --config_path Configs/config_valenciano.yml
```

## Configuración (`Configs/config_valenciano.yml`)

Diferencias principales respecto al `config.yml` original:

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `multispeaker` | `false` | Entrenamos como single-speaker (ver Limitaciones) |
| `n_token` | `178` | Vocabulario char-level IPA |
| `epochs_1st` | `100` | Stage 1 |
| `epochs_2nd` | `100` | Stage 2 (debe ser > `joint_epoch=50`) |
| `diff_epoch` | `20` | Inicio de Fase 2 |
| `joint_epoch` | `50` | Inicio de Fase 3 |
| `TMA_epoch` | `25` | Comienzo del afinado del aligner/pitch |
| `min_length` | `100` | Filtra mels con menos de 100 frames (evita crash del style encoder) |
| `bert_lr` | `1e-5` | LR bajo para afinar PL-BERT (ya pre-entrenado) |
| `PLBERT_dir` | `Utils/PLBERT/` | Directorio con el PL-BERT valenciano |
| `data_params.root_path` | `/media/datos1/.../styletts2_corts_dataset/` | Donde está la carpeta `wavs/` |

Las configs de batch_size y max_len dependen de la GPU. En 4× A100 64GB se usó `batch_size: 8`, `max_len: 512`.

## Inferencia

Tres notebooks disponibles según el escenario:

| Notebook | Cuándo usar |
|----------|-------------|
| [`Demo/Inference_Stage1_Valenciano.ipynb`](Demo/Inference_Stage1_Valenciano.ipynb) (también `test_stage1.py`) | Validar que Stage 1 reconstruye correctamente. Necesita audio real como referencia. |
| [`Colab/Inference_LJSpeech.ipynb`](Colab/Inference_LJSpeech.ipynb) | Inferencia "pura" texto → audio (como single-speaker). El estilo lo genera la difusión a partir del texto. |
| [`Colab/StyleTTS2_Demo_LibriTTS.ipynb`](Colab/StyleTTS2_Demo_LibriTTS.ipynb) | Inferencia con audio de referencia. **Recomendada en este proyecto** porque produce audio más limpio (ver Limitaciones). |

### Configurar la inferencia

Editar el notebook elegido para apuntar a:
- Modelo entrenado: `Models/LJSpeech/epoch_2nd_00100.pth` (o descargado de [`gplsi/StyleTTS2-va`](https://huggingface.co/gplsi/StyleTTS2-va))
- Config: `Models/LJSpeech/config.yml` (que se guarda automáticamente con el modelo durante el entrenamiento)
- Fonemizador: configurado para `ca-va` (ver [t3](../t3/README.md))

### Test de Stage 1 (`test_stage1.py`)

Script independiente que replica el forward pass de `train_first.py` sobre una muestra real, útil para validar que Stage 1 funciona antes de lanzar Stage 2.

```bash
python test_stage1.py \
    --config Configs/config_valenciano.yml \
    --model /media/datos1/tts-valenciano/Models/Valenciano/first_stage.pth \
    --out stage1_reconstruction.wav \
    --sample 0
```

## Bugs encontrados y corregidos

Durante la adaptación al valenciano se encontraron varios problemas en el código original o en su uso con nuestro setup. Quedan documentados aquí porque podrían reaparecer al replicar.

### 1. Mismatch de claves al cargar checkpoints (Issue #254 del repo original)

**Síntoma**: Stage 2 produce NaN loss desde el primer step.

**Causa**: Stage 1 guarda los pesos con el prefijo `module.` (al usar `accelerate`). Stage 2 cargaba con `strict=False` y, si las claves no coincidían exactamente, fallaba **silenciosamente** dejando los pesos en sus valores aleatorios iniciales.

**Fix**: en [`models.py`](models.py), función `load_checkpoint` — primero intentar `strict=True`; si falla, remapear claves manualmente y reintentar (sin silenciar).

### 2. Bug en JDC (squeeze elimina dimensión batch)

**Síntoma**: Crash al usar batch_size=1 por GPU. En sistemas multi-GPU con DataParallel y batch_size pequeño esto ocurre fácilmente.

**Causa**: en [`Utils/JDC/model.py`](Utils/JDC/model.py:137) un `squeeze()` sin argumentos elimina TODAS las dimensiones de tamaño 1, incluida la dimensión batch cuando hay un solo elemento.

**Fix**: cambiar `squeeze()` por `squeeze(-1)` (sólo elimina la última dimensión, que es la de clase).

### 3. Secuencias de texto > 512 tokens

**Síntoma**: PL-BERT crashea con error de `position_embeddings` cuando `texts.shape[1] > 512`.

**Causa**: PL-BERT tiene `max_position_embeddings: 512`. El parámetro `max_len` del config controla la longitud del mel, no la del texto. En frases parlamentarias largas la cadena fonética puede superar 512 caracteres.

**Fix**: en [`train_second.py`](train_second.py) saltar batches con `texts.shape[1] > 512`. Sólo afecta al ~0.33% de las muestras.

### 4. Mels demasiado cortos para el Style Encoder

**Síntoma**: `RuntimeError: Calculated padded input size per channel: (5 x 4). Kernel size: (5 x 5)`.

**Causa**: el Style Encoder tiene 4 bloques de downsample-by-half (16× reducción) seguidos de una conv 5×5 sin padding. Necesita ≥ 80 frames de mel en la dimensión temporal.

**Fix**: en [`train_second.py`](train_second.py) saltar batches con `mel_input_length.min() < 80`. Adicionalmente subir `min_length` en el config a 100. Afecta al ~3.7% de las muestras.

### 5. `pretrained_model` y `first_stage_path`

**Síntoma**: El optimizador empieza con valores extraños (betas raras, lr ya decaído) y el modelo entrena con pesos corruptos de un Stage 2 fallido anterior.

**Causa**: confusión entre los dos parámetros del config:
- `first_stage_path`: ruta al checkpoint de Stage 1 (cargar con `ignore_modules` para los módulos nuevos).
- `pretrained_model`: ruta a un checkpoint de Stage 2 para *resumir* — carga TODO incluido el optimizer state.

Si se pone `pretrained_model: <ruta a first_stage.pth>` se carga `first_stage.pth` dos veces y se restaura un optimizer state inadecuado.

**Fix**: dejar `pretrained_model: ""` cuando se entrena Stage 2 desde cero. Sólo poner valor si se quiere resumir un Stage 2 interrumpido.

## Limitaciones conocidas

### Single-speaker entrenado con datos multi-speaker

El corpus de Corts Valencianes contiene voces de muchos políticos diferentes, pero `multispeaker: false` en el config y todas las muestras tienen `speaker_id = 0`. Esto significa:

- El style encoder aprende una distribución "promedio" de todas las voces.
- La difusión genera estilos muy variables → en inferencia sin audio de referencia (`Inference_LJSpeech`) el audio sale ruidoso e inconsistente.
- Con audio de referencia (`Inference_LibriTTS`) la calidad es mucho mejor porque el estilo se fija desde la referencia y se evita el muestreo difuso.

**Cómo arreglarlo en una iteración futura**: re-anotar `train_list.txt` con identificadores reales de hablante y entrenar con `multispeaker: true` (requeriría re-hacer Stage 1 y Stage 2).

### Diffusion entrena pocos epochs

`diff_epoch = 20` y `joint_epoch = 50`, así que la difusión sólo se entrena 30 epochs (en Fase 2). Eso es suficiente para que la inferencia sea inteligible pero la diversidad estilística podría ser mayor con más epochs.

## Cómo replicar T06 con un dataset propio

Suponiendo que se ha completado T02 (datos en formato StyleTTS2), T03 (fonemizador) y T05 (PL-BERT entrenado):

1. Copiar `Configs/config_valenciano.yml` a `Configs/config_<idioma>.yml`.
2. Ajustar:
   - `data_params.root_path`: directorio con `wavs/`.
   - `data_params.train_data` / `val_data`: rutas a las listas StyleTTS2.
   - `PLBERT_dir`: dónde está el PL-BERT entrenado.
   - `multispeaker`: `true` si el dataset tiene varios hablantes etiquetados.
   - `batch_size` y `max_len`: ajustar a la VRAM disponible.
3. Lanzar Stage 1: `accelerate launch train_first.py --config_path Configs/config_<idioma>.yml`. Esperar ~100 epochs.
4. Lanzar Stage 2: `accelerate launch train_second.py --config_path Configs/config_<idioma>.yml`.
5. Validar Stage 1 con `python test_stage1.py` (opcional pero recomendado).
6. Probar la inferencia con un notebook de `Demo/`.

## Archivos modificados / añadidos en esta carpeta

| Archivo | Origen | Estado |
|---------|--------|--------|
| `models.py` | upstream + fix #1 | modificado |
| `Utils/JDC/model.py` | upstream + fix #2 | modificado |
| `train_second.py` | upstream + fixes #3, #4 | modificado |
| `Configs/config_valenciano.yml` | nuevo | añadido |
| `test_stage1.py` | nuevo | añadido |
| `Demo/Inference_Stage1_Valenciano.ipynb` | nuevo | añadido |
| `Utils/PLBERT/` | nuevo | añadido (= [`gplsi/PL-BERT-va`](https://huggingface.co/gplsi/PL-BERT-va)) |
| `Models/LJSpeech/` | nuevo | añadido (configs y checkpoints; los pesos también en HF) |

## Referencias

- [Paper StyleTTS2 (arXiv 2306.07691)](https://arxiv.org/abs/2306.07691)
- [Repo original yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [`gplsi/StyleTTS2-va`](https://huggingface.co/gplsi/StyleTTS2-va) — modelo final
- [Issue #254 (checkpoint key mismatch)](https://github.com/yl4579/StyleTTS2/issues/254)
