# T02 — Análisis y preparación de datos

## Objetivo

Partiendo del corpus público de Corts Valencianes (audio + transcripción), generar los artefactos en el formato que necesita StyleTTS2 para entrenar:

- Carpeta `wavs/` con un archivo de audio por muestra
- Archivos de texto con el formato `wavs/<id>.wav|<fonemas_IPA>|<speaker_id>` (uno por línea)

Estos archivos son la entrada directa de las dos fases de entrenamiento de StyleTTS2 (T06).

## Inputs

| Input | Origen | Descripción |
|-------|--------|-------------|
| Dataset original | [`projecte-aina/corts_valencianes_asr_a`](https://huggingface.co/datasets/projecte-aina/corts_valencianes_asr_a) | Audio + texto de las Corts Valencianes |
| Versión parquet | [`javiimts/corts-valencianes-asr-parquet`](https://huggingface.co/datasets/javiimts/corts-valencianes-asr-parquet) | Re-subida del original como parquet (el original usa loading scripts, ya no soportados) |

## Outputs

Generados en `/media/datos1/tts-valenciano/styletts2_corts_dataset/`:

| Archivo | Descripción | Tamaño |
|---------|-------------|--------|
| `wavs/` | Audio extraído del dataset y **resampleado a 24 kHz** (formato wav) | ~92 k archivos |
| `wavs_16k_backup/` | Audio original a 16 kHz (backup) | ~92 k archivos |
| `train_list.txt` | Lista de entrenamiento (sin filtrar) | salida directa de `dataset_conversion.ipynb` |
| `val_list.txt` | Lista de validación | salida directa |
| `OOD_texts.txt` | Textos out-of-distribution para entrenamiento adversarial SLM | salida directa |
| `train_list_filtered.txt` | Lista de entrenamiento tras filtrado por longitud | **89 330 muestras** (la usada por StyleTTS2) |
| `val_list_filtered.txt` | Lista de validación tras filtrado | **2 791 muestras** |

El formato de cada línea es: `wavs/<id>.wav|<fonemas_IPA>|<speaker_id>`

Ejemplo:
```
wavs/777d0be485c32fc946d21f5b13eebf5e.wav|el seɲˈoɾ kˈams χeɾˈaɾðo kˈamz...|0
```

## Pipeline

### Paso 1 — Convertir el dataset original a parquet

El dataset original (`projecte-aina/corts_valencianes_asr_a`) usa un loading script de HuggingFace que ya no es compatible con las versiones recientes de la librería `datasets`. Por eso re-subimos el dataset a [`javiimts/corts-valencianes-asr-parquet`](https://huggingface.co/datasets/javiimts/corts-valencianes-asr-parquet) en formato parquet.


### Paso 2 — Generar wavs y listas StyleTTS2

Notebook: [`dataset_conversion.ipynb`](dataset_conversion.ipynb)

Lo que hace, por celda:

1. **Cell 1** — Configura el fonemizador `phonemizer` con backend `espeak` y variante `ca-va` (valenciano). Ver [t3](../t3/README.md) para la justificación de esta elección.
2. **Cell 2** — Carga el dataset desde HuggingFace (`javiimts/corts-valencianes-asr-parquet`).
3. **Cells 3-4** — Inspección de splits y estructura.
4. **Cell 6** — Crea el directorio de salida y la subcarpeta `wavs/`.
5. **Cell 7** — Define `map_generate_batched`: para cada muestra fonemiza el texto, guarda el WAV con su `identifier` y produce una línea con formato StyleTTS2.
6. **Cells 9-10** — Mapea splits del dataset original a los archivos de StyleTTS2:
   - `train_list.txt` ← `clean_train_short` + `other_train_short`
   - `val_list.txt` ← `clean_test_short` + `other_test_short`
   - `OOD_texts.txt` ← `clean_dev_short` + `other_dev_short`

Procesa en paralelo (`num_proc = CPUs - 1`) con batches de 64.

> **Importante (single-speaker)**: el notebook asigna `speaker_id = "0"` a todas las muestras. El corpus tiene en realidad muchos hablantes (políticos), pero entrenamos como single-speaker. Esta es una limitación conocida que afecta a la calidad de la inferencia (ver T06) pero decidimos hacerlo así ya que el dataset original no identifica a los speakers.

### Paso 3 — Resampleo a 24 kHz (ad-hoc)

El corpus original de Corts Valencianes está a **16 kHz**, pero StyleTTS2 espera **24 kHz** (es el sample rate que aparece en `preprocess_params.sr` del config y al que están afinados los modelos auxiliares de `Utils/JDC/` y `Utils/ASR/`).

Después de Paso 2 los audios en `wavs/` están a 16 kHz. Hay que resamplearlos:

```bash
# Mover los originales a un backup
mv /media/datos1/tts-valenciano/styletts2_corts_dataset/wavs \
   /media/datos1/tts-valenciano/styletts2_corts_dataset/wavs_16k_backup

# Resamplear a 24 kHz
python -c "
import os
import librosa
import soundfile as sf
src = '/media/datos1/tts-valenciano/styletts2_corts_dataset/wavs_16k_backup'
dst = '/media/datos1/tts-valenciano/styletts2_corts_dataset/wavs'
os.makedirs(dst, exist_ok=True)
for fn in os.listdir(src):
    if not fn.endswith('.wav'): continue
    y, sr = librosa.load(os.path.join(src, fn), sr=24000)
    sf.write(os.path.join(dst, fn), y, 24000)
"
```

Alternativa: hacer la conversión dentro del notebook en Paso 2 añadiendo un `dataset = dataset.cast_column('audio', Audio(sampling_rate=24000))` antes del `.map(...)`.

### Paso 4 — Filtrado por longitud (ad-hoc)

Las listas generadas en el paso anterior se filtran para descartar audios demasiado cortos o demasiado largos. Esto produce los archivos `*_filtered.txt`, que son los que realmente se usan en el entrenamiento.

El criterio usado, basado en el análisis de [`dataset_exp.ipynb`](dataset_exp.ipynb), es el rango óptimo `1 s ≤ duración ≤ 15 s`.

Operación realizada como paso ad-hoc en terminal (no hay un script dedicado). Para replicar:

```bash
# Filtrar líneas cuyos wavs duran entre 1 y 15 segundos
# (script de ejemplo - adaptar según el entorno)
python -c "
import soundfile as sf, os
root='/media/datos1/tts-valenciano/styletts2_corts_dataset'
with open(f'{root}/train_list.txt') as f:
    for line in f:
        wav = line.split('|')[0]
        info = sf.info(os.path.join(root, wav))
        dur = info.frames / info.samplerate
        if 1.0 <= dur <= 15.0:
            print(line, end='')
" > train_list_filtered.txt
```

### Paso 5 (opcional) — Asegurar 24 kHz en HuggingFace

Script: [`inspect_hf_dataset.py`](inspect_hf_dataset.py)

Inspecciona el dataset en HuggingFace y, opcionalmente, lo actualiza para que los audios estén a 24 kHz (el sample rate que espera StyleTTS2):

```bash
# Solo inspeccionar
python inspect_hf_dataset.py

# Actualizar con audios 24kHz desde un directorio local
python inspect_hf_dataset.py --update --wavs_dir /ruta/a/wavs_24k
```

## Análisis exploratorio del dataset

Notebook: [`dataset_exp.ipynb`](dataset_exp.ipynb)

Útil para entender el dataset antes de entrenar y para tomar la decisión de filtrado:

- **Distribución de duración** (cells 3–4): histograma, box plot y curva acumulativa de duraciones por split.
- **Análisis de rangos óptimos para TTS** (cell 6): conteo de audios en los rangos `<1 s`, `1-15 s`, `15-20 s`, `>20 s`. Justifica el filtrado al rango óptimo `1-15 s` siguiendo el rango de LJSpeech y las recomendaciones del paper StyleTTS2.
- **Análisis de calidad** (cells 8–13): SNR simple (basado en silencios) y DNSMOS (calidad perceptual sin referencia). Permite estimar la calidad media del corpus.

> Las celdas 14–24 contienen experimentos **incompletos** de mejora de calidad de audio que no se aplicaron al pipeline final. Ver sección "Lo que se probó".

## Lo que se probó pero no se usó

| Experimento | Carpeta / archivo | Resultado |
|-------------|-------------------|-----------|
| Transcripción con WhisperX (BSC) | `whisper_catalan.py`, `launch_whisper.sh` | **No usado**: el dataset original ya viene con texto transcrito, no necesitamos reconocer audio. |
| Mejora de calidad con Mel-Band-Roformer | `Mel-Band-Roformer-Vocal-Model/`, `dataset_exp.ipynb` cells 14–17 | **Incompleto**: las celdas marcan `(INCOMPLETO)`. Los intentos de separación de voces que se hicieron no resultaron fiables |
| Mejora de calidad con ClearVoice | `clearervoice/`, `dataset_exp.ipynb` cells 18–24 | **Incompleto** |

## Cómo replicar T02 con un dataset propio

1. Subir el dataset a HuggingFace en formato parquet (o cargarlo desde disco).
2. Editar [`dataset_conversion.ipynb`](dataset_conversion.ipynb):
   - Cambiar el `language` del fonemizador (cell 1) si no es valenciano.
   - Cambiar el `dataset_id` en `load_dataset(...)` (cell 2).
   - Adaptar los nombres de splits (cell 9) a los del dataset propio.
   - Adaptar las columnas en `map_generate_batched` (cell 7) si se llaman distinto.
3. Ejecutar el notebook completo.
4. Filtrar por longitud (Paso 3 arriba).
5. Convertir `speaker_id` (Paso 4 arriba).
6. Verificar el sample rate con `inspect_hf_dataset.py`.

## Dependencias

- `phonemizer` con `espeak-ng` (con soporte de la variante objetivo, p. ej. `ca-va`)
- `datasets` (HuggingFace)
- `huggingface_hub`
- `soundfile`
- `librosa`
- `numpy`, `pandas`, `matplotlib`
- (Opcional, sólo para análisis de calidad) `speechmos`, `speechscore`
