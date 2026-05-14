# Guía de entrenamiento

Guía paso a paso para entrenar StyleTTS2 valenciano desde cero. Si quieres entrenar con otro idioma o corpus, las notas al final de cada paso indican qué hay que cambiar.

---

## 0. Pre-requisitos

```bash
# Clonar el repo
git clone https://github.com/javimosa/styletts2-valenciano.git
cd styletts2-valenciano

# Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate

# Dependencias
pip install -r StyleTTS2/requirements.txt
pip install phonemizer transformers datasets accelerate huggingface_hub python-dotenv soundfile librosa
```

Instala espeak-ng con soporte de `ca-va`:

```bash
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh && ./configure --prefix=/usr/local && make && sudo make install && sudo ldconfig
cd ..

# Verifica
espeak-ng --voices=ca | grep ca-va
```

Configura el token de HuggingFace en un archivo `.env` en la raíz:

```
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## 1. Preparar el corpus

### 1.1 Si usas Corts Valencianes (default)

No tienes que hacer nada — el script de T02 ya apunta a `javiimts/corts-valencianes-asr-parquet`.

### 1.2 Si usas otro corpus

Sube tu dataset a HuggingFace en formato parquet con estas columnas:
- `identifier` (str) — ID único de cada muestra
- `text` (str) — transcripción
- `audio` (Audio) — el audio (cualquier sample rate)

Luego edita [`t2/dataset_conversion.ipynb`](t2/dataset_conversion.ipynb):
- Cell 2: cambia `load_dataset("javiimts/corts-valencianes-asr-parquet")` por tu dataset.
- Cell 9: ajusta los nombres de splits a los del tuyo (`train_splits`, `val_splits`, `ood_splits`).

---

## 2. Generar wavs + listas (T02)

```bash
cd t2/
jupyter notebook dataset_conversion.ipynb
```

Ejecuta todas las celdas. Esto produce en `/media/datos1/tts-valenciano/styletts2_corts_dataset/`:

- `wavs/`: audios extraídos del dataset
- `train_list.txt`: lista de entrenamiento
- `val_list.txt`: lista de validación
- `OOD_texts.txt`: textos out-of-distribution para Stage 2

> Si quieres cambiar el directorio de salida, edita la variable `output_dir` en Cell 6.

### 2.1 Resamplear a 24 kHz (si tu corpus está a otro SR)

StyleTTS2 espera 24 kHz. Si tu audio está a otro sample rate (Corts está a 16 kHz):

```bash
# Mover los originales
DATA_DIR=/media/datos1/tts-valenciano/styletts2_corts_dataset
mv $DATA_DIR/wavs $DATA_DIR/wavs_original_sr

python -c "
import os, librosa, soundfile as sf
src='$DATA_DIR/wavs_original_sr'
dst='$DATA_DIR/wavs'
os.makedirs(dst, exist_ok=True)
for fn in os.listdir(src):
    if not fn.endswith('.wav'): continue
    y, _ = librosa.load(os.path.join(src, fn), sr=24000)
    sf.write(os.path.join(dst, fn), y, 24000)
"
```

### 2.2 Filtrar por duración

Quedarse con audios de entre 1 y 15 segundos:

```bash
DATA_DIR=/media/datos1/tts-valenciano/styletts2_corts_dataset

for split in train val; do
python -c "
import soundfile as sf, os
root='$DATA_DIR'
with open(f'{root}/${split}_list.txt') as f:
    for line in f:
        wav = line.split('|')[0]
        info = sf.info(os.path.join(root, wav))
        dur = info.frames / info.samplerate
        if 1.0 <= dur <= 15.0:
            print(line, end='')
" > $DATA_DIR/${split}_list_filtered.txt
done
```

### 2.3 Copiar las listas a StyleTTS2/Data/

```bash
mkdir -p StyleTTS2/Data
cp /media/datos1/tts-valenciano/styletts2_corts_dataset/train_list_filtered.txt StyleTTS2/Data/train_list.txt
cp /media/datos1/tts-valenciano/styletts2_corts_dataset/val_list_filtered.txt   StyleTTS2/Data/val_list.txt
cp /media/datos1/tts-valenciano/styletts2_corts_dataset/OOD_texts.txt           StyleTTS2/Data/OOD_texts.txt
```

---

## 3. Entrenar PL-BERT (T05)

### 3.1 Preparar el dataset (T05a)

```bash
cd t5/
python T05a_prepare_dataset.py
```

> Produce: `dataset_valenciano/`, `token_maps_valenciano.pkl`, `dataset_config.json`. Dura ~30 min (espeak es el cuello de botella).

> Para otro idioma cambia los splits y el tokenizador editando las constantes al inicio de [`T05a_prepare_dataset.py`](t5/T05a_prepare_dataset.py): `ROBERTA_CA_TOKENIZER` (apunta a un tokenizador del idioma destino) y argumentos `--splits`.

### 3.2 Entrenar (T05b)

```bash
# 1 GPU
python T05b_train_plbert.py

# Multi-GPU (recomendado)
accelerate launch T05b_train_plbert.py
```

> Produce checkpoints en `t5/checkpoints_valenciano/step_*.t7` cada 10 000 pasos. Para nosotros 50 000 pasos fueron suficientes (perplexity ~5.93). Ajustable en la variable `num_steps` dentro del script.

### 3.3 Convertir el checkpoint para StyleTTS2

StyleTTS2 espera tener `step_50000.t7` y `config.yml` dentro de `StyleTTS2/Utils/PLBERT/`. Copia los archivos:

```bash
cp t5/checkpoints_valenciano/step_50000.t7 StyleTTS2/Utils/PLBERT/step_50000.t7
# El config.yml ya existe en esa carpeta — sólo se sustituye si has cambiado los hiperparámetros
```

> Como alternativa: si no quieres entrenar PL-BERT, descarga el nuestro desde HuggingFace:
> ```bash
> huggingface-cli download gplsi/PL-BERT step_50000.t7 config.yml --local-dir StyleTTS2/Utils/PLBERT/
> ```

---

## 4. Configurar StyleTTS2

Edita [`StyleTTS2/Configs/config_valenciano.yml`](StyleTTS2/Configs/config_valenciano.yml). Cambia al menos:

```yaml
log_dir: "/ruta/donde/se/guardarán/checkpoints/Models/Valenciano"
batch_size: 8      # ajustar a la VRAM (8 va bien en A100 64GB)
max_len: 512       # frames de mel; 400-512 va bien

data_params:
  root_path: "/media/datos1/tts-valenciano/styletts2_corts_dataset/"  # donde está wavs/

model_params:
  multispeaker: false   # si tu dataset es single-speaker o no etiqueta speakers
```

> Si tienes un dataset multi-speaker con `speaker_id` etiquetado, pon `multispeaker: true` y asegúrate de que `train_list.txt` tiene IDs reales en la 3ª columna.

---

## 5. Entrenar Stage 1

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path Configs/config_valenciano.yml
```

> Para monitorizar: `tensorboard --logdir <log_dir>/tensorboard`.

---

## 6. Entrenar Stage 2

Antes de lanzar, asegúrate de que el config tiene:

```yaml
first_stage_path: "first_stage.pth"     # nombre del checkpoint de Stage 1 dentro de log_dir
pretrained_model: ""                    # IMPORTANTE: vacío al empezar Stage 2 desde cero
load_only_params: false
epochs_2nd: 100                         # DEBE ser > joint_epoch (50)
diff_epoch: 20
joint_epoch: 50
```

Lanza:

```bash
cd StyleTTS2
accelerate launch train_second.py --config_path Configs/config_valenciano.yml
```

Salidas: `log_dir/epoch_2nd_*.pth` cada `save_freq` epochs. El modelo final es `epoch_2nd_00100.pth`.

### Atajos en caso de error

- **NaN loss desde el step 1**: probablemente Stage 1 no se cargó (mismatch de claves). Verifica que `first_stage_path` apunta al archivo correcto y que existe en `log_dir`.
- **Crash en `predictor_encoder` con kernel size > input**: `min_length` en el config es demasiado bajo (debe ser ≥ 80 frames). Sube a `min_length: 100`.
- **Crash en BERT con `position_embeddings 524`**: alguna muestra tiene > 512 caracteres fonéticos. El fix ya está en `train_second.py` (skip de batches > 512), pero verifica que no hayas hecho un git reset que quite los fixes (ver `StyleTTS2/README_VALENCIANO.md` sección "Bugs").
- **Crash con set_trace en SLURM**: el script tiene un `set_trace()` legítimo cuando detecta NaN; en SLURM crashea porque no es interactivo. Si lo ves, hay un NaN — investiga **antes** del set_trace, no el set_trace en sí.

---

## 7. Reanudar entrenamiento interrumpido

Si Stage 2 se interrumpe y quieres reanudar desde un checkpoint:

```yaml
# En config_valenciano.yml
pretrained_model: "epoch_2nd_00050.pth"   # apunta al último guardado en log_dir
load_only_params: false                   # mantén false para recuperar el optimizer state
```

**Nota importante**: no pongas `pretrained_model: first_stage.pth` para reanudar. Eso provoca cargas dobles y corrupción del optimizador.

---

## 8. Inferencia

Una vez termine el entrenamiento, copia el modelo final a la carpeta de demos:

```bash
mkdir -p StyleTTS2/Models/LJSpeech
cp /ruta/log_dir/epoch_2nd_00100.pth StyleTTS2/Models/LJSpeech/
cp /ruta/log_dir/config.yml          StyleTTS2/Models/LJSpeech/
```

Abre uno de los notebooks de inferencia:

- **Sin audio de referencia** (single-speaker puro): [`StyleTTS2/Demo/Inference_LJSpeech.ipynb`](StyleTTS2/Demo/Inference_LJSpeech.ipynb)
- **Con audio de referencia** (mejor calidad si has entrenado con multi-speaker mezclado): [`StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb`](StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb)

Edita la celda que carga el modelo para que apunte a tu checkpoint y ejecuta.

---

## Resumen comandos en una sola tirada

Asumiendo ya hechos pasos 0-2:

```bash
# PL-BERT
cd t5 && python T05a_prepare_dataset.py && accelerate launch T05b_train_plbert.py && cd ..
cp t5/checkpoints_valenciano/step_50000.t7 StyleTTS2/Utils/PLBERT/

# StyleTTS2
cd StyleTTS2
accelerate launch train_first.py  --config_path Configs/config_valenciano.yml
accelerate launch train_second.py --config_path Configs/config_valenciano.yml
```
