# 🗣️ Desarrollo de StyleTTS2 para el Valenciano

Este proyecto tiene como objetivo adaptar el modelo **[StyleTTS2](https://github.com/yl4579/StyleTTS2)** para generar voz sintética en **valenciano**.

El trabajo incluye la preparación de un corpus de voz valenciano, el entrenamiento del modelo, y la evaluación de la naturalidad y la fidelidad lingüística de las voces generadas.

## Estructura del repositorio

```
styletts2-valenciano/
├── StyleTTS2/              # Fork adaptado de StyleTTS2 para valenciano
│   ├── Configs/            # Configuraciones de entrenamiento
│   ├── Models/             # Arquitecturas de los modelos
│   ├── Utils/              # Utilidades (incluye PL-BERT)
│   ├── Demo/               # Notebooks de inferencia
│   └── train_*.py          # Scripts de entrenamiento
│
├── t1/                     # Tarea 1: Exploración del dataset inglés
│   └── dataset-exploration.ipynb
│
├── t2/                     # Tarea 2: Procesamiento y mejora de audio
│   ├── dataset_exp.ipynb   # Análisis de calidad y duración
│   ├── clearervoice/       # Herramientas de mejora de audio
│   ├── whisper_catalan.py  # Transcripción con Whisper
│   └── espeak-ng/          # Fonetización
│
├── t3/                     # Tarea 3: Pruebas de fonetización
│   └── phonemizer_test.py
│
├── t4/                     # Tarea 4: Tokenizador de fonemas
│   ├── tokenizer_conversion.py
│   ├── tokenizer_conversion_to_hf.py
│   └── tokenizer_coverage.ipynb
│
└── t5/                     # Tarea 5: Entrenamiento PL-BERT
    └── PL-BERT/            # Modelo de representación fonética
        ├── eval.ipynb
        └── Configs/
```

## Requisitos

- Python 3.12.3
- ffmpeg version 7

### Instalación

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Descargar PL-BERT

El modelo requiere los pesos de PL-BERT entrenado para valenciano. Descárgalos desde Hugging Face:

```bash
# Crear directorio para PL-BERT
mkdir -p StyleTTS2/Utils/PLBERT/valenciano

# Descargar los archivos necesarios
cd StyleTTS2/Utils/PLBERT/valenciano
wget https://huggingface.co/javiimts/plbert-valenciano/resolve/main/step_5000.t7
wget https://huggingface.co/javiimts/plbert-valenciano/resolve/main/config.yml
```

O clona el repositorio completo:

```bash
cd StyleTTS2/Utils/PLBERT
git lfs install
git clone https://huggingface.co/javiimts/plbert-valenciano
```

## Entrenamiento

El entrenamiento de StyleTTS2 se realiza en dos etapas.

### Configuración

Antes de entrenar, edita `StyleTTS2/Configs/config_valenciano.yml` y modifica las siguientes rutas:

| Parámetro | Descripción |
|-----------|-------------|
| `log_dir` | Directorio donde se guardarán los checkpoints y logs |
| `data_params.root_path` | Directorio raíz donde se encuentra la carpeta con los `.wav` |

### Stage 1: Pre-training (Acoustic Model)

#### Una sola GPU

El script usa `accelerate` internamente con DDP, pero para **una sola GPU** no es necesario configurar nada. Simplemente ejecuta:

```bash
cd StyleTTS2
python -u train_first.py --config_path Configs/config_valenciano.yml
```

#### Multi-GPU

Para entrenar con **múltiples GPUs**, usa `accelerate launch`:

```bash
cd StyleTTS2
accelerate launch --multi_gpu --num_processes=N train_first.py --config_path Configs/config_valenciano.yml
```

Donde `N` es el número de GPUs a utilizar.

Este stage entrena el decoder, style encoder, text encoder y pitch extractor. La configuración por defecto usa:
- `epochs_1st: 100`
- `batch_size: 8`
- `TMA_epoch: 25` (Text-Mel Alignment empieza en epoch 25)

Los checkpoints se guardan en el directorio especificado en `log_dir` del config.

### Stage 2: Joint Training (con Diffusion)

**Nota:** El autor del repositorio original indica que DDP no funciona correctamente en Stage 2.

Por completar

### Monitorización

Puedes monitorizar el entrenamiento con TensorBoard:

```bash
tensorboard --logdir /ruta/a/Models/Valenciano/tensorboard
```

## Inferencia (síntesis de voz)
Utiliza la libreta ./StyleTTS2/Demo/Inference_LJSpeech.ipynb

## Evaluación
Por completar

## Tareas
Todas las tareas serán escritas y documentadas en la plataforma [Linear](https://linear.app/tts-valenciano)

## Referencias
[StyleTTS2 Paper](https://arxiv.org/abs/2306.07691)

[StyleTTS2 Repositorio](https://github.com/yl4579/StyleTTS2)

[Script Fine Tuning Ejemplo](https://github.com/IIEleven11/StyleTTS2FineTune)
