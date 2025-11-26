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

- Python = 3.10.19  
- Librerías principales:
```bash
pip install -r requirements.txt
```
Debes tener ffmpeg version 7 instalado

## Entrenamiento
Por completar

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
