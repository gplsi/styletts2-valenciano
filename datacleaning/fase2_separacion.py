import os
import sys
import types
import warnings
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

# Activamos la aceleración TF32 para que la GPU vaya más rápido (y quite el warning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from pyannote.audio import Pipeline

# ==========================================
# 1. CONFIGURA TUS RUTAS AQUÍ
# ==========================================
INPUT_DIR = "/home/gplsinas/ALIA/datos/Voz/Voz_2/voces_limpias_sin_musica"     # Los audios limpios de la Fase 1
OUTPUT_DIR = "/home/gplsinas/ALIA/datos/Voz/Voz_2/voces_limpias_segmentadas" # Donde irán los audios cortos

# Definimos el formato ideal para StyleTTS2
TARGET_SAMPLE_RATE = 24000

# Creamos la carpeta de salida si no existe
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def process_audio():
    audio_files = list(Path(INPUT_DIR).glob("*_clean.wav"))
    
    if not audio_files:
        print(f"No se encontraron audios limpios en {INPUT_DIR}")
        return

    print("Cargando el modelo de Pyannote (versión libre de la comunidad)...")
    try:
        # Usamos el modelo que no requiere autenticación ni token
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token="<<INSERT_TOKEN>>"
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("Modelo cargado en GPU 🚀")
        else:
            print("Modelo cargado en CPU 🐢 (Tardará un rato)")
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo.\nDetalle: {e}")
        return

    print(f"\nComenzando segmentación y formateo de {len(audio_files)} archivos...\n")

    for audio_path in audio_files:
        print(f"Analizando turnos y formateando: {audio_path.name} ...")
        
        # 1. Cargamos el audio original
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # --- PREPROCESAMIENTO PARA STYLETTS2 ---
        
        # A) Convertir a MONO
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # B) Resamplear a 24kHz
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = TARGET_SAMPLE_RATE
            
        # ----------------------------------------
        
        # 2. Pasamos el audio optimizado a Pyannote
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        salida = pipeline(audio_in_memory)
        
        # --- DESEMPAQUETADO PYANNOTE 3.3.2 ---
        # Abrimos la "caja" (DiarizeOutput) para sacar la lista de tiempos real
        if hasattr(salida, 'speaker_diarization'):
            diarization = salida.speaker_diarization
        elif hasattr(salida, 'annotation'):
            diarization = salida.annotation
        elif hasattr(salida, 'diarization'):
            diarization = salida.diarization
        elif hasattr(salida, 'itertracks'):
            diarization = salida
        else:
            print(f"⚠️ Formato raro detectado. Atributos: {dir(salida)}")
            diarization = salida
        # --------------------------------------
        
        chunk_count = 0
        
        # 3. Iteramos sobre los turnos detectados para hacer los cortes
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            
            # Filtro: Ignoramos fragmentos de menos de 1 segundo
            if duration < 1.0:
                continue
                
            speaker_dir = Path(OUTPUT_DIR) / f"{audio_path.stem}_{speaker}"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculamos los frames exactos a 24kHz
            start_frame = int(turn.start * sample_rate)
            end_frame = int(turn.end * sample_rate)
            
            # Cortamos la forma de onda
            audio_chunk = waveform[:, start_frame:end_frame]
            
            # Guardamos el archivo final
            chunk_filename = speaker_dir / f"chunk_{chunk_count:04d}.wav"
            torchaudio.save(str(chunk_filename), audio_chunk, sample_rate)
            chunk_count += 1
            
        print(f"✅ {audio_path.name} procesado: {chunk_count} fragmentos perfectos extraídos.")

    print("\n¡Fase 2 Completada con éxito!")

if __name__ == "__main__":
    process_audio()