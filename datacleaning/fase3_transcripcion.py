import os
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ==========================================
# 1. CONFIGURA TUS RUTAS AQUÍ
# ==========================================
INPUT_DIR = "/home/gplsinas/ALIA/datos/Voz/Voz_2/voces_limpias_segmentadas"
OUTPUT_FILE = "/home/gplsinas/ALIA/datos/Voz/Voz_2/metadata.csv"

# Parámetros para StyleTTS2
MIN_DURATION = 1.0  # Segundos mínimos
MAX_DURATION = 15.0 # Segundos máximos

def procesar_transcripciones():
    audio_files = list(Path(INPUT_DIR).rglob("*.wav"))
    
    if not audio_files:
        print(f"❌ No se encontraron audios en {INPUT_DIR}")
        return

    print(f"Se han encontrado {len(audio_files)} fragmentos de audio.")
    
    # ==========================================
    # CARGA DEL MODELO BSC-LT DESDE HUGGING FACE
    # ==========================================
    print("Cargando el modelo BSC-LT/whisper-large-v3-LoS-punctuated en GPU 🚀...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "BSC-LT/whisper-large-v3-LoS-punctuated"

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Creamos el pipeline de reconocimiento de voz
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    except Exception as e:
        print(f"❌ Error al cargar el modelo de Hugging Face: {e}")
        return

    print(f"\nComenzando transcripción y guardando en: {OUTPUT_FILE}\n")
    
    audios_procesados = 0
    audios_descartados = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        
        for i, audio_path in enumerate(audio_files, 1):
            try:
                # 1. Cargamos el audio (que está a 24kHz por la Fase 2)
                waveform, sample_rate = torchaudio.load(str(audio_path))
                duration = waveform.shape[1] / sample_rate
                
                # Filtro de duración
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    audios_descartados += 1
                    continue
                
                print(f"[{i}/{len(audio_files)}] Transcribiendo: {audio_path.name} ({duration:.1f}s)...")
                
                # 2. Whisper SIEMPRE necesita 16000 Hz. Lo resampleamos "al vuelo" en memoria.
                if sample_rate != 16000:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform_16k = resampler(waveform)
                else:
                    waveform_16k = waveform
                    
                # Pasamos el tensor a un array numpy de 1D que es lo que espera el pipeline
                audio_np = waveform_16k.squeeze().numpy()
                
                # 3. Transcribimos usando el pipeline (el modelo de BSC suele detectar el idioma solo, 
                # pero forzamos catalán/valenciano por seguridad si lo permite).
                result = pipe(audio_np, generate_kwargs={"language": "catalan"})
                
                texto = result["text"].strip()
                
                # Si el texto está vacío, lo saltamos
                if not texto:
                    audios_descartados += 1
                    continue
                    
                # 4. Guardamos en el metadata.csv
                linea = f"{audio_path.absolute()}|{texto}\n"
                f.write(linea)
                f.flush() 
                
                audios_procesados += 1
                
            except Exception as e:
                print(f"⚠️ Error procesando {audio_path.name}: {e}")
                audios_descartados += 1

    print("\n" + "="*50)
    print("✅ ¡Fase 3 Completada con éxito!")
    print(f"🎙️ Audios válidos transcritos: {audios_procesados}")
    print(f"🗑️ Audios descartados (por duración o error): {audios_descartados}")
    print(f"📄 Metadatos guardados en: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    procesar_transcripciones()