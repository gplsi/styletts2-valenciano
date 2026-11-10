import os
import subprocess
import shutil
from pathlib import Path

# ==========================================
# 1. CONFIGURA TUS RUTAS AQUÍ
# ==========================================
INPUT_DIR = "/home/gplsi/GPLSI/shared_data/VOZ_2/dictactsgva/VA/2025-07-18/audio" # Carpeta con tus audios con música
OUTPUT_DIR = "/home/gplsinas/ALIA/datos/Voz/Voz_2"    # Carpeta donde irán las voces finales
TEMP_DIR = "/scratch/demucs_temp"        # Carpeta temporal para procesar

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

def extract_vocals():
    audio_files = list(Path(INPUT_DIR).glob("*.wav")) + list(Path(INPUT_DIR).glob("*.mp3"))
    
    if not audio_files:
        print(f"No se encontraron audios en {INPUT_DIR}")
        return

    print(f"Comenzando limpieza de {len(audio_files)} archivos...\n")

    # PREPARAMOS EL ENTORNO: Forzamos a PyTorch/Torchaudio a usar 'soundfile'
    my_env = os.environ.copy()
    my_env["TORCHAUDIO_BACKEND"] = "soundfile"

    for audio_path in audio_files:
        print(f"Procesando: {audio_path.name} ...")
        
        command = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems=vocals",
            "-o", TEMP_DIR,
            str(audio_path)
        ]
        
        # Le inyectamos el entorno modificado 'my_env' al comando
        resultado = subprocess.run(command, capture_output=True, text=True, env=my_env)
        
        if resultado.returncode != 0:
            print(f"⚠️ ERROR INTERNO DE DEMUCS al procesar '{audio_path.name}':")
            error_lines = resultado.stderr.strip().split('\n')
            print('\n'.join(error_lines[-15:])) 
            print("-" * 50 + "\n")
            continue 
        
        demucs_output_folder = Path(TEMP_DIR) / "htdemucs" / audio_path.stem
        vocals_file = demucs_output_folder / "vocals.wav"
        
        if vocals_file.exists():
            final_dest = Path(OUTPUT_DIR) / f"{audio_path.stem}_clean.wav"
            shutil.move(str(vocals_file), str(final_dest))
            print(f"✅ Éxito. Guardado limpio en: {final_dest.name}\n")
        else:
            print(f"❌ Error fantasma: Demucs terminó bien pero no generó vocals.wav para {audio_path.name}\n")

    print("Limpiando directorio temporal...")
    if Path(TEMP_DIR).exists():
        shutil.rmtree(TEMP_DIR)
        
    print("¡Script finalizado!")

if __name__ == "__main__":
    extract_vocals()