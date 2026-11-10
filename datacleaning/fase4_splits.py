import os
import random
import shutil
from pathlib import Path

# ==========================================
# 1. INYECTAR LA LIBRERÍA DE ESPEAK-NG
# ==========================================
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/home/gplsi/rst29/anaconda3/envs/styletts/lib/libespeak-ng.so"
import phonemizer

# ==========================================
# 2. CONFIGURA TUS RUTAS
# ==========================================
BASE_DIR = "/home/gplsinas/ALIA/datos/Voz/Voz_2"
INPUT_METADATA = os.path.join(BASE_DIR, "metadata.csv")
WAVS_DIR = os.path.join(BASE_DIR, "wavs")

# Archivos de salida finales para StyleTTS2
TRAIN_OUT = os.path.join(BASE_DIR, "train_list.txt")
VAL_OUT = os.path.join(BASE_DIR, "val_list.txt")
OOD_OUT = os.path.join(BASE_DIR, "OOD_list.txt")

# Creamos la carpeta wavs si no existe
os.makedirs(WAVS_DIR, exist_ok=True)

# ==========================================
# 3. CONFIGURAR EL MOTOR VALENCIANO
# ==========================================
print("Inicializando el motor fonético eSpeak (ca-va)...")
global_phonemizer_valencia = phonemizer.backend.EspeakBackend(
    language="ca-va",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)

def reorganizar_y_fonematizar():
    if not os.path.exists(INPUT_METADATA):
        print(f"❌ No se encuentra el archivo {INPUT_METADATA}")
        return

    with open(INPUT_METADATA, 'r', encoding='utf-8') as f:
        lineas = [linea.strip() for linea in f if linea.strip()]

    print(f"Leídas {len(lineas)} transcripciones. Copiando y renombrando audios a /wavs ...")

    rutas_nuevas = []
    textos_originales = []

    for linea in lineas:
        partes = linea.split('|')
        if len(partes) != 2:
            continue
            
        ruta_antigua = partes[0]
        texto = partes[1]
        
        if not os.path.exists(ruta_antigua):
            continue

        # Extraemos el nombre de la carpeta y del archivo
        # Ej: "dentista_clean_SPEAKER_00" y "chunk_0001.wav"
        nombre_archivo = os.path.basename(ruta_antigua)
        nombre_carpeta = os.path.basename(os.path.dirname(ruta_antigua))
        
        # Creamos un nombre único para evitar colisiones en la carpeta wavs
        nombre_unico = f"{nombre_carpeta}_{nombre_archivo}"
        ruta_nueva = os.path.join(WAVS_DIR, nombre_unico)
        
        # Copiamos el archivo a su nueva casa
        shutil.copy2(ruta_antigua, ruta_nueva)
        
        rutas_nuevas.append(ruta_nueva)
        textos_originales.append(texto)

    print("Audios organizados. Transformando texto valenciano a fonemas (IPA)...")
    
    # Pre-fonematizar en bloque
    textos_fonematizados = global_phonemizer_valencia.phonemize(
        textos_originales, 
        strip=True, 
        njobs=1
    )

    # Ensamblar el formato exigido por StyleTTS2 (ruta_nueva | fonemas | id=0)
    dataset_formateado = []
    for ruta, texto_fon in zip(rutas_nuevas, textos_fonematizados):
        dataset_formateado.append(f"{ruta}|{texto_fon}|0\n")

    # Barajar y Dividir (90% Train, 5% Val, 5% OOD)
    random.seed(42)
    random.shuffle(dataset_formateado)

    total = len(dataset_formateado)
    val_size = max(1, int(total * 0.05))
    ood_size = max(1, int(total * 0.05))
    
    val_data = dataset_formateado[:val_size]
    ood_data = dataset_formateado[val_size:val_size+ood_size]
    train_data = dataset_formateado[val_size+ood_size:]

    # Guardar en disco
    with open(TRAIN_OUT, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
        
    with open(VAL_OUT, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
        
    with open(OOD_OUT, 'w', encoding='utf-8') as f:
        f.writelines(ood_data)

    print("\n" + "="*50)
    print("✅ ¡DATASET ESTRUCTURADO Y FONEMATIZADO CON ÉXITO!")
    print(f"📁 Todos los audios están ahora en: {WAVS_DIR}")
    print(f"📊 Entrenamiento (train_list.txt): {len(train_data)} audios")
    print(f"🧪 Validación (val_list.txt): {len(val_data)} audios")
    print(f"🔮 Examen OOD (OOD_list.txt): {len(ood_data)} audios")
    print("="*50)

if __name__ == "__main__":
    reorganizar_y_fonematizar()