#!/usr/bin/env python3
"""
Script para inspeccionar y actualizar el dataset de HuggingFace
con audios a 24kHz.

Uso:
    # Solo inspeccionar
    python t2/inspect_hf_dataset.py

    # Actualizar con audios 24kHz
    python t2/inspect_hf_dataset.py --update --wavs_dir /ruta/a/wavs_24k
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset, Audio, Dataset
from huggingface_hub import HfApi, login
import numpy as np


def inspect_dataset(dataset_name):
    """Inspecciona el dataset y muestra información."""

    print("=" * 60)
    print(f"INSPECCIONANDO DATASET: {dataset_name}")
    print("=" * 60)

    # Cargar dataset
    print("\nCargando dataset...")
    dataset = load_dataset(dataset_name)

    # 1. Información general
    print("\n" + "-" * 40)
    print("1. SPLITS DISPONIBLES")
    print("-" * 40)
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} muestras")

    total_samples = sum(len(split) for split in dataset.values())
    print(f"\n  TOTAL: {total_samples} muestras")

    # 2. Estructura del dataset
    print("\n" + "-" * 40)
    print("2. ESTRUCTURA DEL DATASET")
    print("-" * 40)

    first_split = list(dataset.keys())[0]
    print(f"\nColumnas (de '{first_split}'):")
    print(f"  {dataset[first_split].column_names}")

    print(f"\nFeatures:")
    for col, feat in dataset[first_split].features.items():
        print(f"  {col}: {feat}")

    # 3. Examinar audio
    print("\n" + "-" * 40)
    print("3. INFORMACIÓN DE AUDIO")
    print("-" * 40)

    sample = dataset[first_split][0]
    print(f"\nEjemplo del split '{first_split}':")
    print(f"  identifier: {sample['identifier']}")
    print(f"  text: {sample['text'][:80]}...")

    audio = sample['audio']
    if isinstance(audio, dict):
        audio_array = audio.get('array')
        sampling_rate = audio.get('sampling_rate')
    else:
        audio_array = audio['array'] if hasattr(audio, '__getitem__') else None
        sampling_rate = audio['sampling_rate'] if hasattr(audio, '__getitem__') else None

    if audio_array is not None:
        print(f"\n  Audio info:")
        print(f"    - Sample rate: {sampling_rate} Hz")
        print(f"    - Duration: {len(audio_array) / sampling_rate:.2f} segundos")

    # 4. Verificar sample rate
    print("\n" + "-" * 40)
    print("4. VERIFICACIÓN DE SAMPLE RATE")
    print("-" * 40)

    sample_rates = set()
    for i in range(min(10, len(dataset[first_split]))):
        audio = dataset[first_split][i]['audio']
        sr = audio.get('sampling_rate') if isinstance(audio, dict) else audio['sampling_rate']
        if sr:
            sample_rates.add(sr)

    print(f"\n  Sample rates encontrados: {sample_rates}")

    return dataset


def update_dataset_with_24k(dataset_name, wavs_dir, dry_run=False):
    """
    Actualiza el dataset existente reemplazando los audios por versiones a 24kHz.

    Args:
        dataset_name: Nombre del dataset en HuggingFace
        wavs_dir: Directorio con los archivos .wav a 24kHz
        dry_run: Si True, solo muestra qué haría sin subir
    """
    import soundfile as sf

    print("=" * 60)
    print(f"ACTUALIZANDO DATASET: {dataset_name}")
    print(f"Directorio de WAVs 24kHz: {wavs_dir}")
    print("=" * 60)

    wavs_path = Path(wavs_dir)
    if not wavs_path.exists():
        print(f"ERROR: El directorio {wavs_dir} no existe")
        return

    # Listar archivos disponibles
    wav_files = list(wavs_path.glob("*.wav"))
    print(f"\nArchivos .wav encontrados: {len(wav_files)}")

    # Crear mapa de identifier -> path
    wav_map = {f.stem: f for f in wav_files}
    print(f"Mapa de identificadores creado: {len(wav_map)} archivos")

    # Verificar un archivo de ejemplo
    if wav_files:
        sample_wav = wav_files[0]
        data, sr = sf.read(sample_wav)
        print(f"\nMuestra de verificación: {sample_wav.name}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(data) / sr:.2f}s")

        if sr != 24000:
            print(f"\n⚠ ADVERTENCIA: Los archivos no están a 24kHz (están a {sr}Hz)")
            response = input("¿Continuar de todos modos? (s/n): ")
            if response.lower() != 's':
                return

    # Cargar dataset
    print("\nCargando dataset desde HuggingFace...")
    dataset = load_dataset(dataset_name)

    # Procesar cada split
    updated_splits = {}

    for split_name in dataset.keys():
        print(f"\n" + "-" * 40)
        print(f"Procesando split: {split_name}")
        print("-" * 40)

        split_data = dataset[split_name]
        found = 0
        missing = []

        # Verificar cobertura
        for i, sample in enumerate(split_data):
            identifier = sample['identifier']
            if identifier in wav_map:
                found += 1
            else:
                missing.append(identifier)
                if len(missing) <= 5:
                    print(f"  Missing: {identifier}")

        print(f"  Encontrados: {found}/{len(split_data)}")
        if missing:
            print(f"  Faltantes: {len(missing)}")

        if dry_run:
            continue

        # Crear función para cargar audio 24kHz
        def load_audio_24k(example):
            identifier = example['identifier']
            if identifier in wav_map:
                wav_path = str(wav_map[identifier])
                # Devolver path para que datasets lo cargue
                example['audio'] = wav_path
            return example

        # Aplicar el mapeo
        print(f"  Actualizando audios...")
        updated = split_data.map(load_audio_24k, num_proc=4)

        # Cast audio feature a 24kHz
        updated = updated.cast_column("audio", Audio(sampling_rate=24000))

        updated_splits[split_name] = updated
        print(f"  ✓ Split '{split_name}' actualizado")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No se subió nada")
        print("Ejecuta sin --dry_run para subir los cambios")
        print("=" * 60)
        return

    # Subir a HuggingFace
    print("\n" + "=" * 60)
    print("SUBIENDO A HUGGINGFACE")
    print("=" * 60)

    from datasets import DatasetDict
    updated_dataset = DatasetDict(updated_splits)

    print(f"\nSubiendo a {dataset_name}...")
    updated_dataset.push_to_hub(dataset_name)

    print("\n✓ Dataset actualizado exitosamente!")


def main():
    parser = argparse.ArgumentParser(
        description='Inspeccionar y actualizar dataset de HuggingFace con audios 24kHz'
    )
    parser.add_argument('--update', action='store_true',
                        help='Actualizar el dataset (por defecto solo inspecciona)')
    parser.add_argument('--wavs_dir', type=str,
                        help='Directorio con los archivos .wav a 24kHz')
    parser.add_argument('--dry_run', action='store_true',
                        help='Solo mostrar qué se haría sin subir')
    parser.add_argument('--dataset', type=str,
                        default='javiimts/corts-valencianes-asr-parquet',
                        help='Nombre del dataset en HuggingFace')
    args = parser.parse_args()

    # Login
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token)
        print("Logged in to HuggingFace\n")

    if args.update:
        if not args.wavs_dir:
            print("ERROR: Debes especificar --wavs_dir para actualizar")
            return
        update_dataset_with_24k(args.dataset, args.wavs_dir, args.dry_run)
    else:
        inspect_dataset(args.dataset)


if __name__ == "__main__":
    main()
