import os
import sys
import json
import pickle
import argparse
from pathlib import Path

# El dataset de Corts Valencianes está cacheado en /media/datos1
os.environ.setdefault('HF_HOME', '/media/datos1/tts-valenciano/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', '/media/datos1/tts-valenciano/huggingface/datasets')

import phonemizer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

WORD_SEPARATOR_ID = 50255

ROBERTA_CA_TOKENIZER = "projecte-aina/roberta-base-ca-v2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="javiimts/corts-valencianes-asr-parquet",
                   help="ID del dataset en HuggingFace")
    p.add_argument("--splits", nargs="+",
                   default=["clean_train_short", "other_train_short"],
                   help="Splits del dataset a usar")
    p.add_argument("--out-dir", default=".",
                   help="Directorio de salida donde se crean dataset_valenciano/, "
                        "token_maps_valenciano.pkl y dataset_config.json")
    p.add_argument("--limit", type=int, default=None,
                   help="Si se indica, procesar sólo las primeras N muestras "
                        "(útil para debugging)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Procesos en paralelo para fonemización (espeak no escala bien)")
    return p.parse_args()


def build_phonemizer():
    """Configura el fonemizador espeak con la variante valenciana."""
    return phonemizer.backend.EspeakBackend(
        language="ca-va",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )


def first_subword_id(tokenizer, word):
    """
    Tokeniza ' ' + word con RoBERTa-ca y devuelve el ID del primer subword.

    Tokenizar con espacio inicial es la convención de RoBERTa (el primer
    subword lleva el prefijo Ġ que marca el inicio de palabra), y es lo
    que produjo el dataset original publicado.
    """
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if ids else tokenizer.unk_token_id


def process_sample(text, phonem, tokenizer):
    """
    Procesa una muestra: fonemiza la frase completa, alinea palabras con IDs.

    Devuelve (phonemes_list, input_ids_list) si todo es consistente.
    Devuelve None si la fonemización falla o las longitudes no coinciden.
    """
    if not text or not text.strip():
        return None

    words = text.split()
    if not words:
        return None

    try:
        phonemized = phonem.phonemize([text], strip=True, njobs=1)[0]
    except Exception:
        return None

    phonemes_list = phonemized.split()

    if len(phonemes_list) != len(words):
        # Si la fonemización añade/quita palabras (puede pasar con números,
        # abreviaturas o puntuación) saltamos la muestra para mantener la
        # alineación 1:1.
        return None

    input_ids = [first_subword_id(tokenizer, w) for w in words]
    return phonemes_list, input_ids


def build_token_maps(samples):
    """
    Construye el mapeo {roberta_id: {word, token}} a partir de las muestras
    procesadas, asignando enteros consecutivos en orden de primera aparición.

    Añade al final una entrada especial para WORD_SEPARATOR_ID.
    """
    token_maps = {}
    next_token = 0

    for words, phonemes, input_ids in samples:
        for orth_word, rid in zip(words, input_ids):
            if rid not in token_maps:
                token_maps[rid] = {"word": orth_word, "token": next_token}
                next_token += 1

    # Separador como última entrada
    token_maps[WORD_SEPARATOR_ID] = {"word": "<sep>", "token": next_token}
    num_vocab = next_token + 1

    return token_maps, num_vocab


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== T05a: preparar dataset PL-BERT valenciano ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Splits:     {args.splits}")
    print(f"Out dir:    {out_dir}")
    if args.limit:
        print(f"Limit:      {args.limit}")
    print()

    print("Cargando tokenizador RoBERTa-ca...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CA_TOKENIZER)

    print("Inicializando fonemizador espeak ca-va...")
    phonem = build_phonemizer()

    print("Cargando dataset (sólo columnas text/identifier, sin audio)...")
    from datasets import concatenate_datasets
    parts = []
    for split in args.splits:
        ds_split = load_dataset(
            args.dataset, split=split,
            columns=["identifier", "text"],
        )
        print(f"  {split}: {len(ds_split)} muestras")
        parts.append(ds_split)
    full = concatenate_datasets(parts)
    print(f"  total: {len(full)} muestras")

    if args.limit:
        full = full.select(range(min(args.limit, len(full))))
        print(f"  limitado a: {len(full)} muestras")

    # Procesar (lineal, en orden — el orden de primera aparición determina
    # las clases del word_predictor, así que NO podemos paralelizar a ciegas)
    print("\nProcesando muestras (fonemización + tokenización)...")
    samples = []  # list[(words, phonemes, input_ids)]
    discarded = 0
    for i, ex in enumerate(full):
        text = ex["text"]
        result = process_sample(text, phonem, tokenizer)
        if result is None:
            discarded += 1
            continue
        phonemes, input_ids = result
        words = text.split()
        samples.append((words, phonemes, input_ids))

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(full)}  (válidas: {len(samples)}, descartadas: {discarded})")

    print(f"\nTotal: {len(samples)} válidas, {discarded} descartadas")

    print("\nConstruyendo token_maps...")
    token_maps, num_vocab = build_token_maps(samples)
    print(f"  Palabras únicas (incluyendo <sep>): {num_vocab}")

    print("\nGuardando dataset_valenciano/...")
    hf_ds = Dataset.from_dict({
        "phonemes":  [s[1] for s in samples],
        "input_ids": [s[2] for s in samples],
    })
    hf_ds.save_to_disk(str(out_dir / "dataset_valenciano"))

    print("Guardando token_maps_valenciano.pkl...")
    with open(out_dir / "token_maps_valenciano.pkl", "wb") as f:
        pickle.dump(token_maps, f)

    print("Guardando dataset_config.json...")
    config = {
        "word_separator": WORD_SEPARATOR_ID,
        "num_vocab":      num_vocab,
        "tokenizer":      ROBERTA_CA_TOKENIZER,
    }
    with open(out_dir / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== Hecho ===")
    print(f"  - {out_dir/'dataset_valenciano'} ({len(samples)} muestras)")
    print(f"  - {out_dir/'token_maps_valenciano.pkl'} ({num_vocab} entradas)")
    print(f"  - {out_dir/'dataset_config.json'}")


if __name__ == "__main__":
    main()
