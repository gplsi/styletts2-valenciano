"""
Script para extraer el tokenizer WordPiece de un modelo PL-BERT
y guardarlo como un tokenizer independiente de Hugging Face
"""

import os
import torch
import pickle
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer


def extract_tokenizer_from_plbert(model_path, output_path="./extracted_tokenizer"):
    """
    Extrae el tokenizer de un modelo PL-BERT y lo guarda
    
    Args:
        model_path: Ruta al directorio del modelo PL-BERT (debe contener token_maps_bert_cat.pkl)
        output_path: Ruta donde guardar el tokenizer extraído
    """
    
    print(f"Buscando tokenizer en: {model_path}")
    
    # Opción 1: Si tienes el archivo token_maps.pkl
    token_maps_path = os.path.join(model_path, "token_maps_bert_cat.pkl")
    
    if os.path.exists(token_maps_path):
        print(f"✓ Encontrado token_maps en: {token_maps_path}")
        
        # Cargar el token_maps
        with open(token_maps_path, 'rb') as f:
            token_maps = pickle.load(f)
        
        print(f"Token maps contiene: {token_maps.keys() if isinstance(token_maps, dict) else type(token_maps)}")
        
        # Guardar información del tokenizer
        os.makedirs(output_path, exist_ok=True)
        
        # Guardar el pickle original
        import shutil
        shutil.copy(token_maps_path, os.path.join(output_path, "token_maps.pkl"))
        print(f"✓ Token maps copiado a: {output_path}/token_maps.pkl")
        
        # Si es un diccionario, crear un archivo de vocabulario
        if isinstance(token_maps, dict):
            vocab_file = os.path.join(output_path, "vocab.txt")
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for token, idx in sorted(token_maps.items(), key=lambda x: x[1]):
                    f.write(f"{token}\n")
            print(f"✓ Vocabulario guardado en: {vocab_file}")
            print(f"  Tamaño del vocabulario: {len(token_maps)}")
    
    # Opción 2: Si el modelo fue entrenado con un tokenizer de Hugging Face
    # Intentar cargar desde el config.yml para ver qué tokenizer base se usó
    config_path = os.path.join(model_path, "config.yml")
    
    if os.path.exists(config_path):
        print(f"\n✓ Encontrado config.yml")
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        tokenizer_name = config.get('dataset_params', {}).get('tokenizer', 'transfo-xl-wt103')
        print(f"  Tokenizer base mencionado en config: {tokenizer_name}")
        
        # Intentar descargar el tokenizer base de Hugging Face
        try:
            print(f"\nDescargando tokenizer base: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Guardar el tokenizer
            tokenizer.save_pretrained(output_path)
            print(f"✓ Tokenizer base guardado en: {output_path}")
            
        except Exception as e:
            print(f"⚠ No se pudo descargar el tokenizer base: {e}")
    
    # Opción 3: Buscar archivos de tokenizer de Hugging Face directamente
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'vocab.json']
    found_files = [f for f in tokenizer_files if os.path.exists(os.path.join(model_path, f))]
    
    if found_files:
        print(f"\n✓ Encontrados archivos de tokenizer: {found_files}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(output_path)
            print(f"✓ Tokenizer cargado y guardado en: {output_path}")
            
            # Mostrar información del tokenizer
            print(f"\nInformación del tokenizer:")
            print(f"  Tipo: {type(tokenizer).__name__}")
            print(f"  Tamaño vocab: {tokenizer.vocab_size}")
            print(f"  Tokens especiales: {tokenizer.special_tokens_map}")
            
        except Exception as e:
            print(f"⚠ Error al cargar tokenizer: {e}")
    
    print(f"\n{'='*60}")
    print(f"Proceso completado. Archivos guardados en: {output_path}")
    print(f"{'='*60}")
    
    return output_path


def load_tokenizer_from_huggingface(model_name, output_path="./my_tokenizer"):
    """
    Descarga un tokenizer directamente desde Hugging Face
    
    Args:
        model_name: Nombre del modelo en Hugging Face (ej: "rjzevallos/PL-BERT-wp-ca")
        output_path: Ruta donde guardar el tokenizer
    """
    print(f"Descargando tokenizer desde Hugging Face: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Guardar el tokenizer localmente
        os.makedirs(output_path, exist_ok=True)
        tokenizer.save_pretrained(output_path)
        
        print(f"✓ Tokenizer descargado y guardado en: {output_path}")
        print(f"\nInformación del tokenizer:")
        print(f"  Tipo: {type(tokenizer).__name__}")
        print(f"  Tamaño vocab: {tokenizer.vocab_size}")
        print(f"  Tokens especiales: {tokenizer.special_tokens_map}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error al descargar tokenizer: {e}")
        raise
