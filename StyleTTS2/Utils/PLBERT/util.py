import os
import yaml
import torch
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer

def load_plbert(log_dir):
    """
    Carga el modelo PL-BERT (100% fonético) que hemos entrenado.
    Adaptado para StyleTTS2.
    """
    print(f"Cargando PL-BERT desde: {log_dir}")

    # Cargar el config.yml
    config_path = os.path.join(log_dir, "config.yml")
    if not os.path.exists(config_path):
         config_path = "Configs/config.yml" 
    
    try:
        plbert_config = yaml.safe_load(open(config_path))
        model_params = plbert_config['model_params']
    except Exception as e:
        print(f"Advertencia: No se pudo cargar config.yml desde {config_path}. Usando parámetros por defecto o revisa la ruta. Error: {e}")
        return None

    tokenizer = BertTokenizer.from_pretrained("javiimts/bert-ca-va-tokenizer")

    # Creamos la configuración para el modelo BERT
    bert_config = BertConfig(
        vocab_size=len(tokenizer), # este tamaño está actualizado frente al original
        hidden_size=model_params['hidden_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        intermediate_size=model_params['intermediate_size'],
        max_position_embeddings=model_params['max_position_embeddings']
    )
    
    # Creamos el modelo base
    bert = BertModel(bert_config)

    # Extraemos el último checkpoint
    if os.path.isfile(log_dir):
        # Si log_dir es directamente el archivo .t7
        checkpoint_path = log_dir
    else:
        # Si es un directorio, buscamos el último step
        files = os.listdir(log_dir)
        ckpts = [f for f in files if f.startswith("step_")]
        if not ckpts:
            raise FileNotFoundError(f"No se encontraron checkpoints 'step_*.t7' en {log_dir}")
            
        iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
        iters = sorted(iters)[-1]
        checkpoint_path = os.path.join(log_dir, f"step_{iters}.t7")
    

    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Limpiamos el state_dict para quedarnos solo con las claves del BERT ya que StyleTTS2 solo necesita la base
    
    state_dict = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('bert.'): 
            name = k[5:] # Quitamos el prefijo 'bert.'
            new_state_dict[name] = v
            
    if not new_state_dict:
        # Si el diccionario está vacío, quizás guardaste el modelo base directamente
        # o tiene otro formato. Intentamos cargar tal cual si falla lo anterior.
        print("Aviso: No se encontraron claves con prefijo 'bert.'. Intentando cargar claves originales...")
        new_state_dict = state_dict

    # Cargar los pesos del modelo base en el objeto BERT creado anteriormente
    # strict=False por seguridad si sobra alguna clave rara, 
    # pero debería ser strict=True idealmente.
    keys = bert.load_state_dict(new_state_dict, strict=False)
    print(f"Claves cargadas. Missing: {keys.missing_keys}. Unexpected: {keys.unexpected_keys}")
    
    bert.eval()
    return bert