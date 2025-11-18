import os
import shutil
import os.path as osp
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling
)
import os
from dotenv import load_dotenv
import wandb

# Carga las variables del archivo .env
load_dotenv()

os.environ['HF_HOME'] = '/media/datos1/tts-valenciano/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/media/datos1/tts-valenciano/huggingface/datasets'

print("Iniciando sesión en Weights & Biases...")
wandb.login(key=os.getenv('WANDB_API_KEY'))



# --- Cargar Configuración ---
print("Cargando config.yml...")
config_path = "Configs/config.yml" 
config = yaml.safe_load(open(config_path))

# --- Cargar NUESTRO Tokenizador de Fonemas ---
print("Cargando tokenizador desde Hugging Face Hub...")
tokenizer_hub_id = config['dataset_params']['tokenizer_name']
tokenizer = BertTokenizer.from_pretrained(tokenizer_hub_id)

# --- Cargar y Procesar el Dataset Fonetizado ---
print("Cargando dataset fonetizado...")
dataset = load_from_disk(config["data_folder"])

def tokenize_function(examples):
    phoneme_strings = [" ".join(ph_list) for ph_list in examples["phonemes"]]
    return tokenizer(
        phoneme_strings, 
        padding=False, 
        truncation=True, 
        max_length=config['model_params']['max_position_embeddings'], 
        return_special_tokens_mask=False,
    )

print("Tokenizando el dataset en memoria...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4, 
    remove_columns=["phonemes"] 
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=config['dataset_params']['word_mask_prob'] 
)

# --- Función de Entrenamiento --- 
def train():
    
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'], 
        log_with="wandb"
    )

    # Usará el nombre de 'log_dir' como nombre del proyecto
    accelerator.init_trackers(
        project_name=config['log_dir'], 
        config=config # Sube tu config.yml
    )

    # --- Creación del Modelo (sin cargar los pesos todavía!!) --
    accelerator.print('Creando un nuevo modelo BertForMaskedLM...')
    
    model_params = config['model_params']
    bert_config = BertConfig(
        vocab_size=len(tokenizer), 
        hidden_size=model_params['hidden_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        intermediate_size=model_params['intermediate_size'],
        max_position_embeddings=model_params['max_position_embeddings']
    )
    
    model = BertForMaskedLM(bert_config)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # --- 5. Crear el DataLoader ---
    accelerator.print("Creando DataLoader...")
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # --- Lógica de Checkpoint ---
    log_dir = config['log_dir']
    if not osp.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)
        accelerator.print("Directorio de checkpoint no encontrado. Empezando desde cero.")
        start_step = 0
    else:
        # Busca el último checkpoint
        try:
            ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
            iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
            start_step = sorted(iters)[-1]
            checkpoint_path = log_dir + "/step_" + str(start_step) + ".t7"
            
            accelerator.print(f"Resumiendo desde el checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            accelerator.print("No se pudo cargar el checkpoint. Empezando desde cero.")
            start_step = 0

    # --- Preparar con Accelerator ---
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    accelerator.print('Empezando entrenamiento...')
    
    num_steps = config['num_steps']
    log_interval = config['log_interval']
    save_interval = config['save_interval']
    
    running_loss = 0
    curr_steps = start_step

    # Bucle de entrenamiento
    while curr_steps < num_steps:
        for batch in train_loader:        
            
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            curr_steps += 1

            if (curr_steps + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                accelerator.print ('Step [%d/%d], Loss: %.5f'
                        % (curr_steps + 1, num_steps, avg_loss))
                
                # Para que se loguee en Weights & Biases
                accelerator.log({"loss": avg_loss}, step=curr_steps + 1)
                
                running_loss = 0
                
            if (curr_steps + 1) % save_interval == 0:
                accelerator.print('Saving..')
                unwrapped_model = accelerator.unwrap_model(model)
                state = {
                    'net':  unwrapped_model.state_dict(),
                    'step': curr_steps + 1,
                    'optimizer': optimizer.state_dict(),
                }
                accelerator.save(state, log_dir + '/step_' + str(curr_steps + 1) + '.t7')

            if curr_steps >= num_steps:
                accelerator.print("Entrenamiento completado.")
                
                # Con esto le decimos a WB que ya hemos terminado
                accelerator.end_training()
                return


if __name__ == "__main__":
    train()