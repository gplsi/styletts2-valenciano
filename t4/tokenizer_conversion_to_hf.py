import pickle
from transformers import BertTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

print("Iniciando el proceso de extracción del tokenizador...")

# --- Nombres de archivos ---
pickle_file = '../token_maps_bert_cat.pkl'
vocab_file = 'vocab2.txt'

# --- Tokens especiales que EXISTEN en el .pkl ---
UNK_TOKEN_NAME = '[UNK]'  # Confirmado
MASK_TOKEN_NAME = 'M'     # Confirmado

# --- Tokens especiales que VAMOS A AÑADIR ---
# (Estos NO existen en el .pkl y los añadiremos al final)
PAD_TOKEN_NAME = '[PAD]'
SEP_TOKEN_NAME = '[SEP]' 

from huggingface_hub import login

login(os.getenv("HUGGINGFACE_TOKEN"))
try:
    # --- 1. Cargar y extraer vocabulario base ---
    print(f"Cargando {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        token_data = pickle.load(f)
    
    vocab_list = []
    for key, value_dict in sorted(token_data.items()):
        vocab_list.append(value_dict['word'])
    
    print(f"Vocabulario base extraído con {len(vocab_list)} tokens.")

    # --- 2. Añadir los tokens especiales que faltan ---
    print(f"Añadiendo '{PAD_TOKEN_NAME}', y '{SEP_TOKEN_NAME}' al vocabulario...")
    vocab_list.append(PAD_TOKEN_NAME)
    vocab_list.append(SEP_TOKEN_NAME)

    # --- 3. Guardar el vocab.txt COMPLETO ---
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in vocab_list:
            f.write(token + '\n')
    
    # El vocabulario original (29848) + 3 tokens nuevos = 29851
    print(f"Vocabulario completo guardado en {vocab_file}. Tamaño total: {len(vocab_list)}")

    # --- 4. Cargar el tokenizador final ---
    print("Cargando BertTokenizer con la configuración final...")
    tokenizer = BertTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False, 
        
        # Apuntamos a los NOMBRES de nuestros tokens especiales
        pad_token=PAD_TOKEN_NAME,
        sep_token=SEP_TOKEN_NAME,
        unk_token=UNK_TOKEN_NAME,
        mask_token=MASK_TOKEN_NAME,
    )

    print("\n" + "="*30)
    print("¡TOKENIZADOR CREADO CON ÉXITO!")
    print("="*30)

    # --- 5. Verificar los IDs ---
    print(f"Vocabulario total: {len(tokenizer)}")
    # Los nuevos tokens estarán al final
    print(f"Pad token:  '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Cls token:  '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
    print(f"Sep token:  '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
    # Los tokens originales mantienen sus IDs
    print(f"Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    print(f"Unk token:  '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    
    # Verificamos el token 3039 que hemos ignorado
    token_3039 = tokenizer.convert_ids_to_tokens([3039])[0]
    print(f"El token en ID 3039 (ignorado) es: '{token_3039}'")


    # --- 6. Prueba de funcionalidad ---
    text_ejemplo = "kɾˈɛstə kərβ"
    encoded = tokenizer(
        text_ejemplo, 
        max_length=10,
        add_special_tokens=False
    )

    tokenizer.push_to_hub("javiimts/bert-ca-va-tokenizer", private=True)
    
    print("\n--- Prueba de codificación y padding ---")
    print(f"Texto: '{text_ejemplo}'")
    print(f"IDs: {encoded['input_ids']}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")

except Exception as e:
    print(f"Ha ocurrido un error: {e}")