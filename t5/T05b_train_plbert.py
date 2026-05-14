#!/usr/bin/env python3
"""
T05b: Entrenar PL-BERT para valenciano.

Usa el dataloader original de PL-BERT pero con:
- RoBERTa-ca en lugar de TransfoXL para supervisión semántica
- Dataset preparado por T05a

Uso:
    python scripts/T05b_train_plbert.py

Requiere:
    - dataset_valenciano/ (generado por T05a)
    - token_maps_valenciano.pkl (generado por T05a)
    - dataset_config.json (generado por T05a)
"""

import os
import sys
import json
import shutil
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from transformers import AlbertConfig, AlbertModel
from torch.optim import AdamW
from datasets import load_from_disk

from tqdm import tqdm

# Añadir path para imports locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_utils import TextCleaner


# =============================================================================
# MODELO (igual que el original)
# =============================================================================

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=84827, hidden_size=768):
        super().__init__()
        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, num_vocab)

    def forward(self, phonemes, attention_mask=None):
        output = self.encoder(phonemes, attention_mask=attention_mask)
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)
        return tokens_pred, words_pred


# =============================================================================
# DATASET (adaptado del original)
# =============================================================================

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,
                 token_maps_path="token_maps_valenciano.pkl",
                 word_separator=0,
                 token_separator=" ",
                 token_mask="M",
                 max_mel_length=512,
                 word_mask_prob=0.15,
                 phoneme_mask_prob=0.1,
                 replace_prob=0.2):

        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()

        self.word_separator = word_separator
        self.token_separator = token_separator
        self.token_mask = token_mask

        with open(token_maps_path, 'rb') as handle:
            self.token_maps = pickle.load(handle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import numpy as np

        phonemes = self.data[idx]['phonemes']
        input_ids = self.data[idx]['input_ids']

        words = []
        labels = ""
        phoneme = ""

        phoneme_list = ''.join(phonemes)
        masked_index = []

        for z in zip(phonemes, input_ids):
            z = list(z)

            words.extend([z[1]] * len(z[0]))
            words.append(self.word_separator)
            labels += z[0] + " "

            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        phoneme += ''.join([phoneme_list[np.random.randint(0, len(phoneme_list))] for _ in range(len(z[0]))])
                    else:
                        phoneme += z[0]
                else:
                    phoneme += self.token_mask * len(z[0])

                masked_index.extend((np.arange(len(phoneme) - len(z[0]), len(phoneme))).tolist())
            else:
                phoneme += z[0]

            phoneme += self.token_separator

        mel_length = len(phoneme)
        masked_idx = np.array(masked_index)
        masked_index = []

        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start:random_start + self.max_mel_length]
            words = words[random_start:random_start + self.max_mel_length]
            labels = labels[random_start:random_start + self.max_mel_length]

            for m in masked_idx:
                if m >= random_start and m < random_start + self.max_mel_length:
                    masked_index.append(m - random_start)
        else:
            masked_index = masked_idx.tolist()

        phoneme = self.text_cleaner(phoneme)
        labels = self.text_cleaner(labels)

        # Mapear words a tokens internos
        mapped_words = []
        for w in words:
            if w in self.token_maps:
                mapped_words.append(self.token_maps[w]['token'])
            else:
                mapped_words.append(0)

        assert len(phoneme) == len(mapped_words), f"Mismatch: {len(phoneme)} vs {len(mapped_words)}"
        assert len(phoneme) == len(labels), f"Mismatch: {len(phoneme)} vs {len(labels)}"

        phonemes = torch.LongTensor(phoneme)
        labels = torch.LongTensor(labels)
        words = torch.LongTensor(mapped_words)

        return phonemes, words, labels, masked_index


class Collater(object):
    def __init__(self):
        self.text_pad_index = 0

    def __call__(self, batch):
        import numpy as np

        batch_size = len(batch)
        lengths = [b[1].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        max_text_length = max([b[1].shape[0] for b in batch])

        words = torch.zeros((batch_size, max_text_length)).long()
        labels = torch.zeros((batch_size, max_text_length)).long()
        phonemes = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []

        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            text_size = phoneme.size(0)
            words[bid, :text_size] = word
            labels[bid, :text_size] = label
            phonemes[bid, :text_size] = phoneme
            input_lengths.append(text_size)
            masked_indices.append(masked_index)

        return words, labels, phonemes, input_lengths, masked_indices


def build_dataloader(dataset, batch_size=4, num_workers=1, dataset_config={}):
    ds = FilePathDataset(dataset, **dataset_config)
    collate_fn = Collater()
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return data_loader


# =============================================================================
# UTILIDADES
# =============================================================================

def length_to_mask(lengths, device=None):
    max_len = int(max(lengths))
    lengths_tensor = torch.tensor(lengths, device=device)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) >= lengths_tensor.unsqueeze(1)
    return mask


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train():
    # Configuración
    config = {
        'log_dir': 'checkpoints_valenciano',
        'mixed_precision': 'fp16',
        'batch_size': 32,  # Ajustar según GPU
        'num_steps': 500000,
        'log_interval': 100,
        'save_interval': 10000,
        'lr': 1e-4,
        # Modelo
        'model_params': {
            'vocab_size': 178,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 2048,
            'max_position_embeddings': 512,
            'num_hidden_layers': 12,
            'dropout': 0.1,
        },
        # Dataset
        'dataset_params': {
            'token_maps_path': 'token_maps_valenciano.pkl',
            'word_separator': 0,  # Se actualiza después
            'token_separator': ' ',
            'token_mask': 'M',
            'max_mel_length': 512,
            'word_mask_prob': 0.15,
            'phoneme_mask_prob': 0.1,
            'replace_prob': 0.2,
        }
    }

    # Cargar config del dataset
    with open('dataset_config.json', 'r') as f:
        dataset_info = json.load(f)

    config['dataset_params']['word_separator'] = dataset_info['word_separator']
    num_vocab = dataset_info['num_vocab']

    print(f"=== T05b: Entrenar PL-BERT Valenciano ===\n")
    print(f"Word separator: {config['dataset_params']['word_separator']}")
    print(f"Num vocab: {num_vocab}")

    # Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        split_batches=True,
        kwargs_handlers=[ddp_kwargs]
    )

    # Cargar dataset
    accelerator.print("\nCargando dataset...")
    dataset = load_from_disk('dataset_valenciano')
    accelerator.print(f"  {len(dataset)} muestras")

    # Crear dataloader
    accelerator.print("Creando dataloader...")
    train_loader = build_dataloader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        dataset_config=config['dataset_params']
    )

    # Crear modelo
    accelerator.print("Creando modelo...")
    albert_config = AlbertConfig(**config['model_params'])
    bert = AlbertModel(albert_config)
    bert = MultiTaskModel(
        bert,
        num_vocab=num_vocab,
        num_tokens=config['model_params']['vocab_size'],
        hidden_size=config['model_params']['hidden_size']
    )

    optimizer = AdamW(bert.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Checkpoint
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    iters = 0
    load = False

    try:
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        if ckpts:
            iters_list = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
            iters = sorted(iters_list)[-1]
            load = True
    except:
        pass

    if load:
        checkpoint = torch.load(f"{log_dir}/step_{iters}.t7", map_location='cpu')
        state_dict = checkpoint['net']

        # Remover prefijo 'module.' si existe
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        bert.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        accelerator.print(f"Checkpoint cargado: step_{iters}")

    # Preparar con accelerator
    bert, optimizer, train_loader = accelerator.prepare(bert, optimizer, train_loader)

    accelerator.print(f"\nIniciando entrenamiento desde step {iters}...")
    accelerator.print(f"  Total steps: {config['num_steps']}")
    accelerator.print(f"  Batch size: {config['batch_size']}")

    running_loss = 0

    # Bucle infinito sobre el dataset hasta alcanzar num_steps
    while iters < config['num_steps']:
        for _, batch in enumerate(train_loader):
            words, labels, phonemes, input_lengths, masked_indices = batch
            text_mask = length_to_mask(input_lengths, device=phonemes.device)

            tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())

            # Loss vocab (word prediction)
            loss_vocab = 0
            for _pred, _target, _length, _ in zip(words_pred, words, input_lengths, masked_indices):
                loss_vocab += criterion(_pred[:_length], _target[:_length])
            loss_vocab /= words.size(0)

            # Loss token (masked phoneme prediction)
            loss_token = 0
            sizes = 1
            for _pred, _target, _length, _masked in zip(tokens_pred, labels, input_lengths, masked_indices):
                if len(_masked) > 0:
                    _masked = torch.tensor(_masked, device=_pred.device)
                    _target_masked = _target[:_length][_masked]
                    loss_tmp = criterion(_pred[:_length][_masked], _target_masked)
                    loss_token += loss_tmp
                    sizes += 1
            loss_token /= sizes

            loss = loss_vocab + loss_token

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            iters += 1

            if iters % config['log_interval'] == 0:
                avg_loss = running_loss / config['log_interval']
                accelerator.print(f"Step [{iters}/{config['num_steps']}] Loss: {avg_loss:.5f} "
                                f"(vocab: {loss_vocab.item():.5f}, token: {loss_token.item():.5f})")
                running_loss = 0

            if iters % config['save_interval'] == 0:
                accelerator.print("Guardando checkpoint...")
                state = {
                    'net': bert.state_dict(),
                    'step': iters,
                    'optimizer': optimizer.state_dict(),
                }
                accelerator.save(state, f"{log_dir}/step_{iters}.t7")

            if iters >= config['num_steps']:
                accelerator.print("Entrenamiento completado!")
                return


if __name__ == "__main__":
    train()
