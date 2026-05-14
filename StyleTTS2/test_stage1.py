"""
Stage 1 reconstruction test for Valencian StyleTTS2.

This replicates the train_first.py forward pass using a real audio sample
from the validation set, to verify the trained model produces coherent speech.

Usage:
    cd StyleTTS2
    python test_stage1.py [--config Configs/config_valenciano.yml] [--model Models/LJSpeech/first_stage.pth] [--out output.wav] [--sample N]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

from munch import Munch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import *
from utils import *
from meldataset import build_dataloader, preprocess

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Configs/config_valenciano.yml')
    p.add_argument('--model', default='Models/LJSpeech/first_stage.pth')
    p.add_argument('--out', default='stage1_reconstruction.wav')
    p.add_argument('--sample', type=int, default=0, help='Index into val_list.txt')
    return p.parse_args()


def load_model(config, checkpoint_path, device):
    model_params = recursive_munch(config['model_params'])

    # Load auxiliary models
    ASR_config = config.get('ASR_config', 'Utils/ASR/config.yml')
    ASR_path   = config.get('ASR_path',   'Utils/ASR/epoch_00080.pth')
    F0_path    = config.get('F0_path',    'Utils/JDC/bst.t7')

    text_aligner  = load_ASR_models(ASR_path, ASR_config)
    pitch_extractor = load_F0_models(F0_path)

    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(config.get('PLBERT_dir', 'Utils/PLBERT/'))

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('net', ckpt)
    for key, module in model.items():
        if key not in state:
            print(f'  [skip] {key} not in checkpoint')
            continue
        sd = state[key]
        # Strip DDP 'module.' prefix if present
        if any(k.startswith('module.') for k in sd):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        try:
            module.load_state_dict(sd)
            print(f'  [ok] {key}')
        except Exception as e:
            print(f'  [warn] {key}: {e}')

    for key in model:
        model[key] = model[key].to(device).eval()

    return model


def reconstruction_test(audio_path, model, config, device):
    """
    Replicates the train_first.py forward pass:
      mel -> text_aligner -> alignment -> text_encoder -> asr
      mel -> pitch_extractor -> F0
      mel -> log_norm -> N
      mel -> style_encoder -> s
      decoder(asr, F0, N, s) -> audio
    """
    sr = config['preprocess_params']['sr']

    # ---- Load and preprocess audio ----
    wave, _ = librosa.load(audio_path, sr=sr)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    wave = np.concatenate([np.zeros(5000), wave, np.zeros(5000)])

    # preprocess() returns [1, n_mels, time]  (already has batch dim)
    mel = preprocess(wave).to(device)
    # Pad to even length — text_aligner CNN does 2x downsample, odd T causes off-by-one
    if mel.shape[-1] % 2 != 0:
        mel = torch.nn.functional.pad(mel, (0, 1))
    print(f'  mel shape: {mel.shape}')  # [1, 80, T]

    n_down = model.text_aligner.n_down
    mel_input_length = torch.LongTensor([mel.shape[-1]]).to(device)

    with torch.no_grad():
        # ---- Alignment ----
        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)

        # text_aligner(mel, mel_mask, dummy_text) — we pass None for text to get just alignment
        # Actually we need a text input; use a dummy token sequence (zeros)
        # For reconstruction test the alignment is driven by the mel, not the text
        dummy_text_len = mel.shape[-1] // (2 ** n_down)
        dummy_tokens = torch.zeros(1, dummy_text_len, dtype=torch.long).to(device)
        input_lengths = torch.LongTensor([dummy_text_len]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        ppgs, s2s_pred, s2s_attn = model.text_aligner(mel, mask, dummy_tokens)

        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        attn_mask = (~mask).unsqueeze(-1).expand(
            mask.shape[0], mask.shape[1], text_mask.shape[-1]
        ).float().transpose(-1, -2)
        attn_mask = attn_mask * (~text_mask).unsqueeze(-1).expand(
            text_mask.shape[0], text_mask.shape[1], mask.shape[-1]
        ).float()
        attn_mask = (attn_mask < 1)
        s2s_attn.masked_fill_(attn_mask, 0.0)

        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # ---- Encode text via alignment ----
        t_en = model.text_encoder(dummy_tokens, input_lengths, text_mask)
        asr = (t_en @ s2s_attn_mono)

        # ---- F0 (pitch) ----
        F0_real, _, _ = model.pitch_extractor(mel.unsqueeze(1))
        # JDC squeeze() removes batch dim when batch=1 → fix shape to [batch, time]
        if F0_real.dim() == 1:
            F0_real = F0_real.unsqueeze(0)
        print(f'  F0 shape: {F0_real.shape}')  # should be [1, T]

        # ---- Norm ----
        real_norm = log_norm(mel.unsqueeze(1)).squeeze(1)
        print(f'  norm shape: {real_norm.shape}')  # should be [1, T]

        # ---- Style ----
        s = model.style_encoder(mel.unsqueeze(1))
        print(f'  style shape: {s.shape}')

        # ---- Decode ----
        print(f'  asr shape: {asr.shape}')
        y_rec = model.decoder(asr, F0_real, real_norm, s)

    audio_out = y_rec.squeeze().cpu().numpy()
    return audio_out, wave


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    config = yaml.safe_load(open(args.config))

    print('Loading model...')
    model = load_model(config, args.model, device)

    # Pick a sample from val_list
    val_path = config['data_params']['val_data']
    root_path = config['data_params']['root_path']
    with open(val_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    line = lines[args.sample]
    parts = line.split('|')
    audio_rel = parts[0]
    text_info = parts[1] if len(parts) > 1 else '(no text)'
    audio_path = os.path.join(root_path, audio_rel)

    print(f'\nSample #{args.sample}: {audio_path}')
    print(f'Text: {text_info[:80]}...' if len(text_info) > 80 else f'Text: {text_info}')

    print('\nRunning reconstruction...')
    audio_out, audio_orig = reconstruction_test(audio_path, model, config, device)

    sr = config['preprocess_params']['sr']
    sf.write(args.out, audio_out, sr)
    sf.write(args.out.replace('.wav', '_original.wav'), audio_orig, sr)

    print(f'\nSaved: {args.out}')
    print(f'Saved: {args.out.replace(".wav", "_original.wav")}')
    print(f'Output samples: {len(audio_out)} ({len(audio_out)/sr:.2f}s)')


if __name__ == '__main__':
    main()
