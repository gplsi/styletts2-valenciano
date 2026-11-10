import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    """AlbertModel que devuelve solo last_hidden_state (requerido por StyleTTS2)."""
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def load_plbert(log_dir):
    """
    Carga PL-BERT valenciano.

    Args:
        log_dir: Puede ser:
            - Ruta local a directorio con config.yml y step_*.t7
            - ID de HuggingFace (ej: "javiimts/plbert-valenciano")
    """
    # Intentar cargar desde HuggingFace
    if not os.path.exists(log_dir):
        print(f"Cargando PL-BERT desde HuggingFace: {log_dir}")
        model = CustomAlbert.from_pretrained(log_dir)
        model.eval()
        return model

    # Cargar desde directorio local
    print(f"Cargando PL-BERT desde: {log_dir}")

    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    # Buscar checkpoint
    ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(os.path.join(log_dir, f"step_{iters}.t7"), map_location='cpu')
    state_dict = checkpoint['net']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v

    new_state_dict.pop("embeddings.position_ids", None)
    bert.load_state_dict(new_state_dict, strict=False)

    return bert
