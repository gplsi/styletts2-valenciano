import subprocess
import os
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Mel-Band-Roformer denoiser on a folder of WAV files")
    parser.add_argument("--input_dir", default=str(Path(__file__).resolve().parents[1] / "RAW_AUDIO"), help="Input audio folder (default: RAW_AUDIO)")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "DENOISED"), help="Output folder (default: DENOISED)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    denoiser_root = project_root / "tools" / "denoiser" / "Mel-Band-Roformer-Vocal-Model"
    models_dir = denoiser_root / "models" / "mel_band_roformer"
    config_path = denoiser_root / "configs" / "config_vocals_mel_band_roformer.yaml"
    ckpt_path = models_dir / "MelBandRoformer.ckpt"
    inference_py = denoiser_root / "inference.py"

    # Ensure model exists; download if missing
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        download_cmd = f"huggingface-cli download KimberleyJSN/melbandroformer --local-dir {models_dir}"
        subprocess.run(download_cmd, shell=True, check=True)

    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    inference_cmd = (
        f"python {inference_py} "
        f"--config_path {config_path} "
        f"--model_path {ckpt_path} "
        f"--input_folder {args.input_dir} "
        f"--store_dir {args.output_dir}"
    )
    subprocess.run(inference_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
