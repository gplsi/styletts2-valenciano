import os
import subprocess
from pathlib import Path
import argparse

def process_wav_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    wav_files = list(Path(input_dir).glob('*.wav'))

    for wav_file in wav_files:
        input_path = str(wav_file)
        output_path = os.path.join(output_dir, wav_file.name)
    
        # Construct the unsilence command
        command = [
            'unsilence',
            '-ao',  # audio only
            '-sl', '-35',  # silence level
            '-stt', '0.5',  # silence time threshold
            '-sit', '0.3',  # short interval threshold
            '-y',  # 
            input_path,
            output_path
        ]
    
        print(f"Processing: {wav_file.name}")
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed: {wav_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {wav_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove silence from WAV files using 'unsilence'")
    parser.add_argument("--input_dir", default="../Automatic-Audio-Dataset-Maker/RAW_AUDIO", help="Input directory (default: RAW_AUDIO)")
    parser.add_argument("--output_dir", default="../Automatic-Audio-Dataset-Maker/RAW_AUDIO1", help="Output directory (default: RAW_AUDIO1)")
    args = parser.parse_args()

    process_wav_files(args.input_dir, args.output_dir)