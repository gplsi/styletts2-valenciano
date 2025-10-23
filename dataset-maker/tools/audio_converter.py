#!/usr/bin/env python3
import argparse, os, subprocess, sys

def convert(in_path, out_path, sr, mono, bit_depth):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ar", str(sr),
        "-ac", "1" if mono else "2",
        "-sample_fmt", f"s{bit_depth}",
        out_path,
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--bit_depth", type=int, default=16, choices=[16, 24, 32])
    p.add_argument("--mono", action="store_true")
    p.add_argument("--extensions", default=".mp3,.m4a,.flac,.ogg,.aac,.wav")
    args = p.parse_args()

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    failures = 0
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            in_path = os.path.join(root, f)
            rel = os.path.relpath(in_path, args.input_dir)
            base = os.path.splitext(rel)[0] + ".wav"
            out_path = os.path.join(args.output_dir, base)
            if os.path.exists(out_path):
                continue
            rc = convert(in_path, out_path, args.sample_rate, args.mono, args.bit_depth)
            if rc != 0:
                print(f"Failed: {in_path}", file=sys.stderr)
                failures += 1
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()