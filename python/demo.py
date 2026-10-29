#!/usr/bin/env python3
"""LavaSR v2 AX650 demo: enhance/upsample a wav on the board.

Usage:
    python3 demo.py input.wav -o output.wav
    python3 demo.py input.wav -o output.wav --no-denoise
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lavasr_sdk import LavaSR, read_wav, write_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--no-denoise", action="store_true", help="skip the UL-UNAS denoiser")
    ap.add_argument("--cutoff", type=int, default=None,
                    help="LR crossover Hz (default: min(input_sr,16000)/2)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.models)
    model = LavaSR(
        os.path.join(model_dir, "den_encoder.axmodel"),
        os.path.join(model_dir, "den_decoder.axmodel"),
        os.path.join(model_dir, "enhancer.axmodel"),
        model_config=os.path.join(model_dir, "model_config.json"),
        cutoff=args.cutoff,
    )
    x, sr = read_wav(args.input)
    t0 = time.time()
    y, osr = model.enhance(x, sr, denoise=not args.no_denoise)
    dt = time.time() - t0
    out = args.output or os.path.splitext(args.input)[0] + "_enhanced.wav"
    write_wav(out, y, osr)
    dur = len(x) / float(sr)
    print(f"LavaSR v2: {sr} Hz -> {osr} Hz, {dur:.2f}s audio")
    print(f"  denoise : {not args.no_denoise}")
    print(f"  time    : {dt:.3f}s   RTF {dt / max(dur, 1e-9):.4f}")
    print(f"  output  : {out}")


if __name__ == "__main__":
    main()
