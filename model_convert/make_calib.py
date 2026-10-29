#!/usr/bin/env python3
"""Generate real-data calibration tensors for the two LavaSR NPU cores.

denoiser core input : ERB feature (1, 1, 301, 129) from noisy/clean 16 kHz audio
enhancer core input : mel feature  (1, 80, 450)    from denoised/clean 48 kHz audio

Uses the original PyTorch UL-UNAS on GPU/CPU, but the exact same host DSP as the
SDK (lavasr_sdk.audio).  Each sample is written as a float32 npy with the batch
dimension and packed into one tar.gz per model.
"""
import argparse
import glob
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
from lavasr_sdk import audio  # noqa: E402
from lavasr_sdk.io import read_wav  # noqa: E402

DEN_N = 76800
BWE_N = 230400


def list_audio(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("wav", "flac"):
                out.extend(glob.glob(os.path.join(p, "**", f"*.{ext}"), recursive=True))
        else:
            out.append(p)
    return sorted(out)


def add_noise(x, snr_db, rng):
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= rms / (np.sqrt(np.mean(noise * noise) + 1e-12) * (10 ** (snr_db / 20.0)))
    return np.clip(x + noise, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-dir", default="origin/LavaSR")
    ap.add_argument("--hf-dir", default="origin/LavaSR_HF")
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--out-dir", default="build/calib")
    ap.add_argument("--den-samples", type=int, default=256)
    ap.add_argument("--bwe-samples", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    sys.path.insert(0, args.origin_dir)
    from LavaSR.denoiser.ulunas import ULUNAS

    assets = audio.Assets(HERE.parent / "python" / "lavasr_sdk" / "assets")
    denoiser = ULUNAS()
    denoiser.load_state_dict(torch.load(os.path.join(args.hf_dir, "denoiser", "denoiser.bin"),
                                        map_location="cpu"))
    denoiser.eval().to(args.device)

    rng = np.random.default_rng(args.seed)
    files = list_audio(args.audio)
    if not files:
        raise SystemExit("no audio files")
    print(f"{len(files)} source files")

    segments = []
    for f in files:
        x, sr = read_wav(f)
        x16 = audio.resample(x, sr, 16000) if sr != 16000 else x
        if len(x16) < DEN_N:
            x16 = np.pad(x16, (0, DEN_N - len(x16)))
        n_chunks = max(1, len(x16) // DEN_N)
        for k in range(n_chunks):
            start = k * DEN_N
            if start + DEN_N > len(x16):
                start = max(0, len(x16) - DEN_N)
            segments.append(x16[start:start + DEN_N])
    rng.shuffle(segments)
    print(f"{len(segments)} clean 4.8 s segments")

    need = max(args.den_samples, args.bwe_samples)
    while len(segments) < need:
        segments.extend(segments)
    segments = segments[:need]

    noisy_segments = []
    for i, seg in enumerate(segments):
        if i % 2 == 0:
            noisy_segments.append(add_noise(seg, rng.choice([0.0, 5.0, 10.0, 20.0]), rng))
        else:
            noisy_segments.append(seg)
    noisy = np.stack(noisy_segments).astype(np.float32)
    clean = np.stack(segments).astype(np.float32)

    den_out = np.empty_like(noisy)
    with torch.no_grad():
        for i in range(0, len(noisy), 8):
            batch = torch.from_numpy(noisy[i:i + 8]).to(args.device)
            den_out[i:i + 8] = denoiser(batch).cpu().numpy()
    den_out = np.clip(den_out, -1.0, 1.0)

    out_dir = Path(args.out_dir)
    den_dir = out_dir / "denoiser"
    bwe_dir = out_dir / "bwe"
    den_dir.mkdir(parents=True, exist_ok=True)
    bwe_dir.mkdir(parents=True, exist_ok=True)

    den_count = 0
    for i in range(args.den_samples):
        src = noisy[i] if i % 3 else clean[i]
        spec = audio.den_stft(src, assets.den_window)
        feat = audio.den_features(spec, assets)
        np.save(den_dir / f"{den_count:04d}.npy", feat)
        den_count += 1

    bwe_count = 0
    order = list(range(len(noisy)))
    rng.shuffle(order)
    for i in order:
        if bwe_count >= args.bwe_samples:
            break
        src16 = den_out[i] if bwe_count % 3 != 0 else clean[i]
        src48 = audio.resample(src16, 16000, 48000)
        src48 = src48[:BWE_N]
        if len(src48) < BWE_N:
            src48 = np.pad(src48, (0, BWE_N - len(src48)), mode="reflect")
        mel = audio.mel_features(src48, assets)
        np.save(bwe_dir / f"{bwe_count:04d}.npy", mel)
        bwe_count += 1

    for name in ("denoiser", "bwe"):
        d = out_dir / name
        tar_path = out_dir / f"{name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for npy in sorted(d.glob("*.npy")):
                tar.add(npy, arcname=npy.name)
        print(f"{name}: {len(list(d.glob('*.npy')))} samples -> {tar_path}")


if __name__ == "__main__":
    main()
