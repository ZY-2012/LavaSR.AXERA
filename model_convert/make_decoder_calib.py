#!/usr/bin/env python3
"""Build multi-input calibration tensors for the denoiser decoder.

Runs the exported encoder ONNX on every denoiser calibration sample and saves
the resulting feat/en0..en4 tensors, one tar.gz per decoder input.
"""
import argparse
import shutil
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def pack(folder, tar_path):
    with tarfile.open(tar_path, "w:gz") as tar:
        for npy in sorted(folder.glob("*.npy")):
            tar.add(npy, arcname=npy.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="build/onnx/denoiser_encoder.sim.onnx")
    ap.add_argument("--den-calib", default="build/calib/denoiser")
    ap.add_argument("--out-dir", default="build/calib/decoder")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    names = ["feat", "en0", "en1", "en2", "en3", "en4"]
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    files = sorted(Path(args.den_calib).glob("*.npy"))
    if args.limit:
        files = files[: args.limit]
    for i, f in enumerate(files):
        x = np.load(f).astype(np.float32)
        ys = sess.run(None, {"erb_feat": x})
        for n, y in zip(names, ys):
            np.save(out / n / f"{i:04d}.npy", y.astype(np.float32))
    for n in names:
        pack(out / n, out / f"{n}.tar.gz")
        print(n, len(list((out / n).glob("*.npy"))))


if __name__ == "__main__":
    main()
