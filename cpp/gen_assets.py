#!/usr/bin/env python3
"""Generate cpp/assets.cpp / cpp/assets.h from python/lavasr_sdk/assets."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "python", "lavasr_sdk", "assets")


def _lit(v):
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def fmt(arr):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    lines = []
    for i in range(0, len(a), 8):
        lines.append("    " + ", ".join(_lit(v) for v in a[i:i + 8]) + ",")
    return "\n".join(lines)


def main():
    mel_fb = np.load(os.path.join(ASSETS, "mel_fb.npy"))
    mel_win = np.load(os.path.join(ASSETS, "mel_window.npy"))
    den_win = np.load(os.path.join(ASSETS, "den_window.npy"))
    erb_fc = np.load(os.path.join(ASSETS, "erb_fc.npy"))
    ierb_fc = np.load(os.path.join(ASSETS, "ierb_fc.npy"))
    dp = np.load(os.path.join(ASSETS, "dpgrnn.npz"))

    h = ['#pragma once', '#include <cstddef>', '',
         f'extern const float kMelFb[{mel_fb.size}];',
         f'extern const float kMelWindow[{mel_win.size}];',
         f'extern const float kDenWindow[{den_win.size}];',
         f'extern const float kErbFc[{erb_fc.size}];',
         f'extern const float kIerbFc[{ierb_fc.size}];',
         'struct DpAsset { const char* name; const float* data; int size; };',
         'extern const DpAsset kDpAssets[];',
         'extern const int kDpAssetCount;', '']
    with open(os.path.join(ROOT, "cpp", "assets.h"), "w") as f:
        f.write("\n".join(h))

    c = ['#include "assets.h"', '']
    c.append(f'const float kMelFb[{mel_fb.size}] = {{\n{fmt(mel_fb)}\n}};')
    c.append(f'const float kMelWindow[{mel_win.size}] = {{\n{fmt(mel_win)}\n}};')
    c.append(f'const float kDenWindow[{den_win.size}] = {{\n{fmt(den_win)}\n}};')
    c.append(f'const float kErbFc[{erb_fc.size}] = {{\n{fmt(erb_fc)}\n}};')
    c.append(f'const float kIerbFc[{ierb_fc.size}] = {{\n{fmt(ierb_fc)}\n}};')
    entries = []
    for i, key in enumerate(sorted(dp.files)):
        arr = dp[key]
        c.append(f'static const float kDp_{i}[{arr.size}] = {{\n{fmt(arr)}\n}};')
        entries.append(f'    {{"{key}", kDp_{i}, {arr.size}}},')
    c.append('const DpAsset kDpAssets[] = {\n' + "\n".join(entries) + '\n};')
    c.append(f'const int kDpAssetCount = {len(entries)};')
    with open(os.path.join(ROOT, "cpp", "assets.cpp"), "w") as f:
        f.write("\n".join(c) + "\n")
    print("generated cpp/assets.h / cpp/assets.cpp")


if __name__ == "__main__":
    main()
