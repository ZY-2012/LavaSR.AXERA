# LavaSR.AXERA C++ runtime

AX650 C++17 runtime for the three LavaSR v2 axmodels. No Python, torch or
onnxruntime dependency. Host DSP (STFT / mel / iSTFT / DPGRNN / LR merge) is
implemented in `dsp.hpp` / `dpgrnn.hpp`; KissFFT is bundled under
`third_party/kissfft` (BSD-3-Clause).

## Build

```bash
bash cpp/download_toolchains.sh       # Arm GNU 9.2 + AX650 BSP (once)
TOOLCHAIN_ROOT=$PWD/cpp/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=$PWD/cpp/toolchains/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

The binary is written to `cpp/bin/lavasr_ax650` and copied to `release/bin/`
for the HuggingFace package.

## Run on board

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/lavasr_ax650 models/den_encoder.axmodel models/den_decoder.axmodel \
    models/enhancer.axmodel samples/libri_16k.wav output.wav [--no-denoise] [--bench]
```

`--bench` averages 3 runs. `LAVASR_PROFILE=1` prints stage timings.
The build uses OpenMP (Cortex-A55 8 cores) for the frame/batch-parallel host DSP,
a 5-smooth LR-merge FFT length, cached AX I/O buffers and AX Engine V2 runtime.

## Results (AX650N, NPU3)

| input | denoise | time | RTF |
|---|---:|---:|---:|
| LibriSpeech 17.75 s | yes | 1.08 s | 0.061 |
| LibriSpeech 17.75 s | no | 0.70 s | 0.039 |

C++ vs ONNX reference: SNR 46.0 dB (denoise) / 47.6 dB (no denoise).

`assets.cpp` is generated from `python/lavasr_sdk/assets`:

```bash
python cpp/gen_assets.py
```

Regenerate it only if the Python assets change.
