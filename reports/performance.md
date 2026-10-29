# LavaSR v2 AX650 performance

Board: AX650N, NPU3, BSP `V3.10.2_20260323224507`, AX Engine `2.12.0s`.
`RTF = total enhance() wall time / input audio duration`; lower is better.
No NPU simulation was used; all numbers are from the board.

## Models

| axmodel | size | NPU latency |
|---|---:|---:|
| `den_encoder.axmodel` | 11.32 MB | ~17.7 ms / 4.8 s chunk |
| `den_decoder.axmodel` | 11.80 MB | ~16.4 ms / 4.8 s chunk |
| `enhancer.axmodel` | 15.07 MB | ~5.7 ms / 4.8 s chunk |
| **total** | **38.2 MB** | |

NPU latency is per 4.8 s chunk, measured by wrapping the three sessions in the
Python SDK. Host DSP (STFT/mel/iSTFT/DPGRNN/LR merge) dominates the wall time.

## Python SDK

| input | duration | denoise | time | RTF |
|---|---:|---|---:|---:|
| LibriSpeech 16 kHz | 17.75 s | yes | 4.93 s | **0.278** |
| LibriSpeech 16 kHz | 17.75 s | no | 1.82 s | **0.103** |
| LibriSpeech 16 kHz | 30.86 s | yes | 8.73 s | 0.283 |
| LibriSpeech 16 kHz | 30.86 s | no | 3.21 s | 0.104 |
| LibriSpeech 16 kHz | 3.0 s | yes | 0.95 s | 0.315 |
| LibriSpeech 16 kHz | 3.0 s | no | 0.32 s | 0.108 |
| noisy 48 kHz | 10.0 s | yes | 2.86 s | 0.286 |
| noisy 48 kHz | 10.0 s | no | 1.05 s | 0.105 |
| LibriSpeech 8 kHz | 17.75 s | yes | 4.87 s | 0.274 |
| LibriSpeech 24 kHz | 17.75 s | yes | 5.01 s | 0.283 |
| LibriSpeech 48 kHz | 17.75 s | yes | 5.01 s | 0.282 |

Host breakdown for the 17.75 s / denoise case: denoiser DSP + DPGRNN 3.07 s,
BWE mel/iSTFT 1.35 s, LR merge 0.45 s, resampling 0.10 s; NPU calls 0.24 s total.

## C++ SDK

`bin/lavasr_ax650` (cross-compiled, AX650 aarch64) uses the same axmodels and
embedded assets:

| input | duration | denoise | time | RTF |
|---|---:|---|---:|---:|
| LibriSpeech 16 kHz | 17.75 s | yes | 5.75 s | **0.324** |
| LibriSpeech 16 kHz | 17.75 s | no | 3.37 s | **0.190** |

C++ host breakdown: denoiser 2.09 s, BWE 2.27 s, LR merge 1.11 s, resampling
0.06 s. The Python/numpy SDK is faster on this board for the mel/iSTFT stage
(BLAS/vectorized ops); the C++ SDK is self-contained and has no Python/numpy
runtime dependency.

## Runtime requirements

- Board runtime: AX650 BSP with `ax_engine` / `ax_sys` (tested with 3.10.2).
- Python SDK: `numpy`, `pyaxengine` (`axengine`), optional `soundfile`.
- C++ SDK: prebuilt `bin/lavasr_ax650`; runtime libs in `/soc/lib`.
- No onnxruntime / torch on the board.
