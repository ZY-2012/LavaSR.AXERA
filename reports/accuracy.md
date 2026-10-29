# LavaSR v2 AX650 accuracy

Board: AX650N, BSP `V3.10.2_20260323224507`, AX Engine `2.12.0s`, Pulsar2 `8.0`.
Reference: original PyTorch LavaSR v2 (`enhancer_v2` + `denoiser`) and the exported
ONNX cores + host DSP. Metrics are computed on the 48 kHz waveform.

## Quantization transparency (board vs ONNX)

SNR is `10*log10(sum(ref^2)/sum((board-ref)^2))`. Cosine is on the raw waveform.

| case | input | denoise | board | cosine | SNR |
|---|---|---|---|---|---|
| LibriSpeech 17.75 s | 16 kHz | yes | Python | 0.9999878 | 45.95 dB |
| LibriSpeech 17.75 s | 16 kHz | no | Python | 0.9999912 | 47.65 dB |
| LibriSpeech 17.75 s | 8 kHz | yes | Python | 0.9999919 | 47.87 dB |
| LibriSpeech 17.75 s | 24 kHz | yes | Python | 0.9999864 | 45.57 dB |
| LibriSpeech 17.75 s | 48 kHz | yes | Python | 0.9999863 | 45.52 dB |
| LibriSpeech 30.86 s | 16 kHz | yes | Python | 0.9999483 | 39.79 dB |
| LibriSpeech 30.86 s | 16 kHz | no | Python | 0.9999844 | 45.06 dB |
| LibriSpeech 3.0 s | 16 kHz | yes | Python | 0.9999899 | 46.95 dB |
| LibriSpeech 3.0 s | 16 kHz | no | Python | 0.9999965 | 51.65 dB |
| noisy speech 10 s | 48 kHz | yes | Python | 0.9998354 | 34.71 dB |
| noisy speech 10 s | 48 kHz | no | Python | 0.9999513 | 40.10 dB |
| LibriSpeech 17.75 s | 16 kHz | yes | C++ | 0.9999858 | 45.94 dB |
| LibriSpeech 17.75 s | 16 kHz | no | C++ | 0.9999896 | 47.62 dB |

The C++ and Python SDKs use the same models; differences are only host FFT /
resampler rounding.

## Float reference

The ONNX + host-DSP pipeline (before quantization) vs the original PyTorch
implementation is **cosine 0.99995, SNR 39.6 dB** on the 17.75 s LibriSpeech
sample. The remaining difference is the 4.8 s chunking (50% overlap for BWE,
25% for the denoiser) and the fast zero-padded LR merge (~60 dB vs the exact
full-length FFT).

## Quantization scheme

- All three axmodels: **U16 activations, S8 weights, U16 output** (`MinMax`).
- Calibration: real speech (LibriSpeech + AISHELL-1), noisy variants mixed in;
  256 samples for encoder/enhancer, 64 for the decoder.
- The large FA GRU inside every `cTFA` block is exported as an unrolled
  MatMul/Add/Sigmoid/Tanh graph. Pulsar2 8.0's `AxDynamicQuantizedGRU` for that
  shape returns `Failed to run model` on this BSP; the manual expansion runs and
  is numerically identical (cosine 1.0 vs PyTorch).
