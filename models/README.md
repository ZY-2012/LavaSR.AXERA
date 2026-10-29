# Models

This directory intentionally contains no axmodel files in the GitHub repository.

Download the prebuilt AX650 models from:

<https://huggingface.co/HY-2012/LavaSR.AXERA/tree/main/models>

- `den_encoder.axmodel`
- `den_decoder.axmodel`
- `enhancer.axmodel`
- `model_meta.json`
- `model_config.json`

The models are compiled with Pulsar2 8.0 for AX650 / NPU3, U16 activations /
S8 weights / U16 outputs. See `reports/accuracy.md` and `reports/performance.md`.
