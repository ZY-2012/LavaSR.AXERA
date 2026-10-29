# Model conversion

Reproduces the AX650 axmodels from the upstream LavaSR v2 checkpoint.

## 1. Dependencies

```bash
pip install -r model_convert/requirements.txt
```

Also install the upstream source and weights:

```bash
git clone https://github.com/ysharma3501/LavaSR origin/LavaSR
hf download YatharthS/LavaSR --local-dir origin/LavaSR_HF
```

## 2. Export ONNX

```bash
python model_convert/export_onnx.py \
  --origin-dir origin/LavaSR \
  --hf-dir origin/LavaSR_HF \
  --out-dir build/onnx \
  --asset-dir python/lavasr_sdk/assets
```

`export_onnx.py` does three important things:

1. Splits UL-UNAS into `den_encoder` + DPGRNN (host) + `den_decoder`.
2. Replaces every `cTFA` FA GRU with an unrolled `MatMul/Add/Sigmoid/Tanh`
   graph. Pulsar2 8.0's ONNX GRU lowering for this shape fails on the board.
3. Exports the Vocos backbone + head projection as `enhancer_core`.

It also writes the runtime assets (`mel_fb`, windows, ERB matrices,
`dpgrnn.npz`) and `model_config.json`.

## 3. Calibration data

```bash
python model_convert/make_calib.py \
  --origin-dir origin/LavaSR --hf-dir origin/LavaSR_HF \
  --audio <wav-files-or-dirs> --out-dir build/calib

python model_convert/make_decoder_calib.py
```

`make_calib.py` uses real speech (clean + noisy variants) and the same host DSP
as the SDK. It writes `build/calib/denoiser.tar.gz` and
`build/calib/bwe.tar.gz`; `make_decoder_calib.py` runs the exported encoder and
writes the six decoder input tars.

## 4. Pulsar2 8.0

```bash
hf download AXERA-TECH/Pulsar2 8.0/ax_pulsar2_8.0.tar.gz --local-dir .
docker load -i ax_pulsar2_8.0.tar.gz    # -> pulsar2:8.0

python model_convert/prepare_compile.py
PULSAR2_IMAGE=pulsar2:8.0 HASP_HOME=<license-dir> bash model_convert/compile.sh
```

`HASP_HOME` only applies to the driverless Pulsar2 license used on our build
host; a normal Sentinel/ACC setup can omit it.

Outputs:

```
build/compile/den_encoder/model.axmodel
build/compile/den_decoder/model.axmodel
build/compile/enhancer/model.axmodel
```

Copy them to `release/models/` as `den_encoder.axmodel`,
`den_decoder.axmodel`, `enhancer.axmodel`.
