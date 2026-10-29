#!/usr/bin/env python3
"""Export LavaSR v2 neural cores to static ONNX for AX650.

Pipeline split:

    host:  UL-UNAS STFT / ERB features
    NPU :  denoiser encoder  (erb_feat -> feat + 5 skip tensors)
    host:  DPGRNN (2 blocks; Pulsar2 cannot lower its large-batch GRU)
    NPU :  denoiser decoder  (feat + skips -> ERB mask)
    host:  mask application / ISTFT / 16k->48k resample
    host:  Vocos mel features
    NPU :  Vocos backbone + head linear (mel -> log-mag / phase)
    host:  exp / cos / sin / ISTFT / Linkwitz-Riley merge
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEN_CHUNK_SEC = 4.8
BWE_CHUNK_SEC = 4.8
DEN_SR = 16000
BWE_SR = 48000
DEN_T = int(DEN_CHUNK_SEC * DEN_SR) // 256 + 1          # 301
BWE_T = int(BWE_CHUNK_SEC * BWE_SR) // 512              # 450
BWE_N = BWE_T * 512                                     # 230400

DEN_SKIP_SHAPES = [
    (1, 12, DEN_T, 65),
    (1, 24, DEN_T, 33),
    (1, 24, DEN_T, 33),
    (1, 32, DEN_T, 33),
    (1, 16, DEN_T, 33),
]




class ManualGRU(torch.nn.Module):
    """Unrolled bidirectional GRU (Pulsar2's GRU lowering is broken on this BSP).

    Only used for the tiny FA GRU (input/hidden = 4, sequence 17 or 33).  The
    recurrence is expanded into MatMul/Add/Sigmoid/Tanh, which Pulsar2 lowers
    correctly, instead of emitting an ONNX GRU node.
    """

    def __init__(self, gru):
        super().__init__()
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        self.bidirectional = gru.bidirectional
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            setattr(self, name, torch.nn.Parameter(getattr(gru, name).detach().clone()))
        if gru.bidirectional:
            for name in ("weight_ih_l0_reverse", "weight_hh_l0_reverse",
                         "bias_ih_l0_reverse", "bias_hh_l0_reverse"):
                setattr(self, name, torch.nn.Parameter(getattr(gru, name).detach().clone()))

    def _run(self, x, reverse=False):
        suffix = "_reverse" if reverse else ""
        w_ih = getattr(self, "weight_ih_l0" + suffix)
        w_hh = getattr(self, "weight_hh_l0" + suffix)
        b_ih = getattr(self, "bias_ih_l0" + suffix)
        b_hh = getattr(self, "bias_hh_l0" + suffix)
        b, t, _ = x.shape
        hsize = self.hidden_size
        h = x.new_zeros(b, hsize)
        out = []
        steps = range(t - 1, -1, -1) if reverse else range(t)
        for i in steps:
            gx = x[:, i] @ w_ih.T + b_ih
            gh = h @ w_hh.T + b_hh
            r = torch.sigmoid(gx[:, :hsize] + gh[:, :hsize])
            z = torch.sigmoid(gx[:, hsize:2 * hsize] + gh[:, hsize:2 * hsize])
            n = torch.tanh(gx[:, 2 * hsize:] + r * gh[:, 2 * hsize:])
            h = (1.0 - z) * n + z * h
            out.append(h)
        y = torch.stack(out, dim=1)
        if reverse:
            y = torch.flip(y, dims=[1])
        return y

    def forward(self, x):
        y = self._run(x, False)
        if self.bidirectional:
            y = torch.cat([y, self._run(x, True)], dim=-1)
        return y, y


def patch_fa_gru(model):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "FA" and isinstance(getattr(m, "gru", None), torch.nn.GRU):
            m.gru = ManualGRU(m.gru)
            n += 1
    return n


class DenEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, feat):
        x, en_outs = self.encoder(feat)
        return (x, *en_outs)


class DenDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, feat, en0, en1, en2, en3, en4):
        return self.decoder(feat, [en0, en1, en2, en3, en4])


class BWECore(torch.nn.Module):
    """Vocos backbone + ISTFT-head projection (mel -> log-mag / phase)."""

    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.out = vocos.head.out

    def forward(self, mel):
        x = self.backbone(mel)
        return self.out(x).transpose(1, 2)


def load_models(origin_dir: Path, hf_dir: Path):
    sys.path.insert(0, str(origin_dir))
    from vocos import Vocos
    from LavaSR.denoiser.ulunas import ULUNAS

    vocos = Vocos.from_hparams(str(hf_dir / "enhancer_v2" / "config.yaml"))
    state = torch.load(hf_dir / "enhancer_v2" / "pytorch_model.bin", map_location="cpu")
    vocos.load_state_dict(state)
    vocos.eval()

    denoiser = ULUNAS()
    denoiser.load_state_dict(torch.load(hf_dir / "denoiser" / "denoiser.bin", map_location="cpu"))
    denoiser.eval()
    return vocos, denoiser


def export_model(module, args, path, input_names, output_names, opset=17):
    module = module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
        )
    import onnx
    import onnxsim

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    sim, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim failed for {path}")
    sim_path = path.with_name(path.stem + ".sim.onnx")
    onnx.save(sim, str(sim_path))
    return sim_path


def save_assets(vocos, denoiser, out_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    from lavasr_sdk.dpgrnn import extract_params

    out_dir.mkdir(parents=True, exist_ok=True)
    fe = vocos.feature_extractor
    np.save(out_dir / "mel_fb.npy", fe.mel_spec.mel_scale.fb.detach().numpy().astype(np.float32))
    np.save(out_dir / "mel_window.npy", torch.hann_window(2048).numpy().astype(np.float32))
    np.save(out_dir / "den_window.npy", torch.hann_window(512).numpy().astype(np.float32))
    erb = denoiser.erb
    np.save(out_dir / "erb_fc.npy", erb.erb_fc.weight.detach().numpy().astype(np.float32))
    np.save(out_dir / "ierb_fc.npy", erb.ierb_fc.weight.detach().numpy().astype(np.float32))
    extract_params(denoiser, out_dir / "dpgrnn.npz")
    cfg = {
        "denoiser": {
            "sample_rate": DEN_SR,
            "n_fft": 512,
            "hop": 256,
            "win": 512,
            "chunk_samples": int(DEN_CHUNK_SEC * DEN_SR),
            "frames": DEN_T,
            "erb_low": 65,
            "erb_high": 64,
            "bins": 257,
            "overlap": 0.25,
        },
        "enhancer": {
            "sample_rate": BWE_SR,
            "model_sample_rate": fe.mel_spec.sample_rate,
            "n_fft": 2048,
            "hop": 512,
            "win": 2048,
            "n_mels": 80,
            "chunk_samples": BWE_N,
            "frames": BWE_T,
            "bins": 1025,
            "padding": "same",
            "overlap": 0.5,
        },
    }
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-dir", default="origin/LavaSR")
    ap.add_argument("--hf-dir", default="origin/LavaSR_HF")
    ap.add_argument("--out-dir", default="build/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--asset-dir", default="python/lavasr_sdk/assets")
    args = ap.parse_args()

    origin_dir = Path(args.origin_dir)
    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocos, denoiser = load_models(origin_dir, hf_dir)
    print(f"patched {patch_fa_gru(denoiser)} FA GRUs")
    den_encoder = DenEncoder(denoiser)
    den_decoder = DenDecoder(denoiser)
    bwe_core = BWECore(vocos)

    den_example = torch.zeros(1, 1, DEN_T, 129, dtype=torch.float32)
    bwe_example = torch.zeros(1, 80, BWE_T, dtype=torch.float32)
    dec_example = (torch.zeros(1, 16, DEN_T, 33),) + tuple(
        torch.zeros(*s) for s in DEN_SKIP_SHAPES
    )

    enc_onnx = export_model(
        den_encoder, (den_example,), out_dir / "denoiser_encoder.onnx",
        ["erb_feat"], ["feat", "en0", "en1", "en2", "en3", "en4"], args.opset,
    )
    dec_onnx = export_model(
        den_decoder, dec_example, out_dir / "denoiser_decoder.onnx",
        ["feat", "en0", "en1", "en2", "en3", "en4"], ["mask"], args.opset,
    )
    bwe_onnx = export_model(
        bwe_core, (bwe_example,), out_dir / "enhancer_core.onnx",
        ["mel"], ["head_out"], args.opset,
    )

    meta = {
        "model_name": "LavaSR-v2",
        "denoiser_encoder": {
            "onnx": enc_onnx.name,
            "inputs": [{"name": "erb_feat", "shape": [1, 1, DEN_T, 129], "dtype": "float32"}],
            "outputs": [{"name": "feat", "shape": [1, 16, DEN_T, 33], "dtype": "float32"}]
            + [{"name": f"en{i}", "shape": list(s), "dtype": "float32"} for i, s in enumerate(DEN_SKIP_SHAPES)],
        },
        "denoiser_decoder": {
            "onnx": dec_onnx.name,
            "inputs": [{"name": "feat", "shape": [1, 16, DEN_T, 33], "dtype": "float32"}]
            + [{"name": f"en{i}", "shape": list(s), "dtype": "float32"} for i, s in enumerate(DEN_SKIP_SHAPES)],
            "outputs": [{"name": "mask", "shape": [1, 1, DEN_T, 129], "dtype": "float32"}],
        },
        "enhancer": {
            "onnx": bwe_onnx.name,
            "inputs": [{"name": "mel", "shape": [1, 80, BWE_T], "dtype": "float32"}],
            "outputs": [{"name": "head_out", "shape": [1, 2050, BWE_T], "dtype": "float32"}],
            "chunk_seconds": BWE_CHUNK_SEC,
            "chunk_samples": BWE_N,
        },
        "denoiser": {
            "chunk_seconds": DEN_CHUNK_SEC,
            "chunk_samples": int(DEN_CHUNK_SEC * DEN_SR),
            "dpgrnn": "host numpy (assets/dpgrnn.npz)",
        },
        "opset": args.opset,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    save_assets(vocos, denoiser, Path(args.asset_dir))
    print(f"exported {enc_onnx}, {dec_onnx}, {bwe_onnx}")
    print(f"assets -> {args.asset_dir}")


if __name__ == "__main__":
    main()
