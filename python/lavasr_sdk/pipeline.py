"""LavaSR v2 end-to-end pipeline: UL-UNAS denoiser + Vocos BWE, NPU cores.

Host: STFT / ERB / mel / mask application / exp / iSTFT / LR merge / resampling
      + the tiny UL-UNAS DPGRNN (Pulsar2 cannot lower its large-batch GRU).
NPU : denoiser encoder/decoder and the Vocos enhancer core, all in fixed 4.8 s
      chunks with 50% overlap.
"""
import json
import os

import numpy as np

from . import audio
from .dpgrnn import DPGRNN
from .io import read_wav, write_wav  # re-export

DEN_CHUNK = 4.8
BWE_CHUNK = 4.8


def _pad_to(x, n, mode="reflect"):
    if len(x) >= n:
        return x
    pad = n - len(x)
    if mode == "reflect" and len(x) > 1:
        return np.pad(x, (0, pad), mode="reflect")
    return np.pad(x, (0, pad), mode="constant")


def _starts(length, chunk, hop):
    if length <= chunk:
        return [0], 0
    pad = (hop - (length - chunk) % hop) % hop
    starts = list(range(0, length + pad - chunk + 1, hop))
    if starts[-1] + chunk < length + pad:
        starts.append(length + pad - chunk)
    return starts, pad


class Core:
    """Thin wrapper around an axengine session (board) or onnxruntime (host)."""

    def __init__(self, path, backend="auto"):
        self.path = str(path)
        self.sess = None
        self.backend = backend
        self.input_names = []
        self.output_names = []
        if backend in ("auto", "axengine"):
            try:
                import axengine as axe

                self.sess = axe.InferenceSession(self.path)
                self.input_names = [i.name for i in self.sess.get_inputs()]
                try:
                    self.output_names = [o.name for o in self.sess.get_outputs()]
                except Exception:
                    self.output_names = []
                self.backend = "axengine"
            except Exception:
                if backend == "axengine":
                    raise
        if self.sess is None:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.sess.get_inputs()]
            self.output_names = [o.name for o in self.sess.get_outputs()]
            self.backend = "onnxruntime"

    def run(self, feed):
        names = self.input_names
        if isinstance(feed, dict):
            data = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()}
        else:
            data = {names[0]: np.ascontiguousarray(feed, dtype=np.float32)}
        return self.sess.run(None, data)

    def __call__(self, x):
        return self.run(x)[0]


class LavaSR:
    def __init__(self, denoiser_encoder, denoiser_decoder, enhancer_model,
                 model_config=None, asset_dir=None, cutoff=None, backend="auto"):
        self.assets = audio.Assets(asset_dir)
        cfg_path = model_config
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(denoiser_encoder), "model_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.den_cfg = self.cfg["denoiser"]
        self.bwe_cfg = self.cfg["enhancer"]
        self.den_encoder = Core(denoiser_encoder, backend=backend)
        self.den_decoder = Core(denoiser_decoder, backend=backend)
        self.bwe_core = Core(enhancer_model, backend=backend)
        asset_dir = asset_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        self.dpgrnn = DPGRNN(os.path.join(asset_dir, "dpgrnn.npz"))
        self.cutoff = cutoff
        self._den_win = self.assets.den_window
        self._mel_win = self.assets.mel_window
        self._den_hop = int(self.den_cfg["chunk_samples"] * (1 - self.den_cfg["overlap"]))
        self._bwe_hop = int(self.bwe_cfg["chunk_samples"] * (1 - self.bwe_cfg["overlap"]))

    def _denoise_chunk(self, c):
        spec = audio.den_stft(c, self._den_win)
        feat = audio.den_features(spec, self.assets)
        enc = self.den_encoder.run({"erb_feat": feat})
        dp = self.dpgrnn(enc[0])
        feed = {"feat": dp, "en0": enc[1], "en1": enc[2], "en2": enc[3], "en3": enc[4], "en4": enc[5]}
        mask = self.den_decoder.run(feed)[0][0, 0]
        m = audio.den_mask_expand(mask, self.assets)
        return audio.den_istft(spec * m, self._den_win, len(c))

    def denoise_16k(self, x):
        n = int(self.den_cfg["chunk_samples"])
        hop = self._den_hop
        starts, pad = _starts(len(x), n, hop)
        xp = _pad_to(x, len(x) + pad)
        chunks = []
        for s in starts:
            c = xp[s:s + n]
            if len(c) < n:
                c = _pad_to(c, n)
            chunks.append(self._denoise_chunk(c))
        return audio.crossfade(chunks, hop, n, len(x))

    def _enhance_chunk(self, c):
        mel = audio.mel_features(c, self.assets)
        head = self.bwe_core(mel)[0]
        bins = self.bwe_cfg["bins"]
        mag = np.exp(np.clip(head[:bins], None, 1e3))
        phase = head[bins:]
        spec = (mag * np.cos(phase) + 1j * mag * np.sin(phase)).T
        return audio.bwe_istft(spec, self._mel_win)

    def enhance_48k(self, x):
        n = int(self.bwe_cfg["chunk_samples"])
        hop = self._bwe_hop
        out_len = (len(x) // 512) * 512
        starts, pad = _starts(len(x), n, hop)
        xp = _pad_to(x, len(x) + pad)
        chunks = []
        for s in starts:
            c = xp[s:s + n]
            if len(c) < n:
                c = _pad_to(c, n)
            chunks.append(self._enhance_chunk(c))
        y = audio.crossfade(chunks, hop, n, len(xp))
        return y[:out_len]

    def enhance(self, wav, sr, denoise=True):
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = np.ascontiguousarray(x)
        x16 = audio.resample(x, int(sr), 16000) if int(sr) != 16000 else x
        if denoise:
            x16 = self.denoise_16k(x16)
        x48 = audio.resample(x16, 16000, 48000)
        generated = self.enhance_48k(x48)
        cutoff = self.cutoff
        if cutoff is None:
            cutoff = min(int(sr), 16000) // 2
        original = x48[: len(generated)]
        return audio.lr_merge(generated, original, cutoff), 48000
