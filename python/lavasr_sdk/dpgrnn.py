"""Numpy implementation of UL-UNAS DPGRNN (grouped dual-path RNN).

Pulsar2 cannot lower the large-batch GRU transposes in this block to AX650
(``AxTranspose: size 16 cannot explode into 256``), so the two DPGRNN blocks
run on the host between the NPU encoder and decoder.  They are tiny: 2 blocks,
6 GRU directions, ~60 k parameters, < 5 ms for a 4.8 s chunk.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru(x, p, prefix, reverse=False):
    w_ih = p[prefix + ".weight_ih_l0" + ("_reverse" if reverse else "")]
    w_hh = p[prefix + ".weight_hh_l0" + ("_reverse" if reverse else "")]
    b_ih = p[prefix + ".bias_ih_l0" + ("_reverse" if reverse else "")]
    b_hh = p[prefix + ".bias_hh_l0" + ("_reverse" if reverse else "")]
    if reverse:
        x = x[:, ::-1]
    b, t, _ = x.shape
    hsize = w_hh.shape[1]
    gx = x @ w_ih.T + b_ih
    h = np.zeros((b, hsize), dtype=np.float32)
    out = np.empty((b, t, hsize), dtype=np.float32)
    for i in range(t):
        gh = h @ w_hh.T + b_hh
        r = _sigmoid(gx[:, i, :hsize] + gh[:, :hsize])
        z = _sigmoid(gx[:, i, hsize:2 * hsize] + gh[:, hsize:2 * hsize])
        n = np.tanh(gx[:, i, 2 * hsize:] + r * gh[:, 2 * hsize:])
        h = (1.0 - z) * n + z * h
        out[:, i] = h
    if reverse:
        out = out[:, ::-1]
    return out


def _grnn(x, p, prefix):
    x1, x2 = np.split(x, 2, axis=-1)
    y1 = _gru(x1, p, prefix + ".rnn1")
    if (prefix + ".rnn1.weight_ih_l0_reverse") in p:
        y1b = _gru(x1, p, prefix + ".rnn1", reverse=True)
        y1 = np.concatenate([y1, y1b], axis=-1)
    y2 = _gru(x2, p, prefix + ".rnn2")
    if (prefix + ".rnn2.weight_ih_l0_reverse") in p:
        y2b = _gru(x2, p, prefix + ".rnn2", reverse=True)
        y2 = np.concatenate([y2, y2b], axis=-1)
    return np.concatenate([y1, y2], axis=-1)


def _layer_norm(x, w, b):
    mean = x.mean(axis=(-2, -1), keepdims=True)
    var = x.var(axis=(-2, -1), keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-8) * w + b


def _linear(x, p, prefix):
    return x @ p[prefix + ".weight"].T + p[prefix + ".bias"]


class DPGRNN:
    def __init__(self, npz_path, blocks=2):
        self.p = dict(np.load(npz_path))
        self.blocks = blocks

    def _one(self, x, i):
        p = self.p
        pre = f"b{i}."
        x = x.transpose(0, 2, 3, 1)  # (B,C,T,F) -> (B,T,F,C)
        b, t, f, c = x.shape
        intra = _grnn(x.reshape(b * t, f, c), p, pre + "intra_rnn")
        intra = _linear(intra, p, pre + "intra_fc").reshape(b, t, f, c)
        intra = _layer_norm(intra, p[pre + "intra_ln.weight"], p[pre + "intra_ln.bias"])
        intra_out = x + intra

        x = intra_out.transpose(0, 2, 1, 3)  # (B,F,T,C)
        inter = _grnn(x.reshape(b * f, t, c), p, pre + "inter_rnn")
        inter = _linear(inter, p, pre + "inter_fc").reshape(b, f, t, c)
        inter = inter.transpose(0, 2, 1, 3)
        inter = _layer_norm(inter, p[pre + "inter_ln.weight"], p[pre + "inter_ln.bias"])
        out = intra_out + inter
        return out.transpose(0, 3, 1, 2)  # (B,C,T,F)

    def __call__(self, x):
        for i in range(self.blocks):
            x = self._one(x, i)
        return x


def extract_params(model, out_path):
    """Extract DPGRNN weights from a torch ULUNAS model into a small .npz."""
    sd = model.state_dict()
    out = {}
    for i, block in enumerate(model.dpgrnn):
        pre = f"b{i}."
        for part in ("intra_rnn", "inter_rnn"):
            for rnn in ("rnn1", "rnn2"):
                p = f"dpgrnn.{i}.{part}.{rnn}"
                for suffix in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                    key = f"{p}.{suffix}"
                    if key in sd:
                        out[pre + f"{part}.{rnn}.{suffix}"] = sd[key].detach().numpy().astype(np.float32)
                    if key + "_reverse" in sd:
                        out[pre + f"{part}.{rnn}.{suffix}_reverse"] = sd[key + "_reverse"].detach().numpy().astype(np.float32)
        for part in ("intra_fc", "inter_fc"):
            p = f"dpgrnn.{i}.{part}"
            out[pre + f"{part}.weight"] = sd[p + ".weight"].detach().numpy().astype(np.float32)
            out[pre + f"{part}.bias"] = sd[p + ".bias"].detach().numpy().astype(np.float32)
        for part in ("intra_ln", "inter_ln"):
            p = f"dpgrnn.{i}.{part}"
            out[pre + f"{part}.weight"] = sd[p + ".weight"].detach().numpy().astype(np.float32)
            out[pre + f"{part}.bias"] = sd[p + ".bias"].detach().numpy().astype(np.float32)
    np.savez(out_path, **out)
    return len(out)
