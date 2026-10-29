"""Small wav I/O helpers (soundfile if present, wave fallback)."""
import wave

import numpy as np


def read_wav(path):
    try:
        import soundfile as sf

        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), int(sr)
    except Exception:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / 8388608.0
        else:
            raise RuntimeError(f"unsupported wav sample width: {sw}")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(path, x, int(sr), subtype="PCM_16")
        return
    except Exception:
        pcm = (x * 32767.0).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm.tobytes())
