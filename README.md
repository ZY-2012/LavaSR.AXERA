# LavaSR.AXERA

[LavaSR v2](https://github.com/ysharma3501/LavaSR) 的 AX650（NPU3）量化与板端部署工程。

LavaSR 是一个轻量语音超分 / 带宽扩展模型：输入 8–48 kHz 低质量语音，输出
48 kHz 清晰语音。本工程把其中的 **UL-UNAS 降噪器**和 **Vocos BWE 增强器**
（`enhancer_v2`）拆成 3 个 axmodel，STFT / mel / iSTFT / DPGRNN / LR merge 留在
主机 CPU，提供 Python 与 C++ 两套板端 SDK。

- 量化模型与预编译可执行文件：[HuggingFace: HY-2012/LavaSR.AXERA](https://huggingface.co/HY-2012/LavaSR.AXERA)
- 上游模型：[ysharma3501/LavaSR](https://github.com/ysharma3501/LavaSR) /
  [YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)

## 结果

板端 AX650N，Pulsar2 8.0，NPU3，不跑仿真：

| 模型 | 大小 | 单 chunk NPU 延迟 |
|---|---:|---:|
| `den_encoder.axmodel` | 11.32 MB | ~17.7 ms / 4.8 s |
| `den_decoder.axmodel` | 11.80 MB | ~16.4 ms / 4.8 s |
| `enhancer.axmodel` | 15.07 MB | ~5.7 ms / 4.8 s |
| 合计 | 38.2 MB | |

| 链路 | 17.75 s 音频耗时 | RTF |
|---|---:|---:|
| Python：降噪 + BWE | 4.93 s | **0.278** |
| Python：仅 BWE | 1.82 s | **0.103** |
| C++：降噪 + BWE | 4.22 s | **0.238** |
| C++：仅 BWE | 2.52 s | **0.142** |

板端 Python SDK vs ONNX 参考：**cosine 0.99999，SNR 45.9 dB**（17.75 s
LibriSpeech）；ONNX 主机链路 vs 原始 PyTorch：**cosine 0.99995，SNR 39.6 dB**。
细节见 [`reports/accuracy.md`](reports/accuracy.md) 和
[`reports/performance.md`](reports/performance.md)。

## 试听：超分前后对比

以下音频均在 AX650 板端用本仓库模型生成（Python SDK）。

**1. 干净 16 kHz -> 48 kHz（仅 BWE）**

| 超分前（16 kHz 升采样到 48 kHz） | LavaSR 超分后（48 kHz） |
|---|---|
| <audio controls src="samples/demo_clean_before_48k.wav"></audio> | <audio controls src="samples/demo_clean_after_48k.wav"></audio> |
| [▶ 播放/下载 before](samples/demo_clean_before_48k.wav) | [▶ 播放/下载 after](samples/demo_clean_after_48k.wav) |

8–24 kHz 高频能量占比：**-44.8 dB -> -33.4 dB（+11.4 dB）**。

**2. 带噪 16 kHz -> 降噪 + 48 kHz（denoise + BWE）**

| 超分前（10 dB SNR 噪声） | LavaSR 降噪+超分后 |
|---|---|
| <audio controls src="samples/demo_noisy_before_48k.wav"></audio> | <audio controls src="samples/demo_noisy_after_48k.wav"></audio> |
| [▶ 播放/下载 before](samples/demo_noisy_before_48k.wav) | [▶ 播放/下载 after](samples/demo_noisy_after_48k.wav) |

低带（<8 kHz）相对干净参考的 SNR：**10.4 dB -> 18.4 dB（+8.0 dB）**。

原始 16 kHz 输入：`samples/demo_clean_16k.wav`、`samples/demo_noisy_16k.wav`。

## 架构

```text
16 kHz wav
  ├─ host:  UL-UNAS STFT / ERB 特征
  ├─ NPU :  den_encoder  (erb_feat -> feat + 5 个 skip)
  ├─ host:  DPGRNN       (2 个双路径 RNN 块，numpy/C++)
  ├─ NPU :  den_decoder  (feat + skips -> ERB mask)
  ├─ host:  mask / ISTFT / 16k->48k 重采样
  ├─ host:  Vocos mel 特征 (44.1 kHz 配置, 48 kHz 样本)
  ├─ NPU :  enhancer     (mel -> log-mag + phase)
  └─ host:  exp / cos / sin / ISTFT / Linkwitz-Riley 合并 -> 48 kHz
```

固定 4.8 s chunk：降噪器 25% overlap，BWE 50% overlap，Hann 交叉淡化。

### 量化注意事项：FA GRU

每个 `cTFA` 块里的频率注意力 GRU（输入 `(151, 33, 4)`/`(301, 33, 4)`，
bidirectional）如果直接导出为 ONNX `GRU`，Pulsar2 8.0 产物在板端会报
`Failed to run model`（同板 GTCRN 的小 GRU 正常）。本工程在导出时把它展开成
`MatMul/Add/Sigmoid/Tanh` 的 17/33 步循环，数值与原 GRU 一致（cosine 1.0），
Pulsar2 可以正常编译、板端可以正常推理。

DPGRNN 的两个块包含 batch 较大的 GRU，Pulsar2 同样无法可靠 lower，因此放在
主机 numpy / C++ 中执行（模型仅 ~60k 参数，耗时可忽略）。

## 目录

```
LavaSR.AXERA/
├── model_convert/          # ONNX 导出 / 校准 / Pulsar2 编译（GitHub only）
│   ├── export_onnx.py      # 含 FA GRU 展开、onnxsim
│   ├── make_calib.py       # 真实语音校准数据
│   ├── make_decoder_calib.py
│   ├── prepare_compile.py
│   └── compile.sh
├── python/
│   ├── lavasr_sdk/         # 纯 numpy + pyaxengine 运行时
│   └── demo.py
├── cpp/                    # C++ 运行时源码 + 嵌入式 assets
│   └── bin/lavasr_ax650    # AX650 预编译可执行文件
├── samples/                # 8k/16k/24k/48k + 噪声示例
├── reports/
└── release/                # HF 发布包（gitignore）
```

## 板端快速开始

推荐直接从 [HuggingFace 仓库](https://huggingface.co/HY-2012/LavaSR.AXERA)
下载 `models/`、`bin/`、`python/` 与 `samples/`，然后：

```bash
# C++（默认）
bash run_ax650.sh samples/libri_16k.wav output.wav

# Python
bash run_ax650.sh samples/libri_16k.wav output.wav --python

# 跳过 UL-UNAS 降噪，只跑 BWE
bash run_ax650.sh samples/libri_16k.wav output.wav --no-denoise
```

输出为 48 kHz 单声道 wav。输入支持 8–48 kHz；`--cutoff` 可调
Linkwitz-Riley 交叉频率（默认 `min(input_sr,16000)/2`）。

### Python API

```python
from lavasr_sdk import LavaSR, read_wav, write_wav

model = LavaSR(
    "models/den_encoder.axmodel",
    "models/den_decoder.axmodel",
    "models/enhancer.axmodel",
    model_config="models/model_config.json",
)
x, sr = read_wav("input.wav")
y, out_sr = model.enhance(x, sr, denoise=True)
write_wav("output.wav", y, out_sr)
```

依赖：`numpy`、`pyaxengine`（板端 `axengine`）、可选 `soundfile`。
不依赖 torch / onnxruntime。

### C++

```bash
./bin/lavasr_ax650 models/den_encoder.axmodel models/den_decoder.axmodel \
    models/enhancer.axmodel input.wav output.wav [--no-denoise] [--bench]
```

交叉编译：

```bash
bash cpp/download_toolchains.sh
TOOLCHAIN_ROOT=... BSP_MSP_DIR=... bash cpp/build_ax650.sh
```

## 模型转换（可复现）

```bash
# 1. 原始模型
git clone https://github.com/ysharma3501/LavaSR origin/LavaSR
hf download YatharthS/LavaSR --local-dir origin/LavaSR_HF

# 2. 导出静态 ONNX（自动展开 FA GRU + onnxsim）
python model_convert/export_onnx.py \
  --origin-dir origin/LavaSR --hf-dir origin/LavaSR_HF \
  --out-dir build/onnx --asset-dir python/lavasr_sdk/assets

# 3. 校准数据（LibriSpeech + AISHELL-1 等真实语音）
python model_convert/make_calib.py \
  --origin-dir origin/LavaSR --hf-dir origin/LavaSR_HF \
  --audio <wav-dir-or-files> --out-dir build/calib
python model_convert/make_decoder_calib.py

# 4. Pulsar2 8.0 编译
python model_convert/prepare_compile.py
PULSAR2_IMAGE=pulsar2:8.0 HASP_HOME=<license-dir> bash model_convert/compile.sh
```

Pulsar2 8.0 镜像：

```bash
hf download AXERA-TECH/Pulsar2 8.0/ax_pulsar2_8.0.tar.gz --local-dir .
docker load -i ax_pulsar2_8.0.tar.gz   # -> pulsar2:8.0
```

## 限制

- 当前只发布 AX650 / NPU3 版本；其他芯片需要重新编译。
- 降噪器 DPGRNN 在 CPU 上执行；BWE 主模型在 NPU 上。
- 4.8 s 分块会引入极小的边界差异；如需严格逐样本对齐，需要改为状态化流式实现。
- `enhancer_v2` 的 mel 配置是 44.1 kHz，但输入样本是 48 kHz（上游原样行为），
  本工程严格复现，不做重采样修正。

## 参考

- [LavaSR](https://github.com/ysharma3501/LavaSR)
- [Vocos](https://github.com/gemelo-ai/vocos)
- [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas)
- [Magnetar](https://github.com/AXERA-TECH/Magnetar)
- [Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2)

## License

代码遵循 Apache-2.0；上游 LavaSR 模型与代码为 Apache-2.0。KissFFT 为
BSD-3-Clause。
