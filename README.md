# LavaSR.AXERA

[LavaSR v2](https://github.com/ysharma3501/LavaSR) 的 AX650（NPU3）量化与板端部署工程。输入 8–48 kHz 语音，输出 48 kHz；提供 3 个 axmodel、Python/C++ SDK 和预编译可执行文件。

- HuggingFace: https://huggingface.co/HY-2012/LavaSR.AXERA
- GitHub: https://github.com/ZY-2012/LavaSR.AXERA
- 上游: https://github.com/ysharma3501/LavaSR / https://huggingface.co/YatharthS/LavaSR

## 目录

- 快速开始
- 试听结果
- 性能
- 架构
- 模型转换
- 数据来源
- License

## 快速开始（AX650 板端）

从 HuggingFace 下载 `models/`、`bin/`、`python/`、`samples/`，在包根目录运行：

```bash
# C++（默认）
bash run_ax650.sh samples/listen/8k/OSR_us_000_0010_input_8k.wav output.wav --no-denoise

# Python SDK
bash run_ax650.sh samples/listen/8k/OSR_us_000_0010_input_8k.wav output.wav --python --no-denoise
```

输入支持 8–48 kHz，输出固定 48 kHz。`--no-denoise` 仅做 BWE；去掉会同时跑 UL-UNAS 降噪。

## 试听结果

所有音频在 AX650 板端生成；每个样本为 **8k/16k 输入 -> LavaSR 48k 输出**（BWE only，`denoise=False`）。
LSD 口径：22.05 kHz、N_FFT 2048 / hop 512、低频 10% 能量对齐，越低越好；Open Speech/Samplelib 只有 8k 源，没有 48k 参考，LSD 为 —。

### 8 kHz 输入（Open Speech / Samplelib）

#### English（随机保留 4/25）

**OSR_us_000_0010**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_us_000_0010_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_us_000_0010_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_us_000_0010.png" width="260"> |

**OSR_us_000_0013**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_us_000_0013_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_us_000_0013_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_us_000_0013.png" width="260"> |

**OSR_us_000_0057**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_us_000_0057_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_us_000_0057_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_us_000_0057.png" width="260"> |

**OSR_us_000_0059**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_us_000_0059_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_us_000_0059_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_us_000_0059.png" width="260"> |

#### Chinese（4 条）

**OSR_cn_000_0072**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_cn_000_0072_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_cn_000_0072_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_cn_000_0072.png" width="260"> |

**OSR_cn_000_0073**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_cn_000_0073_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_cn_000_0073_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_cn_000_0073.png" width="260"> |

**OSR_cn_000_0074**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_cn_000_0074_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_cn_000_0074_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_cn_000_0074.png" width="260"> |

**OSR_cn_000_0075**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/OSR_cn_000_0075_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/OSR_cn_000_0075_output_48k.wav) | — | <img src="samples/listen/spectrograms/OSR_cn_000_0075.png" width="260"> |

#### Samplelib（1 分钟，频谱图为前 10 s）

**samplelib_speech_1m**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/samplelib_speech_1m_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/samplelib_speech_1m_output_48k.wav) | — | <img src="samples/listen/spectrograms/samplelib_speech_1m.png" width="260"> |

### 16 kHz 参照（人声）

同一批人声降采样到 16 kHz 后输入；8k 行为官方 8k 输入，16k 行为 16k 输入。

**speech_2**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/speech_2_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/speech_2_output_48k.wav) | 2.09 | <img src="samples/listen/spectrograms/speech_2.png" width="260"> |
| 16k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/16k/speech_2_input_16k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from16k/speech_2_output_48k.wav) | 0.74 | 同上 |

**speech_3**

| 版本 | 输入 | LavaSR 输出 | LSD | 频谱图 |
|---|---|---|---|---|
| 8k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/8k/speech_3_input_8k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from8k/speech_3_output_48k.wav) | 1.80 | <img src="samples/listen/spectrograms/speech_3.png" width="260"> |
| 16k | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/16k/speech_3_input_16k.wav) | [▶](https://huggingface.co/HY-2012/LavaSR.AXERA/resolve/main/samples/listen/48k_from16k/speech_3_output_48k.wav) | 1.49 | 同上 |

> LSD 只对 speech_2/3 有效（有 48k 参考）；Open Speech/Samplelib 无 48k 参考，故为 —。

## 性能

3 个 axmodel：`den_encoder` 11.32 MB、`den_decoder` 11.80 MB、`enhancer` 15.07 MB，共 38.2 MB。

| 链路 | Python RTF | C++ RTF |
|---|---:|---:|
| 降噪 + BWE | 0.278 | **0.061** |
| 仅 BWE | 0.103 | **0.039** |

## 架构

```text
16k wav -> 16k STFT/ERB -> NPU den_encoder -> host DPGRNN
        -> NPU den_decoder -> mask/ISTFT -> 16k->48k
        -> 48k mel -> NPU enhancer -> exp/cos/sin/ISTFT -> LR merge -> 48k
```

固定 4.8 s chunk：降噪 25% overlap，BWE 50% overlap。STFT / mel / iSTFT / DPGRNN / LR merge 在主机 CPU；三个神经网络核在 AX650 NPU。

## 模型转换

```bash
python model_convert/export_onnx.py --origin-dir origin/LavaSR --hf-dir origin/LavaSR_HF
python model_convert/make_calib.py --audio <wavs> --out-dir build/calib
python model_convert/make_decoder_calib.py
python model_convert/prepare_compile.py
PULSAR2_IMAGE=pulsar2:8.0 bash model_convert/compile.sh
```

详见 `model_convert/README.md`（Pulsar2 8.0）。

## 数据来源

- Open Speech Repository（English / Chinese 8k）：http://voiptroubleshooter.com/open_speech/index.html
- Samplelib speech 1 min：https://samplelib.com/mp3/sample-speech-1m.mp3
- 16 kHz 参照：公开人声样例经 48k->16k 降采样。

## License

Apache-2.0；KissFFT BSD-3-Clause。
