#!/usr/bin/env python3
"""Generate Pulsar2 8.0 configs for the three LavaSR AX650 submodels.

Expected repository layout (the repo root is mounted at /workspace):

    build/onnx/denoiser_encoder.sim.onnx
    build/onnx/denoiser_decoder.sim.onnx
    build/onnx/enhancer_core.sim.onnx
    build/calib/denoiser.tar.gz
    build/calib/decoder/{feat,en0,en1,en2,en3,en4}.tar.gz
    build/calib/bwe.tar.gz
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model_convert" / "configs"


def base_config(input_onnx, output_dir, input_shapes, calibs, size):
    return {
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU3",
        "input": f"/workspace/build/onnx/{input_onnx}",
        "output_dir": f"/workspace/build/compile/{output_dir}",
        "output_name": "model.axmodel",
        "work_dir": f"/workspace/build/compile/{output_dir}_work",
        "input_shapes": input_shapes,
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": False,
            "model_check": True,
        },
        "quant": {
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "highest_mix_precision": False,
            "layer_configs": [{
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "U16",
                "weight_data_type": "S8",
                "output_data_type": "U16",
            }],
            "input_configs": [{
                "tensor_name": name,
                "calibration_dataset": f"/workspace/build/calib/{path}",
                "calibration_format": "Numpy",
                "calibration_size": size,
                "calibration_mean": [],
                "calibration_std": [],
            } for name, path in calibs],
        },
        "input_processors": [],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {
        "den_encoder": base_config(
            "denoiser_encoder.sim.onnx", "den_encoder", "erb_feat:1x1x301x129",
            [("erb_feat", "denoiser.tar.gz")], 256),
        "den_decoder": base_config(
            "denoiser_decoder.sim.onnx", "den_decoder",
            "feat:1x16x301x33,en0:1x12x301x65,en1:1x24x301x33,"
            "en2:1x24x301x33,en3:1x32x301x33,en4:1x16x301x33",
            [("feat", "decoder/feat.tar.gz"), ("en0", "decoder/en0.tar.gz"),
             ("en1", "decoder/en1.tar.gz"), ("en2", "decoder/en2.tar.gz"),
             ("en3", "decoder/en3.tar.gz"), ("en4", "decoder/en4.tar.gz")], 64),
        "enhancer": base_config(
            "enhancer_core.sim.onnx", "enhancer", "mel:1x80x450",
            [("mel", "bwe.tar.gz")], 256),
    }
    for name, cfg in configs.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
