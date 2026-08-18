from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationMethod

from neural_denoise.deploy import (
    OnnxDenoiser,
    as_images,
    as_input,
    benchmark,
    export_onnx,
    quantize_int8,
)
from neural_denoise.neural import DnCNN


def test_onnx_matches_torch_and_int8_runs(tmp_path: Path, rng: np.random.Generator) -> None:
    torch.manual_seed(0)
    model = DnCNN(depth=5, width=16).eval()
    fp32 = tmp_path / "model.onnx"
    export_onnx(model, fp32, size=32)
    images = rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
    inputs = as_input(images[:1])
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()
    denoiser = OnnxDenoiser(fp32, threads=1)
    assert np.allclose(denoiser.raw(inputs), expected, atol=1e-4)
    assert np.array_equal(as_images(expected)[0], denoiser.denoise(images[0]))
    calibration = [as_input(images[index : index + 1]) for index in range(6)]
    int8 = tmp_path / "model_int8.onnx"
    quantize_int8(fp32, int8, calibration, CalibrationMethod.MinMax, chunk=3)
    quantized = OnnxDenoiser(int8, threads=1)
    result = quantized.denoise_batch(images)
    assert result.shape == images.shape and result.dtype == np.uint8
    timing = benchmark(quantized.denoise, list(images), warmup=2)
    assert timing["ms_median"] > 0
