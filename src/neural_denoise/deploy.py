import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from neural_denoise.config import BLOCK_SIZE


def as_input(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    return np.ascontiguousarray(values)


def as_images(values: np.ndarray) -> np.ndarray:
    pixels = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    return pixels.transpose(0, 2, 3, 1)


def export_onnx(model: torch.nn.Module, path: Path, size: int = BLOCK_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["noisy"],
        output_names=["clean"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
    )


class ArrayReader(CalibrationDataReader):
    def __init__(self, batches: list[np.ndarray]) -> None:
        self.data = batches
        self.set_range(0, len(batches))

    def __len__(self) -> int:
        return len(self.data)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.batches = iter(self.data[start_index:end_index])

    def get_next(self) -> dict | None:
        batch = next(self.batches, None)
        return None if batch is None else {"noisy": batch}


def quantize_int8(
    source: Path,
    target: Path,
    calibration: list[np.ndarray],
    method: CalibrationMethod,
    chunk: int = 16,
) -> None:
    prepared = target.with_name(target.stem + "_prepared.onnx")
    quant_pre_process(str(source), str(prepared))
    quantize_static(
        str(prepared),
        str(target),
        ArrayReader(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        extra_options={"CalibStridedMinMax": chunk},
    )
    prepared.unlink()


class OnnxDenoiser:
    def __init__(self, path: Path, threads: int = 0) -> None:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    def raw(self, inputs: np.ndarray) -> np.ndarray:
        return self.session.run(["clean"], {"noisy": inputs})[0]

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return as_images(self.raw(as_input(image[None])))[0]

    def denoise_batch(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self.denoise(image) for image in images])


def benchmark(
    run: Callable[[np.ndarray], object], images: list[np.ndarray], warmup: int = 5
) -> dict[str, float]:
    for image in images[:warmup]:
        run(image)
    times = []
    for image in images:
        start = time.perf_counter()
        run(image)
        times.append(1000.0 * (time.perf_counter() - start))
    return {
        "ms_median": float(np.median(times)),
        "ms_p90": float(np.percentile(times, 90)),
    }


def file_megabytes(path: Path) -> float:
    return path.stat().st_size / 1e6
