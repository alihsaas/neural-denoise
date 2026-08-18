import argparse
import json
import os
import platform
import time

import cv2
import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import CalibrationMethod

from neural_denoise.cache import load_crops
from neural_denoise.classical import make_denoiser
from neural_denoise.config import (
    CALIBRATION_CROPS,
    CHECKPOINT_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SPEED_BLOCKS,
)
from neural_denoise.deploy import (
    OnnxDenoiser,
    as_input,
    benchmark,
    export_onnx,
    file_megabytes,
    quantize_int8,
)
from neural_denoise.evaluation import (
    claim_test_pass,
    evaluate_blocks,
    print_summary,
    save_scores,
    summarize,
    write_csv,
    write_json,
)
from neural_denoise.neural import denoise_batch, load_checkpoint
from neural_denoise.studies import load_test_blocks, validation_groups

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
}
BM3D_SPEED_BLOCKS = 16


def best_run(runs: list[dict], model: str) -> str:
    return max(
        (run for run in runs if run["model"] == model), key=lambda run: run["best_validation_psnr"]
    )["name"]


def gpu_timer(model: torch.nn.Module, device: torch.device):
    def run(image: np.ndarray) -> None:
        denoise_batch(model, image[None], device)
        torch.cuda.synchronize()

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = RESULTS_DIR / "deploy"
    runs = json.loads((RESULTS_DIR / "learned" / "runs.json").read_text(encoding="ascii"))["runs"]
    chosen = best_run(runs, "dncnn")
    print(f"Exporting {chosen}, the DnCNN seed with the best validation PSNR")
    model, _ = load_checkpoint(CHECKPOINT_DIR / f"{chosen}.pt", torch.device("cpu"))
    fp32_path = MODELS_DIR / "dncnn_fp32.onnx"
    export_onnx(model, fp32_path)

    calibration_crops = np.asarray(load_crops("calibration")[:CALIBRATION_CROPS, 0])
    calibration = [as_input(crop[None]) for crop in calibration_crops]
    noisy, clean = validation_groups()

    calibration_rows = []
    for name, method in CALIBRATION_METHODS.items():
        path = MODELS_DIR / f"dncnn_int8_{name}.onnx"
        started = time.perf_counter()
        quantize_int8(fp32_path, path, calibration, method)
        denoiser = OnnxDenoiser(path)
        psnr_values, _ = evaluate_blocks(denoiser.denoise_batch, noisy, clean)
        calibration_rows.append(
            {
                "calibration": name,
                "validation_psnr": float(psnr_values.mean()),
                "seconds": time.perf_counter() - started,
            }
        )
        print(calibration_rows[-1])
    write_csv(calibration_rows, output / "calibration.csv")
    best_calibration = max(calibration_rows, key=lambda row: row["validation_psnr"])["calibration"]
    int8_path = MODELS_DIR / f"dncnn_int8_{best_calibration}.onnx"

    test_noisy, test_clean = load_test_blocks()
    rows = []
    for label, path in (("onnx_fp32", fp32_path), ("onnx_int8", int8_path)):
        claim_test_pass(f"deploy/{label}", force=arguments.force)
        denoiser = OnnxDenoiser(path)
        psnr_values, ssim_values = evaluate_blocks(
            denoiser.denoise_batch, test_noisy, test_clean, label
        )
        save_scores(psnr_values, ssim_values, output / f"scores_{label}.npz")
        row = summarize(label, psnr_values, ssim_values)
        row["megabytes"] = file_megabytes(path)
        rows.append(row)
    write_csv(rows, output / "summary.csv")
    print_summary(rows)

    flat = np.asarray(test_noisy).reshape(-1, *test_noisy.shape[2:])
    picks = np.sort(np.random.default_rng(0).choice(len(flat), SPEED_BLOCKS, replace=False))
    images = [np.ascontiguousarray(flat[index]) for index in picks]
    del flat
    classical = json.loads((RESULTS_DIR / "classical" / "chosen.json").read_text(encoding="ascii"))
    all_threads = os.cpu_count() or 1
    speed_rows = []
    for threads in (1, all_threads):
        cv2.setNumThreads(threads)
        torch.set_num_threads(threads)
        timed = {
            "nlm": (make_denoiser("nlm", classical["nlm"]), images),
            "bm3d": (make_denoiser("bm3d", classical["bm3d"]), images[:BM3D_SPEED_BLOCKS]),
            "onnx_fp32": (OnnxDenoiser(fp32_path, threads).denoise, images),
            "onnx_int8": (OnnxDenoiser(int8_path, threads).denoise, images),
        }
        for name, (run, items) in timed.items():
            timing = benchmark(run, items)
            speed_rows.append({"method": name, "device": "cpu", "threads": threads, **timing})
            print(speed_rows[-1])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for model_name in ("dncnn", "dncnn_small"):
            if not any(run["model"] == model_name for run in runs):
                continue
            gpu_model, _ = load_checkpoint(
                CHECKPOINT_DIR / f"{best_run(runs, model_name)}.pt", device
            )
            timing = benchmark(gpu_timer(gpu_model, device), images)
            speed_rows.append(
                {"method": f"torch_{model_name}", "device": "cuda", "threads": "", **timing}
            )
            print(speed_rows[-1])
    write_csv(speed_rows, output / "speed.csv")
    write_json(
        {
            "exported": chosen,
            "calibration": best_calibration,
            "cpu": platform.processor(),
            "logical_cpus": all_threads,
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        output / "environment.json",
    )


if __name__ == "__main__":
    main()
