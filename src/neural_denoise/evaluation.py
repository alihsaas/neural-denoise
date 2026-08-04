import csv
import json
import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from neural_denoise.config import TEST_LOG_PATH
from neural_denoise.metrics import cluster_bootstrap_interval, psnr, ssim

BatchDenoiser = Callable[[np.ndarray], np.ndarray]
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def score_images(clean: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psnr_values = np.array(
        [psnr(truth, guess) for truth, guess in zip(clean, estimate, strict=True)]
    )
    ssim_values = np.array(
        [ssim(truth, guess) for truth, guess in zip(clean, estimate, strict=True)]
    )
    return psnr_values, ssim_values


def denoise_each(denoise: Callable[[np.ndarray], np.ndarray], images: np.ndarray) -> np.ndarray:
    return np.stack([denoise(image) for image in images])


def score_group(
    denoise: Callable[[np.ndarray], np.ndarray], noisy: np.ndarray, clean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return score_images(clean, denoise_each(denoise, noisy))


def evaluate_blocks(
    denoise_batch: BatchDenoiser, noisy: np.ndarray, clean: np.ndarray, label: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    psnr_rows, ssim_rows = [], []
    for index in range(len(noisy)):
        estimate = denoise_batch(np.asarray(noisy[index]))
        psnr_values, ssim_values = score_images(np.asarray(clean[index]), estimate)
        psnr_rows.append(psnr_values)
        ssim_rows.append(ssim_values)
        if label and (index + 1) % 10 == 0:
            print(f"{label}: {index + 1}/{len(noisy)} groups")
    return np.stack(psnr_rows), np.stack(ssim_rows)


@contextmanager
def single_thread_children() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def evaluate_blocks_parallel(
    denoise: Callable[[np.ndarray], np.ndarray],
    noisy: np.ndarray,
    clean: np.ndarray,
    workers: int,
    label: str = "",
    chunk: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if workers <= 1:
        return evaluate_blocks(lambda images: denoise_each(denoise, images), noisy, clean, label)
    groups, per_group = noisy.shape[:2]
    psnr_values = np.zeros((groups, per_group))
    ssim_values = np.zeros((groups, per_group))
    tasks = [(group, start) for group in range(groups) for start in range(0, per_group, chunk)]
    pending: deque = deque()
    done = 0
    with single_thread_children(), ProcessPoolExecutor(max_workers=workers) as pool:
        for position in range(len(tasks) + 1):
            if position < len(tasks):
                group, start = tasks[position]
                stop = start + chunk
                future = pool.submit(
                    score_group,
                    denoise,
                    np.asarray(noisy[group, start:stop]),
                    np.asarray(clean[group, start:stop]),
                )
                pending.append((group, start, future))
            while pending and (len(pending) >= 4 * workers or position == len(tasks)):
                group, start, future = pending.popleft()
                chunk_psnr, chunk_ssim = future.result()
                psnr_values[group, start : start + len(chunk_psnr)] = chunk_psnr
                ssim_values[group, start : start + len(chunk_ssim)] = chunk_ssim
                done += 1
                if label and done % 40 == 0:
                    print(f"{label}: {done}/{len(tasks)} chunks")
    return psnr_values, ssim_values


def summarize(method: str, psnr_values: np.ndarray, ssim_values: np.ndarray) -> dict:
    psnr_low, psnr_high = cluster_bootstrap_interval(psnr_values)
    ssim_low, ssim_high = cluster_bootstrap_interval(ssim_values)
    return {
        "method": method,
        "images": int(psnr_values.shape[0]),
        "blocks": int(psnr_values.size),
        "psnr": float(psnr_values.mean()),
        "psnr_ci_low": psnr_low,
        "psnr_ci_high": psnr_high,
        "ssim": float(ssim_values.mean()),
        "ssim_ci_low": ssim_low,
        "ssim_ci_high": ssim_high,
    }


def claim_test_pass(method: str, path: Path = TEST_LOG_PATH, force: bool = False) -> None:
    passes = json.loads(path.read_text(encoding="ascii")) if path.exists() else {}
    if method in passes and not force:
        raise SystemExit(
            f"{method} already used the test set ({passes[method]} pass). Use --force to rerun."
        )
    passes[method] = passes.get(method, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passes, indent=2), encoding="ascii")


def save_scores(psnr_values: np.ndarray, ssim_values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, psnr=psnr_values, ssim=ssim_values)


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["psnr"], payload["ssim"]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.4f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def print_summary(rows: list[dict]) -> None:
    print(f"{'method':28s} {'PSNR':>7s} {'95% CI':>15s} {'SSIM':>7s} {'95% CI':>15s}")
    for row in rows:
        psnr_interval = f"{row['psnr_ci_low']:.2f}-{row['psnr_ci_high']:.2f}"
        ssim_interval = f"{row['ssim_ci_low']:.4f}-{row['ssim_ci_high']:.4f}"
        print(
            f"{row['method']:28s} {row['psnr']:7.2f} {psnr_interval:>15s}"
            f" {row['ssim']:7.4f} {ssim_interval:>15s}"
        )
