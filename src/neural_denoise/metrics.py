import cv2
import numpy as np

from neural_denoise.config import BOOTSTRAP_SAMPLES

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def psnr(clean: np.ndarray, estimate: np.ndarray, data_range: float = 255.0) -> float:
    error = np.mean((clean.astype(np.float64) - estimate.astype(np.float64)) ** 2)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / error))


def gaussian_filter(values: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(
        values, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA, borderType=cv2.BORDER_REFLECT
    )


def ssim_channel(clean: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    mean_x, mean_y = gaussian_filter(clean), gaussian_filter(estimate)
    var_x = gaussian_filter(clean * clean) - mean_x**2
    var_y = gaussian_filter(estimate * estimate) - mean_y**2
    covariance = gaussian_filter(clean * estimate) - mean_x * mean_y
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    margin = SSIM_WINDOW // 2
    ratio = numerator / denominator
    return float(ratio[margin:-margin, margin:-margin].mean())


def ssim(clean: np.ndarray, estimate: np.ndarray, data_range: float = 255.0) -> float:
    clean = clean.astype(np.float64)
    estimate = estimate.astype(np.float64)
    if clean.ndim == 2:
        return ssim_channel(clean, estimate, data_range)
    return float(
        np.mean(
            [
                ssim_channel(clean[..., channel], estimate[..., channel], data_range)
                for channel in range(clean.shape[-1])
            ]
        )
    )


def cluster_bootstrap_interval(
    values: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    image_means = values.reshape(len(values), -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    count = len(image_means)
    draws = image_means[rng.integers(0, count, (samples, count))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_bootstrap_interval(
    first: np.ndarray, second: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    low, high = cluster_bootstrap_interval(difference, samples, seed)
    return float(difference.mean()), low, high
