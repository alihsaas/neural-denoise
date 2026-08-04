import math

import numpy as np
from helpers import add_noise, smooth_image

from neural_denoise.metrics import cluster_bootstrap_interval, paired_bootstrap_interval, psnr, ssim


def test_psnr_of_unit_error_and_identical_images() -> None:
    clean = np.full((16, 16, 3), 100, dtype=np.uint8)
    assert math.isclose(psnr(clean, clean + 1), 10 * math.log10(255.0**2), rel_tol=1e-12)
    assert psnr(clean, clean) == float("inf")


def test_psnr_does_not_wrap_uint8() -> None:
    clean = np.zeros((8, 8), dtype=np.uint8)
    estimate = np.full((8, 8), 255, dtype=np.uint8)
    assert math.isclose(psnr(clean, estimate), 0.0, abs_tol=1e-12)


def test_ssim_is_one_for_identical_and_falls_with_noise(rng: np.random.Generator) -> None:
    clean = smooth_image(rng)
    assert math.isclose(ssim(clean, clean), 1.0, rel_tol=1e-9)
    light = ssim(clean, add_noise(clean, 5.0, rng))
    heavy = ssim(clean, add_noise(clean, 30.0, rng))
    assert 1.0 > light > heavy > 0.0


def test_ssim_is_symmetric(rng: np.random.Generator) -> None:
    clean = smooth_image(rng)
    noisy = add_noise(clean, 15.0, rng)
    assert math.isclose(ssim(clean, noisy), ssim(noisy, clean), rel_tol=1e-9)


def test_cluster_bootstrap_resamples_whole_images(rng: np.random.Generator) -> None:
    values = rng.normal(30.0, 1.0, (40, 32)) + rng.normal(0.0, 2.0, (40, 1))
    low, high = cluster_bootstrap_interval(values)
    assert low < values.mean() < high
    image_means = values.mean(axis=1)
    standard_error = image_means.std(ddof=1) / math.sqrt(len(image_means))
    assert math.isclose(high - low, 2 * 1.96 * standard_error, rel_tol=0.25)


def test_paired_interval_finds_a_constant_gain(rng: np.random.Generator) -> None:
    base = rng.normal(30.0, 3.0, (40, 32))
    gain, low, high = paired_bootstrap_interval(base + 0.5, base)
    assert math.isclose(gain, 0.5, rel_tol=1e-9)
    assert math.isclose(low, 0.5, rel_tol=1e-9) and math.isclose(high, 0.5, rel_tol=1e-9)
