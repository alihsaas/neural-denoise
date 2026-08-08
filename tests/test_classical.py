import numpy as np
import pytest
from helpers import add_noise, smooth_image

from neural_denoise.classical import bm3d_denoise, identity, make_denoiser, nlm_denoise
from neural_denoise.metrics import psnr


def test_nlm_improves_psnr_and_keeps_shape(rng: np.random.Generator) -> None:
    clean = smooth_image(rng)
    noisy = add_noise(clean, 15.0, rng)
    result = nlm_denoise(noisy, 10.0)
    assert result.shape == clean.shape and result.dtype == np.uint8
    assert psnr(clean, result) > psnr(clean, noisy) + 3.0


def test_nlm_keeps_colour_order(rng: np.random.Generator) -> None:
    red = np.zeros((48, 48, 3), dtype=np.uint8)
    red[..., 0] = 200
    result = nlm_denoise(red, 5.0)
    assert result[..., 0].mean() > 190 and result[..., 2].mean() < 10


def test_bm3d_improves_psnr(rng: np.random.Generator) -> None:
    clean = smooth_image(rng, (64, 64))
    noisy = add_noise(clean, 20.0, rng)
    result = bm3d_denoise(noisy, 20.0)
    assert result.shape == clean.shape and result.dtype == np.uint8
    assert psnr(clean, result) > psnr(clean, noisy) + 3.0


def test_make_denoiser_builds_each_method(rng: np.random.Generator) -> None:
    image = smooth_image(rng, (48, 48))
    assert make_denoiser("noisy") is identity
    assert np.array_equal(make_denoiser("noisy")(image), image)
    assert make_denoiser("nlm", 5.0)(image).shape == image.shape
    with pytest.raises(ValueError):
        make_denoiser("median")
