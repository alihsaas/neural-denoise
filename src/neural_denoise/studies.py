import numpy as np

from neural_denoise.cache import load_crops
from neural_denoise.config import VALIDATION_CROPS_PER_IMAGE
from neural_denoise.datasets import load_blocks


def validation_groups(
    per_image: int = VALIDATION_CROPS_PER_IMAGE,
) -> tuple[np.ndarray, np.ndarray]:
    crops = np.asarray(load_crops("validation"))
    shape = (len(crops) // per_image, per_image, *crops.shape[2:])
    return crops[:, 0].reshape(shape), crops[:, 1].reshape(shape)


def load_test_blocks() -> tuple[np.ndarray, np.ndarray]:
    return load_blocks("test_noisy"), load_blocks("test_clean")


def subset_groups(
    noisy: np.ndarray, clean: np.ndarray, count: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    flat_noisy = noisy.reshape(-1, *noisy.shape[2:])
    flat_clean = clean.reshape(-1, *clean.shape[2:])
    picks = np.sort(np.random.default_rng(seed).choice(len(flat_noisy), count, replace=False))
    return flat_noisy[picks][None], flat_clean[picks][None]
