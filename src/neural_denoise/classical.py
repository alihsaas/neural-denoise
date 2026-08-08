from collections.abc import Callable
from functools import partial

import bm3d
import cv2
import numpy as np

from neural_denoise.config import NLM_SEARCH, NLM_TEMPLATE

Denoiser = Callable[[np.ndarray], np.ndarray]


def identity(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8).copy()


def nlm_denoise(
    image: np.ndarray, h: float, template: int = NLM_TEMPLATE, search: int = NLM_SEARCH
) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    result = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, template, search)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def bm3d_denoise(image: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64) / 255.0
    result = bm3d.bm3d_rgb(values, sigma / 255.0)
    return np.clip(np.rint(np.asarray(result) * 255.0), 0, 255).astype(np.uint8)


def make_denoiser(method: str, parameter: float | None = None) -> Denoiser:
    if method == "noisy":
        return identity
    if method == "nlm":
        return partial(nlm_denoise, h=float(parameter))
    if method == "bm3d":
        return partial(bm3d_denoise, sigma=float(parameter))
    raise ValueError(f"Unknown classical method: {method}")
