import cv2
import numpy as np


def smooth_image(rng: np.random.Generator, shape: tuple[int, int] = (96, 96)) -> np.ndarray:
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.zeros((*shape, 3), dtype=np.float64)
    image[..., 0] = 128 + 60 * np.sin(columns / 9.0)
    image[..., 1] = 100 + 0.8 * rows
    image[..., 2] = 160 - 50 * np.cos((rows + columns) / 14.0)
    for _ in range(3):
        centre = (int(rng.integers(10, shape[1] - 10)), int(rng.integers(10, shape[0] - 10)))
        colour = tuple(float(value) for value in rng.integers(30, 220, 3))
        cv2.circle(image, centre, int(rng.integers(5, 15)), colour, -1)
    return np.clip(image, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
