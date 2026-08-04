import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from neural_denoise.config import CACHE_DIR
from neural_denoise.datasets import Pair, load_pair


def crop_positions(
    height: int, width: int, size: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    if height < size or width < size:
        raise ValueError(f"Image {height} x {width} is smaller than the crop {size}")
    rows = rng.integers(0, height - size + 1, count)
    columns = rng.integers(0, width - size + 1, count)
    return np.stack([rows, columns], axis=1)


def take_crops(image: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    return np.stack([image[row : row + size, column : column + size] for row, column in positions])


def build_crop_cache(
    name: str,
    pairs: Sequence[Pair],
    size: int,
    per_image: int,
    seed: int,
    cache_dir: Path = CACHE_DIR,
    load: Callable[[Pair], tuple[np.ndarray, np.ndarray]] = load_pair,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.npy"
    shape = (len(pairs) * per_image, 2, size, size, 3)
    crops = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    rng = np.random.default_rng(seed)
    for index, pair in enumerate(pairs):
        noisy, clean = load(pair)
        positions = crop_positions(*noisy.shape[:2], size, per_image, rng)
        start = index * per_image
        crops[start : start + per_image, 0] = take_crops(noisy, positions, size)
        crops[start : start + per_image, 1] = take_crops(clean, positions, size)
        del noisy, clean
    crops.flush()
    del crops
    details = {
        "crops": shape[0],
        "size": size,
        "per_image": per_image,
        "seed": seed,
        "instances": [pair.instance.name for pair in pairs],
    }
    (cache_dir / f"{name}.json").write_text(json.dumps(details, indent=2), encoding="ascii")
    return path


def crop_record(name: str, cache_dir: Path = CACHE_DIR) -> dict:
    return json.loads((cache_dir / f"{name}.json").read_text(encoding="ascii"))


def load_crops(name: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    return np.load(cache_dir / f"{name}.npy", mmap_mode="r")
