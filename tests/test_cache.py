from pathlib import Path

import numpy as np

from neural_denoise.cache import build_crop_cache, crop_positions, load_crops
from neural_denoise.datasets import Pair, parse_instance


def test_crop_positions_stay_inside_image(rng: np.random.Generator) -> None:
    positions = crop_positions(50, 70, 32, 200, rng)
    assert positions[:, 0].min() >= 0 and positions[:, 0].max() <= 18
    assert positions[:, 1].min() >= 0 and positions[:, 1].max() <= 38


def test_cache_keeps_noisy_and_clean_aligned(tmp_path: Path, rng: np.random.Generator) -> None:
    images = {}
    pairs = []
    for index in range(3):
        name = f"{index + 1:04d}_001_S6_00100_00060_3200_L"
        rows, columns = np.mgrid[0:60, 0:80]
        clean = np.stack([rows, columns, np.full_like(rows, index)], axis=-1).astype(np.uint8)
        images[name] = (clean + 1, clean)
        pairs.append(Pair(parse_instance(name), Path("unused.zip"), name, name))
    path = build_crop_cache(
        "train", pairs, 16, 5, 0, tmp_path, load=lambda pair: images[pair.instance.name]
    )
    assert path.exists() and (tmp_path / "train.json").exists()
    crops = load_crops("train", tmp_path)
    assert crops.shape == (15, 2, 16, 16, 3)
    assert np.array_equal(crops[:, 0].astype(np.int16) - crops[:, 1], np.ones((15, 16, 16, 3)))
    assert np.all(crops[5:10, 1, ..., 2] == 1)
    rows = crops[:, 1, :, 0, 0].astype(np.int16)
    assert np.all(np.diff(rows, axis=1) == 1)
