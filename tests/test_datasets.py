import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.io import savemat

from neural_denoise.datasets import (
    convert_blocks,
    find_pairs,
    load_blocks,
    load_pair,
    missing_files,
    parse_instance,
    refresh_blocks,
)
from neural_denoise.splits import load_split, make_split, save_split

NAMES = [
    "0001_001_S6_00100_00060_3200_L",
    "0002_001_S6_00100_00020_3200_N",
    "0003_002_GP_00400_00100_4400_L",
    "0004_003_IP_01600_01000_5500_H",
]


def write_sidd_zip(archive: Path, rng: np.random.Generator) -> None:
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("SIDD_Small_sRGB_Only/ReadMe.txt", "readme")
        for name in reversed(NAMES):
            folder = f"SIDD_Small_sRGB_Only/Data/{name}/"
            clean = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
            noisy = np.clip(clean.astype(np.int16) + 3, 0, 255).astype(np.uint8)
            for file_name, image in (("GT_SRGB_010.PNG", clean), ("NOISY_SRGB_010.PNG", noisy)):
                encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1]
                bundle.writestr(folder + file_name, encoded.tobytes())
            bundle.writestr(folder + "GT_RAW_010.MAT", b"raw")


def test_parse_instance_reads_every_field() -> None:
    instance = parse_instance("0003_002_GP_00400_00100_4400_L")
    assert instance.scene_instance == 3 and instance.scene == 2 and instance.phone == "GP"
    assert instance.iso == 400 and instance.shutter == 100 and instance.illuminant == 4400
    assert instance.brightness == "L"
    with pytest.raises(ValueError):
        parse_instance("not_a_scene")


def test_find_pairs_reads_aligned_rgb_from_zip(tmp_path: Path, rng: np.random.Generator) -> None:
    archive = tmp_path / "SIDD_Small_sRGB_Only.zip"
    write_sidd_zip(archive, rng)
    pairs = find_pairs(archive)
    assert [pair.instance.name for pair in pairs] == NAMES
    assert pairs[0].noisy.endswith("NOISY_SRGB_010.PNG")
    noisy, clean = load_pair(pairs[0])
    assert noisy.shape == clean.shape == (40, 60, 3)
    assert np.array_equal(noisy, np.clip(clean.astype(np.int16) + 3, 0, 255).astype(np.uint8))


def test_find_pairs_rejects_a_folder_without_clean_image(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"SIDD_Small_sRGB_Only/Data/{NAMES[0]}/NOISY_SRGB_010.PNG", b"x")
    with pytest.raises(ValueError):
        find_pairs(archive)


def test_convert_blocks_round_trips_mat(tmp_path: Path, rng: np.random.Generator) -> None:
    blocks = rng.integers(0, 256, (3, 2, 16, 16, 3), dtype=np.uint8)
    savemat(str(tmp_path / "blocks.mat"), {"ValidationNoisyBlocksSrgb": blocks})
    shape = convert_blocks(
        tmp_path / "blocks.mat", "ValidationNoisyBlocksSrgb", tmp_path / "test_noisy.npy"
    )
    assert shape == blocks.shape
    loaded = load_blocks("test_noisy", tmp_path)
    assert isinstance(loaded, np.memmap) and np.array_equal(loaded, blocks)


def test_refresh_blocks_replaces_stale_conversions(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    key = "ValidationNoisyBlocksSrgb"
    mat_path = tmp_path / "blocks.mat"
    target = tmp_path / "test_noisy.npy"
    stale = rng.integers(0, 256, (1, 1, 16, 16, 3), dtype=np.uint8)
    np.save(target, stale)
    first = rng.integers(0, 256, (3, 2, 16, 16, 3), dtype=np.uint8)
    savemat(str(mat_path), {key: first})
    refresh_blocks(mat_path, key, target)
    assert np.array_equal(np.load(target), first)

    np.save(target, stale)
    refresh_blocks(mat_path, key, target)
    assert np.array_equal(np.load(target), stale)

    second = rng.integers(0, 256, (2, 2, 16, 16, 3), dtype=np.uint8)
    savemat(str(mat_path), {key: second})
    refresh_blocks(mat_path, key, target)
    assert np.array_equal(np.load(target), second)


def test_missing_files_lists_every_needed_file(tmp_path: Path) -> None:
    assert len(missing_files(tmp_path)) == 3
    (tmp_path / "ValidationGtBlocksSrgb.mat").write_bytes(b"")
    assert "ValidationGtBlocksSrgb.mat" not in missing_files(tmp_path)


def test_split_is_disjoint_deterministic_and_saved(tmp_path: Path) -> None:
    names = [f"{index:04d}_001_S6_00100_00060_3200_L" for index in range(1, 41)]
    split = make_split(names, 8, 2026)
    assert split == make_split(list(reversed(names)), 8, 2026)
    assert len(split["validation"]) == 8 and len(split["train"]) == 32
    assert not set(split["train"]) & set(split["validation"])
    save_split(split, tmp_path / "split.json")
    assert load_split(tmp_path / "split.json") == split
    with pytest.raises(ValueError):
        make_split(names[:3], 3)
