import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from helpers import add_noise, smooth_image

from neural_denoise.classical import nlm_denoise
from neural_denoise.evaluation import (
    claim_test_pass,
    denoise_each,
    evaluate_blocks,
    evaluate_blocks_parallel,
    load_scores,
    save_scores,
    single_thread_children,
    summarize,
)
from neural_denoise.metrics import psnr
from neural_denoise.studies import subset_groups


def make_groups(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean = np.stack([np.stack([smooth_image(rng, (32, 32)) for _ in range(3)]) for _ in range(4)])
    noisy = np.stack(
        [np.stack([add_noise(image, 12.0, rng) for image in group]) for group in clean]
    )
    return noisy, clean


def test_identity_scores_match_direct_psnr(rng: np.random.Generator) -> None:
    noisy, clean = make_groups(rng)
    psnr_values, ssim_values = evaluate_blocks(lambda images: images.copy(), noisy, clean)
    assert psnr_values.shape == ssim_values.shape == (4, 3)
    assert np.isclose(psnr_values[2, 1], psnr(clean[2, 1], noisy[2, 1]))
    row = summarize("noisy", psnr_values, ssim_values)
    assert row["blocks"] == 12 and row["images"] == 4
    assert row["psnr_ci_low"] <= row["psnr"] <= row["psnr_ci_high"]


def test_parallel_matches_serial(rng: np.random.Generator) -> None:
    noisy, clean = make_groups(rng)
    denoise = partial(nlm_denoise, h=8.0)
    serial = evaluate_blocks(lambda images: denoise_each(denoise, images), noisy, clean)
    parallel = evaluate_blocks_parallel(denoise, noisy, clean, workers=2, chunk=2)
    assert np.allclose(serial[0], parallel[0]) and np.allclose(serial[1], parallel[1])


def test_subset_groups_picks_distinct_blocks(rng: np.random.Generator) -> None:
    noisy, clean = make_groups(rng)
    small_noisy, small_clean = subset_groups(noisy, clean, 5)
    assert small_noisy.shape == (1, 5, 32, 32, 3) and small_clean.shape == small_noisy.shape
    flat = noisy.reshape(-1, 32, 32, 3)
    assert all(any(np.array_equal(item, block) for block in flat) for item in small_noisy[0])


def test_test_pass_is_claimed_once(tmp_path: Path) -> None:
    log = tmp_path / "passes.json"
    claim_test_pass("classical/nlm", log)
    with pytest.raises(SystemExit):
        claim_test_pass("classical/nlm", log)
    claim_test_pass("classical/nlm", log, force=True)
    claim_test_pass("classical/bm3d", log)


def test_scores_round_trip(tmp_path: Path, rng: np.random.Generator) -> None:
    values = rng.random((4, 3))
    save_scores(values, values / 2, tmp_path / "scores.npz")
    loaded_psnr, loaded_ssim = load_scores(tmp_path / "scores.npz")
    assert np.array_equal(loaded_psnr, values) and np.array_equal(loaded_ssim, values / 2)


def test_children_get_single_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with single_thread_children():
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1" and os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8" and "OMP_NUM_THREADS" not in os.environ
