from pathlib import Path

import numpy as np
import pytest

from neural_denoise.comparison import collect_scores, gain_rows
from neural_denoise.evaluation import save_scores


def write_scores(path: Path, psnr_values: np.ndarray) -> None:
    save_scores(psnr_values, np.zeros_like(psnr_values), path)


def test_gains_are_paired_against_bm3d(tmp_path: Path, rng: np.random.Generator) -> None:
    baseline = rng.normal(30.0, 2.0, (6, 4))
    better = baseline + 1.5 + rng.normal(0.0, 0.1, baseline.shape)
    write_scores(tmp_path / "classical" / "scores_bm3d.npz", baseline)
    write_scores(tmp_path / "classical" / "scores_nlm.npz", baseline - 2.0)
    write_scores(tmp_path / "learned" / "scores_dncnn_seed0.npz", better)
    scores = collect_scores(tmp_path)
    assert {method: stage for method, (stage, _) in scores.items()} == {
        "bm3d": "classical",
        "nlm": "classical",
        "dncnn_seed0": "learned",
    }
    rows = {row["method"]: row for row in gain_rows(scores)}
    assert set(rows) == {"nlm", "dncnn_seed0"}
    assert rows["nlm"]["gain_over_bm3d"] == pytest.approx(-2.0)
    learned = rows["dncnn_seed0"]
    assert learned["gain_over_bm3d"] == pytest.approx(float((better - baseline).mean()))
    assert learned["gain_ci_low"] <= learned["gain_over_bm3d"] <= learned["gain_ci_high"]


def test_scores_need_unique_names_and_a_baseline(tmp_path: Path) -> None:
    values = np.full((2, 2), 30.0)
    write_scores(tmp_path / "learned" / "scores_dncnn_seed0.npz", values)
    with pytest.raises(ValueError):
        gain_rows(collect_scores(tmp_path))
    write_scores(tmp_path / "deploy" / "scores_dncnn_seed0.npz", values)
    with pytest.raises(ValueError):
        collect_scores(tmp_path)
