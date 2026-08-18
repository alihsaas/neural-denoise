from pathlib import Path

import numpy as np

from neural_denoise.evaluation import load_scores
from neural_denoise.metrics import paired_bootstrap_interval

STAGES = ("classical", "learned", "deploy")
BASELINE = "bm3d"


def collect_scores(
    results_dir: Path, stages: tuple[str, ...] = STAGES
) -> dict[str, tuple[str, np.ndarray]]:
    scores: dict[str, tuple[str, np.ndarray]] = {}
    for stage in stages:
        for path in sorted((results_dir / stage).glob("scores_*.npz")):
            method = path.stem.removeprefix("scores_")
            if method in scores:
                raise ValueError(f"{method} has scores in both {scores[method][0]} and {stage}")
            psnr_values, _ = load_scores(path)
            scores[method] = (stage, psnr_values)
    return scores


def gain_rows(scores: dict[str, tuple[str, np.ndarray]], baseline: str = BASELINE) -> list[dict]:
    if baseline not in scores:
        raise ValueError(f"No scores for the baseline {baseline}")
    reference = scores[baseline][1]
    rows = []
    for method, (stage, psnr_values) in scores.items():
        if method == baseline:
            continue
        gain, low, high = paired_bootstrap_interval(psnr_values, reference)
        rows.append(
            {
                "stage": stage,
                "method": method,
                f"gain_over_{baseline}": gain,
                "gain_ci_low": low,
                "gain_ci_high": high,
            }
        )
    return rows
