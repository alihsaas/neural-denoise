import json
from pathlib import Path

import numpy as np

from neural_denoise.config import SPLIT_SEED, VALIDATION_INSTANCES


def make_split(
    names: list[str], validation_count: int = VALIDATION_INSTANCES, seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    if validation_count >= len(names):
        raise ValueError("Validation would leave no training images")
    order = np.random.default_rng(seed).permutation(sorted(names))
    validation = sorted(str(value) for value in order[:validation_count])
    train = sorted(str(value) for value in order[validation_count:])
    return {"train": train, "validation": validation}


def save_split(split: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SPLIT_SEED,
        "counts": {name: len(values) for name, values in split.items()},
        **split,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    return {name: payload[name] for name in ("train", "validation")}
