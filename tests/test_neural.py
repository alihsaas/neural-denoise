from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import add_noise, smooth_image

from neural_denoise.neural import (
    DnCNN,
    RunSettings,
    augment_pairs,
    check_run_matches,
    denoise_batch,
    dihedral,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    to_images,
    to_tensor,
    train_model,
    validation_psnr,
)


def synthetic_patches(rng: np.random.Generator, count: int = 16, size: int = 32) -> np.ndarray:
    clean = np.stack([smooth_image(rng, (size, size)) for _ in range(count)])
    noisy = np.stack([add_noise(image, 20.0, rng) for image in clean])
    return np.stack([noisy, clean], axis=1)


def test_dncnn_keeps_shape_and_subtracts_residual() -> None:
    torch.manual_seed(0)
    model = DnCNN(depth=5, width=16).eval()
    inputs = torch.rand(2, 3, 24, 40)
    with torch.no_grad():
        outputs = model(inputs)
        residual = model.body(inputs)
    assert outputs.shape == inputs.shape
    assert torch.allclose(outputs, inputs - residual)
    assert parameter_count(DnCNN(17, 64)) == 558403


def test_tensor_round_trip_is_exact(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, (3, 8, 10, 3), dtype=np.uint8)
    assert np.array_equal(to_images(to_tensor(images)), images)


def test_dihedral_gives_eight_distinct_views_and_keeps_pairs(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    views = {dihedral(image, code).tobytes() for code in range(8)}
    assert len(views) == 8
    pairs = synthetic_patches(rng, 6, 32)
    moved = augment_pairs(pairs, rng)
    for before, after in zip(pairs, moved, strict=True):
        code = next(
            code for code in range(8) if np.array_equal(dihedral(before[0], code), after[0])
        )
        assert np.array_equal(dihedral(before[1], code), after[1])


def test_short_training_improves_psnr_and_resumes(tmp_path: Path, rng: np.random.Generator) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = synthetic_patches(rng, 64, 32)
    validation = synthetic_patches(rng, 8, 32)
    settings = RunSettings("tiny", 0, depth=5, width=16, epochs=4, batch_size=8, learning_rate=3e-3)

    def evaluate(model: torch.nn.Module) -> float:
        return validation_psnr(model, validation[:, 0], validation[:, 1], device)

    calls = []

    def interrupted(model: torch.nn.Module) -> float:
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return evaluate(model)

    resume = tmp_path / "tiny_last.pt"
    with pytest.raises(KeyboardInterrupt):
        train_model(settings, patches, interrupted, device, resume)
    partial_history = torch.load(resume, weights_only=False)["history"]
    assert len(partial_history) == 2

    model, history, best_epoch = train_model(settings, patches, evaluate, device, resume)
    assert len(history) == 4 and 1 <= best_epoch <= 4
    assert history[:2] == partial_history
    noisy_psnr = validation_psnr(torch.nn.Identity(), validation[:, 0], validation[:, 1], device)
    assert max(entry["validation_psnr"] for entry in history) > noisy_psnr + 0.5

    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, settings, history, best_epoch)
    loaded, payload = load_checkpoint(path, device)
    assert payload["best_epoch"] == best_epoch
    original = denoise_batch(model, validation[:, 0], device)
    assert np.array_equal(denoise_batch(loaded, validation[:, 0], device), original)
    assert original.shape == validation[:, 0].shape and original.dtype == np.uint8


def test_checkpoints_refuse_other_settings_or_crops(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    device = torch.device("cpu")
    patches = synthetic_patches(rng, 16, 32)
    settings = RunSettings("tiny", 0, depth=3, width=8, epochs=1, batch_size=8)
    source = {"train": {"crops": 16}, "validation": {"crops": 4}}
    resume = tmp_path / "tiny_last.pt"
    model, history, best_epoch = train_model(
        settings, patches, lambda _: 0.0, device, resume, source
    )

    longer = RunSettings("tiny", 0, depth=3, width=8, epochs=2, batch_size=8)
    with pytest.raises(SystemExit):
        train_model(longer, patches, lambda _: 0.0, device, resume, source)
    with pytest.raises(SystemExit):
        train_model(settings, patches, lambda _: 0.0, device, resume, {"train": {"crops": 32}})

    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, settings, history, best_epoch, source)
    _, payload = load_checkpoint(path, device)
    check_run_matches(path, payload, settings, source)
    with pytest.raises(SystemExit):
        check_run_matches(path, payload, longer, source)
    with pytest.raises(SystemExit):
        check_run_matches(path, payload, settings, None)
