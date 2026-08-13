import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class RunSettings:
    name: str
    seed: int
    depth: int = 17
    width: int = 64
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class DnCNN(nn.Module):
    def __init__(self, depth: int = 17, width: int = 64, channels: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        last = nn.Conv2d(width, channels, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.body = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy - self.body(noisy)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    values = np.asarray(images, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    return torch.from_numpy(np.ascontiguousarray(values))


def to_images(values: torch.Tensor) -> np.ndarray:
    pixels = (values.float().clamp(0.0, 1.0) * 255.0).round().byte()
    return pixels.permute(0, 2, 3, 1).cpu().numpy()


def dihedral(images: np.ndarray, code: int) -> np.ndarray:
    turned = np.rot90(images, code % 4, axes=(-3, -2))
    return turned[..., ::-1, :] if code >= 4 else turned


def augment_pairs(pairs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes = rng.integers(0, 8, len(pairs))
    return np.stack([dihedral(pair, int(code)) for pair, code in zip(pairs, codes, strict=True)])


@torch.no_grad()
def denoise_batch(
    model: nn.Module, noisy: np.ndarray, device: torch.device, batch_size: int = 16
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(noisy), batch_size):
        inputs = to_tensor(noisy[start : start + batch_size]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs.append(to_images(model(inputs)))
    return np.concatenate(outputs)


def validation_psnr(
    model: nn.Module, noisy: np.ndarray, clean: np.ndarray, device: torch.device
) -> float:
    values = []
    for start in range(0, len(noisy), 32):
        estimate = denoise_batch(model, noisy[start : start + 32], device).astype(np.float32)
        truth = np.asarray(clean[start : start + 32], dtype=np.float32)
        errors = ((estimate - truth) ** 2).reshape(len(truth), -1).mean(axis=1, dtype=np.float64)
        values.append(10.0 * np.log10(255.0**2 / np.maximum(errors, 1e-12)))
    return float(np.mean(np.concatenate(values)))


def run_identity(settings: RunSettings, source: dict | None) -> dict:
    return {"settings": asdict(settings), "source": source or {}}


def check_run_matches(
    path: Path, payload: dict, settings: RunSettings, source: dict | None
) -> None:
    expected = run_identity(settings, source)
    if {key: payload.get(key) for key in expected} != expected:
        raise SystemExit(
            f"{path} was made with other settings or other crops. Move it away to train again."
        )


def make_optimiser(
    model: nn.Module, settings: RunSettings, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=settings.learning_rate,
        total_steps=settings.epochs * steps_per_epoch,
        pct_start=0.05,
    )
    return optimiser, schedule


def train_model(
    settings: RunSettings,
    patches: np.ndarray,
    evaluate: Callable[[nn.Module], float],
    device: torch.device,
    resume_path: Path | None = None,
    source: dict | None = None,
) -> tuple[nn.Module, list[dict], int]:
    seed_everything(settings.seed)
    model = DnCNN(settings.depth, settings.width).to(device)
    steps_per_epoch = math.ceil(len(patches) / settings.batch_size)
    optimiser, schedule = make_optimiser(model, settings, steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict] = []
    best_score, best_epoch, best_state = -math.inf, 0, None
    first_epoch = 1
    if resume_path is not None and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        check_run_matches(resume_path, saved, settings, source)
        model.load_state_dict(saved["state"])
        optimiser.load_state_dict(saved["optimiser"])
        schedule.load_state_dict(saved["schedule"])
        scaler.load_state_dict(saved["scaler"])
        history, best_score, best_epoch = saved["history"], saved["best_score"], saved["best_epoch"]
        best_state = saved["best_state"]
        first_epoch = saved["epoch"] + 1
        print(f"{settings.name}: resuming after epoch {saved['epoch']}")
    loss_function = nn.MSELoss()
    for epoch in range(first_epoch, settings.epochs + 1):
        model.train()
        rng = np.random.default_rng((settings.seed, epoch))
        order = rng.permutation(len(patches))
        total, started = 0.0, time.perf_counter()
        for start in range(0, len(order), settings.batch_size):
            chosen = np.sort(order[start : start + settings.batch_size])
            pairs = augment_pairs(np.asarray(patches[chosen]), rng)
            noisy = to_tensor(pairs[:, 0]).to(device, non_blocking=True)
            clean = to_tensor(pairs[:, 1]).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                estimate = model(noisy)
            loss = loss_function(estimate.float(), clean)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item() * len(chosen)
        score = evaluate(model)
        entry = {
            "epoch": epoch,
            "train_mse": total / len(order),
            "validation_psnr": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(entry)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(
            f"{settings.name} epoch {epoch:3d}  mse {entry['train_mse']:.6f}"
            f"  validation PSNR {score:.3f}  {entry['seconds']:.0f} s"
        )
        if resume_path is not None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **run_identity(settings, source),
                    "epoch": epoch,
                    "state": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "schedule": schedule.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                },
                resume_path,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def save_checkpoint(
    path: Path,
    model: nn.Module,
    settings: RunSettings,
    history: list[dict],
    best_epoch: int,
    source: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **run_identity(settings, source),
            "history": history,
            "best_epoch": best_epoch,
            "state": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    settings = payload["settings"]
    model = DnCNN(settings["depth"], settings["width"]).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload
