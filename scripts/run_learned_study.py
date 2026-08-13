import argparse
from functools import partial

import numpy as np
import torch

from neural_denoise.cache import crop_record, load_crops
from neural_denoise.config import CHECKPOINT_DIR, LEARNED_MODELS, RESULTS_DIR, SEEDS
from neural_denoise.evaluation import (
    claim_test_pass,
    evaluate_blocks,
    print_summary,
    save_scores,
    summarize,
    write_csv,
    write_json,
)
from neural_denoise.neural import (
    RunSettings,
    check_run_matches,
    denoise_batch,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    train_model,
    validation_psnr,
)
from neural_denoise.studies import load_test_blocks, validation_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RunSettings.epochs)
    parser.add_argument("--models", nargs="+", default=list(LEARNED_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    patches = load_crops("train")
    source = {"train": crop_record("train"), "validation": crop_record("validation")}
    noisy, clean = validation_groups()
    flat_noisy = noisy.reshape(-1, *noisy.shape[2:])
    flat_clean = clean.reshape(-1, *clean.shape[2:])
    evaluate = partial(validation_psnr, noisy=flat_noisy, clean=flat_clean, device=device)
    output = RESULTS_DIR / "learned"

    runs = []
    for model_name in arguments.models:
        depth, width = LEARNED_MODELS[model_name]
        for seed in arguments.seeds:
            settings = RunSettings(f"{model_name}_seed{seed}", seed, depth, width, arguments.epochs)
            final_path = CHECKPOINT_DIR / f"{settings.name}.pt"
            if final_path.exists():
                model, payload = load_checkpoint(final_path, device)
                check_run_matches(final_path, payload, settings, source)
                history, best_epoch = payload["history"], payload["best_epoch"]
                print(f"{settings.name}: reusing {final_path.name}")
            else:
                model, history, best_epoch = train_model(
                    settings,
                    patches,
                    evaluate,
                    device,
                    CHECKPOINT_DIR / f"{settings.name}_last.pt",
                    source,
                )
                save_checkpoint(final_path, model, settings, history, best_epoch, source)
            write_csv(history, output / f"history_{settings.name}.csv")
            runs.append(
                {
                    "name": settings.name,
                    "model": model_name,
                    "seed": seed,
                    "depth": depth,
                    "width": width,
                    "parameters": parameter_count(model),
                    "best_epoch": best_epoch,
                    "best_validation_psnr": max(entry["validation_psnr"] for entry in history),
                }
            )
    write_json({"runs": runs}, output / "runs.json")

    test_noisy, test_clean = load_test_blocks()
    rows = []
    for run in runs:
        claim_test_pass(f"learned/{run['name']}", force=arguments.force)
        model, _ = load_checkpoint(CHECKPOINT_DIR / f"{run['name']}.pt", device)
        psnr_values, ssim_values = evaluate_blocks(
            lambda images, model=model: denoise_batch(model, images, device),
            test_noisy,
            test_clean,
            run["name"],
        )
        save_scores(psnr_values, ssim_values, output / f"scores_{run['name']}.npz")
        row = summarize(run["name"], psnr_values, ssim_values)
        row["parameters"] = run["parameters"]
        row["best_epoch"] = run["best_epoch"]
        rows.append(row)
    write_csv(rows, output / "summary.csv")
    print_summary(rows)
    for model_name in arguments.models:
        values = [row["psnr"] for row in rows if row["method"].startswith(f"{model_name}_seed")]
        print(f"{model_name}: mean test PSNR over seeds {np.mean(values):.3f}")


if __name__ == "__main__":
    main()
