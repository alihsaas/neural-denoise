import argparse
import time

from neural_denoise.classical import make_denoiser
from neural_denoise.config import (
    BM3D_SIGMA_GRID,
    BM3D_TUNING_CROPS,
    BM3D_WORKERS,
    NLM_H_GRID,
    NLM_WORKERS,
    RESULTS_DIR,
)
from neural_denoise.evaluation import (
    claim_test_pass,
    evaluate_blocks_parallel,
    print_summary,
    save_scores,
    summarize,
    write_csv,
    write_json,
)
from neural_denoise.studies import load_test_blocks, subset_groups, validation_groups

GRIDS = {"nlm": NLM_H_GRID, "bm3d": BM3D_SIGMA_GRID}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlm-workers", type=int, default=NLM_WORKERS)
    parser.add_argument("--bm3d-workers", type=int, default=BM3D_WORKERS)
    parser.add_argument("--bm3d-crops", type=int, default=BM3D_TUNING_CROPS)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = RESULTS_DIR / "classical"
    noisy, clean = validation_groups()
    workers = {"noisy": 1, "nlm": arguments.nlm_workers, "bm3d": arguments.bm3d_workers}
    tuning_rows, chosen = [], {}
    for method, grid in GRIDS.items():
        if method == "bm3d":
            method_noisy, method_clean = subset_groups(noisy, clean, arguments.bm3d_crops)
        else:
            method_noisy, method_clean = noisy, clean
        best = None
        for value in grid:
            started = time.perf_counter()
            psnr_values, ssim_values = evaluate_blocks_parallel(
                make_denoiser(method, value), method_noisy, method_clean, workers[method]
            )
            row = {
                "method": method,
                "parameter": value,
                "crops": int(psnr_values.size),
                "validation_psnr": float(psnr_values.mean()),
                "validation_ssim": float(ssim_values.mean()),
                "seconds": time.perf_counter() - started,
            }
            tuning_rows.append(row)
            write_csv(tuning_rows, output / "tuning.csv")
            print(row, flush=True)
            if best is None or row["validation_psnr"] > best["validation_psnr"]:
                best = row
        chosen[method] = best["parameter"]
    write_csv(tuning_rows, output / "tuning.csv")
    write_json(chosen, output / "chosen.json")

    test_noisy, test_clean = load_test_blocks()
    rows = []
    for method, parameter in [("noisy", None), *chosen.items()]:
        claim_test_pass(f"classical/{method}", force=arguments.force)
        psnr_values, ssim_values = evaluate_blocks_parallel(
            make_denoiser(method, parameter), test_noisy, test_clean, workers[method], method
        )
        save_scores(psnr_values, ssim_values, output / f"scores_{method}.npz")
        row = summarize(method, psnr_values, ssim_values)
        row["parameter"] = "" if parameter is None else parameter
        rows.append(row)
    write_csv(rows, output / "summary.csv")
    print_summary(rows)


if __name__ == "__main__":
    main()
