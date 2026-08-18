from neural_denoise.comparison import BASELINE, collect_scores, gain_rows
from neural_denoise.config import RESULTS_DIR
from neural_denoise.evaluation import write_csv


def main() -> None:
    rows = gain_rows(collect_scores(RESULTS_DIR))
    write_csv(rows, RESULTS_DIR / "gains.csv")
    print(f"{'method':28s} {'gain over ' + BASELINE:>16s} {'95% CI':>15s}")
    for row in rows:
        interval = f"{row['gain_ci_low']:.2f}-{row['gain_ci_high']:.2f}"
        print(f"{row['method']:28s} {row[f'gain_over_{BASELINE}']:16.2f} {interval:>15s}")


if __name__ == "__main__":
    main()
