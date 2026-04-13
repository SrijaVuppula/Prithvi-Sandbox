"""
patch_masking_study/scripts/plot_degradation_curve.py

Reads results.csv and plots degradation curves:
  - PSNR vs mask ratio (global + masked-region)
  - MAE vs mask ratio (global + masked-region)
  - SSIM vs mask ratio (global + masked-region)
  - Summary: masked-region only, all 3 metrics in one figure

One line per backbone, all on the same plot for comparison.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = REPO_ROOT / "patch_masking_study" / "outputs"
CSV_PATH    = OUTPUT_DIR / "results.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_COLORS = {
    "tiny":  "#4C72B0",
    "100M":  "#DD8452",
    "300M":  "#55A868",
    "600M":  "#C44E52",
}

METRICS = [
    ("psnr", "masked_psnr", "PSNR (dB)", "higher is better"),
    ("mae",  "masked_mae",  "MAE",        "lower is better"),
    ("ssim", "masked_ssim", "SSIM",       "higher is better"),
]


def load_results(csv_path: Path) -> dict:
    data = defaultdict(lambda: defaultdict(dict))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backbone   = row["backbone"]
            mask_ratio = float(row["mask_ratio"])
            for key, val in row.items():
                if key not in ("backbone", "mask_ratio"):
                    try:
                        data[backbone][mask_ratio][key] = float(val)
                    except ValueError:
                        data[backbone][mask_ratio][key] = None
    return data


def plot_metric(
    data: dict,
    global_key: str,
    masked_key: str,
    ylabel: str,
    direction: str,
    save_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    titles = ["Global (entire image)", "Masked region only"]
    keys   = [global_key, masked_key]

    for ax, key, title in zip(axes, keys, titles):
        for backbone, color in BACKBONE_COLORS.items():
            if backbone not in data:
                continue
            ratios = sorted(data[backbone].keys())
            values = [data[backbone][r].get(key) for r in ratios]
            x_pct  = [int(r * 100) for r in ratios]

            ax.plot(
                x_pct, values,
                marker="o", linewidth=2, markersize=6,
                color=color, label=backbone,
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Mask Ratio (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Backbone", fontsize=10)
        ax.text(
            0.98, 0.02, direction,
            transform=ax.transAxes,
            fontsize=9, color="gray",
            ha="right", va="bottom",
        )

    fig.suptitle(
        f"{ylabel} vs Mask Ratio — Prithvi EO 2.0 Patch Masking Study",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_summary(data: dict, save_path: Path):
    """
    Single figure with 3 panels — masked-region metrics only.
    This is the money figure: honest degradation across all backbones.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_info = [
        ("masked_psnr", "Masked PSNR (dB)", "higher is better"),
        ("masked_mae",  "Masked MAE",        "lower is better"),
        ("masked_ssim", "Masked SSIM",       "higher is better"),
    ]

    for ax, (key, ylabel, direction) in zip(axes, metrics_info):
        for backbone, color in BACKBONE_COLORS.items():
            if backbone not in data:
                continue
            ratios = sorted(data[backbone].keys())
            values = [data[backbone][r].get(key) for r in ratios]
            x_pct  = [int(r * 100) for r in ratios]

            ax.plot(
                x_pct, values,
                marker="o", linewidth=2.5, markersize=7,
                color=color, label=backbone,
            )

        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Mask Ratio (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Backbone", fontsize=10)
        ax.text(
            0.98, 0.02, direction,
            transform=ax.transAxes,
            fontsize=9, color="gray",
            ha="right", va="bottom",
        )

    fig.suptitle(
        "Masked-Region Reconstruction Quality vs Mask Ratio — Prithvi EO 2.0",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: results.csv not found at {CSV_PATH}")
        print("Run run_patch_experiment.py first.")
        return

    print(f"Loading results from {CSV_PATH}")
    data = load_results(CSV_PATH)
    print(f"Found backbones: {list(data.keys())}")

    for global_key, masked_key, ylabel, direction in METRICS:
        save_path = FIGURES_DIR / f"degradation_{global_key}.png"
        plot_metric(data, global_key, masked_key, ylabel, direction, save_path)

    plot_summary(data, FIGURES_DIR / "degradation_summary.png")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
