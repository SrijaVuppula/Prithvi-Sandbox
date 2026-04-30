"""
plot_block_results.py
---------------------
Generates figures for the block masking study.
Run after aggregate_block_results.py.

Usage
-----
  cd ~/Prithvi/Prithvi-Sandbox
  python multi_tile_generalization/block_masking_study/scripts/plot_block_results.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR  = SCRIPT_DIR.parent
OUT_DIR    = STUDY_DIR / "outputs"
FIG_DIR    = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["tiny", "100M", "300M", "600M"]
RATIOS    = [0.20, 0.40, 0.60, 0.80]
COLORS    = {"tiny": "#4C8BE0", "100M": "#E07C4C", "300M": "#4CBF7A", "600M": "#C04CD1"}
MARKERS   = {"tiny": "o", "100M": "s", "300M": "^", "600M": "D"}
N_SAMPLES  = 500 * 5  # chips x trials

RANDOM_PSNR = {
    "tiny": {0.20: 34.57, 0.40: 33.31, 0.60: 31.61, 0.80: 29.10},
    "100M": {0.20: 35.85, 0.40: 34.16, 0.60: 32.11, 0.80: 29.34},
    "300M": {0.20: 36.32, 0.40: 34.58, 0.60: 32.43, 0.80: 29.55},
    "600M": {0.20: 37.32, 0.40: 35.56, 0.60: 33.28, 0.80: 30.19},
}


def load_summary():
    path = OUT_DIR / "block_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run aggregate_block_results.py first: {path}")
    data = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["backbone"]][float(row["mask_ratio"])] = {
                "mean": float(row["block_psnr_mean"]),
                "std":  float(row["block_psnr_std"]),
                "se":   float(row["block_psnr_std"]) / np.sqrt(N_SAMPLES),
            }
    return data


def _base_plot(ax, data, yerr_key, title_suffix):
    x = [int(r * 100) for r in RATIOS]
    x_arr = np.array(x, dtype=float)
    offsets = {"tiny": -2, "100M": -0.7, "300M": +0.7, "600M": +2}

    for bb in BACKBONES:
        if bb not in data:
            continue
        means = np.array([data[bb].get(r, {}).get("mean", np.nan) for r in RATIOS])
        errs  = np.array([data[bb].get(r, {}).get(yerr_key, np.nan) for r in RATIOS]) if yerr_key else None
        x_off = x_arr + offsets[bb]

        if errs is not None:
            ax.errorbar(x_off, means, yerr=errs,
                        marker=MARKERS[bb], color=COLORS[bb],
                        linewidth=2, markersize=7, capsize=4, capthick=1.5,
                        elinewidth=1.5, label=bb)
        else:
            ax.plot(x_off, means,
                    marker=MARKERS[bb], color=COLORS[bb],
                    linewidth=2, markersize=7, label=bb)

    ax.axhline(30, color="red", linestyle="--", linewidth=1.2, label="30 dB threshold")
    ax.set_xlabel("Block Mask Ratio (%)", fontsize=12)
    ax.set_ylabel("Block PSNR (dB)", fontsize=12)
    ax.set_title(f"Block Masking — PSNR Degradation by Backbone\n{title_suffix}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_clean_lines(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    _base_plot(ax, data, yerr_key=None,
               title_suffix="(500 chips × 5 trials, contiguous rectangular block)")
    fig.tight_layout()
    path = FIG_DIR / "fig1a_block_degradation_clean.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_std_error_bars(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    _base_plot(ax, data, yerr_key="se",
               title_suffix="(500 chips × 5 trials, ±1 standard error)")
    fig.tight_layout()
    path = FIG_DIR / "fig1b_block_degradation_stderr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_block_vs_random(data):
    fig, axes = plt.subplots(1, len(RATIOS), figsize=(14, 5), sharey=False)
    for ax, ratio in zip(axes, RATIOS):
        bb_labels, block_vals, random_vals = [], [], []
        for bb in BACKBONES:
            if bb not in data or ratio not in data[bb]:
                continue
            bb_labels.append(bb)
            block_vals.append(data[bb][ratio]["mean"])
            random_vals.append(RANDOM_PSNR.get(bb, {}).get(ratio, np.nan))
        x_pos = np.arange(len(bb_labels))
        w = 0.35
        ax.bar(x_pos - w/2, block_vals,  w, color=[COLORS[b] for b in bb_labels], alpha=0.85)
        ax.bar(x_pos + w/2, random_vals, w, color=[COLORS[b] for b in bb_labels], alpha=0.4, hatch="//")
        ax.set_title(f"{int(ratio*100)}% masked", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bb_labels, fontsize=9)
        ax.set_ylabel("PSNR (dB)" if ax == axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3)
    block_patch  = mpatches.Patch(color="grey", alpha=0.85, label="Block masking")
    random_patch = mpatches.Patch(color="grey", alpha=0.4, hatch="//", label="Random masking")
    fig.legend(handles=[block_patch, random_patch], loc="upper right", fontsize=10)
    fig.suptitle("Block vs Random Masking — PSNR by Backbone and Ratio", fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "fig2_block_vs_random.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_difficulty_gap(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(RATIOS))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(BACKBONES)) * width
    for i, bb in enumerate(BACKBONES):
        if bb not in data:
            continue
        gaps = [RANDOM_PSNR.get(bb, {}).get(r, np.nan) - data[bb].get(r, {}).get("mean", np.nan)
                for r in RATIOS]
        ax.bar(x + offsets[i], gaps, width, label=bb, color=COLORS[bb], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Block Mask Ratio (%)", fontsize=12)
    ax.set_ylabel("PSNR Drop: Random → Block (dB)", fontsize=12)
    ax.set_title("How Much Harder Is Block Masking?\n"
                 "(positive = block masking harder, as expected for cloud removal)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r*100)}%" for r in RATIOS])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig3_difficulty_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    data = load_summary()
    plot_clean_lines(data)
    plot_std_error_bars(data)
    plot_block_vs_random(data)
    plot_difficulty_gap(data)
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
