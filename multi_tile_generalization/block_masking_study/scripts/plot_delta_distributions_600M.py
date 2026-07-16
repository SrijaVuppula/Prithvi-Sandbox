"""
plot_delta_distributions_600M.py
----------------------------------
KDE distribution of delta (random - block PSNR) across the 500 study chips,
for the 600M backbone. One subplot per mask ratio (20/40/60/80%).

Delta convention: delta = random_psnr - block_psnr
    delta > 0  ->  block_harder
    delta < 0  ->  random_harder

y-axis is shared/fixed across all 4 subplots per project convention
(cross-condition comparability).

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/plot_delta_distributions_600M.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, zero_line, COLORS, FIGSIZE, FONT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
OUT_DIR   = IN_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "600M"
RATIOS   = [0.2, 0.4, 0.6, 0.8]


def trimmed_mean(x, pct=0.05):
    x = np.sort(x)
    k = int(len(x) * pct)
    return x[k: len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def main():
    apply_style()

    df = pd.read_csv(IN_DIR / f"results_fixed_{BACKBONE}.csv")
    agg = (df.groupby(["chip", "mask_ratio"], as_index=False)
             .agg(block_psnr=("block_psnr", "mean"),
                  random_psnr=("random_psnr", "mean")))
    agg["delta"] = agg["random_psnr"] - agg["block_psnr"]

    kdes, deltas, stats = {}, {}, {}
    for ratio in RATIOS:
        d = agg.loc[agg["mask_ratio"] == ratio, "delta"].values
        deltas[ratio] = d
        kdes[ratio] = gaussian_kde(d)
        stats[ratio] = {
            "mean": d.mean(),
            "median": np.median(d),
            "trimmed": trimmed_mean(d),
            "pct_block_harder": (d > 0).mean() * 100,
        }

    x_min = min(d.min() for d in deltas.values()) - 0.5
    x_max = max(d.max() for d in deltas.values()) + 0.5
    xs = np.linspace(x_min, x_max, 400)
    y_max = max(kdes[r](xs).max() for r in RATIOS) * 1.15

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.15, hspace=0.4, top=0.90, bottom=0.09,
                        left=0.08, right=0.97)
    axes = axes.flatten()

    for i, ratio in enumerate(RATIOS):
        ax = axes[i]
        d = deltas[ratio]
        s = stats[ratio]
        ys = kdes[ratio](xs)

        ax.plot(xs, ys, color=COLORS["blue"], linewidth=1.3)
        ax.fill_between(xs, ys, color=COLORS["blue"], alpha=0.15)

        zero_line(ax)
        ax.axvline(s["mean"], color=COLORS["vermillion"], linestyle="--",
                   linewidth=1.0, label=f"Mean {s['mean']:.2f}")
        ax.axvline(s["median"], color=COLORS["orange"], linestyle=":",
                   linewidth=1.0, label=f"Median {s['median']:.2f}")
        ax.axvline(s["trimmed"], color=COLORS["green"], linestyle="-.",
                   linewidth=1.0, label=f"Trimmed {s['trimmed']:.2f}")

        ax.set_ylim(0, y_max)
        style_ax(ax,
                 xlabel="Delta = Random PSNR \u2212 Block PSNR (dB)" if i >= 2 else None,
                 ylabel="Density" if i % 2 == 0 else None,
                 title=f"{int(ratio*100)}% mask ratio")
        ax.legend(fontsize=FONT["annot"], loc="upper left",
                  handlelength=1.4, borderaxespad=0.4, labelspacing=0.3)

        ax.text(0.97, 0.06,
                f"{s['pct_block_harder']:.1f}% block harder\n"
                f"{100 - s['pct_block_harder']:.1f}% random harder",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT["annot"],
                bbox=dict(boxstyle="round", facecolor="white",
                         edgecolor="#cccccc", alpha=0.9))

    fig.suptitle(f"Delta distribution across 500 chips \u2014 {BACKBONE} backbone",
                fontsize=FONT["title"] + 1)

    out_path = OUT_DIR / f"fig_delta_distributions_{BACKBONE}.png"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
