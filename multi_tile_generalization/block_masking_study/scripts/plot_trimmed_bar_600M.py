"""
plot_trimmed_bar_600M.py
--------------------------
Bar chart comparing the original mean delta (random - block PSNR) to the
5% trimmed mean delta, per mask ratio, for the 600M backbone. Answers:
does the block-harder finding survive removing the top/bottom 5% of
chips by delta (outlier robustness check)?

Delta convention: delta = random_psnr - block_psnr
    delta > 0  ->  block_harder

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/plot_trimmed_bar_600M.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, zero_line, COLORS, FIGSIZE, FONT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
OUT_DIR   = IN_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE = "600M"
RATIOS   = [0.2, 0.4, 0.6, 0.8]

# neutral pairing: primary accent (blue) for the headline result,
# muted gray for the trimmed/robustness-check series
COLOR_ORIGINAL = COLORS["blue"]
COLOR_TRIMMED  = "#9B9B9B"


def trimmed_mean(x, pct=0.05):
    x = np.sort(x)
    k = int(len(x) * pct)
    return x[k: len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def main():
    apply_style()

    df = pd.read_csv(IN_DIR / f"results_{BACKBONE}.csv")
    agg = (df.groupby(["chip", "mask_ratio"], as_index=False)
             .agg(block_psnr=("block_psnr", "mean"),
                  random_psnr=("random_psnr", "mean")))
    agg["delta"] = agg["random_psnr"] - agg["block_psnr"]

    original, trimmed = [], []
    for ratio in RATIOS:
        d = agg.loc[agg["mask_ratio"] == ratio, "delta"].values
        original.append(d.mean())
        trimmed.append(trimmed_mean(d))

    x = np.arange(len(RATIOS))
    width = 0.34

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

    bars1 = ax.bar(x - width / 2, original, width, label="Original mean",
                   color=COLOR_ORIGINAL)
    bars2 = ax.bar(x + width / 2, trimmed, width, label="5% trimmed mean",
                   color=COLOR_TRIMMED)

    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=FONT["annot"])

    group_max = [max(o, t) for o, t in zip(original, trimmed)]
    y_top = max(group_max)
    for i, (o, t, gm) in enumerate(zip(original, trimmed, group_max)):
        diff = o - t
        ax.text(x[i], gm + y_top * 0.10, f"\u0394={diff:+.2f} dB",
                ha="center", va="bottom", fontsize=FONT["annot"], color="#555555")

    zero_line(ax)
    ax.set_ylim(0, y_top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r*100)}%" for r in RATIOS])
    style_ax(ax, xlabel="Mask ratio",
             ylabel="Delta = Random PSNR \u2212 Block PSNR (dB)",
             title=f"Block vs random delta: original vs 5% trimmed mean \u2014 {BACKBONE}")

    ax.legend(fontsize=FONT["legend"], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig.subplots_adjust(right=0.80, top=0.88, bottom=0.12, left=0.09)
    out_path = OUT_DIR / f"fig_trimmed_bar_{BACKBONE}.png"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    for r, o, t in zip(RATIOS, original, trimmed):
        print(f"  {int(r*100)}%: original={o:.3f}  trimmed={t:.3f}  diff={o - t:+.3f}")


if __name__ == "__main__":
    main()
