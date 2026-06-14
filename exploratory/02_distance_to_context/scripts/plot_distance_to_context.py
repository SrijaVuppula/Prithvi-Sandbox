"""
Experiment 2 — Plot: per-patch error vs distance to nearest visible patch.
Run after compute_distance_errors.py has finished.
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO    = os.path.expanduser("~/Prithvi/Prithvi-Sandbox")
IN_CSV  = os.path.join(REPO, "exploratory/02_distance_to_context/outputs/distance_errors.csv")
OUT_DIR = os.path.join(REPO, "exploratory/02_distance_to_context/outputs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.expanduser("~/Prithvi"))
from plot_style import apply_style, style_ax, zero_line, COLOR_SEQ, MARKERS, FONT, FIGSIZE

apply_style()

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)
print(f"Loaded {len(df)} rows")
print(df.head(3))

ratios  = sorted(df["ratio_pct"].unique())
colors  = COLOR_SEQ[:len(ratios)]
markers = MARKERS[:len(ratios)]

# ── Aggregate: mean MAE per (ratio, distance) ──────────────────────────────────
agg = (df.groupby(["ratio_pct", "distance"])["patch_mae"]
         .agg(mean="mean", sem=lambda x: x.std()/np.sqrt(len(x)))
         .reset_index())

print("\nMax distance per ratio:")
print(df.groupby("ratio_pct")["distance"].max())

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

for ratio, color, marker in zip(ratios, colors, markers):
    sub = agg[agg["ratio_pct"] == ratio].sort_values("distance")
    ax.plot(sub["distance"], sub["mean"],
            color=color, marker=marker,
            linewidth=2.0, markersize=6.5,
            label=f"{ratio}% mask", zorder=3)
    ax.fill_between(sub["distance"],
                    sub["mean"] - sub["sem"],
                    sub["mean"] + sub["sem"],
                    alpha=0.10, color=color)

style_ax(ax,
    xlabel="Distance to Nearest Visible Patch (patches)",
    ylabel="Mean Patch MAE",
    title="Reconstruction Error Rises with Distance from Visible Context"
)

ax.set_xticks(sorted(df["distance"].unique()))
ax.legend(title="Mask Ratio", loc="lower right")

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_distance_to_context.png")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")
