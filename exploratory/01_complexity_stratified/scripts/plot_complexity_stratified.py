"""
Experiment 1 — Complexity-Stratified Crossover (publication style)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = os.path.expanduser("~/Prithvi/Prithvi-Sandbox")
COMP_CSV = os.path.join(REPO, "multi_tile_generalization/block_masking_study/outputs/chip_complexity_v2.csv")
SANITY   = os.path.join(REPO, "multi_tile_generalization/block_masking_study/outputs/sanity_check_block_vs_random.xlsx")
OUT_DIR  = os.path.join(REPO, "exploratory/01_complexity_stratified/outputs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & merge ───────────────────────────────────────────────────────────────
comp = pd.read_csv(COMP_CSV)
comp["chip_key"] = comp["chip"].str.replace("_merged.tif", "", regex=False).str.strip()
comp = comp[["chip_key", "complexity"]]

df = pd.read_excel(SANITY, sheet_name="600M")
df["chip_key"] = df["chip"].str.replace("_merged.tif", "", regex=False).str.strip()
df["delta"]    = df["delta_random_minus_block"]

def to_pct(x):
    v = float(str(x).replace("%", "").strip())
    return int(v * 100) if v < 1.5 else int(v)
df["ratio_pct"] = df["mask_ratio"].apply(to_pct)

merged = df.merge(comp, on="chip_key", how="inner")
merged["quartile"] = pd.qcut(
    merged["complexity"], q=4,
    labels=["Q1 (Uniform)", "Q2", "Q3", "Q4 (Complex)"]
)

agg = (merged
       .groupby(["ratio_pct", "quartile"])["delta"]
       .mean()
       .reset_index())

# ── Style: CVPR-clean ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "legend.frameon":     False,
    "legend.fontsize":    10,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# Okabe-Ito colorblind-safe palette (used in Nature, CVPR, NeurIPS papers)
COLORS  = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]
LABELS  = ["Q1 (Uniform)", "Q2", "Q3", "Q4 (Complex)"]
RATIOS  = [20, 40, 60, 80]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

# Zero line — thin, gray, behind everything
ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)

# One line per quartile
for label, color, marker in zip(LABELS, COLORS, MARKERS):
    sub = agg[agg["quartile"] == label].sort_values("ratio_pct")
    ax.plot(
        sub["ratio_pct"], sub["delta"],
        color=color, marker=marker,
        linewidth=2.0, markersize=6.5,
        label=label, zorder=3
    )

# Axis
ax.set_xlabel("Mask Ratio (%)", fontsize=10)
ax.set_ylabel("Δ PSNR  (Random − Block)  [dB]", fontsize=10)
ax.set_xticks(RATIOS)
ax.set_xticklabels([f"{r}%" for r in RATIOS])
ax.set_xlim(17, 83)
ax.set_ylim(-1.5, 2.4)
ax.tick_params(labelsize=8)

# Legend inside plot, bottom left — no frame
ax.legend(title="Scene Complexity",title_fontsize=8, fontsize=8,
          loc="lower left", handlelength=1.8)

# Minimal annotations — just the key finding
ax.annotate("Q1 crosses over\nbetween 60–80%",
            xy=(65, 0.78), fontsize=7, color=COLORS[0],
            ha="left", va="center")
ax.annotate("Q2–Q4 cross over\nbetween 40–60%",
            xy=(42, -0.55), fontsize=7, color="#666666",
            ha="left", va="top")

ax.set_title(
    "Scene Complexity Shifts the Geometry Crossover",
    fontsize=10, pad=8
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_complexity_stratified_crossover.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
