"""
generate_psnr_table.py
Generates a clean plain PSNR comparison table as a PNG image.

Run from repo root:
  python multi_tile_generalization/block_masking_study/scripts/generate_psnr_table.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

data = [
    ("20%", 41.3, 40.4),
    ("40%", 37.3, 35.2),
    ("60%", 34.1, 31.1),
    ("80%", 31.1, 25.9),
]

headers = ["Mask Ratio", "Random PSNR (dB)", "Block PSNR (dB)", "Delta (dB)"]
rows = []
for ratio, rand, block in data:
    delta = round(block - rand, 1)
    rows.append([ratio, f"{rand:.1f}", f"{block:.1f}", f"{delta:+.1f}"])

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=headers,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.0)

# Header row styling
for col in range(len(headers)):
    cell = table[0, col]
    cell.set_facecolor("#222222")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#555555")

# Data rows styling
for row in range(1, len(rows) + 1):
    for col in range(len(headers)):
        cell = table[row, col]
        cell.set_facecolor("white")
        cell.set_text_props(color="black")
        cell.set_edgecolor("#aaaaaa")

fig.patch.set_facecolor("white")
plt.title("600M Backbone — Random vs Block Masking PSNR\n(chip_003_062_merged.tif, middle frame)",
          fontsize=11, pad=12, color="black")

out_path = Path("multi_tile_generalization/block_masking_study/outputs/figures/psnr_table_600M.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")