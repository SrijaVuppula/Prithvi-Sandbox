"""
Figure 1: the two occlusion geometries on the HLS multispectral input at r = 20%.

Single-column figure (3.35 in). Rows: Contiguous / Scattered. Columns: the three
temporal frames (spring, summer, fall) as 14x14 patch grids. Only the summer frame
is masked; both geometries hide the same 39 of 196 patches.

Layout follows the reference papers (HyperFM Fig. 6): light-gray header band for
column titles, rotated row labels on the left. Tiles and palette follow Prithvi-EO-2.0
Fig. 3 (tokens as tiles with thin separators). Fonts match the paper body (Times) via
~/Prithvi/paper_style.py, at true print size.

Masking logic is reused unchanged from temporal_gap_masker.py.

Usage:   python fig1_geometries.py
Output:  figures/fig1_geometries.pdf (LaTeX), figures/fig1_geometries.png (preview)
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "masking"))
sys.path.insert(0, os.path.expanduser("~/Prithvi"))

from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask  # noqa: E402
from paper_style import apply_paper_style, COLORS, PAPER_FONT, COL_W  # noqa: E402

apply_paper_style()

MASKED = COLORS["vermillion"]
VISIBLE = "#E3E3E3"             # visible tiles: neutral light gray
CMAP = ListedColormap([VISIBLE, MASKED])
BAND = "#EDEDED"                # header / row-label bands
DARK = "#4d4d4d"                # outline of the masked (target) frame
LIGHT = "#bdbdbd"               # outline of context frames

RATIO = 0.20
PATCH_SIZE = 16
IMG_SIZE = 224
NUM_FRAMES = 3
FRAME_IDX = 1                   # summer
GRID = IMG_SIZE // PATCH_SIZE
TOKENS_PER_FRAME = GRID * GRID
TRIAL_SEED = 42
COLS = ["Spring (context)", "Summer (masked)", "Fall (context)"]


def masked_patch_grid(masked_global, frame_idx, grid, tokens_per_frame):
    offset = frame_idx * tokens_per_frame
    local = (masked_global - offset).cpu().numpy()
    mask_grid = np.zeros((grid, grid), dtype=bool)
    mask_grid[local // grid, local % grid] = True
    return mask_grid


def draw_patch_grid(ax, mask_grid, is_target):
    n = mask_grid.shape[0]
    ax.imshow(mask_grid.astype(int), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(DARK if is_target else LIGHT)
        s.set_linewidth(1.0 if is_target else 0.5)


def band(fig, x0, y0, x1, y1):
    fig.add_artist(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                      facecolor=BAND, edgecolor="none", zorder=0))


def main():
    _, _, idx_block = build_block_noise_mask(
        RATIO, patch_size=PATCH_SIZE, img_size=IMG_SIZE, num_frames=NUM_FRAMES,
        frame_idx=FRAME_IDX, trial_seed=TRIAL_SEED)
    _, _, idx_rand = build_random_noise_mask(
        RATIO, patch_size=PATCH_SIZE, img_size=IMG_SIZE, num_frames=NUM_FRAMES,
        frame_idx=FRAME_IDX, trial_seed=TRIAL_SEED, n_summer_masked=len(idx_block))
    n_masked = len(idx_block)
    assert n_masked == len(idx_rand) == 39

    rows = [
        ("Contiguous", masked_patch_grid(idx_block, FRAME_IDX, GRID, TOKENS_PER_FRAME)),
        ("Scattered",  masked_patch_grid(idx_rand,  FRAME_IDX, GRID, TOKENS_PER_FRAME)),
    ]
    empty = np.zeros((GRID, GRID), dtype=bool)

    fig = plt.figure(figsize=(COL_W, 2.55))
    gs = GridSpec(2, 3, figure=fig, left=0.105, right=0.995, top=0.895, bottom=0.115,
                  wspace=0.06, hspace=0.06)

    axes = [[None] * 3 for _ in range(2)]
    for r, (_, summer_grid) in enumerate(rows):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            draw_patch_grid(ax, summer_grid if c == FRAME_IDX else empty, c == FRAME_IDX)
            ax.apply_aspect()
            axes[r][c] = ax

    # column header bands (top row only)
    for c, title in enumerate(COLS):
        pos = axes[0][c].get_position()
        y0, y1 = pos.y1 + 0.012, pos.y1 + 0.085
        band(fig, pos.x0, y0, pos.x1, y1)
        fig.text((pos.x0 + pos.x1) / 2, (y0 + y1) / 2, title, ha="center", va="center",
                 fontsize=PAPER_FONT["label"] - 1, fontweight="bold")

    # rotated row-label bands (left)
    for r, (name, _) in enumerate(rows):
        pos = axes[r][0].get_position()
        x0, x1 = 0.008, 0.075
        band(fig, x0, pos.y0, x1, pos.y1)
        fig.text((x0 + x1) / 2, (pos.y0 + pos.y1) / 2, name, rotation=90, ha="center",
                 va="center", fontsize=PAPER_FONT["label"] - 1, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=VISIBLE, edgecolor=LIGHT, linewidth=0.5, label="Visible"),
        mpatches.Patch(facecolor=MASKED, edgecolor=LIGHT, linewidth=0.5,
                       label=f"Masked ({n_masked} of {TOKENS_PER_FRAME} patches, "
                             rf"$r = {int(RATIO * 100)}\%$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 0.0), fontsize=PAPER_FONT["legend"],
               handlelength=1.0, handleheight=0.9, columnspacing=1.6, borderaxespad=0.2)

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig1_geometries.{ext}"
        fig.savefig(p, dpi=300)          # no bbox_inches="tight": keep the true 3.35 in width
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
