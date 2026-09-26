"""
Figure 1: the two occlusion geometries on each axis at r = 20%.
Top: 14x14 spatial patch grid, contiguous block vs scattered, 39 patches each.
Bottom: 291-band spectral strip, contiguous run vs scattered, 58 bands each.

Reuses (unchanged logic):
  - build_block_noise_mask / build_random_noise_mask (temporal_gap_masker.py)
  - masked_patch_grid (from visualize_error_maps.py)
  - draw_band_strip logic (from render_reconstruction_composites.py), reimplemented
    standalone here so this script needs no torch/terratorch/checkpoint.

Usage:
    python fig1_geometries.py
Output:
    figures/fig1_geometries.pdf
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "masking"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask  # noqa: E402
from pace_band_wavelengths import BAND_WAVELENGTH_NM, BAND_ORIGIN  # noqa: E402

# ---- spatial setup: r=20% on the 196-patch (14x14) grid -> 39 patches, per Sec. 3.2 ----
RATIO = 0.20
PATCH_SIZE = 16
IMG_SIZE = 224
NUM_FRAMES = 3
FRAME_IDX = 1  # summer
GRID = IMG_SIZE // PATCH_SIZE           # 14
TOKENS_PER_FRAME = GRID * GRID          # 196
TRIAL_SEED = 42
FILL_COLOR = "#404040"


def masked_patch_grid(masked_global, frame_idx, grid, tokens_per_frame):
    offset = frame_idx * tokens_per_frame
    local = (masked_global - offset).cpu().numpy()
    rows = local // grid
    cols = local % grid
    mask_grid = np.zeros((grid, grid), dtype=bool)
    mask_grid[rows, cols] = True
    return mask_grid


def draw_patch_grid(ax, mask_grid, title):
    grid = mask_grid.shape[0]
    ax.set_xlim(0, grid)
    ax.set_ylim(grid, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(grid):
        for c in range(grid):
            face = FILL_COLOR if mask_grid[r, c] else "white"
            rect = mpatches.Rectangle((c, r), 1, 1, facecolor=face,
                                       edgecolor="black", linewidth=0.4)
            ax.add_patch(rect)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


# ---- spectral setup: r=20% on 291 bands -> 58 bands, per Sec. 3.2 ----
N_BANDS = 291
spec_rng = np.random.default_rng(42)  # project convention: seed 42 for spectral trials


def contiguous_band_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    start = spec_rng.integers(0, N_BANDS - n_mask + 1)
    return np.arange(start, start + n_mask)


def scattered_band_mask(frac):
    n_mask = int(round(N_BANDS * frac))
    return spec_rng.choice(N_BANDS, size=n_mask, replace=False)


ORIGIN_COLOR = {"blue": "#cfe3f7", "red": "#f7d9cf", "swir": "#e0e0e0"}


def draw_band_strip(ax, masked_idx, title):
    n = N_BANDS
    strip = np.zeros((1, n, 3))
    for i in range(n):
        strip[0, i] = matplotlib.colors.to_rgb(ORIGIN_COLOR[BAND_ORIGIN[i]])
    strip_img = np.tile(strip, (20, 1, 1))
    ax.imshow(strip_img, aspect="auto", extent=[0, n, 0, 1])
    for i in masked_idx:
        ax.axvline(i, color="black", linewidth=0.6, alpha=0.85)
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel("band index (hidden bands marked black)", fontsize=8)
    tick_idx = [0, 50, 100, 150, 200, 250, 290]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{BAND_WAVELENGTH_NM[i]:.0f}nm" for i in tick_idx], fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def main():
    # --- spatial masks, count-matched at exactly 39 patches (same seed, paired trial) ---
    noise_block, gratio_block, idx_block = build_block_noise_mask(
        RATIO, patch_size=PATCH_SIZE, img_size=IMG_SIZE, num_frames=NUM_FRAMES,
        frame_idx=FRAME_IDX, trial_seed=TRIAL_SEED)
    noise_rand, gratio_rand, idx_rand = build_random_noise_mask(
        RATIO, patch_size=PATCH_SIZE, img_size=IMG_SIZE, num_frames=NUM_FRAMES,
        frame_idx=FRAME_IDX, trial_seed=TRIAL_SEED, n_summer_masked=len(idx_block))
    assert len(idx_block) == len(idx_rand) == 39, \
        f"expected 39 patches each, got block={len(idx_block)} scattered={len(idx_rand)}"

    grid_block = masked_patch_grid(idx_block, FRAME_IDX, GRID, TOKENS_PER_FRAME)
    grid_rand  = masked_patch_grid(idx_rand,  FRAME_IDX, GRID, TOKENS_PER_FRAME)

    # --- spectral masks, count-matched at exactly 58 bands ---
    band_contig  = contiguous_band_mask(RATIO)
    band_scatter = scattered_band_mask(RATIO)
    assert len(band_contig) == len(band_scatter) == 58, \
        f"expected 58 bands each, got contiguous={len(band_contig)} scattered={len(band_scatter)}"

    # --- assemble figure: 2 rows x 2 cols, single-column width ---
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.8),
                              gridspec_kw={"height_ratios": [2.2, 1]})
    draw_patch_grid(axes[0, 0], grid_block, "Contiguous")
    draw_patch_grid(axes[0, 1], grid_rand,  "Scattered")
    draw_band_strip(axes[1, 0], band_contig,  "Contiguous")
    draw_band_strip(axes[1, 1], band_scatter, "Scattered")

    fig.tight_layout()
    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig1_geometries.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
