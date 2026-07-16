"""
visualize_error_maps.py

Generates 4 separate figures (one per mask ratio: 20%, 40%, 60%, 80%).
Each figure: 2 rows (Random, Block) x 3 columns (Ground Truth, Reconstructed RGB, Error Table).
Error Table = 16x16 grid of colored cells with per-patch error value printed inside.

FIXED 2026-07-16: masking now uses temporal_gap_masker.build_block_noise_mask /
build_random_noise_mask instead of this script's own local noise builders (old
Bug 1: nominal ratio passed straight through to TerraTorch's GLOBAL len_keep,
letting spring/fall context erode as ratio rose).

MULTI-TRIAL 2026-07-16: single-trial PSNR is noisy and can show a misleading
crossover on one chip (flagged after the original Session 11 run showed block
"easier" than random at 20% on chip_217_425 -- consistent with the population-
level crossover artifact, not necessarily real). Now runs --n_trials (default
50, matching the project's adopted standard for random-masking variance) per
ratio per chip, reporting mean+-std PSNR/delta across trials. The rendered
images/error tables still come from ONE representative trial (trial 0) --
this script's purpose is a concrete visual example, not a distribution plot.
"""

import argparse
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import torch
import rasterio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "masking"))

from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

RATIOS       = [0.20, 0.40, 0.60, 0.80]
RATIO_LABELS = ["20", "40", "60", "80"]
RGB_BANDS    = [2, 1, 0]
FRAME_IDX    = 1
N_FRAMES     = 3
IMG_SIZE     = 224
SEED         = 42


def load_chip(chip_path):
    with rasterio.open(chip_path) as src:
        data = src.read()
    return data.reshape(3, 6, 224, 224).astype(np.float32)


def normalise_chip(chip_np, mean, std):
    normed = (chip_np - mean[None, :, None, None]) / std[None, :, None, None]
    t = torch.tensor(normed, dtype=torch.float32)
    return t.permute(1, 0, 2, 3)  # (C, T, H, W)


def make_rgb(frame_np, percentile=2):
    rgb = frame_np[RGB_BANDS].astype(np.float32)
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(3):
        lo = np.percentile(rgb[i], percentile)
        hi = np.percentile(rgb[i], 100 - percentile)
        out[:, :, i] = (np.clip((rgb[i] - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
    return out


def bbox_from_masked_global(masked_global, frame_idx, grid, tokens_per_frame):
    """Recover (top, left, bh, bw) in patch-grid coords from the masker's
    global token indices. Only valid for a contiguous rectangular block."""
    offset = frame_idx * tokens_per_frame
    local = (masked_global - offset).cpu().numpy()
    rows = local // grid
    cols = local % grid
    top, left = int(rows.min()), int(cols.min())
    bh, bw = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
    return top, left, bh, bw


def run_forward(model, chip_norm, noise, mask_ratio, mean_hls, std_hls, device, frame_idx=1):
    x         = chip_norm.unsqueeze(0).to(device)
    noise_dev = noise.to(device)
    _, _, rec_img, _, _ = run_masked_forward(
        model=model, x=x,
        temporal_coords=None, location_coords=None,
        mask_ratio=mask_ratio, noise=noise_dev,
    )
    pred_norm = rec_img[0, :, frame_idx, :, :]
    mean_t    = torch.tensor(mean_hls, dtype=torch.float32).reshape(-1, 1, 1)
    std_t     = torch.tensor(std_hls,  dtype=torch.float32).reshape(-1, 1, 1)
    pred_unit = torch.clamp((pred_norm * std_t + mean_t) / 10000.0, 0.0, 1.0)
    return pred_unit.numpy()  # (C, H, W) in [0,1]


def compute_psnr(err_map):
    mse = np.mean(err_map ** 2)
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def compute_patch_errors(err_hw, patch_size, grid):
    """Compute mean error per patch cell → (grid, grid) array."""
    table = np.zeros((grid, grid), dtype=np.float32)
    for pr in range(grid):
        for pc in range(grid):
            py, px = pr * patch_size, pc * patch_size
            table[pr, pc] = err_hw[py:py+patch_size, px:px+patch_size].mean()
    return table


def draw_error_table(ax, table, vmax, patch_size, block_rect=None, title=""):
    """
    Draw a grid-of-cells table where each cell is colored by error value
    and has the numeric value printed inside.
    block_rect: (top, left, bh, bw) in patch-grid coords
    title: full pre-built title string (caller controls content/lines).
    """
    grid = table.shape[0]
    cmap = plt.cm.hot
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    ax.set_xlim(0, grid)
    ax.set_ylim(grid, 0)  # top-to-bottom
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")

    for pr in range(grid):
        for pc in range(grid):
            val  = table[pr, pc]
            color = cmap(norm(val))

            rect = mpatches.FancyBboxPatch(
                (pc, pr), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="#1a1a2e",
                linewidth=0.4,
            )
            ax.add_patch(rect)

            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_color  = "white" if brightness < 0.55 else "black"

            ax.text(pc + 0.5, pr + 0.5, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=3.8, color=txt_color, fontweight="bold",
                    path_effects=[
                        pe.withStroke(
                            linewidth=0.5,
                            foreground="black" if txt_color=="white" else "white"
                        )
                    ])

    if block_rect is not None:
        bt, bl, bh, bw = block_rect
        border = mpatches.FancyBboxPatch(
            (bl, bt), bw, bh,
            boxstyle="square,pad=0",
            facecolor="none",
            edgecolor="cyan",
            linewidth=2.5,
        )
        ax.add_patch(border)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, format="%.3f")
    cb.ax.tick_params(colors="white", labelsize=7)

    ax.set_title(title, color="yellow", fontsize=10, fontweight="bold", pad=6)


def draw_rgb(ax, rgb_img, title="", block_rect=None, patch_size=14):
    ax.imshow(rgb_img)
    if block_rect is not None:
        bt, bl, bh, bw = block_rect
        rect = plt.Rectangle(
            (bl * patch_size, bt * patch_size), bw * patch_size, bh * patch_size,
            linewidth=2.5, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Trials per ratio: {args.n_trials}")

    chip_np      = load_chip(args.chip_path)
    ground_truth = chip_np[FRAME_IDX]

    with open(Path(args.backbone_dir) / "config.json") as f:
        cfg = json.load(f)
    pcfg       = cfg["pretrained_cfg"]
    mean_hls   = np.array(pcfg["mean"], dtype=np.float32)
    std_hls    = np.array(pcfg["std"],  dtype=np.float32)
    patch_size = pcfg["patch_size"][1]
    grid       = 224 // patch_size
    tokens_per_frame = grid * grid

    print("Loading 600M model...")
    model, _, _, _, _ = load_prithvi_from_terratorch(
        backbone_name       = "prithvi_eo_v2_600",
        base_dir            = args.backbone_dir,
        checkpoint_filename = "Prithvi_EO_V2_600M_TL.pt",
        num_frames          = 3,
        device              = device,
    )
    model.eval()

    chip_norm = normalise_chip(chip_np, mean_hls, std_hls)
    gt_01     = np.clip(ground_truth / 10000.0, 0, 1)
    gt_rgb    = make_rgb(ground_truth)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row_idx, (ratio, label) in enumerate(zip(RATIOS, RATIO_LABELS)):
        print(f"\nGenerating figure for {label}% masking ({args.n_trials} trials)...")

        rand_psnrs, block_psnrs, deltas = [], [], []
        rep = None  # representative trial (trial 0) data, for drawing

        for t in range(args.n_trials):
            trial_seed = SEED + row_idx * 10000 + t

            noise_block, gratio_block, idx_block = build_block_noise_mask(
                ratio, patch_size=patch_size, img_size=IMG_SIZE, num_frames=N_FRAMES,
                frame_idx=FRAME_IDX, trial_seed=trial_seed)
            noise_rand, gratio_rand, idx_rand = build_random_noise_mask(
                ratio, patch_size=patch_size, img_size=IMG_SIZE, num_frames=N_FRAMES,
                frame_idx=FRAME_IDX, trial_seed=trial_seed, n_summer_masked=len(idx_block))
            assert abs(gratio_block - gratio_rand) < 1e-9, "block/random ratio mismatch"

            with torch.no_grad():
                recon_rand = run_forward(model, chip_norm, noise_rand.unsqueeze(0), gratio_rand,
                                         mean_hls, std_hls, device, FRAME_IDX)
            err_rand  = np.mean(np.abs(gt_01 - recon_rand), axis=0)
            psnr_rand = compute_psnr(err_rand)

            with torch.no_grad():
                recon_block = run_forward(model, chip_norm, noise_block.unsqueeze(0), gratio_block,
                                          mean_hls, std_hls, device, FRAME_IDX)
            err_block  = np.mean(np.abs(gt_01 - recon_block), axis=0)
            psnr_block = compute_psnr(err_block)

            rand_psnrs.append(psnr_rand)
            block_psnrs.append(psnr_block)
            deltas.append(psnr_rand - psnr_block)

            if t == 0:
                bt, bl, bh, bw = bbox_from_masked_global(idx_block, FRAME_IDX, grid, tokens_per_frame)
                rep = dict(
                    recon_rand=recon_rand, recon_block=recon_block,
                    table_rand=compute_patch_errors(err_rand, patch_size, grid),
                    table_block=compute_patch_errors(err_block, patch_size, grid),
                    psnr_rand=psnr_rand, psnr_block=psnr_block,
                    bt=bt, bl=bl, bh=bh, bw=bw,
                )

        rand_arr  = np.array(rand_psnrs)
        block_arr = np.array(block_psnrs)
        delta_arr = np.array(deltas)
        pct_block_harder = (delta_arr > 0).mean() * 100

        print(f"  random = {rand_arr.mean():.2f}±{rand_arr.std():.2f} dB, "
              f"block = {block_arr.mean():.2f}±{block_arr.std():.2f} dB, "
              f"delta = {delta_arr.mean():+.2f}±{delta_arr.std():.2f} dB, "
              f"{pct_block_harder:.0f}% of trials block harder")

        vmax = max(rep["table_rand"].max(), rep["table_block"].max(), 0.001)

        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor("#1a1a2e")

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3, figure=fig,
                               width_ratios=[1, 1, 1.35],
                               hspace=0.4, wspace=0.15)

        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

        fig.suptitle(
            f"600M  —  Mask Ratio {label}%  |  Random (top) vs Block (bottom)\n"
            f"Ground truth / reconstruction / error table from representative trial 0  |  "
            f"cyan = block boundary  |  corrected masking: spring/fall 100% visible, matched patch count",
            color="white", fontsize=12, fontweight="bold", y=1.01
        )

        row_labels = ["Random Masking", "Block Masking"]
        for r, rl in enumerate(row_labels):
            axes[r][0].set_ylabel(rl, color="white", fontsize=13,
                                  fontweight="bold", rotation=90, labelpad=10)

        title_rand = (f"Random — trial 0: {rep['psnr_rand']:.1f} dB\n"
                      f"mean±std (n={args.n_trials}): {rand_arr.mean():.1f}±{rand_arr.std():.1f} dB")
        title_block = (f"Block — trial 0: {rep['psnr_block']:.1f} dB\n"
                       f"mean±std (n={args.n_trials}): {block_arr.mean():.1f}±{block_arr.std():.1f} dB  "
                       f"({pct_block_harder:.0f}% trials block harder)")

        # Row 0 — Random
        draw_rgb(axes[0][0], gt_rgb, title="Ground Truth")
        draw_rgb(axes[0][1], make_rgb(rep["recon_rand"] * 10000.0), title="Reconstructed (trial 0)")
        draw_error_table(axes[0][2], rep["table_rand"], vmax, patch_size,
                         block_rect=None, title=title_rand)

        # Row 1 — Block
        draw_rgb(axes[1][0], gt_rgb, title="Ground Truth")
        draw_rgb(axes[1][1], make_rgb(rep["recon_block"] * 10000.0), title="Reconstructed (trial 0)",
                 block_rect=(rep["bt"], rep["bl"], rep["bh"], rep["bw"]), patch_size=patch_size)
        draw_error_table(axes[1][2], rep["table_block"], vmax, patch_size,
                         block_rect=(rep["bt"], rep["bl"], rep["bh"], rep["bw"]), title=title_block)

        out_path = out_dir / f"error_table_mask{label}pct.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved -> {out_path}")

    print("\nAll figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip_path",    required=True)
    parser.add_argument("--backbone_dir", required=True)
    parser.add_argument("--output_dir",   default="outputs/figures")
    parser.add_argument("--n_trials",     type=int, default=50)
    args = parser.parse_args()
    main(args)
