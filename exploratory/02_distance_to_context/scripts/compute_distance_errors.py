"""
Experiment 2 — Distance-to-Context Error Analysis
For each masked patch inside the block, records its distance to the nearest
visible patch and its reconstruction MAE. Saves results to CSV.
Run: cd ~/Prithvi/Prithvi-Sandbox && ~/.venv/bin/python exploratory/02_distance_to_context/scripts/compute_distance_errors.py
"""

import sys, os
import numpy as np
import pandas as pd
import torch
import csv

REPO = os.path.expanduser("~/Prithvi/Prithvi-Sandbox")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "patch_masking_study"))
sys.path.insert(0, os.path.join(REPO, "multi_tile_generalization/block_masking_study/masking"))
sys.path.insert(0, os.path.join(REPO, "multi_tile_generalization"))
sys.path.insert(0, os.path.join(REPO, "multi_tile_generalization", "block_masking_study"))
sys.path.insert(0, os.path.join(REPO, "multi_tile_generalization"))

import rasterio

def load_chip(path):
    """Load merged.tif → numpy (T, C, H, W) in raw HLS scale."""
    with rasterio.open(path) as src:
        data = src.read()  # (18, 224, 224)
    return data.reshape(3, 6, 224, 224).astype('float32')

from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from block_masker import build_block_noise_mask

# ── Config ─────────────────────────────────────────────────────────────────────
BACKBONE_DIR = os.path.expanduser("~/Prithvi/prithvi_600M")
CHIP_DIR     = os.path.join(REPO, "multi_tile_generalization/study_chips_500")
OUT_DIR      = os.path.join(REPO, "exploratory/02_distance_to_context/outputs")
OUT_CSV      = os.path.join(OUT_DIR, "distance_errors.csv")
os.makedirs(OUT_DIR, exist_ok=True)

RATIOS      = [0.20, 0.40, 0.60, 0.80]
N_TRIALS    = 3
PATCH_SIZE  = 14   # 600M uses 14×14 patches
GRID_SIZE   = 16   # 224 / 14 = 16
MIDDLE_FRAME = 1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model once ────────────────────────────────────────────────────────────
print("Loading 600M model...")
model, bands, mean, std, _ = load_prithvi_from_terratorch(
    backbone_name="prithvi_eo_v2_600",
    base_dir=BACKBONE_DIR,
    checkpoint_filename="Prithvi_EO_V2_600M_TL.pt",
    num_frames=3,
    device=DEVICE,
)
model.eval()
mean = torch.tensor(mean, device=DEVICE).view(1, len(mean), 1, 1, 1)
std  = torch.tensor(std,  device=DEVICE).view(1, len(std),  1, 1, 1)

# ── Chip list ──────────────────────────────────────────────────────────────────
chips = sorted([f for f in os.listdir(CHIP_DIR) if f.endswith("_merged.tif")])
print(f"Found {len(chips)} chips")

# ── Distance helper ────────────────────────────────────────────────────────────
def patch_distances(block_idx, grid=GRID_SIZE):
    """
    Given flat block patch indices on a grid×grid grid, returns a dict:
    { flat_idx: distance_to_nearest_visible_patch }
    Distance = min steps to reach a non-masked patch (Chebyshev in any direction).
    For a solid rectangular block this simplifies to:
      dist(r,c) = min(r-r0, r1-r, c-c0, c1-c) + 1
    """
    rows = block_idx // grid
    cols = block_idx % grid
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    dist_map = {}
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        d = min(r - r0, r1 - r, c - c0, c1 - c) + 1
        dist_map[r * grid + c] = d
    return dist_map

# ── Main loop ──────────────────────────────────────────────────────────────────
rows_written = 0
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["chip", "ratio_pct", "trial", "patch_idx", "distance", "patch_mae"])

    for chip_i, chip_file in enumerate(chips):
        chip_path = os.path.join(CHIP_DIR, chip_file)
        try:
            raw = load_chip(chip_path)           # (T, C, H, W) numpy, raw HLS scale
        except Exception as e:
            print(f"  Skip {chip_file}: {e}")
            continue

        # Normalise → tensor (1, C, T, H, W)
        x = torch.tensor(raw, dtype=torch.float32, device=DEVICE)  # (T,C,H,W)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)                      # (1,C,T,H,W)
        x_norm = (x - mean) / std

        # Ground truth middle frame in [0,1] (denormalised)
        gt_norm = x_norm[0, :, MIDDLE_FRAME, :, :]                   # (C,H,W)
        gt_raw  = (gt_norm * std[0,:,0,0,0].view(-1,1,1)
                   + mean[0,:,0,0,0].view(-1,1,1)) / 10000.0         # (C,H,W) in [0,1]
        gt_raw  = gt_raw.cpu().numpy()

        for ratio in RATIOS:
            ratio_pct = int(ratio * 100)
            for trial in range(N_TRIALS):
                # Build block mask
                num_patches = GRID_SIZE * GRID_SIZE   # 256 for 600M
                try:
                    noise, ids_keep, ids_restore = build_block_noise_mask(
                        mask_ratio=ratio,
                        patch_size=PATCH_SIZE,
                        img_size=224,
                        num_frames=3,
                        frame_idx=MIDDLE_FRAME,
                        trial_seed=trial,
                    )
                    # masked patches in middle frame have noise >= 2.0
                    frame_offset = MIDDLE_FRAME * (GRID_SIZE * GRID_SIZE)
                    frame_noise  = noise[frame_offset: frame_offset + GRID_SIZE * GRID_SIZE]
                    block_idx    = (frame_noise >= 2.0).nonzero(as_tuple=True)[0].numpy()
                    actual_ratio = len(block_idx) / (GRID_SIZE * GRID_SIZE)
                    noise        = noise.unsqueeze(0)  # run_masked_forward expects (B, total_tokens)
                except Exception as e:
                    print(f"  Mask error {chip_file} r={ratio_pct} t={trial}: {e}")
                    continue

                # Forward pass
                try:
                    with torch.no_grad():
                        _, _, recon_norm, _, _ = run_masked_forward(
                            model, x_norm,
                            temporal_coords=None,
                            location_coords=None,
                            mask_ratio=actual_ratio,
                            noise=noise,
                        )  # rec_img: composite recon, already on CPU  # (1, C, T, H, W) normalised
                except Exception as e:
                    print(f"  Fwd error {chip_file} r={ratio_pct} t={trial}: {e}")
                    continue

                # Denormalise reconstruction middle frame (rec_img is already on CPU)
                recon_mid = recon_norm[0, :, MIDDLE_FRAME, :, :]
                std_cpu   = std[0, :, 0, 0, 0].view(-1, 1, 1).cpu()
                mean_cpu  = mean[0, :, 0, 0, 0].view(-1, 1, 1).cpu()
                recon_raw = (recon_mid * std_cpu + mean_cpu) / 10000.0
                recon_raw = recon_raw.numpy()

                # Distance map for this block
                dist_map = patch_distances(block_idx)

                # Per-patch MAE for each masked patch
                for flat_idx, dist in dist_map.items():
                    r_p = (flat_idx // GRID_SIZE) * PATCH_SIZE
                    c_p = (flat_idx %  GRID_SIZE) * PATCH_SIZE
                    gt_patch    = gt_raw[:,   r_p:r_p+PATCH_SIZE, c_p:c_p+PATCH_SIZE]
                    recon_patch = recon_raw[:, r_p:r_p+PATCH_SIZE, c_p:c_p+PATCH_SIZE]
                    mae = float(np.abs(gt_patch - recon_patch).mean())
                    writer.writerow([chip_file, ratio_pct, trial,
                                     flat_idx, dist, mae])
                    rows_written += 1

        if (chip_i + 1) % 50 == 0:
            print(f"  {chip_i+1}/500 chips done — {rows_written} rows written")

print(f"\nDone. {rows_written} rows → {OUT_CSV}")
