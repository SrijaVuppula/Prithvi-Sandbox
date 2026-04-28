"""
run_block_masking.py
--------------------
Master runner for the block masking study.
500 chips x 4 ratios x 4 backbones x 5 trials = 40,000 forward passes.
Results flushed to CSV after every row so no progress is lost on crash.

Usage
-----
  cd ~/Prithvi/Prithvi-Sandbox
  source ~/.venv/bin/activate
  python multi_tile_generalization/block_masking_study/scripts/run_block_masking.py
"""

import csv
import json
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR  = SCRIPT_DIR.parent
MTG_DIR    = STUDY_DIR.parent
REPO_ROOT  = MTG_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MTG_DIR))
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR / 'masking'))
sys.path.insert(0, str(STUDY_DIR / 'metrics'))

from block_masker import build_block_noise_mask, block_mask_to_pixel_map
from evaluate_block_masked import compute_block_metrics
from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = STUDY_DIR / "config" / "block_masking_config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def resolve(path_str):
    return Path(os.path.expanduser(path_str))

# ---------------------------------------------------------------------------
# Normalise / denormalise
# ---------------------------------------------------------------------------
def normalise_chip(chip_raw, mean, std):
    m = torch.tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
    s = torch.tensor(std,  dtype=torch.float32).reshape(1, -1, 1, 1)
    return (chip_raw - m) / s

def denormalise(tensor, mean, std):
    m = torch.tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
    s = torch.tensor(std,  dtype=torch.float32).reshape(-1, 1, 1)
    return tensor * s + m

def to_unit(tensor_hls):
    arr = tensor_hls.numpy() if isinstance(tensor_hls, torch.Tensor) else tensor_hls
    return np.clip(arr / 10000.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Forward pass with block noise injected
# ---------------------------------------------------------------------------
def run_forward(model, chip_norm, noise, device, mask_ratio, mean_hls, std_hls, frame_idx=1):
    x = chip_norm.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]
    noise_dev = noise.unsqueeze(0).to(device)
    loss, pred_img, rec_img, mask_img, x_cpu = run_masked_forward(
        model=model,
        x=x,
        temporal_coords=None,
        location_coords=None,
        mask_ratio=mask_ratio,
        noise=noise_dev,
    )
    # rec_img: [1, C, T, H, W] normalised — extract middle frame and denormalize
    pred_norm = rec_img[0, :, frame_idx, :, :]          # [C, H, W]
    mean_t = torch.tensor(mean_hls, dtype=torch.float32).reshape(-1, 1, 1)
    std_t  = torch.tensor(std_hls,  dtype=torch.float32).reshape(-1, 1, 1)
    pred_unit = torch.clamp((pred_norm * std_t + mean_t) / 10000.0, 0.0, 1.0)
    return pred_unit.numpy()  # [C, H, W] in [0,1]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg    = load_config()
    device = torch.device(cfg["compute"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir   = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    chips_dir  = resolve(cfg["data"]["chips_dir"])
    n_chips    = cfg["data"]["n_chips"]
    chip_seed  = cfg["data"]["chip_seed"]
    img_size   = cfg["data"]["img_size"]
    n_frames   = cfg["data"]["n_frames"]
    frame_idx  = cfg["data"]["masked_frame_idx"]
    n_bands    = cfg["data"]["n_bands"]
    ratios     = cfg["masking"]["ratios"]
    n_trials   = cfg["masking"]["trials_per_chip"]
    backbones  = cfg["backbones"]

    all_chips = sorted([p for p in chips_dir.glob("chip_*.tif")
                        if not p.name.startswith("._")])
    rng = random.Random(chip_seed)
    sampled = rng.sample(all_chips, min(n_chips, len(all_chips)))
    print(f"Chips available: {len(all_chips)},  sampled: {len(sampled)}")

    import rasterio

    for bb_name, bb_cfg in backbones.items():
        ckpt       = resolve(bb_cfg["checkpoint"])
        config_f   = resolve(bb_cfg["config"])
        patch_size = bb_cfg["patch_size"]

        print(f"\n{'='*60}")
        print(f"Backbone: {bb_name}  |  patch_size={patch_size}")
        print(f"{'='*60}")

        model, _, mean_hls, std_hls, _ = load_prithvi_from_terratorch(
            backbone_name=bb_name,
            base_dir=ckpt.parent,
            checkpoint_filename=ckpt.name,
            num_frames=n_frames,
            device=device,
        )

        mean_hls = np.array(mean_hls)
        std_hls  = np.array(std_hls)
        if len(mean_hls) > n_bands:
            mean_hls = mean_hls[:n_bands]
            std_hls  = std_hls[:n_bands]

        csv_path   = out_dir / f"results_{bb_name}.csv"
        fieldnames = [
            "backbone", "chip", "mask_ratio", "trial", "trial_seed",
            "block_h_patches", "block_w_patches", "block_area_frac",
            "global_mae", "global_psnr", "global_ssim",
            "block_mae",  "block_psnr",  "block_ssim",
        ]

        csv_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()

            total = len(sampled) * len(ratios) * n_trials
            done  = 0

            for chip_path in sampled:
                try:
                    with rasterio.open(chip_path) as src:
                        data = src.read()
                    chip_raw = torch.tensor(
                        data.reshape(n_frames, n_bands, img_size, img_size),
                        dtype=torch.float32,
                    )
                except Exception as e:
                    print(f"  SKIP {chip_path.name}: {e}")
                    continue

                gt_unit   = to_unit(chip_raw[frame_idx])
                chip_norm = normalise_chip(chip_raw, mean_hls, std_hls)

                for ratio in ratios:
                    for trial in range(n_trials):
                        trial_seed = hash((chip_path.name, ratio, trial)) % (2**31)

                        try:
                            noise, _, _ = build_block_noise_mask(
                                mask_ratio=ratio,
                                patch_size=patch_size,
                                img_size=img_size,
                                num_frames=n_frames,
                                frame_idx=frame_idx,
                                trial_seed=trial_seed,
                            )
                            pixel_mask = block_mask_to_pixel_map(
                                noise, patch_size, img_size, n_frames, frame_idx
                            )

                            recon_unit = run_forward(model, chip_norm, noise, device, ratio, mean_hls, std_hls, frame_idx)

                            metrics = compute_block_metrics(
                                recon_unit, gt_unit, pixel_mask.numpy()
                            )

                            rows_m = int(pixel_mask.any(dim=1).sum().item()) // patch_size
                            cols_m = int(pixel_mask.any(dim=0).sum().item()) // patch_size

                            writer.writerow({
                                "backbone":        bb_name,
                                "chip":            chip_path.name,
                                "mask_ratio":      ratio,
                                "trial":           trial,
                                "trial_seed":      trial_seed,
                                "block_h_patches": rows_m,
                                "block_w_patches": cols_m,
                                "block_area_frac": round(metrics["block_area_frac"], 4),
                                "global_mae":      round(metrics["global_mae"],  6),
                                "global_psnr":     round(metrics["global_psnr"], 4),
                                "global_ssim":     round(metrics["global_ssim"], 4),
                                "block_mae":       round(metrics["block_mae"],   6),
                                "block_psnr":      round(metrics["block_psnr"],  4),
                                "block_ssim":      round(metrics["block_ssim"],  4),
                            })
                            fh.flush()

                        except Exception as e:
                            print(f"  ERROR {chip_path.name} ratio={ratio} "
                                  f"trial={trial}: {e}")

                        done += 1
                        if done % 100 == 0:
                            print(f"  [{bb_name}] {done}/{total} done")

        print(f"  Saved: {csv_path}")

    print("\nAll backbones complete.")


if __name__ == "__main__":
    main()
