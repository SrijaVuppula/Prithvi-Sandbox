"""
resume_block_masking.py
-----------------------
Resumes the block masking experiment from where it stopped.
Skips chips already present in the output CSVs and appends new results.

Run from repo root:
    tmux new -s block_resume
    source ~/.venv/bin/activate
    python multi_tile_generalization/block_masking_study/scripts/resume_block_masking.py
"""
import csv
import os
import sys
import random
from pathlib import Path
import numpy as np
import torch
import yaml
import rasterio

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

CONFIG_PATH = STUDY_DIR / "config" / "block_masking_config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def resolve(path_str):
    return Path(os.path.expanduser(path_str))

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

def run_forward(model, chip_norm, noise, device, mask_ratio, mean_hls, std_hls, frame_idx=1):
    x = chip_norm.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    noise_dev = noise.unsqueeze(0).to(device)
    loss, pred_img, rec_img, mask_img, x_cpu = run_masked_forward(
        model=model,
        x=x,
        temporal_coords=None,
        location_coords=None,
        mask_ratio=mask_ratio,
        noise=noise_dev,
    )
    pred_norm = rec_img[0, :, frame_idx, :, :]
    mean_t = torch.tensor(mean_hls, dtype=torch.float32).reshape(-1, 1, 1)
    std_t  = torch.tensor(std_hls,  dtype=torch.float32).reshape(-1, 1, 1)
    return torch.clamp((pred_norm * std_t + mean_t) / 10000.0, 0.0, 1.0)

def get_completed_chips(csv_path):
    """Return set of chip filenames already in the CSV."""
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:
                completed.add(parts[1])
    return completed

def main():
    cfg       = load_config()
    device    = torch.device(cfg["compute"]["device"] if torch.cuda.is_available() else "cpu")
    chips_dir = resolve(cfg["data"]["chips_dir"])
    n_chips   = cfg["data"]["n_chips"]
    chip_seed = cfg["data"]["chip_seed"]
    ratios    = cfg["masking"]["ratios"]
    n_trials  = cfg["masking"]["trials_per_chip"]
    frame_idx = 1
    out_dir   = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct original sample order
    all_chips = sorted([p for p in chips_dir.glob("chip_*.tif") if "merged" in p.name])
    rng       = random.Random(chip_seed)
    sampled   = rng.sample(all_chips, min(n_chips, len(all_chips)))
    print(f"Total sampled: {len(sampled)}")

    backbones = cfg["backbones"]

    for bb_name, bb_cfg in backbones.items():
        csv_path  = out_dir / f"results_{bb_name}.csv"
        completed = get_completed_chips(csv_path)
        remaining = [p for p in sampled if p.name not in completed]

        print(f"\n── {bb_name} ── completed: {len(completed)}, remaining: {len(remaining)}")
        if not remaining:
            print(f"  {bb_name} already complete, skipping.")
            continue

        checkpoint = resolve(bb_cfg["checkpoint"])
        config     = resolve(bb_cfg["config"])

        print(f"  Loading model {bb_name}...")
        model, _, mean_hls, std_hls, _ = load_prithvi_from_terratorch(
            backbone_name=bb_name,
            base_dir=str(checkpoint.parent),
            checkpoint_filename=checkpoint.name,
            num_frames=3,
            device=str(device),
        )
        model.eval()

        fieldnames = [
            "backbone", "chip", "mask_ratio", "trial", "trial_seed",
            "block_h_patches", "block_w_patches", "block_area_frac",
            "global_mae", "global_psnr", "global_ssim",
            "block_mae", "block_psnr", "block_ssim",
        ]

        # Open CSV in append mode
        with open(csv_path, "a", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            # Write header only if file is new
            if csv_path.stat().st_size == 0:
                writer.writeheader()

            total = len(remaining) * len(ratios) * n_trials
            done  = 0

            for chip_path in remaining:
                try:
                    with rasterio.open(chip_path) as src:
                        data = src.read()
                    chip_raw = torch.tensor(
                        data.reshape(3, 6, 224, 224).astype("float32"),
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
                                patch_size=bb_cfg["patch_size"],
                                img_size=224,
                                num_frames=3,
                                frame_idx=frame_idx,
                                trial_seed=trial_seed,
                            )
                            pixel_map = block_mask_to_pixel_map(
                                noise, bb_cfg["patch_size"], 224, 3, frame_idx
                            )
                            rows_m = int(pixel_map.any(dim=1).sum().item()) // bb_cfg["patch_size"]
                            cols_m = int(pixel_map.any(dim=0).sum().item()) // bb_cfg["patch_size"]
                            patch_size = bb_cfg["patch_size"]
                            grid = 224 // patch_size
                            actual_ratio = (rows_m * cols_m) / (grid * grid)
                            recon_unit   = run_forward(
                                model, chip_norm, noise, device,
                                actual_ratio, mean_hls, std_hls, frame_idx,
                            )
                            metrics = compute_block_metrics(recon_unit, gt_unit, pixel_map.numpy())
                            writer.writerow({
                                "backbone":        bb_name,
                                "chip":            chip_path.name,
                                "mask_ratio":      ratio,
                                "trial":           trial,
                                "trial_seed":      trial_seed,
                                "block_h_patches": rows_m,
                                "block_w_patches": cols_m,
                                "block_area_frac": round(actual_ratio, 3),
                                "global_mae":      round(metrics["global_mae"],  6),
                                "global_psnr":     round(metrics["global_psnr"], 4),
                                "global_ssim":     round(metrics["global_ssim"], 4),
                                "block_mae":       round(metrics["block_mae"],   6),
                                "block_psnr":      round(metrics["block_psnr"],  4),
                                "block_ssim":      round(metrics["block_ssim"],  4),
                            })
                            fout.flush()
                            done += 1
                            if done % 200 == 0:
                                print(f"  [{bb_name}] {done}/{total} rows written")
                        except Exception as e:
                            print(f"  ERROR {chip_path.name} ratio={ratio} trial={trial}: {e}")

        print(f"  {bb_name} done. CSV: {csv_path}")

    print("\nAll backbones complete.")

if __name__ == "__main__":
    main()
