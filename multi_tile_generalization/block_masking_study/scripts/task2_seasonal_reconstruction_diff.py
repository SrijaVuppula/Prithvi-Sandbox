"""
Task 2: For 2-3 high-seasonal-variation chips, compute:
  BEFORE = natural pixel diff between spring/summer/fall (no masking involved)
  AFTER  = reconstruction error (masked-frame reconstruction vs true summer)
at each standard mask ratio, both maskers, 600M backbone.
"""
import sys
import os
import json
import numpy as np
import torch
import rasterio

sys.path.insert(0, ".")
sys.path.insert(0, "multi_tile_generalization/block_masking_study/masking")

from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

CHIPS = ["chip_075_347", "chip_282_263", "chip_064_354"]
CHIP_DIR = "multi_tile_generalization/study_chips_500"
BACKBONE_DIR = os.path.expanduser("~/Prithvi/prithvi_600M")
CHECKPOINT_FILENAME = "Prithvi_EO_V2_600M_TL.pt"

RATIOS = [0.20, 0.40, 0.60, 0.80]
FRAME_IDX = 1
N_FRAMES = 3
IMG_SIZE = 224
N_TRIALS = 5
SEED = 42

# --- sanity checks before touching the GPU ---
for chip in CHIPS:
    p = os.path.join(CHIP_DIR, f"{chip}_merged.tif")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing chip file: {p}")
cfg_path = os.path.join(BACKBONE_DIR, "config.json")
ckpt_path = os.path.join(BACKBONE_DIR, CHECKPOINT_FILENAME)
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"Missing config: {cfg_path} — check BACKBONE_DIR path")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Missing checkpoint: {ckpt_path} — check CHECKPOINT_FILENAME")


def load_chip(path):
    with rasterio.open(path) as src:
        data = src.read()
    return data.reshape(3, 6, 224, 224).astype(np.float32)


def normalise_chip(chip_np, mean, std):
    normed = (chip_np - mean[None, :, None, None]) / std[None, :, None, None]
    t = torch.tensor(normed, dtype=torch.float32)
    return t.permute(1, 0, 2, 3)  # (C, T, H, W)


def run_forward(model, chip_norm, noise, mask_ratio, mean_hls, std_hls, device, frame_idx=1):
    x = chip_norm.unsqueeze(0).to(device)
    noise_dev = noise.to(device)
    _, _, rec_img, _, _ = run_masked_forward(
        model=model, x=x,
        temporal_coords=None, location_coords=None,
        mask_ratio=mask_ratio, noise=noise_dev,
    )
    pred_norm = rec_img[0, :, frame_idx, :, :]
    mean_t = torch.tensor(mean_hls, dtype=torch.float32).reshape(-1, 1, 1)
    std_t = torch.tensor(std_hls, dtype=torch.float32).reshape(-1, 1, 1)
    pred_unit = torch.clamp((pred_norm * std_t + mean_t) / 10000.0, 0.0, 1.0)
    return pred_unit.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg_path) as f:
        cfg = json.load(f)
    pcfg = cfg["pretrained_cfg"]
    mean_hls = np.array(pcfg["mean"], dtype=np.float32)
    std_hls = np.array(pcfg["std"], dtype=np.float32)
    patch_size = pcfg["patch_size"][1]

    print("Loading 600M model...")
    model, _, _, _, _ = load_prithvi_from_terratorch(
        backbone_name="prithvi_eo_v2_600",
        base_dir=BACKBONE_DIR,
        checkpoint_filename=CHECKPOINT_FILENAME,
        num_frames=3,
        device=device,
    )
    model.eval()

    for chip_name in CHIPS:
        chip_path = os.path.join(CHIP_DIR, f"{chip_name}_merged.tif")
        chip_np = load_chip(chip_path)
        spring, summer, fall = chip_np[0], chip_np[1], chip_np[2]

        spring_01 = np.clip(spring / 10000.0, 0, 1)
        summer_01 = np.clip(summer / 10000.0, 0, 1)
        fall_01 = np.clip(fall / 10000.0, 0, 1)

        before_ss = np.mean(np.abs(summer_01 - spring_01))
        before_sf = np.mean(np.abs(fall_01 - summer_01))

        chip_norm = normalise_chip(chip_np, mean_hls, std_hls)

        print(f"\n=== {chip_name} ===")
        print(f"  BEFORE (natural diff, whole frame): spring->summer={before_ss:.4f}  summer->fall={before_sf:.4f}")

        for ratio in RATIOS:
            block_errs, rand_errs = [], []
            for t in range(N_TRIALS):
                trial_seed = SEED + int(ratio * 100) * 1000 + t
                noise_block, gratio_block, idx_block = build_block_noise_mask(
                    ratio, patch_size=patch_size, img_size=IMG_SIZE, num_frames=N_FRAMES,
                    frame_idx=FRAME_IDX, trial_seed=trial_seed)
                noise_rand, gratio_rand, idx_rand = build_random_noise_mask(
                    ratio, patch_size=patch_size, img_size=IMG_SIZE, num_frames=N_FRAMES,
                    frame_idx=FRAME_IDX, trial_seed=trial_seed, n_summer_masked=len(idx_block))

                with torch.no_grad():
                    recon_block = run_forward(model, chip_norm, noise_block.unsqueeze(0), gratio_block,
                                               mean_hls, std_hls, device, FRAME_IDX)
                    recon_rand = run_forward(model, chip_norm, noise_rand.unsqueeze(0), gratio_rand,
                                              mean_hls, std_hls, device, FRAME_IDX)

                block_errs.append(np.mean(np.abs(summer_01 - recon_block)))
                rand_errs.append(np.mean(np.abs(summer_01 - recon_rand)))

            block_mean, block_std = np.mean(block_errs), np.std(block_errs)
            rand_mean, rand_std = np.mean(rand_errs), np.std(rand_errs)

            print(f"  ratio={int(ratio*100):>2}% | AFTER block={block_mean:.4f}+/-{block_std:.4f}"
                  f" | AFTER random={rand_mean:.4f}+/-{rand_std:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
