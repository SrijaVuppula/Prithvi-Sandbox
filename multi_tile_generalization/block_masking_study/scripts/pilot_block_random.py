"""
pilot_block_random.py
------------------------
Fast paired block-vs-random pilot using the CORRECTED maskers
(spring/fall fully visible, matched summer count). 600M, 3 pilot chips,
N trials each. Computes PSNR over the masked summer pixels for both
maskers and reports delta = random - block per ratio.

Reads the crossover directly: delta > 0 => block harder; delta < 0 => random
harder. A sign flip from low to high ratio == the crossover survives the fix.

Writes: outputs/pilot_block_random.csv   (nothing old is touched)
Run from repo root with the venv active.
"""

import sys, csv
from pathlib import Path
import numpy as np
import torch
import rasterio

REPO  = Path("~/Prithvi/Prithvi-Sandbox").expanduser()
STUDY = REPO / "multi_tile_generalization" / "block_masking_study"
MASK  = STUDY / "masking"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patch_masking_study"))
sys.path.insert(0, str(MASK))

from terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import (
    build_block_noise_mask  as fixed_block,
    build_random_noise_mask as fixed_random,
    pixel_map,
)

BB, PATCH   = "600M", 14
BASE        = Path("~/Prithvi/prithvi_600M").expanduser()
CKPT        = "Prithvi_EO_V2_600M_TL.pt"
IMG, BANDS  = 224, 6
T, FRAME    = 3, 1
RATIOS      = [0.20, 0.40, 0.60, 0.80]
N_TRIALS    = 50            # Session-14 adopted standard (E=0.5 dB)
CHIPS       = ["chip_217_425", "chip_268_410", "chip_105_452"]  # complex / mid / simple
OUT_CSV     = STUDY / "outputs" / "pilot_block_random.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_chip(name):
    for folder in ["study_chips_500", "training_chips"]:
        d = REPO / "multi_tile_generalization" / folder
        hits = sorted(d.glob(f"{name}*.tif")) if d.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def load_norm_and_gt(path, mean, std):
    with rasterio.open(path) as src:
        data = src.read()
    raw = torch.tensor(data.reshape(T, BANDS, IMG, IMG), dtype=torch.float32)
    gt_unit = np.clip(raw[FRAME].numpy() / 10000.0, 0.0, 1.0)          # (C,H,W)
    m = torch.tensor(mean[:BANDS]).reshape(1, -1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(1, -1, 1, 1)
    x = ((raw - m) / s).permute(1, 0, 2, 3).unsqueeze(0).to(device)    # (1,C,T,H,W)
    return x, gt_unit


@torch.no_grad()
def recon_unit(x, noise, ratio_global, mean, std):
    noise_dev = noise.unsqueeze(0).to(device)
    _, _, rec_img, _, _ = run_masked_forward(
        model=model, x=x, temporal_coords=None, location_coords=None,
        mask_ratio=ratio_global, noise=noise_dev,
    )
    pred = rec_img[0, :, FRAME, :, :]                                  # (C,H,W) normalised
    m = torch.tensor(mean[:BANDS]).reshape(-1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(-1, 1, 1)
    return torch.clamp((pred * s + m) / 10000.0, 0.0, 1.0).numpy()


def masked_psnr(recon, gt, pmask):
    m = pmask.numpy() if isinstance(pmask, torch.Tensor) else pmask
    if m.sum() == 0:
        return float("nan")
    d = recon[:, m] - gt[:, m]
    mse = float(np.mean(d ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


print(f"Loading {BB} on {device} ...")
model, _, mean, std, sp = load_prithvi_from_terratorch(
    backbone_name=BB, base_dir=BASE, checkpoint_filename=CKPT,
    num_frames=T, device=device)
assert sp == PATCH

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
rows = []
summary = {}   # (chip, ratio) -> (block_psnr, random_psnr)

for name in CHIPS:
    path = find_chip(name)
    x, gt = load_norm_and_gt(path, mean, std)
    print(f"\n{name}  ({path.name})")
    for r in RATIOS:
        b_vals, rnd_vals = [], []
        for t in range(N_TRIALS):
            seed = (hash((name, r, t)) % (2**31))
            # fixed block
            nb, grb, idxb = fixed_block(r, PATCH, IMG, T, FRAME, trial_seed=seed)
            rb = recon_unit(x, nb, grb, mean, std)
            pm_b = pixel_map(idxb, PATCH, IMG, T, FRAME)
            b_vals.append(masked_psnr(rb, gt, pm_b))
            # fixed random, matched summer count
            nr, grr, idxr = fixed_random(r, PATCH, IMG, T, FRAME, trial_seed=seed,
                                         n_summer_masked=len(idxb))
            rr = recon_unit(x, nr, grr, mean, std)
            pm_r = pixel_map(idxr, PATCH, IMG, T, FRAME)
            rnd_vals.append(masked_psnr(rr, gt, pm_r))
            rows.append([name, r, t, b_vals[-1], rnd_vals[-1]])
        bp, rp = np.nanmean(b_vals), np.nanmean(rnd_vals)
        summary[(name, r)] = (bp, rp)
        print(f"  {int(r*100):>3}%  block={bp:6.2f}  random={rp:6.2f}  "
              f"delta(rand-block)={rp-bp:+6.2f}  "
              f"{'block harder' if rp>bp else 'random harder'}")

with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["chip", "mask_ratio", "trial", "block_psnr", "random_psnr"])
    w.writerows(rows)

print("\n===== CROSSOVER CHECK (delta = random - block, averaged) =====")
print(f"{'chip':>14} | " + " ".join(f"{int(r*100):>7}%" for r in RATIOS))
for name in CHIPS:
    deltas = [summary[(name, r)][1] - summary[(name, r)][0] for r in RATIOS]
    flip = any(deltas[i] > 0 and deltas[j] < 0 for i in range(len(deltas)) for j in range(i+1, len(deltas)))
    line = " ".join(f"{d:+7.2f}" for d in deltas)
    print(f"{name:>14} | {line}   {'<- SIGN FLIP (crossover present)' if flip else '<- no flip'}")
print(f"\nSaved raw trials to {OUT_CSV}")
print("Positive delta at low ratio + negative at high ratio = crossover survives the fix.")
