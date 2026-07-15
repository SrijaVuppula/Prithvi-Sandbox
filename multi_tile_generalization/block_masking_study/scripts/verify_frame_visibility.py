"""
verify_frame_visibility.py
--------------------------
Reads the ACTUAL token mask the encoder produces (out of _encode_with_noise)
for one chip on the 600M backbone, and reports per-frame visible % for:

  (1) OLD block masker  (block_masker.build_block_noise_mask, per-frame ratio)
  (2) FIXED block masker (temporal_gap_masker, global ratio)
  (3) FIXED random-matched masker (same summer count as the fixed block trial)

Expected outcome:
  * OLD  -> spring/fall visibility DROPS as ratio rises (reproduces the handoff
           table: at 80%, spring ~28%, summer ~7%, fall ~25%)
  * FIXED -> spring = fall = 100% at every ratio; summer masked at the nominal
           ratio; fixed block and fixed random mask the SAME summer count.

Run from repo root:
    cd ~/Prithvi/Prithvi-Sandbox
    source ~/.venv/bin/activate
    python multi_tile_generalization/block_masking_study/scripts/verify_frame_visibility.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import rasterio

# -- paths --------------------------------------------------------------------
REPO   = Path("~/Prithvi/Prithvi-Sandbox").expanduser()
STUDY  = REPO / "multi_tile_generalization" / "block_masking_study"
MASK   = STUDY / "masking"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patch_masking_study"))
sys.path.insert(0, str(MASK))

from terratorch_loader import load_prithvi_from_terratorch, _encode_with_noise
from block_masker import build_block_noise_mask as old_block            # OLD
from temporal_gap_masker import (                                 # FIXED
    build_block_noise_mask  as fixed_block,
    build_random_noise_mask as fixed_random,
)

# -- settings (600M, patch=14 -> 256 tokens/frame, 768 total) -----------------
BB          = "600M"
PATCH       = 14
BASE        = Path("~/Prithvi/prithvi_600M").expanduser()
CKPT        = "Prithvi_EO_V2_600M_TL.pt"
IMG, BANDS  = 224, 6
T, FRAME    = 3, 1                      # mask the middle (summer) frame
PPF         = (IMG // PATCH) ** 2       # 256
RATIOS      = [0.20, 0.40, 0.60, 0.80]
SEED        = 0
CHIP_NAME   = "chip_105_452"           # simple chip used in the handoff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_chip():
    for folder in ["study_chips_500", "training_chips"]:
        d = REPO / "multi_tile_generalization" / folder
        hits = sorted(d.glob(f"{CHIP_NAME}*.tif")) if d.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{CHIP_NAME} not found in study_chips_500/ or training_chips/")


def load_chip_norm(path, mean, std):
    with rasterio.open(path) as src:
        data = src.read()                                   # (18, 224, 224)
    raw = torch.tensor(data.reshape(T, BANDS, IMG, IMG), dtype=torch.float32)
    m = torch.tensor(mean[:BANDS]).reshape(1, -1, 1, 1)
    s = torch.tensor(std[:BANDS]).reshape(1, -1, 1, 1)
    norm = (raw - m) / s                                     # (T, C, H, W)
    return norm.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)


def frame_visibility(mask_row):
    """mask_row: (768,) with 1=masked, 0=kept. Return per-frame visible %."""
    vis = []
    for f in range(T):
        seg = mask_row[f * PPF:(f + 1) * PPF]
        vis.append((seg == 0).float().mean().item() * 100.0)
    summer_masked = int((mask_row[FRAME * PPF:(FRAME + 1) * PPF] == 1).sum().item())
    return vis, summer_masked


@torch.no_grad()
def measure(x, noise, mask_ratio):
    noise_dev = noise.unsqueeze(0).to(device)
    _, mask, _ = _encode_with_noise(
        model, x, temporal_coords=None, location_coords=None,
        mask_ratio=mask_ratio, noise=noise_dev,
    )
    return frame_visibility(mask[0].cpu())


# -- run ----------------------------------------------------------------------
print(f"Loading {BB} on {device} ...")
model, _, mean, std, spatial_patch = load_prithvi_from_terratorch(
    backbone_name=BB, base_dir=BASE, checkpoint_filename=CKPT,
    num_frames=T, device=device,
)
assert spatial_patch == PATCH, f"patch mismatch: {spatial_patch} != {PATCH}"

chip = find_chip()
print(f"Chip: {chip.name}\n")
x = load_chip_norm(chip, mean, std)

hdr = f"{'nominal':>7} | {'masker':>13} | {'spring':>7} {'summer':>7} {'fall':>7} | {'summer masked':>13}"
print(hdr); print("-" * len(hdr))

ok = True
for r in RATIOS:
    # (1) OLD block: per-frame ratio passed to the model (the bug)
    n_old, _, _ = old_block(mask_ratio=r, patch_size=PATCH, img_size=IMG,
                            num_frames=T, frame_idx=FRAME, trial_seed=SEED)
    (s1, m1, f1), sm1 = measure(x, n_old, r)

    # (2) FIXED block: global ratio passed to the model
    n_fb, gr_fb, idx_fb = fixed_block(mask_ratio=r, patch_size=PATCH, img_size=IMG,
                                      num_frames=T, frame_idx=FRAME, trial_seed=SEED)
    (s2, m2, f2), sm2 = measure(x, n_fb, gr_fb)

    # (3) FIXED random, matched to the fixed block's summer count
    n_fr, gr_fr, idx_fr = fixed_random(mask_ratio=r, patch_size=PATCH, img_size=IMG,
                                       num_frames=T, frame_idx=FRAME, trial_seed=SEED,
                                       n_summer_masked=len(idx_fb))
    (s3, m3, f3), sm3 = measure(x, n_fr, gr_fr)

    print(f"{int(r*100):>6}% | {'OLD block':>13} | {s1:6.1f} {m1:6.1f} {f1:6.1f} | {sm1:>13}")
    print(f"{'':>7} | {'FIXED block':>13} | {s2:6.1f} {m2:6.1f} {f2:6.1f} | {sm2:>13}")
    print(f"{'':>7} | {'FIXED random':>13} | {s3:6.1f} {m3:6.1f} {f3:6.1f} | {sm3:>13}")
    print()

    if not (abs(s2 - 100) < 0.01 and abs(f2 - 100) < 0.01):
        print(f"  !! FIXED block: spring/fall not fully visible at {int(r*100)}% "
              f"(spring={s2:.1f}, fall={f2:.1f}) -- investigate before big run")
        ok = False
    if not (abs(s3 - 100) < 0.01 and abs(f3 - 100) < 0.01):
        print(f"  !! FIXED random: spring/fall not fully visible at {int(r*100)}%")
        ok = False
    if sm2 != sm3:
        print(f"  !! summer counts NOT matched at {int(r*100)}%: block={sm2} random={sm3}")
        ok = False

print("=" * len(hdr))
if ok:
    print("PASS: fixed maskers keep spring & fall 100% visible and match summer counts.")
    print("Safe to proceed to the full paired rerun.")
else:
    print("FAIL: see !! lines above. Do NOT launch the 40k-pass rerun yet.")
