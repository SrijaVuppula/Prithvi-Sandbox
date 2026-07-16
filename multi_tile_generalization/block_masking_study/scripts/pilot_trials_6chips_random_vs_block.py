"""
6-chip comparison: random vs block PSNR, 50 trials per chip x ratio.
Chips: rank 1, 2 (complex), rank 250, 251 (medium), rank 499, 500 (simple).
Random columns placed LEFT of block columns in the summary output.

FIXED 2026-07-16: uses temporal_gap_masker.build_block_noise_mask /
build_random_noise_mask instead of the old ad hoc hard-tie noise functions.
Random is matched to block via n_summer_masked so both mask the exact same
number of summer patches per trial -- only geometry differs. The GLOBAL
ratio returned by the masker (not the nominal ratio) is passed to
run_masked_forward.
"""
import os, sys, csv, warnings
import numpy as np, torch, rasterio
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "patch_masking_study"))
sys.path.insert(0, os.path.join(ROOT, "multi_tile_generalization/block_masking_study/masking"))

from patch_masking_study.terratorch_loader import load_prithvi_from_terratorch, run_masked_forward
from temporal_gap_masker import build_block_noise_mask, build_random_noise_mask

CHIPS = [
    ("chip_140_470_merged.tif", 1,   "complex_rank1_new"),
    ("chip_217_425_merged.tif", 2,   "complex_rank2_original"),
    ("chip_182_606_merged.tif", 250, "medium_rank250_new"),
    ("chip_268_410_merged.tif", 251, "medium_rank251_original"),
    ("chip_282_562_merged.tif", 499, "simple_rank499_new_FLAGGED_cloud_contamination"),
    ("chip_105_452_merged.tif", 500, "simple_rank500_original"),
]
CHIP_DIR = "multi_tile_generalization/study_chips_500"
OUT_TRIALS  = "multi_tile_generalization/block_masking_study/outputs/pilot_trials_6chips_random_vs_block.csv"
OUT_SUMMARY = "multi_tile_generalization/block_masking_study/outputs/pilot_summary_6chips_random_vs_block.csv"
BDIR, CKPT = "~/Prithvi/prithvi_600M", "Prithvi_EO_V2_600M_TL.pt"
RATIOS = [0.2, 0.4, 0.6, 0.8]
N_TRIALS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(OUT_TRIALS), exist_ok=True)

def load_chip(path):
    with rasterio.open(os.path.expanduser(path)) as src:
        return src.read().astype(np.float32)

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

model, _, mean, std, ps = load_prithvi_from_terratorch(
    backbone_name="prithvi_eo_v2_600", base_dir=os.path.expanduser(BDIR),
    checkpoint_filename=CKPT, num_frames=3, device=DEVICE)
model.eval()

m = np.tile(np.array(mean, dtype=np.float32), 3).reshape(-1, 1, 1)
s = np.tile(np.array(std, dtype=np.float32), 3).reshape(-1, 1, 1)
mc = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
sc = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

rows = []
for chip, rank, role in CHIPS:
    raw = load_chip(f"{CHIP_DIR}/{chip}")
    norm = (raw - m) / (s + 1e-6)
    x = torch.tensor(norm).reshape(3, 6, 224, 224).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    gt = raw[6:12]  # summer, raw HLS
    gtn = gt / 10000.0

    for ratio in RATIOS:
        for t in range(N_TRIALS):
            # block first -- its n_summer_masked count drives the matched random trial
            noise_b, gratio_b, idx_b = build_block_noise_mask(
                ratio, patch_size=ps, img_size=224, num_frames=3,
                frame_idx=1, trial_seed=t)
            noise_r, gratio_r, idx_r = build_random_noise_mask(
                ratio, patch_size=ps, img_size=224, num_frames=3,
                frame_idx=1, trial_seed=t, n_summer_masked=len(idx_b))
            assert abs(gratio_b - gratio_r) < 1e-9, "block/random ratio mismatch"

            for mtype, noise, gratio in [("block", noise_b, gratio_b),
                                          ("random", noise_r, gratio_r)]:
                nt = noise.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = run_masked_forward(model, x, None, None, gratio, nt)
                rec = out[2].squeeze(0).cpu().numpy()
                rec_summer = rec[:, 1] * sc + mc
                rcn = np.clip(rec_summer / 10000.0, 0, 1)
                rows.append(dict(chip=chip, rank=rank, role=role, ratio=ratio,
                                  mask_type=mtype, trial=t,
                                  global_ratio=round(gratio, 4),
                                  psnr=round(psnr(gtn, rcn), 4)))
        print(f"{chip} (rank {rank}) {int(ratio*100)}% done")

with open(OUT_TRIALS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("Wrote", OUT_TRIALS)

import pandas as pd
df = pd.DataFrame(rows)
piv = df.pivot_table(index=["chip", "rank", "role", "ratio"], columns="mask_type",
                      values="psnr", aggfunc="mean").reset_index()
piv = piv[["chip", "rank", "role", "ratio", "random", "block"]]
piv["delta_random_minus_block"] = piv["random"] - piv["block"]
piv = piv.sort_values(["rank", "ratio"])
piv.to_csv(OUT_SUMMARY, index=False)
print("Wrote", OUT_SUMMARY)
print(piv.to_string(index=False))
