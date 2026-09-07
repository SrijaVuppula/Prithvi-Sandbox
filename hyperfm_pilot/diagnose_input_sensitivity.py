"""
Direct diagnostic: does the fine-tuned model's prediction actually change
when the input changes drastically, or has it collapsed to something close
to a constant/mean prediction regardless of what's masked?

Runs ONE tile through the model twice -- 5% masked and 80% masked, same
geometry -- and compares raw prediction tensors directly (not through the
loss/metric pipeline). If the two predictions are nearly identical to each
other despite wildly different visible inputs, that's direct evidence of
collapse, independent of any PSNR/MAE computation.

Also compares each prediction against a trivial "broadcast this tile's own
per-band mean spectrum to every pixel" baseline -- if the model's error
against ground truth is close to this baseline's error, that's further
confirmation the model isn't doing meaningfully better than guessing the
mean.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from controlnet_encoder_adapter import ControlNetEncoderAdapter  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402
from trainable_forward import run_masked_forward_trainable  # noqa: E402
from masked_reconstruction import apply_band_mask, unpatchify_bands  # noqa: E402
from pace_tile_loader import load_pace_tile  # noqa: E402

BACKBONE = "100M"
CHECKPOINT = Path.home() / "Prithvi" / "hyperfm_pilot" / "checkpoints" / "100M" / "best.pt"
SPEC = dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768)
TILE = "cvpr_dataset/hsi/PACE_OCI.20250220T014900.L1B.V3_image_078.npy"  # same rep. tile used earlier
GEOMETRY = "scattered"
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoints = list(SPEC["base_dir"].glob("*.pt"))
model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
    backbone_name=BACKBONE, base_dir=str(SPEC["base_dir"]),
    checkpoint_filename=checkpoints[0].name, num_frames=1, device=device,
)
model.eval()

enc_adapter = ControlNetEncoderAdapter(
    model.encoder, embed_dim=SPEC["embed_dim"], patch_size=SPEC["patch_size"],
).to(device)
dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=SPEC["patch_size"]).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
enc_adapter.band_adapter.hint_encoder.load_state_dict(ckpt["hint_encoder_state"])
dec_adapter.spectral_head.load_state_dict(ckpt["spectral_head_state"])
enc_adapter.eval()
dec_adapter.eval()
print(f"loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")

target_cube, valid_mask = load_pace_tile(TILE, target_size=112)
target_cube = target_cube.to(device)
valid_mask = valid_mask.to(device)
H, W = target_cube.shape[-2:]


def run_forward(ratio, rng):
    masked_cube, band_mask = apply_band_mask(
        target_cube, mask_ratio=ratio, geometry=GEOMETRY, rng=rng, num_bands=291,
    )
    placeholder = torch.zeros(1, 6, 1, H, W, device=device)
    enc_adapter.set_pace_cube(masked_cube, band_mask.unsqueeze(0))
    with torch.no_grad():
        run_masked_forward_trainable(
            model, placeholder, temporal_coords=None, location_coords=None,
            mask_ratio=0.0, noise=None,
        )
        patch_tokens = dec_adapter.last_output[:, 1:, :]
        grid = H // SPEC["patch_size"]
        prediction = unpatchify_bands(
            patch_tokens, num_channels=291, patch_size=SPEC["patch_size"],
            num_patches_h=grid, num_patches_w=grid,
        ).squeeze(2)
    return prediction, band_mask


rng1 = np.random.default_rng(123)
rng2 = np.random.default_rng(456)
pred_05, mask_05 = run_forward(0.05, rng1)
pred_80, mask_80 = run_forward(0.80, rng2)

# --- Check 1: how different are the two predictions from each other? ---
diff_between_predictions = (pred_05 - pred_80)[0][valid_mask[0]].abs().mean().item()

# --- Check 2: how different is each prediction from ground truth,
#              restricted to the bands actually masked in that run? ---
score_mask_05 = mask_05.view(-1, 1, 1).expand_as(valid_mask[0]) & valid_mask[0]
score_mask_80 = mask_80.view(-1, 1, 1).expand_as(valid_mask[0]) & valid_mask[0]

gt_diff_05 = (pred_05[0] - target_cube[0])[score_mask_05].abs().mean().item()
gt_diff_80 = (pred_80[0] - target_cube[0])[score_mask_80].abs().mean().item()

# --- Check 3: trivial baseline -- broadcast this tile's own per-band mean
#              (computed from GROUND TRUTH's valid pixels) to every pixel,
#              scored over the same masked-band positions as each run ---
per_band_mean = target_cube[0].masked_fill(~valid_mask[0], 0).sum(dim=(-1, -2)) / \
    valid_mask[0].sum(dim=(-1, -2)).clamp(min=1)
baseline = per_band_mean.view(-1, 1, 1).expand_as(target_cube[0])
baseline_diff_05 = (baseline - target_cube[0])[score_mask_05].abs().mean().item()
baseline_diff_80 = (baseline - target_cube[0])[score_mask_80].abs().mean().item()

print(f"\nTile: {TILE}")
print(f"Geometry: {GEOMETRY}\n")
print(f"MAE(pred@5%masked, pred@80%masked) = {diff_between_predictions:.6f}   <- how much output changes with input (full cube)")
print(f"MAE(pred@5%masked,  ground truth)  [masked bands only] = {gt_diff_05:.6f}")
print(f"MAE(pred@80%masked, ground truth)  [masked bands only] = {gt_diff_80:.6f}")
print(f"MAE(trivial baseline, ground truth) @5%masked bands  = {baseline_diff_05:.6f}   <- 'model does nothing' baseline")
print(f"MAE(trivial baseline, ground truth) @80%masked bands = {baseline_diff_80:.6f}   <- 'model does nothing' baseline")

print("\n--- Interpretation ---")
if diff_between_predictions < 0.3 * max(gt_diff_05, gt_diff_80):
    print("Prediction barely changes despite drastically different masked input.")
    print("-> Consistent with a collapsed / near-constant output (input insensitivity).")
else:
    print("Prediction changes substantially with input.")
    print("-> Model IS input-sensitive; check masked-band error trend below.")

if abs(gt_diff_05 - baseline_diff_05) < 0.3 * baseline_diff_05 and abs(gt_diff_80 - baseline_diff_80) < 0.3 * baseline_diff_80:
    print("Model's masked-band error is close to the trivial per-band-mean baseline at BOTH ratios.")
    print("-> Model isn't beating a naive constant-per-band guess by much on the hidden bands.")
else:
    print("Model's masked-band error differs meaningfully from the trivial baseline.")

if gt_diff_80 - gt_diff_05 > 0.1 * gt_diff_05:
    print("Masked-band error meaningfully worse at 80% than 5% -- ratio-sensitivity present.")
else:
    print("Masked-band error is flat between 5% and 80% -- ratio-sensitivity NOT present.")
