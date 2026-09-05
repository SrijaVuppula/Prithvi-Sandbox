"""
Per-band (spectral) error profile for one representative tile, across all
8 ratio x geometry combinations, using the trained ControlNet adapter
checkpoint. Companion to eval_finetuned_adapter.py's --save_errormaps path,
which collapses the band axis (mean over masked bands -> spatial map only).
This keeps the band axis instead (mean over spatial dims, valid pixels only)
so we can check whether error concentrates in specific spectral regions
(O2-A absorption ~bands 221-227, SWIR-tail ~bands 283-290) vs. being uniform.

Usage:
    python spectral_error_by_band.py --backbone 100M --checkpoint checkpoints/100M/best.pt
"""
import argparse
import csv
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

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}
HSI_LIST = Path("hsi_diverse_100.txt")
SCORES_CSV = Path("hsi_diverse_100_scores.csv")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
GEOMETRIES = ["contiguous", "scattered"]
O2A_LO, O2A_HI = 221, 227
SWIR_LO, SWIR_HI = 283, 290


def squeeze_to_2d(t):
    while t.dim() > 2:
        t = t[0]
    return t.bool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["100M", "600M"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = BACKBONE_SPECS[args.backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = list(spec["base_dir"].glob("*.pt"))
    assert len(checkpoints) == 1, f"expected one .pt in {spec['base_dir']}, found: {checkpoints}"
    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=args.backbone, base_dir=str(spec["base_dir"]),
        checkpoint_filename=checkpoints[0].name, num_frames=1, device=device,
    )
    model.eval()
    enc_adapter = ControlNetEncoderAdapter(
        model.encoder, embed_dim=spec["embed_dim"], patch_size=spec["patch_size"],
    ).to(device)
    dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    assert ckpt["backbone"] == args.backbone, f"checkpoint is for {ckpt['backbone']}, not {args.backbone}"
    enc_adapter.band_adapter.hint_encoder.load_state_dict(ckpt["hint_encoder_state"])
    dec_adapter.spectral_head.load_state_dict(ckpt["spectral_head_state"])
    print(f"loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    enc_adapter.eval()
    dec_adapter.eval()

    scores = []
    with open(SCORES_CSV) as f:
        for row in csv.DictReader(f):
            scores.append((row["tile"], float(row["variance_score"])))
    scores.sort(key=lambda x: x[1])
    representative_tile = scores[len(scores) // 2][0]
    print(f"Representative tile: {representative_tile}")

    target_cube, valid_mask = load_pace_tile(representative_tile, target_size=112)
    target_cube = target_cube.to(device)
    valid_mask = valid_mask.to(device)
    H, W = target_cube.shape[-2:]
    vm2d = squeeze_to_2d(valid_mask)

    rng = np.random.default_rng(args.seed)
    profiles = {}

    with torch.no_grad():
        for ratio in RATIOS:
            for geometry in GEOMETRIES:
                masked_cube, band_mask = apply_band_mask(
                    target_cube, mask_ratio=ratio, geometry=geometry,
                    rng=rng, num_bands=N_BANDS,
                )
                placeholder = torch.zeros(1, 6, 1, H, W, device=device)
                enc_adapter.set_pace_cube(masked_cube)
                run_masked_forward_trainable(
                    model, placeholder, temporal_coords=None, location_coords=None,
                    mask_ratio=0.0, noise=None,
                )
                patch_tokens = dec_adapter.last_output[:, 1:, :]
                grid = H // spec["patch_size"]
                prediction = unpatchify_bands(
                    patch_tokens, num_channels=N_BANDS, patch_size=spec["patch_size"],
                    num_patches_h=grid, num_patches_w=grid,
                ).squeeze(2)
                err = (prediction - target_cube)[0].abs()  # (291, H, W)

                per_band_err = np.array([
                    err[b][vm2d].mean().item() for b in range(N_BANDS)
                ])
                masked_bands_idx = band_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                masked_profile = per_band_err[masked_bands_idx]
                overall = masked_profile.mean()

                o2a_idx = [b for b in masked_bands_idx if O2A_LO <= b <= O2A_HI]
                swir_idx = [b for b in masked_bands_idx if SWIR_LO <= b <= SWIR_HI]
                rest_idx = [b for b in masked_bands_idx if b not in o2a_idx and b not in swir_idx]
                o2a_mean = per_band_err[o2a_idx].mean() if o2a_idx else float("nan")
                swir_mean = per_band_err[swir_idx].mean() if swir_idx else float("nan")
                rest_mean = per_band_err[rest_idx].mean() if rest_idx else float("nan")

                top5 = masked_profile.argsort()[::-1][:5]
                top5_bands = masked_bands_idx[top5]

                key = f"{geometry}_ratio{int(ratio*100)}"
                profiles[key] = per_band_err
                print(f"\n=== {key} === n_masked_bands={len(masked_bands_idx)}, overall(masked)={overall:.6f}")
                print(f"  O2-A[{O2A_LO}-{O2A_HI}] n={len(o2a_idx)} mean={o2a_mean:.6f} ({o2a_mean/overall:.2f}x)")
                print(f"  SWIR-tail[{SWIR_LO}-{SWIR_HI}] n={len(swir_idx)} mean={swir_mean:.6f} ({swir_mean/overall:.2f}x)")
                print(f"  rest n={len(rest_idx)} mean={rest_mean:.6f}")
                print("  top5 masked bands (idx:err): " + ", ".join(f"{b}:{per_band_err[b]:.5f}" for b in top5_bands))

    Path(f"results_finetuned/{args.backbone}").mkdir(parents=True, exist_ok=True); np.savez(f"results_finetuned/{args.backbone}/spectral_error_profile.npz", **profiles)
    print(f"\nSaved full per-band profiles to spectral_error_profile_{args.backbone}.npz")


if __name__ == "__main__":
    main()
