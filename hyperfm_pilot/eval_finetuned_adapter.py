"""
Fine-tuned-model eval: scores a trained ControlNet band adapter against the
same 100-tile / 50-trial / 4-ratio / 2-geometry protocol as the Step 1
zero-shot pilot (spectral_ratio_sweep.py), so the two are directly
comparable -- same tiles (hsi_diverse_100.txt), same ratios, same
geometries, same trial count, same MAE/RMSE/PSNR formulas (PSNR assumes
peak=1.0, matching the pilot -- not recalibrated to true data range, kept
identical on purpose for apples-to-apples comparison).

Differences from the pilot, all necessary rather than incidental:
- Reconstruction method: frozen Prithvi + trained ControlNet adapter,
  not wavelength-based linear interpolation.
- Tiles go through pace_tile_loader.load_pace_tile (NaN->0, reflect-pad
  96->112) since the model needs padded input; the pilot worked on raw
  96x96 arrays directly. Scoring uses valid_mask to exclude both NaN and
  padded-border pixels, so the *effective* scored region matches the
  pilot's un-padded 96x96 tiles despite the model needing padding.
- Exact mask draws are NOT bit-identical to the pilot's (different rng
  call order) -- not necessary; only ratio, geometry, tile set, and trial
  count need to match for a fair comparison, not the literal random draws.

Usage:
    python eval_finetuned_adapter.py --backbone 100M \
        --checkpoint checkpoints/100M/best.pt
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


def compute_metrics(prediction, target, score_mask):
    """Same MAE/RMSE/PSNR formulas as the zero-shot pilot's metrics()
    (peak=1.0 assumed for PSNR), computed over score_mask positions only."""
    diff = (prediction - target)[score_mask].detach().cpu().numpy()
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    psnr = 10 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else np.inf
    return mae, rmse, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["100M", "600M"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--ratios", type=str, default="0.2,0.4,0.6,0.8",
                         help="comma-separated mask ratios, e.g. '0.05' for a single low-ratio check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_errormaps", action="store_true")
    args = parser.parse_args()

    spec = BACKBONE_SPECS[args.backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone: {args.backbone}, device: {device}")

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
    assert ckpt["backbone"] == args.backbone, \
        f"checkpoint is for {ckpt['backbone']}, not {args.backbone}"
    enc_adapter.band_adapter.hint_encoder.load_state_dict(ckpt["hint_encoder_state"])
    dec_adapter.spectral_head.load_state_dict(ckpt["spectral_head_state"])
    print(f"loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.6f})")
    enc_adapter.eval()
    dec_adapter.eval()

    rng = np.random.default_rng(args.seed)
    ratios = [float(r) for r in args.ratios.split(",")]
    tile_files = [l.strip() for l in HSI_LIST.read_text().splitlines() if l.strip()]

    errormap_dir = Path("errormaps_finetuned") / args.backbone
    representative_tile = None
    if args.save_errormaps:
        errormap_dir.mkdir(parents=True, exist_ok=True)
        scores = []
        with open(SCORES_CSV) as f:
            for row in csv.DictReader(f):
                scores.append((row["tile"], float(row["variance_score"])))
        scores.sort(key=lambda x: x[1])
        representative_tile = scores[len(scores) // 2][0]
        print(f"Representative tile for error maps: {representative_tile}")

    rows = []
    with torch.no_grad():
        for tf in tile_files:
            tile_id = Path(tf).stem
            target_cube, valid_mask = load_pace_tile(tf, target_size=112)
            target_cube = target_cube.to(device)
            valid_mask = valid_mask.to(device)
            H, W = target_cube.shape[-2:]
            is_representative = (tf == representative_tile)

            for ratio in ratios:
                for geometry in GEOMETRIES:
                    for trial in range(args.n_trials):
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

                        band_mask_expanded = band_mask.view(1, -1, 1, 1).expand_as(target_cube)
                        score_mask = band_mask_expanded & valid_mask

                        mae, rmse, psnr = compute_metrics(prediction, target_cube, score_mask)
                        rows.append([tile_id, geometry, ratio, trial, mae, rmse, psnr])

                        if is_representative and trial == 0:
                            err = prediction - target_cube
                            err_map = err[0][band_mask].abs().mean(dim=0).detach().cpu().numpy()
                            np.save(errormap_dir / f"{geometry}_ratio{int(ratio*100)}.npy", err_map)

    output_path = args.output or f"spectral_ratio_sweep_results_finetuned_{args.backbone}.csv"
    with open(output_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile", "mask_type", "ratio", "trial", "mae", "rmse", "psnr"])
        w.writerows(rows)

    print("\n=== Median PSNR / MAE by ratio and mask type (fine-tuned) ===")
    for ratio in ratios:
        for geometry in GEOMETRIES:
            vals = [r for r in rows if r[1] == geometry and r[2] == ratio]
            psnrs = np.array([r[6] for r in vals])
            maes = np.array([r[4] for r in vals])
            print(f"ratio={ratio:.1f}  {geometry:10s}  median PSNR={np.nanmedian(psnrs):6.2f}dB"
                  f"  median MAE={np.nanmedian(maes):.4f}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
