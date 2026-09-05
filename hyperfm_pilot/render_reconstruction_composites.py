"""
For the representative tile, render one 4-panel composite per ratio x
geometry combo: Ground Truth (RGB) | Reconstructed (RGB) | Error map |
Band-mask strip (which of the 291 bands were hidden -- the spectral
equivalent of a spatial mask box, since PACE masking hides bands, not
pixels). Saved into the same directory as the .npy error maps.

Usage:
    python render_reconstruction_composites.py --backbone 100M \
        --checkpoint ~/Prithvi/hyperfm_pilot/checkpoints/100M/best.pt
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from controlnet_encoder_adapter import ControlNetEncoderAdapter  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402
from trainable_forward import run_masked_forward_trainable  # noqa: E402
from masked_reconstruction import apply_band_mask, unpatchify_bands  # noqa: E402
from pace_tile_loader import load_pace_tile  # noqa: E402
from pace_band_wavelengths import BAND_WAVELENGTH_NM, BAND_ORIGIN  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}
SCORES_CSV = Path("hsi_diverse_100_scores.csv")
N_BANDS = 291
RATIOS = [0.2, 0.4, 0.6, 0.8]
GEOMETRIES = ["contiguous", "scattered"]

RGB_IDX = {"red": 143, "green": 96, "blue": 56}  # 649.6nm, 549.9nm, 449.7nm

ORIGIN_COLOR = {"blue": "#cfe3f7", "red": "#f7d9cf", "swir": "#e0e0e0"}


def squeeze_to_2d(t):
    while t.dim() > 2:
        t = t[0]
    return t.bool()


def stretch_rgb(cube, vm2d):
    """cube: (291, H, W). Returns (H, W, 3) in [0,1], 2-98 percentile stretch
    per channel computed over valid pixels only."""
    out = np.zeros((cube.shape[-2], cube.shape[-1], 3), dtype=np.float32)
    for i, ch in enumerate(["red", "green", "blue"]):
        band = cube[RGB_IDX[ch]].cpu().numpy()
        valid_vals = band[vm2d.cpu().numpy()]
        lo, hi = np.percentile(valid_vals, [2, 98])
        out[..., i] = np.clip((band - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


def draw_band_strip(ax, masked_idx):
    n = N_BANDS
    strip = np.zeros((1, n, 3))
    for i in range(n):
        strip[0, i] = matplotlib.colors.to_rgb(ORIGIN_COLOR[BAND_ORIGIN[i]])
    strip_img = np.tile(strip, (20, 1, 1))
    ax.imshow(strip_img, aspect="auto", extent=[0, n, 0, 1])
    for i in masked_idx:
        ax.axvline(i, color="black", linewidth=0.6, alpha=0.85)
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel("band index (hidden bands marked black)")
    # wavelength ticks
    tick_idx = [0, 50, 100, 150, 200, 250, 290]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{BAND_WAVELENGTH_NM[i]:.0f}nm" for i in tick_idx], fontsize=7)


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

    gt_rgb = stretch_rgb(target_cube[0], vm2d)

    rng = np.random.default_rng(args.seed)
    out_dir = Path("results_finetuned") / args.backbone / "errormaps"

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

                recon_rgb = stretch_rgb(prediction[0], vm2d)
                err = (prediction - target_cube)[0][band_mask].abs().mean(dim=0).detach().cpu().numpy()
                masked_idx = band_mask.nonzero(as_tuple=True)[0].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                                          gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
                axes[0].imshow(gt_rgb); axes[0].set_title("Ground Truth"); axes[0].axis("off")
                axes[1].imshow(recon_rgb); axes[1].set_title("Reconstructed"); axes[1].axis("off")
                im = axes[2].imshow(err, cmap="inferno")
                axes[2].set_title("Error (masked bands, |diff|)"); axes[2].axis("off")
                fig.colorbar(im, ax=axes[2], fraction=0.046)
                draw_band_strip(axes[3], masked_idx)
                axes[3].set_title(f"masked bands (n={len(masked_idx)})")

                key = f"{geometry}_ratio{int(ratio*100)}"
                fig.suptitle(f"{key} — {args.backbone}", fontsize=13)
                out_path = out_dir / f"composite_{key}.png"
                fig.savefig(out_path, dpi=130, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {out_path}")


if __name__ == "__main__":
    main()
