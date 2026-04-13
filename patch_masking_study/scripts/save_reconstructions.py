"""
patch_masking_study/scripts/save_reconstructions.py

Saves side-by-side reconstruction images for all backbone x mask ratio combinations.
Each image shows 4 panels:
  Original | Masked Input | Reconstruction | Ground Truth

Uses TerraTorch for clean model loading.
"""

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baseline_study.data.hls_loader import load_sample_from_inference_module
from baseline_study.metrics.evaluate import denorm_all_bands
from patch_masking_study.terratorch_loader import (
    load_prithvi_from_terratorch,
    run_masked_forward,
)
from patch_masking_study.masking.patch_masker import build_noise_for_mask_ratio

RGB_BANDS = [2, 1, 0]


def to_rgb(frame_chw: torch.Tensor, mean: list, std: list) -> np.ndarray:
    denormed = denorm_all_bands(frame_chw, mean, std)
    rgb = denormed[RGB_BANDS].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


def save_reconstruction_plot(
    x_cpu: torch.Tensor,
    rec_img: torch.Tensor,
    mask_img: torch.Tensor,
    mean: list,
    std: list,
    frame_idx: int,
    backbone: str,
    mask_ratio: float,
    save_path: Path,
):
    original      = x_cpu[0, :, frame_idx, :, :]
    reconstructed = rec_img[0, :, frame_idx, :, :]

    masked_input = original.clone()
    mask_2d = mask_img[0, :, frame_idx, :, :]
    masked_input[mask_2d == 1] = 0.0

    orig_rgb = to_rgb(original,     mean, std)
    mask_rgb = to_rgb(masked_input, mean, std)
    rec_rgb  = to_rgb(reconstructed, mean, std)
    gt_rgb   = to_rgb(original,     mean, std)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (orig_rgb, "Original"),
        (mask_rgb, f"Masked Input\n({int(mask_ratio*100)}% hidden)"),
        (rec_rgb,  "Reconstruction"),
        (gt_rgb,   "Ground Truth"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Backbone: {backbone} | Mask Ratio: {int(mask_ratio*100)}%",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def run_visualizations(config_path: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = torch.device(cfg["model"]["device"])
    backbones   = cfg["model"]["backbones"]
    mask_ratios = cfg["experiment"]["mask_ratios"]
    seq_len     = cfg["experiment"]["sequence_length"]
    frame_idx   = cfg["experiment"]["frame_idx"]
    output_dir  = Path(cfg["experiment"]["output_dir"])
    tile        = cfg["data"]["tiles"][0]

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for backbone in backbones:
        name       = backbone["name"]
        base_dir   = Path(backbone["base_dir"])
        checkpoint = backbone["checkpoint"]

        print(f"\n{'='*60}")
        print(f"  Backbone: {name}")
        print(f"{'='*60}")

        model, bands, mean, std, patch_size = load_prithvi_from_terratorch(
            backbone_name=name,
            base_dir=base_dir,
            checkpoint_filename=checkpoint,
            num_frames=seq_len,
            device=device,
        )

        file_paths = tile["files"][:seq_len]
        x, temporal_coords, location_coords, _ = load_sample_from_inference_module(
            base_dir=base_dir,
            file_paths=file_paths,
            mean=mean,
            std=std,
            device=device,
        )

        for mask_ratio in mask_ratios:
            print(f"  mask_ratio={mask_ratio:.2f} ...", end=" ", flush=True)

            noise, _, _ = build_noise_for_mask_ratio(
                x, mask_ratio, patch_size, device, seed=42
            )

            loss, pred_img, rec_img, mask_img, x_cpu = run_masked_forward(
                model=model,
                x=x,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
                mask_ratio=mask_ratio,
                noise=noise,
            )

            save_path = plots_dir / f"{name}_mask{int(mask_ratio*100):02d}.png"
            save_reconstruction_plot(
                x_cpu=x_cpu,
                rec_img=rec_img,
                mask_img=mask_img,
                mean=mean,
                std=std,
                frame_idx=frame_idx,
                backbone=name,
                mask_ratio=mask_ratio,
                save_path=save_path,
            )

        del model
        torch.cuda.empty_cache()

    print(f"\nAll reconstruction images saved to {plots_dir}")


if __name__ == "__main__":
    config_path = REPO_ROOT / "patch_masking_study" / "config" / "patch_experiment_config.yaml"
    run_visualizations(config_path)
