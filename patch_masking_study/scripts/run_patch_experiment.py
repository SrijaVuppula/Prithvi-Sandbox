"""
patch_masking_study/scripts/run_patch_experiment.py

Master script that loops through all backbones and mask ratios,
runs the forward pass, computes global + masked-region metrics,
and saves results to CSV and JSON sidecars.

Uses TerraTorch for clean model loading — no source patching required.
"""

import sys
import csv
import json
import yaml
import torch
import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baseline_study.data.hls_loader import load_sample_from_inference_module
from patch_masking_study.terratorch_loader import (
    load_prithvi_from_terratorch,
    run_masked_forward,
)
from patch_masking_study.masking.patch_masker import (
    build_noise_for_mask_ratio,
    get_masked_token_indices,
)
from patch_masking_study.metrics.evaluate_masked import (
    evaluate_reconstruction_with_masked_metrics,
)


def run_experiment(config_path: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = torch.device(cfg["model"]["device"])
    backbones   = cfg["model"]["backbones"]
    mask_ratios = cfg["experiment"]["mask_ratios"]
    seq_len     = cfg["experiment"]["sequence_length"]
    frame_idx   = cfg["experiment"]["frame_idx"]
    output_dir  = Path(cfg["experiment"]["output_dir"])
    tile        = cfg["data"]["tiles"][0]

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    csv_path   = output_dir / "results.csv"
    fieldnames = [
        "backbone", "mask_ratio",
        "mae", "psnr", "ssim",
        "masked_mae", "masked_psnr", "masked_ssim",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for backbone in backbones:
        name       = backbone["name"]
        base_dir   = Path(backbone["base_dir"])
        checkpoint = backbone["checkpoint"]

        print(f"\n{'='*60}")
        print(f"  Backbone: {name}")
        print(f"{'='*60}")

        # ── Load model via TerraTorch ─────────────────────────────────────────
        model, bands, mean, std, patch_size = load_prithvi_from_terratorch(
            backbone_name=name,
            base_dir=base_dir,
            checkpoint_filename=checkpoint,
            num_frames=seq_len,
            device=device,
        )

        # ── Load data ─────────────────────────────────────────────────────────
        file_paths = tile["files"][:seq_len]
        x, temporal_coords, location_coords, _ = load_sample_from_inference_module(
            base_dir=base_dir,
            file_paths=file_paths,
            mean=mean,
            std=std,
            device=device,
        )

        for mask_ratio in mask_ratios:
            print(f"\n  mask_ratio={mask_ratio:.2f} ...", end=" ", flush=True)

            # Build noise tensor
            noise, _, _ = build_noise_for_mask_ratio(
                x, mask_ratio, patch_size, device, seed=42
            )

            # Forward pass via TerraTorch
            loss, pred_img, rec_img, mask_img, x_cpu = run_masked_forward(
                model=model,
                x=x,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
                mask_ratio=mask_ratio,
                noise=noise,
            )

            # Extract target frame
            gt_frame   = x_cpu[0, :, frame_idx, :, :]
            pred_frame = rec_img[0, :, frame_idx, :, :]

            # Masked token indices for masked-region metrics
            masked_indices = get_masked_token_indices(noise)

            # Compute metrics
            metrics = evaluate_reconstruction_with_masked_metrics(
                pred_frame=pred_frame,
                gt_frame=gt_frame,
                mean=mean,
                std=std,
                masked_token_indices=masked_indices,
                patch_size=patch_size,
                T=seq_len,
            )

            print(f"PSNR={metrics['psnr']:.2f} | masked_PSNR={metrics['masked_psnr']:.2f}")

            # Save JSON sidecar
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id    = f"{timestamp}_{name}_mask{int(mask_ratio*100):02d}"
            result    = {"backbone": name, "mask_ratio": mask_ratio, **metrics}

            with open(runs_dir / f"{run_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)

        # Free GPU memory before next backbone
        del model
        torch.cuda.empty_cache()

    print(f"\n\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    config_path = REPO_ROOT / "patch_masking_study" / "config" / "patch_experiment_config.yaml"
    run_experiment(config_path)
