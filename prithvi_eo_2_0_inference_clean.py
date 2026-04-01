from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


DEFAULT_EXAMPLES = [
    "Mexico_HLS.S30.T13REM.2018026T173609.v2.0_cropped.tif",
    "Mexico_HLS.S30.T13REM.2018106T172859.v2.0_cropped.tif",
    "Mexico_HLS.S30.T13REM.2018201T172901.v2.0_cropped.tif",
    "Mexico_HLS.S30.T13REM.2018266T173029.v2.0_cropped.tif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prithvi EO 2.0 tiny-TL frame-masking inference on a GPU server."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory containing config.json, inference.py, prithvi_mae.py, checkpoint, and examples/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots and JSON outputs will be written.",
    )
    parser.add_argument(
        "--frames-to-mask",
        type=int,
        nargs="+",
        default=[1],
        help="Frame indices to fully mask. Example: --frames-to-mask 1 3",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit TIFF filenames relative to base-dir/examples.",
    )
    parser.add_argument(
        "--evaluate-all-single-frames",
        action="store_true",
        help="Also evaluate each frame masked individually and save summary JSON.",
    )
    return parser.parse_args()


def safe_replace(text: str, old: str, new: str, description: str) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new), True
    if new in text:
        return text, False
    raise RuntimeError(f"Could not find expected text for patch: {description}")


def patch_inference_py(inference_path: Path) -> None:
    text = inference_path.read_text()
    changed_any = False

    text, changed = safe_replace(
        text,
        'temporal_coords = torch.Tensor(temporal_coords, device=device).unsqueeze(0)',
        'temporal_coords = torch.tensor(temporal_coords, dtype=torch.float32, device=device).unsqueeze(0)',
        'temporal_coords tensor creation',
    )
    changed_any = changed_any or changed

    text, changed = safe_replace(
        text,
        'location_coords = torch.Tensor(location_coords[0], device=device).unsqueeze(0)',
        'location_coords = torch.tensor(location_coords[0], dtype=torch.float32, device=device).unsqueeze(0)',
        'location_coords tensor creation',
    )
    changed_any = changed_any or changed

    if changed_any:
        inference_path.write_text(text)
        print(f"Patched {inference_path.name} for torch.tensor compatibility.")
    else:
        print(f"No compatibility patch needed for {inference_path.name}.")


def patch_prithvi_mae(prithvi_path: Path) -> None:
    text = prithvi_path.read_text()
    changed_any = False

    text, changed = safe_replace(
        text,
        "        self, x: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio=0.75\n    ):",
        "        self, x: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio=0.75,\n        noise: torch.Tensor | None = None,\n    ):",
        "encoder forward signature",
    )
    changed_any = changed_any or changed

    text, changed = safe_replace(
        text,
        "        x, mask, ids_restore = self.random_masking(x, mask_ratio)",
        "        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise=noise)",
        "encoder random_masking call",
    )
    changed_any = changed_any or changed

    text, changed = safe_replace(
        text,
        "        self,\n        pixel_values: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio: float = None,\n    ):",
        "        self,\n        pixel_values: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio: float = None,\n        noise: torch.Tensor | None = None,\n    ):",
        "model forward signature",
    )
    changed_any = changed_any or changed

    text, changed = safe_replace(
        text,
        "        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)",
        "        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio, noise=noise)",
        "model encoder call",
    )
    changed_any = changed_any or changed

    if changed_any:
        prithvi_path.write_text(text)
        print(f"Patched {prithvi_path.name} to support custom frame masking.")
    else:
        print(f"No custom-noise patch needed for {prithvi_path.name}.")


def validate_paths(base_dir: Path) -> None:
    required = [
        base_dir / "config.json",
        base_dir / "inference.py",
        base_dir / "prithvi_mae.py",
        base_dir / "Prithvi_EO_V2_tiny_TL.pt",
        base_dir / "examples",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files/directories:\n" + "\n".join(missing)
        )


def get_input_files(base_dir: Path, requested_files: list[str] | None) -> list[Path]:
    names = requested_files if requested_files else DEFAULT_EXAMPLES
    files = [base_dir / "examples" / name for name in names]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input TIFF files:\n" + "\n".join(missing))
    return files


def import_local_modules(base_dir: Path):
    sys.path.insert(0, str(base_dir))
    from prithvi_mae import PrithviMAE  # type: ignore
    from inference import load_example  # type: ignore
    return PrithviMAE, load_example


def load_model(base_dir: Path, checkpoint_filename: str,
               num_frames: int, device: torch.device):
    PrithviMAE, _ = import_local_modules(base_dir)

    config_path = base_dir / "config.json"
    checkpoint_path = base_dir / checkpoint_filename

    with config_path.open("r") as f:
        config = yaml.safe_load(f)["pretrained_cfg"]

    bands = config["bands"]
    mean = config["mean"]
    std = config["std"]
    config.update(num_frames=num_frames, in_chans=len(bands))

    model = PrithviMAE(**config).to(device)

    load_kwargs = {"map_location": device}
    try:
        state_dict = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        state_dict = torch.load(checkpoint_path, **load_kwargs)

    for key in list(state_dict.keys()):
        if key == "encoder.pos_embed":
            state_dict[key] = model.encoder.pos_embed
        elif key == "decoder.decoder_pos_embed":
            state_dict[key] = model.decoder.decoder_pos_embed

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, bands, mean, std


def load_sample(
    base_dir: Path,
    data_files: list[Path],
    mean: list[float],
    std: list[float],
    device: torch.device,
):
    _, load_example = import_local_modules(base_dir)
    input_data, temporal_coords, location_coords, meta_data = load_example(
        file_paths=[str(path) for path in data_files],
        mean=mean,
        std=std,
    )

    x = torch.tensor(input_data, dtype=torch.float32, device=device)
    temporal_coords = torch.tensor(
        temporal_coords, dtype=torch.float32, device=device
    ).unsqueeze(0)
    location_coords = torch.tensor(
        location_coords[0], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return x, temporal_coords, location_coords, meta_data


def validate_frame_indices(frames_to_mask: Iterable[int], num_frames: int) -> list[int]:
    frames = sorted(set(frames_to_mask))
    bad = [idx for idx in frames if idx < 0 or idx >= num_frames]
    if bad:
        raise ValueError(
            f"Invalid frame indices {bad}. Valid range is 0 to {num_frames - 1}."
        )
    return frames


def build_noise(x: torch.Tensor, frames_to_mask: list[int], device: torch.device):
    patch_size = 16
    tokens_per_frame = (x.shape[-2] // patch_size) * (x.shape[-1] // patch_size)
    seq_length = x.shape[2] * tokens_per_frame

    noise = torch.zeros((1, seq_length), dtype=torch.float32, device=device)
    for frame_idx in frames_to_mask:
        start = frame_idx * tokens_per_frame
        end = (frame_idx + 1) * tokens_per_frame
        noise[:, start:end] = 1.0

    return noise, tokens_per_frame, seq_length


@torch.no_grad()
def run_masked_inference(
    model,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    frames_to_mask: list[int],
    device: torch.device,
):
    noise, tokens_per_frame, seq_length = build_noise(x, frames_to_mask, device)
    mask_ratio = len(frames_to_mask) / x.shape[2]

    loss, pred, mask = model(
        x,
        temporal_coords=temporal_coords,
        location_coords=location_coords,
        mask_ratio=mask_ratio,
        noise=noise,
    )

    orig_h, orig_w = x.shape[-2], x.shape[-1]

    mask_img = model.unpatchify(
        mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]).detach().cpu(),
        image_size=(orig_h, orig_w),
    )
    pred_img = model.unpatchify(
        pred.detach().cpu(),
        image_size=(orig_h, orig_w),
    )

    x_cpu = x.detach().cpu()
    rec_img = x_cpu.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]
    mask_by_frame = mask.view(1, x.shape[2], tokens_per_frame)[0]

    return {
        "loss": float(loss),
        "pred": pred,
        "mask": mask,
        "x_cpu": x_cpu,
        "rec_img": rec_img,
        "mask_by_frame": mask_by_frame,
        "tokens_per_frame": tokens_per_frame,
        "seq_length": seq_length,
        "mask_ratio": mask_ratio,
    }


def tensor_to_rgb(
    img_cthw: torch.Tensor,
    frame_idx: int,
    bands: list[str],
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    rgb_channels = [bands.index("B04"), bands.index("B03"), bands.index("B02")]
    mean_t = torch.tensor(np.asarray(mean)[:, None, None], dtype=torch.float32)
    std_t = torch.tensor(np.asarray(std)[:, None, None], dtype=torch.float32)

    img = img_cthw[:, frame_idx, :, :].clone()
    img = (img * std_t) + mean_t
    img = img[rgb_channels, :, :]
    img = torch.clamp(img / 3000.0, 0, 1)
    return img.permute(1, 2, 0).numpy()


def save_comparison_plot(
    out_path: Path,
    x_cpu: torch.Tensor,
    rec_img: torch.Tensor,
    frames_to_mask: list[int],
    bands: list[str],
    mean: list[float],
    std: list[float],
) -> None:
    num_frames = x_cpu.shape[2]
    fig, axes = plt.subplots(2, num_frames, figsize=(4 * num_frames, 8))
    if num_frames == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for t in range(num_frames):
        axes[0, t].imshow(tensor_to_rgb(x_cpu[0], t, bands, mean, std))
        axes[0, t].set_title(f"original t{t}")
        axes[0, t].axis("off")

        title = f"reconstructed t{t}"
        if t in frames_to_mask:
            title += " <-- fully masked"
        axes[1, t].imshow(tensor_to_rgb(rec_img[0], t, bands, mean, std))
        axes[1, t].set_title(title)
        axes[1, t].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(
    x_cpu: torch.Tensor,
    rec_img: torch.Tensor,
    mask_by_frame: torch.Tensor,
    frames_to_mask: list[int],
    loss: float,
) -> dict:
    per_frame_mae = {}
    mean_mask_value_by_frame = {}

    for t in range(x_cpu.shape[2]):
        mae = torch.mean(torch.abs(rec_img[0, :, t] - x_cpu[0, :, t])).item()
        per_frame_mae[t] = mae
        mean_mask_value_by_frame[t] = mask_by_frame[t].float().mean().item()

    return {
        "frames_to_mask": frames_to_mask,
        "loss": loss,
        "per_frame_mae": per_frame_mae,
        "mean_mask_value_by_frame": mean_mask_value_by_frame,
    }


def evaluate_single_frame_masking(
    model,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    device: torch.device,
) -> list[dict]:
    results = []
    for frame_idx in range(x.shape[2]):
        run = run_masked_inference(
            model=model,
            x=x,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
            frames_to_mask=[frame_idx],
            device=device,
        )
        mae = torch.mean(
            torch.abs(run["rec_img"][0, :, frame_idx] - run["x_cpu"][0, :, frame_idx])
        ).item()
        results.append(
            {
                "masked_frame": frame_idx,
                "loss": run["loss"],
                "mae": mae,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_paths(base_dir)
    patch_inference_py(base_dir / "inference.py")
    patch_prithvi_mae(base_dir / "prithvi_mae.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_files = get_input_files(base_dir, args.input_files)
    print("Input files:")
    for path in data_files:
        print(f"  - {path.name}")

    model, bands, mean, std = load_model(
        base_dir=base_dir,
        num_frames=len(data_files),
        device=device,
    )

    x, temporal_coords, location_coords, _ = load_sample(
        base_dir=base_dir,
        data_files=data_files,
        mean=mean,
        std=std,
        device=device,
    )

    frames_to_mask = validate_frame_indices(args.frames_to_mask, x.shape[2])
    run = run_masked_inference(
        model=model,
        x=x,
        temporal_coords=temporal_coords,
        location_coords=location_coords,
        frames_to_mask=frames_to_mask,
        device=device,
    )

    metrics = compute_metrics(
        x_cpu=run["x_cpu"],
        rec_img=run["rec_img"],
        mask_by_frame=run["mask_by_frame"],
        frames_to_mask=frames_to_mask,
        loss=run["loss"],
    )
    metrics["tokens_per_frame"] = run["tokens_per_frame"]
    metrics["seq_length"] = run["seq_length"]
    metrics["mask_ratio"] = run["mask_ratio"]

    save_comparison_plot(
        out_path=output_dir / "comparison.png",
        x_cpu=run["x_cpu"],
        rec_img=run["rec_img"],
        frames_to_mask=frames_to_mask,
        bands=bands,
        mean=mean,
        std=std,
    )

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMain run summary")
    print(f"  loss: {metrics['loss']:.6f}")
    print(f"  frames_to_mask: {metrics['frames_to_mask']}")
    print(f"  mask_ratio: {metrics['mask_ratio']:.4f}")
    print("  mean mask value by frame:")
    for frame_idx, value in metrics["mean_mask_value_by_frame"].items():
        print(f"    frame {frame_idx}: {value:.4f}")
    print("  per-frame MAE:")
    for frame_idx, value in metrics["per_frame_mae"].items():
        print(f"    frame {frame_idx}: {value:.6f}")
    print(f"\nSaved: {output_dir / 'comparison.png'}")
    print(f"Saved: {output_dir / 'metrics.json'}")

    if args.evaluate_all_single_frames:
        single_frame_results = evaluate_single_frame_masking(
            model=model,
            x=x,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
            device=device,
        )
        with (output_dir / "single_frame_results.json").open("w") as f:
            json.dump(single_frame_results, f, indent=2)
        print(f"Saved: {output_dir / 'single_frame_results.json'}")
        print("\nSingle-frame masking summary")
        for row in single_frame_results:
            print(row)


if __name__ == "__main__":
    main()
