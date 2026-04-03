"""
inference/runner.py

Reusable inference primitives for the Prithvi EO 2.0 baseline sweep.
All patching, model loading, and forward-pass logic lives here.
"""

from __future__ import annotations
from pathlib import Path
import sys
import torch


# ── Source patching ──────────────────────────────────────────────────────────

def _safe_replace(text: str, old: str, new: str, description: str) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new), True
    if new in text:
        return text, False
    raise RuntimeError(f"Could not find expected text for patch: {description}")


def patch_inference_py(inference_path: Path) -> None:
    text = inference_path.read_text()
    changed_any = False

    text, changed = _safe_replace(
        text,
        'temporal_coords = torch.Tensor(temporal_coords, device=device).unsqueeze(0)',
        'temporal_coords = torch.tensor(temporal_coords, dtype=torch.float32, device=device).unsqueeze(0)',
        'temporal_coords tensor creation',
    )
    changed_any = changed_any or changed

    text, changed = _safe_replace(
        text,
        'location_coords = torch.Tensor(location_coords[0], device=device).unsqueeze(0)',
        'location_coords = torch.tensor(location_coords[0], dtype=torch.float32, device=device).unsqueeze(0)',
        'location_coords tensor creation',
    )
    changed_any = changed_any or changed

    if changed_any:
        inference_path.write_text(text)
        print(f"  Patched {inference_path.name} for torch.tensor compatibility.")
    else:
        print(f"  {inference_path.name} already patched.")


def patch_prithvi_mae(prithvi_path: Path) -> None:
    text = prithvi_path.read_text()
    changed_any = False

    patches = [
        (
            "        self, x: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio=0.75\n    ):",
            "        self, x: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio=0.75,\n        noise: torch.Tensor | None = None,\n    ):",
            "encoder forward signature",
        ),
        (
            "        x, mask, ids_restore = self.random_masking(x, mask_ratio)",
            "        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise=noise)",
            "encoder random_masking call",
        ),
        (
            "        self,\n        pixel_values: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio: float = None,\n    ):",
            "        self,\n        pixel_values: torch.Tensor,\n        temporal_coords: None | torch.Tensor = None,\n        location_coords: None | torch.Tensor = None,\n        mask_ratio: float = None,\n        noise: torch.Tensor | None = None,\n    ):",
            "model forward signature",
        ),
        (
            "        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)",
            "        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio, noise=noise)",
            "model encoder call",
        ),
    ]

    for old, new, desc in patches:
        text, changed = _safe_replace(text, old, new, desc)
        changed_any = changed_any or changed

    if changed_any:
        prithvi_path.write_text(text)
        print(f"  Patched {prithvi_path.name} for custom frame masking.")
    else:
        print(f"  {prithvi_path.name} already patched.")


def apply_all_patches(base_dir: Path) -> None:
    """Call this once at the start of any script that uses Prithvi."""
    patch_inference_py(base_dir / "inference.py")
    patch_prithvi_mae(base_dir / "prithvi_mae.py")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    base_dir: Path,
    checkpoint_filename: str,
    num_frames: int,
    device: torch.device,
):
    import importlib.util, yaml

    # Load prithvi_mae.py directly by path to avoid any import collisions
    mae_path = Path(base_dir) / "prithvi_mae.py"
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    prithvi_mae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prithvi_mae)
    PrithviMAE = prithvi_mae.PrithviMAE

    with (Path(base_dir) / "config.json").open() as f:
        config = yaml.safe_load(f)["pretrained_cfg"]

    bands = config["bands"]
    mean  = config["mean"]
    std   = config["std"]
    config.update(num_frames=num_frames, in_chans=len(bands))

    model = PrithviMAE(**config).to(device)

    checkpoint_path = Path(base_dir) / checkpoint_filename
    try:
        state_dict = torch.load(
            checkpoint_path, weights_only=True, map_location=device
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    for key in list(state_dict.keys()):
        if key == "encoder.pos_embed":
            state_dict[key] = model.encoder.pos_embed
        elif key == "decoder.decoder_pos_embed":
            state_dict[key] = model.decoder.decoder_pos_embed

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  Loaded {checkpoint_filename} ({num_frames} frames) on {device}")
    patch_size = config.get("patch_size", [1, 16, 16])
    spatial_patch = patch_size[1] if isinstance(patch_size, list) else patch_size
    print(f"  Loaded {checkpoint_filename} ({num_frames} frames) patch={spatial_patch} on {device}")
    return model, bands, mean, std, spatial_patch


# ── Forward pass ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_one_condition(
    model,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    frame_idx: int,
    patch_size: int,
    device: torch.device,
) -> dict:
    """
    Single masked forward pass. Masks one full frame, returns reconstruction.

    Returns dict with keys: loss, mask_ratio, x_cpu, rec_img
    """
    from masking.temporal_masker import build_noise_from_frame_idx

    noise, _, _ = build_noise_from_frame_idx(x, frame_idx, patch_size, device)
    mask_ratio  = 1.0 / x.shape[2]

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
    pred_img = model.unpatchify(pred.detach().cpu(), image_size=(orig_h, orig_w))

    x_cpu   = x.detach().cpu()
    rec_img = x_cpu.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]

    return {
        "loss":       float(loss),
        "mask_ratio": mask_ratio,
        "x_cpu":      x_cpu,
        "rec_img":    rec_img,
    }