"""
Full training-step verification, real data end to end: load a real PACE
tile -> apply spectral band masking (contiguous or scattered, matching the
Step 1 pilot's geometry) -> forward through both adapters with SPATIAL
masking disabled (mask_ratio=0 -- only the band masking constitutes the
task) -> unpatchify the 291-band prediction back to pixel space ->
masked_reconstruction_loss (scored only on hidden bands x valid pixels) ->
backward -> confirm gradient isolation still holds with the real loss, not
just dec_adapter.last_output.sum().
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
from pace_tile_loader import load_pace_tile, list_pace_tiles  # noqa: E402
from masked_reconstruction import apply_band_mask, unpatchify_bands, masked_reconstruction_loss  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

tiles = list_pace_tiles()
if not tiles:
    raise SystemExit("no .npy tiles found")
tile_path = tiles[0]
print(f"using real tile: {tile_path.name}")

target_cube, valid_mask = load_pace_tile(tile_path, target_size=112)
target_cube = target_cube.to(device)
valid_mask = valid_mask.to(device)
print(f"tile shape: {tuple(target_cube.shape)}, valid fraction: {valid_mask.float().mean().item():.4f}")

rng = np.random.default_rng(123)
MASK_RATIO = 0.2
GEOMETRY = "contiguous"
masked_cube, band_mask = apply_band_mask(target_cube, mask_ratio=MASK_RATIO, geometry=GEOMETRY, rng=rng)
print(f"band mask: {GEOMETRY}, {band_mask.sum().item()}/291 bands hidden")

for name, spec in BACKBONE_SPECS.items():
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    base_dir = spec["base_dir"]
    checkpoints = list(base_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        print(f"Expected exactly one .pt file in {base_dir}, found: {checkpoints}")
        continue

    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=name, base_dir=str(base_dir),
        checkpoint_filename=checkpoints[0].name, num_frames=1, device=device,
    )
    print(f"loaded OK -- spatial_patch={spatial_patch}")

    try:
        enc_adapter = ControlNetEncoderAdapter(
            model.encoder, embed_dim=spec["embed_dim"], patch_size=spec["patch_size"],
        ).to(device)
        dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)
    except Exception as e:
        print(f"adapter init FAILED: {type(e).__name__}: {e}")
        continue

    model.zero_grad()
    enc_adapter.zero_grad()
    dec_adapter.zero_grad()

    placeholder = torch.zeros(1, 6, 1, 112, 112, device=device)
    enc_adapter.set_pace_cube(masked_cube)  # model only ever sees the MASKED cube

    try:
        run_masked_forward_trainable(
            model, placeholder, temporal_coords=None, location_coords=None,
            mask_ratio=0.0, noise=None,  # spatial masking disabled -- band masking is the task
        )
    except Exception as e:
        print(f"forward FAILED: {type(e).__name__}: {e}")
        continue

    patch_tokens = dec_adapter.last_output[:, 1:, :]  # drop CLS token before unpatchify
    grid = 112 // spec["patch_size"]
    prediction = unpatchify_bands(
        patch_tokens, num_channels=291, patch_size=spec["patch_size"],
        num_patches_h=grid, num_patches_w=grid,
    )
    prediction = prediction.squeeze(2)  # drop the size-1 temporal dim -> (B, 291, H, W)
    print(f"unpatchified prediction shape: {tuple(prediction.shape)} (target: {tuple(target_cube.shape)})")
    if prediction.shape != target_cube.shape:
        print("SHAPE MISMATCH between prediction and target -- stopping here for this backbone")
        continue

    try:
        loss, n_scored = masked_reconstruction_loss(prediction, target_cube, band_mask, valid_mask)
    except RuntimeError as e:
        print(f"loss FAILED: {e}")
        continue
    print(f"masked reconstruction loss: {loss.item():.6f} (scored over {n_scored:,} pixel-band positions)")

    loss.backward()
    hint_grad = sum(p.grad.abs().sum().item() for p in enc_adapter.band_adapter.hint_encoder.parameters()
                     if p.grad is not None)
    head_grad = sum(p.grad.abs().sum().item() for p in dec_adapter.spectral_head.parameters()
                     if p.grad is not None)
    native_grads_nonzero = [n for n, p in model.named_parameters() if p.grad is not None]

    print(f"hint_encoder grad norm: {hint_grad:.6f}")
    print(f"spectral_head grad norm: {head_grad:.6f}")
    if native_grads_nonzero:
        print(f"WARNING -- {len(native_grads_nonzero)} native model params got gradients")
    else:
        print("native model fully frozen -- gradient isolation holds with the real loss")
