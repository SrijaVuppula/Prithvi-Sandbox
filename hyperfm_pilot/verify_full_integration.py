"""
Full end-to-end integration test: ControlNetEncoderAdapter (input side) +
SpectralOutputAdapter (output side), both wired onto the real model via
hooks, driven through run_masked_forward_trainable -- a gradient-enabled
replica of run_masked_forward (which is @torch.no_grad()-decorated and
therefore unusable for training; see trainable_forward.py for why).

Checks, per backbone:
  1. The call succeeds and produces a 291-band output at the decoder.
  2. Gradients from that output flow ONLY into the two new trainable
     pieces (band_adapter.hint_encoder, spectral_head) -- every native
     model parameter (encoder blocks, decoder blocks, both frozen
     patch_embed/decoder_pred) gets no gradient.
  3. Trainable parameter count, for a sense of how light this is to train.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402
from controlnet_encoder_adapter import ControlNetEncoderAdapter  # noqa: E402
from spectral_output_head import SpectralOutputAdapter  # noqa: E402
from trainable_forward import run_masked_forward_trainable  # noqa: E402

BACKBONE_SPECS = {
    "100M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_100M", patch_size=16, embed_dim=768),
    "600M": dict(base_dir=Path.home() / "Prithvi" / "prithvi_600M", patch_size=14, embed_dim=1280),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

for name, spec in BACKBONE_SPECS.items():
    print(f"{'=' * 60}\n{name}\n{'=' * 60}")
    base_dir = spec["base_dir"]

    checkpoints = list(base_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        print(f"Expected exactly one .pt file in {base_dir}, found: {checkpoints}")
        continue

    model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
        backbone_name=name,
        base_dir=str(base_dir),
        checkpoint_filename=checkpoints[0].name,
        num_frames=1,
        device=device,
    )
    print(f"loaded OK -- spatial_patch={spatial_patch}, bands={bands}")

    try:
        enc_adapter = ControlNetEncoderAdapter(
            model.encoder, embed_dim=spec["embed_dim"], patch_size=spec["patch_size"],
        ).to(device)
        dec_adapter = SpectralOutputAdapter(model.decoder, patch_size=spec["patch_size"]).to(device)
    except Exception as e:
        print(f"adapter init FAILED: {type(e).__name__}: {e}")
        continue
    print("both adapters attached OK")

    pace_cube = torch.randn(1, 291, 224, 224, device=device)
    placeholder = torch.zeros(1, 6, 1, 224, 224, device=device)
    enc_adapter.set_pace_cube(pace_cube)

    try:
        result = run_masked_forward_trainable(
            model, placeholder, temporal_coords=None, location_coords=None,
            mask_ratio=0.2, noise=None,
        )
    except Exception as e:
        print(f"run_masked_forward_trainable FAILED: {type(e).__name__}: {e}")
        continue

    if dec_adapter.last_output is None:
        print("decoder hook never fired.")
        continue
    print(f"end-to-end 291-band output shape: {tuple(dec_adapter.last_output.shape)}")

    # --- gradient isolation across the WHOLE model ---
    model.zero_grad()
    enc_adapter.zero_grad()
    dec_adapter.zero_grad()

    pace_cube2 = torch.randn(1, 291, 224, 224, device=device)
    enc_adapter.set_pace_cube(pace_cube2)
    run_masked_forward_trainable(model, placeholder, temporal_coords=None, location_coords=None,
                                  mask_ratio=0.2, noise=None)
    loss = dec_adapter.last_output.sum()
    loss.backward()

    hint_grad = sum(p.grad.abs().sum().item() for p in enc_adapter.band_adapter.hint_encoder.parameters()
                     if p.grad is not None)
    head_grad = sum(p.grad.abs().sum().item() for p in dec_adapter.spectral_head.parameters()
                     if p.grad is not None)
    native_grads_nonzero = [n for n, p in model.named_parameters() if p.grad is not None]

    print(f"hint_encoder grad norm: {hint_grad:.4f}")
    print(f"spectral_head grad norm: {head_grad:.4f}")
    if native_grads_nonzero:
        print(f"WARNING -- {len(native_grads_nonzero)} native model params got gradients, e.g.: "
              f"{native_grads_nonzero[:5]}")
    else:
        print("native model fully frozen -- no native param received a gradient")

    trainable = sum(p.numel() for p in enc_adapter.band_adapter.hint_encoder.parameters()) + \
        sum(p.numel() for p in dec_adapter.spectral_head.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} ({100 * trainable / total:.3f}% of the {total:,} total)")
