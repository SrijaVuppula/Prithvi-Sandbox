"""
Diagnostic-only: isolate the Conv3d padding error from the adapter run.
100M only -- both backbones share the same PatchEmbed class, so whatever
this finds applies to 600M too.
"""
import inspect
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))
from terratorch_loader import load_prithvi_from_terratorch  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = Path.home() / "Prithvi" / "prithvi_100M"

model, bands, mean, std, spatial_patch = load_prithvi_from_terratorch(
    backbone_name="100M",
    base_dir=str(base_dir),
    checkpoint_filename="Prithvi_EO_V2_100M_TL.pt",
    num_frames=1,
    device=device,
)

patch_embed = model.encoder.patch_embed

print("=" * 60)
print("patch_embed repr:")
print(patch_embed)
print("\npatch_embed.proj repr:")
print(patch_embed.proj)
print(f"\nproj.kernel_size={patch_embed.proj.kernel_size}, "
      f"stride={patch_embed.proj.stride}, padding={patch_embed.proj.padding}")

print("\n" + "=" * 60)
print("PatchEmbed.forward source:")
try:
    print(inspect.getsource(type(patch_embed).forward))
except Exception as e:
    print(f"(couldn't get source: {e})")

print("\n" + "=" * 60)
print("Attempt 1: call patch_embed.proj directly (bypass PatchEmbed.forward)")
x = torch.randn(1, 6, 1, 224, 224, device=device)
try:
    out = patch_embed.proj(x)
    print(f"proj(x) OK -- shape {tuple(out.shape)}")
except Exception:
    print("proj(x) FAILED:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Attempt 2: call patch_embed(x) -- the full wrapped forward")
try:
    out = patch_embed(x)
    print(f"patch_embed(x) OK -- shape {tuple(out.shape)}")
except Exception:
    print("patch_embed(x) FAILED, full traceback:")
    traceback.print_exc()
