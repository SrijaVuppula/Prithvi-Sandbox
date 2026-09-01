"""
Diagnostic: inspect the decoder side of the loaded Prithvi model to design
the output-side band-expansion head (291-band counterpart to the encoder's
input-side ControlNet adapter). 100M only -- both backbones share the same
decoder class, same reasoning as diagnose_patch_embed.py.

Looking for:
  - the decoder's final prediction head (something like
    decoder_pred: Linear(decoder_embed_dim, patch_size^2 * 6))
  - the unpatchify/patchify convention -- does the flattened prediction
    order band-then-space or space-then-band? Needed to correctly compose
    a 291-band expansion head from the frozen 6-band one.
"""
import inspect
import sys
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

print("=" * 60)
print("Top-level model attributes:")
print([n for n, _ in model.named_children()])

decoder = getattr(model, "decoder", None)
if decoder is None:
    print("model.decoder not found -- full model repr:")
    print(model)
else:
    print("\n" + "=" * 60)
    print("decoder repr:")
    print(decoder)

    print("\n" + "=" * 60)
    print("submodules with 'pred' or 'head' in the name:")
    for n, m in decoder.named_modules():
        if "pred" in n.lower() or "head" in n.lower():
            print(f"  decoder.{n}: {type(m).__name__} -- {m}")

    print("\n" + "=" * 60)
    print("submodules with 'embed' in the name (decoder input projection):")
    for n, m in decoder.named_modules():
        if "embed" in n.lower():
            print(f"  decoder.{n}: {type(m).__name__} -- {m}")

print("\n" + "=" * 60)
print("Looking for an unpatchify method:")
for holder_name, holder in [("model", model), ("model.encoder", model.encoder), ("decoder", decoder)]:
    if holder is None:
        continue
    fn = getattr(holder, "unpatchify", None)
    if fn is not None:
        print(f"\nFound {holder_name}.unpatchify:")
        try:
            print(inspect.getsource(fn))
        except Exception as e:
            print(f"(couldn't get source: {e})")

print("\n" + "=" * 60)
print("Looking for a patchify method (often documents the same convention):")
for holder_name, holder in [("model", model), ("model.encoder", model.encoder), ("decoder", decoder)]:
    if holder is None:
        continue
    fn = getattr(holder, "patchify", None)
    if fn is not None:
        print(f"\nFound {holder_name}.patchify:")
        try:
            print(inspect.getsource(fn))
        except Exception as e:
            print(f"(couldn't get source: {e})")
