"""
Full ControlNet-adapter integration: wires ControlNetBandAdapter (input
side, spectral_band_adapter.py) and SpectralOutputAdapter (output side,
spectral_output_head.py) so both run through a single forward pass, without
either side reimplementing any encoder/decoder internals (masking,
temporal/location embeddings, transformer blocks). Use with
run_masked_forward_trainable (trainable_forward.py), not the real
run_masked_forward -- that one is @torch.no_grad()-decorated.

Encoder side: a forward hook on patch_embed REPLACES its output with the
ControlNet-adapted tokens (native 6-band path + 291-band hint), computed
from a real PACE cube stashed via set_pace_cube() before the call --
run_masked_forward's own input argument only ever carries 6 bands by
construction (that's all patch_embed's frozen Conv3d accepts), so there's
no other path to hand it the full spectrum for the hint branch. The whole
encoder (blocks, cls_token, norm, position/temporal/location embeddings) is
frozen, not just patch_embed -- mirrors the decoder side's full freeze.

Decoder side: unchanged from spectral_output_head.py -- observes
decoder_pred's input via hook, doesn't touch decoder internals.
"""

import torch
import torch.nn as nn

from spectral_band_adapter import ControlNetBandAdapter


class ControlNetEncoderAdapter(nn.Module):
    """
    Wraps a Prithvi ENCODER (not just patch_embed) so patch_embed's output
    is replaced by the ControlNet-adapted tokens computed from a real
    291-band PACE cube, while every other encoder parameter (blocks,
    cls_token, norm, position/temporal/location embeddings) is frozen --
    mirrors SpectralOutputAdapter freezing the whole decoder, not just
    decoder_pred. Takes the full encoder (not patch_embed alone) so that
    freezing scope is correct; patch_embed is extracted internally.

    Usage:
        enc_adapter = ControlNetEncoderAdapter(model.encoder,
                                                embed_dim=..., patch_size=...)
        enc_adapter.set_pace_cube(real_pace_tile)              # (B,291,H,W)
        placeholder = torch.zeros(B, 6, 1, H, W, device=...)   # shape-only,
                                                                 # content ignored
        result = run_masked_forward_trainable(model, placeholder, ...)

    The hook removes and re-attaches itself during its own callback --
    band_adapter.forward() internally calls this SAME patch_embed module to
    get the frozen 6-band path, which would otherwise re-trigger the hook
    (infinite recursion).
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, patch_size: int,
                 in_bands: int = 291, pace_swir_centers_nm=None, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        patch_embed = encoder.patch_embed
        self.patch_embed = patch_embed
        self.band_adapter = ControlNetBandAdapter(
            patch_embed=patch_embed, embed_dim=embed_dim, patch_size=patch_size,
            in_bands=in_bands, pace_swir_centers_nm=pace_swir_centers_nm,
        )
        self._pace_cube = None
        self._hook_handle = patch_embed.register_forward_hook(self._replace)

    def set_pace_cube(self, pace_cube):
        """pace_cube: (B, 291, H, W). Call before each forward pass that
        invokes the wrapped patch_embed (e.g. each run_masked_forward call)."""
        self._pace_cube = pace_cube

    def _replace(self, module, inputs, output):
        if self._pace_cube is None:
            raise RuntimeError(
                "ControlNetEncoderAdapter.set_pace_cube() must be called "
                "before the encoder forward pass -- no PACE cube stashed."
            )
        self._hook_handle.remove()  # avoid recursion: band_adapter calls this same module
        try:
            result = self.band_adapter(self._pace_cube)
        finally:
            self._hook_handle = module.register_forward_hook(self._replace)
        return result

    def remove_hook(self):
        self._hook_handle.remove()
