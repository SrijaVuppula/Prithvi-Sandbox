"""
Output-side spectral expansion head -- 291-band counterpart to the
encoder-side ControlNet band adapter (spectral_band_adapter.py).

Confirmed from the real model (Aug 31, model.patchify/unpatchify source,
100M): decoder_pred is a Linear(decoder_embed_dim, T*P*P*6), and the
flattened output is rearranged as "(t h w) (s p q c)" -- for each spatial
patch position, the 6 channel values are the innermost (fastest-varying)
block. This means expanding 6 -> 291 bands is a per-position linear
expansion; no interleaving ambiguity.

Design:
  - decoder_blocks, decoder_embed, decoder_norm, and the ORIGINAL
    decoder_pred all stay frozen, exactly as-is.
  - A forward hook on decoder_pred captures its input tensor (the decoder's
    final hidden states, post decoder_norm) without touching or
    reimplementing anything about how the decoder assembles that tensor
    (mask tokens, temporal/location embeddings, etc.).
  - A new trainable SpectralOutputHead runs on that same captured tensor,
    producing a 291-band prediction in the same flattened-patch layout.
  - Initialization: composed from the frozen decoder_pred's weight/bias,
    expanded per spatial position via a (291, 6) row-normalized
    inverse-distance spectral-proximity matrix (same physical basis as the
    encoder side's resampling matrix, just normalized the other way, and
    inverse-distance rather than Gaussian -- see build_pace_expansion_matrix
    docstring for why). At init, this head's prediction for each PACE band
    is a sensible blend of the pretrained 6-band prediction -- not random
    noise. Training moves it from there.
"""

import torch
import torch.nn as nn

from spectral_band_adapter import HLS_BAND_CENTERS_NM, build_pace_wavelength_array


def build_pace_expansion_matrix(pace_wavelengths_nm, power=2.0, eps=1e-3):
    """
    (291, 6) matrix used to INITIALIZE the 291-band decoder head from the
    frozen 6-band one, via inverse-distance weighting (deliberately NOT the
    Gaussian used in build_hls_resampling_matrix on the encoder side).

    The encoder-side Gaussian works because each of the 6 HLS centers
    always has PACE bands within a few nm in the continuous blue/red FPA
    ranges -- every column has close neighbors. Going the other direction,
    many of the 291 PACE bands sit far from EVERY HLS center (e.g. the
    ~900-1600 nm gap where none of the 6 centers fall); a narrow Gaussian
    underflows to exactly zero for those rows, and row-normalizing then
    divides ~0 by ~0. Inverse-distance weighting never vanishes, so every
    one of the 291 rows gets a defined (if lower-confidence, for
    far-from-everything bands) blend of the 6 values.
    """
    centers = torch.tensor(list(HLS_BAND_CENTERS_NM.values()), dtype=torch.float32)  # (6,)
    dist = (pace_wavelengths_nm.unsqueeze(1) - centers.unsqueeze(0)).abs()  # (291, 6)
    weights = 1.0 / (dist + eps) ** power
    weights = weights / weights.sum(dim=1, keepdim=True)  # row-normalize, never divides by ~0
    return weights  # (291, 6)


class SpectralOutputHead(nn.Module):
    """
    291-band counterpart to a frozen decoder_pred: Linear(decoder_embed_dim,
    temporal_patch*patch_size*patch_size*6).

    forward() takes the SAME input tensor decoder_pred would receive
    (decoder hidden states, post decoder_norm) and returns a 291-band
    prediction in the identical flattened-patch layout, so it can be
    unpatchified with the same rearrange pattern just substituting
    num_channels=291 for 6.
    """

    def __init__(self, decoder_pred: nn.Linear, patch_size: int, native_bands: int = 6,
                 out_bands: int = 291, temporal_patch: int = 1,
                 pace_swir_centers_nm=None):
        super().__init__()
        num_positions = temporal_patch * patch_size * patch_size
        expected_in = num_positions * native_bands
        assert decoder_pred.out_features == expected_in, (
            f"decoder_pred.out_features={decoder_pred.out_features} != "
            f"expected {expected_in} (temporal_patch={temporal_patch} * "
            f"patch_size={patch_size}^2 * native_bands={native_bands}) -- "
            "the patchify rearrange convention may have changed; re-check "
            "model.patchify's source before trusting this head."
        )
        decoder_embed_dim = decoder_pred.in_features
        self.num_positions = num_positions
        self.out_bands = out_bands

        self.proj = nn.Linear(decoder_embed_dim, num_positions * out_bands)

        # --- Composed initialization, not random ---
        wavelengths = build_pace_wavelength_array(pace_swir_centers_nm)
        expansion_matrix = build_pace_expansion_matrix(wavelengths)  # (291, 6)
        # Built fresh on CPU by default; decoder_pred.weight may already be on
        # CUDA (model loaded with device=cuda before this head is constructed).
        # This matrix is only used transiently here, not kept as a buffer, so
        # move it to match rather than registering it.
        expansion_matrix = expansion_matrix.to(decoder_pred.weight.device)

        with torch.no_grad():
            w = decoder_pred.weight.reshape(num_positions, native_bands, decoder_embed_dim)
            b = decoder_pred.bias.reshape(num_positions, native_bands)

            # per-position: (291,6) x (6,decoder_embed_dim) -> (291,decoder_embed_dim)
            new_w = torch.einsum("oc,pce->poe", expansion_matrix, w)  # (num_positions, 291, embed_dim)
            new_b = torch.einsum("oc,pc->po", expansion_matrix, b)    # (num_positions, 291)

            self.proj.weight.copy_(new_w.reshape(num_positions * out_bands, decoder_embed_dim))
            self.proj.bias.copy_(new_b.reshape(num_positions * out_bands))

    def forward(self, decoder_hidden_states):
        """
        decoder_hidden_states: (B, N, decoder_embed_dim) -- same tensor
        decoder_pred normally consumes.
        returns: (B, N, num_positions * out_bands) per the same flattened
        "(s p q c)" convention as the original decoder_pred.
        """
        return self.proj(decoder_hidden_states)


class SpectralOutputAdapter(nn.Module):
    """
    Wraps a loaded Prithvi decoder with the 291-band head via a forward
    hook on decoder_pred, so the decoder's own forward pass (mask tokens,
    temporal/location embeddings, block stack) runs completely untouched --
    only its final linear layer's input gets intercepted and reused.

    Usage: attach once, then call the decoder exactly as you already do
    elsewhere (e.g. via run_masked_forward) -- no change to that call.
    After it returns, read outputs off this adapter:

        adapter = SpectralOutputAdapter(model.decoder, patch_size=cfg_patch_size)
        ... run your existing forward pass (run_masked_forward(...)) ...
        pred_291 = adapter.last_output          # (B, N, patch^2*291)
        pred_6_native = adapter.last_native_output  # (B, N, patch^2*6), unchanged

    freeze_decoder=True (default) freezes every existing decoder parameter
    (decoder_embed, decoder_blocks, decoder_norm, decoder_pred) -- only the
    new spectral_head is trainable, matching the encoder-side freeze.
    """

    def __init__(self, decoder: nn.Module, patch_size: int, native_bands: int = 6,
                 out_bands: int = 291, temporal_patch: int = 1,
                 pace_swir_centers_nm=None, freeze_decoder: bool = True):
        super().__init__()
        self.decoder = decoder
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        self.spectral_head = SpectralOutputHead(
            decoder_pred=decoder.decoder_pred,
            patch_size=patch_size,
            native_bands=native_bands,
            out_bands=out_bands,
            temporal_patch=temporal_patch,
            pace_swir_centers_nm=pace_swir_centers_nm,
        )

        self.last_output = None
        self.last_native_output = None
        self._hook_handle = decoder.decoder_pred.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        decoder_pred_input = inputs[0]  # (B, N, decoder_embed_dim)
        self.last_native_output = output  # (B, N, patch^2*6) -- untouched
        self.last_output = self.spectral_head(decoder_pred_input)  # (B, N, patch^2*291)

    def remove_hook(self):
        self._hook_handle.remove()
