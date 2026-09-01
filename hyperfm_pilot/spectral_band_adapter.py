"""
ControlNet-style zero-init spectral band adapter.

Bridges PACE-OCI's 291-band hyperspectral cube to Prithvi EO 2.0's native
6-band (HLS-equivalent) input, per Kriti's recommendation (Zhang et al.,
ICCV 2023, arXiv:2302.05543).

Design:
  - Frozen path: PACE bands are spectrally resampled down to 6 HLS-equivalent
    band centers (Blue/Green/Red/NIR/SWIR1/SWIR2) and fed through Prithvi's
    existing (frozen) patch embedding + encoder, unchanged.
  - Hint path: a small trainable CNN ("SpectralHintEncoder") consumes the
    FULL 291-band cube and produces a per-patch-token hint tensor at the
    same spatial resolution as the frozen patch embeddings. Its final
    projection layer is zero-initialized, so at step 0 the hint is exactly
    zero and the frozen backbone runs identically to unmodified Prithvi.
  - The hint is added to the frozen patch embeddings before they enter the
    (frozen) transformer blocks. Only the hint encoder is trained.

One thing in here is still marked NEEDS VERIFICATION -- the PACE discrete
SWIR band wavelengths (placeholders, not read from real data). Everything
else has been confirmed against the real 100M/600M checkpoints on cuda2
(Aug 31 run of verify_band_adapter.py).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Band mapping: PACE's 291 bands -> 6 HLS-equivalent centers
# ---------------------------------------------------------------------------

HLS_BAND_CENTERS_NM = {
    "blue": 480,
    "green": 560,
    "red": 655,
    "nir": 865,
    "swir1": 1610,
    "swir2": 2200,
}

# PACE-OCI L1B band layout, per HyperFM250K: 291 bands total --
#   idx   0-118 : blue FPA, continuous ~314-605 nm  (119 bands)
#   idx 119-281 : red  FPA, continuous ~600-895 nm  (163 bands)
#   idx 282-290 : discrete SWIR, ~940-2258 nm        (9 bands)
# NEEDS VERIFICATION: the 9 discrete SWIR centers below are placeholders,
# NOT read from your actual data. Get the real array (should ship as
# metadata with HyperFM250K, or from PACE L1B band-center docs) and replace
# PACE_SWIR_CENTERS_NM.
PACE_SWIR_CENTERS_NM = [940, 1038, 1250, 1378, 1615, 1878, 2130, 2210, 2258]


def build_pace_wavelength_array(pace_swir_centers_nm=None):
    """Full 291-length wavelength array for PACE bands, nm."""
    blue = torch.linspace(314, 605, 119)
    red = torch.linspace(600, 895, 163)
    swir = torch.tensor(pace_swir_centers_nm or PACE_SWIR_CENTERS_NM, dtype=torch.float32)
    assert swir.numel() == 9, "expected 9 discrete SWIR bands"
    return torch.cat([blue, red, swir])  # (291,)


def build_hls_resampling_matrix(pace_wavelengths_nm, sigma_nm=15.0):
    """
    (291, 6) matrix mapping PACE bands -> 6 HLS-equivalent bands via a
    Gaussian spectral response centered on each HLS band, normalized to
    sum to 1 per output band. Applied as: hls_bands = pace_cube @ matrix.
    """
    centers = torch.tensor(list(HLS_BAND_CENTERS_NM.values()), dtype=torch.float32)  # (6,)
    dist = pace_wavelengths_nm.unsqueeze(1) - centers.unsqueeze(0)  # (291, 6)
    weights = torch.exp(-0.5 * (dist / sigma_nm) ** 2)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weights  # (291, 6)


def pace_to_hls_equivalent(pace_cube, resampling_matrix):
    """
    pace_cube: (B, 291, H, W)
    resampling_matrix: (291, 6)
    returns: (B, 6, H, W)
    """
    b, c, h, w = pace_cube.shape
    flat = pace_cube.permute(0, 2, 3, 1).reshape(-1, c)              # (B*H*W, 291)
    hls = flat @ resampling_matrix                                   # (B*H*W, 6)
    return hls.reshape(b, h, w, 6).permute(0, 3, 1, 2).contiguous()  # (B, 6, H, W)


# ---------------------------------------------------------------------------
# 2. Trainable hint encoder -- full 291-band cube -> per-patch hint tensor
# ---------------------------------------------------------------------------

class SpectralHintEncoder(nn.Module):
    """
    Small CNN, downsamples the full PACE cube to the same spatial grid as
    Prithvi's patch embeddings (H/patch_size, W/patch_size) and projects to
    embed_dim channels. Final conv is zero-initialized (ControlNet-style):
    at init, this module outputs exactly zero for any input, so the frozen
    backbone starts out completely undisturbed.

    Downsampling is a stride-1 feature stack followed by ONE conv with
    kernel_size=patch_size, stride=patch_size -- exact for any patch_size,
    not just powers of 2. This matters because 100M and 600M use different
    patch sizes (confirmed: 16 for 100M, 14 for 600M, via each backbone's
    own config.json -- don't hardcode).
    """

    def __init__(self, in_bands: int, embed_dim: int, patch_size: int,
                 hidden: int = 64, n_feature_layers: int = 2):
        super().__init__()
        layers = []
        c_in = in_bands
        for _ in range(n_feature_layers):
            layers += [
                nn.Conv2d(c_in, hidden, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(min(8, hidden), hidden),
                nn.SiLU(),
            ]
            c_in = hidden
        self.features = nn.Sequential(*layers)

        # Exact-grid downsample to match the frozen patch_embed's token grid,
        # whatever patch_size that backbone uses (14 for 600M, 16 for 100M).
        self.downsample = nn.Conv2d(c_in, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Zero-init final 1x1 projection -- the ControlNet trick.
        self.zero_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, pace_cube):
        """
        pace_cube: (B, 291, H, W)
        returns: (B, embed_dim, H/patch_size, W/patch_size), all zeros at init.
        """
        x = self.features(pace_cube)
        x = self.downsample(x)
        return self.zero_proj(x)


# ---------------------------------------------------------------------------
# 3. Full adapter -- wires frozen path + hint path together
# ---------------------------------------------------------------------------

class ControlNetBandAdapter(nn.Module):
    """
    Wraps a frozen Prithvi patch-embed call with the spectral hint branch.

    Usage (matches your existing pattern of calling model.encoder() directly
    rather than the exposed forward()):

        adapter = ControlNetBandAdapter(patch_embed=model.encoder.patch_embed,
                                         embed_dim=cfg["embed_dim"],
                                         patch_size=cfg["patch_size"])
        tokens = adapter(pace_cube)   # same shape self.patch_embed would emit
        # feed `tokens` into model.encoder.blocks(...) same as today

    Confirmed via verify_band_adapter.py (Aug 31 run, both 100M and 600M):
    patch_embed.proj is Conv3d, config.json patch_size is [1, 16, 16] /
    [1, 14, 14] -- the leading 1 is the temporal patch. patch_embed always
    needs a (B, C, T, H, W) input, T=1 for single-date PACE tiles.
    """

    def __init__(self, patch_embed: nn.Module, embed_dim: int, patch_size: int,
                 in_bands: int = 291, pace_swir_centers_nm=None, freeze_patch_embed=True):
        super().__init__()
        self.patch_embed = patch_embed
        if freeze_patch_embed:
            for p in self.patch_embed.parameters():
                p.requires_grad = False

        wavelengths = build_pace_wavelength_array(pace_swir_centers_nm)
        self.register_buffer("resampling_matrix", build_hls_resampling_matrix(wavelengths))

        self.hint_encoder = SpectralHintEncoder(
            in_bands=in_bands, embed_dim=embed_dim, patch_size=patch_size
        )

    def forward(self, pace_cube):
        """
        pace_cube: (B, 291, H, W) -- single-date PACE tile.
        returns tokens shaped to match whatever self.patch_embed normally emits.
        """
        hls_equivalent = pace_to_hls_equivalent(pace_cube, self.resampling_matrix)  # (B, 6, H, W)
        hls_equivalent = hls_equivalent.unsqueeze(2)  # (B, 6, 1, H, W) -- patch_embed is Conv3d, T=1

        frozen_grad_enabled = any(p.requires_grad for p in self.patch_embed.parameters())
        with torch.set_grad_enabled(frozen_grad_enabled):
            frozen_tokens = self.patch_embed(hls_equivalent)  # (B, N, embed_dim) or (B, embed_dim, h, w)

        hint = self.hint_encoder(pace_cube)  # (B, embed_dim, h, w)

        if frozen_tokens.dim() == 3:  # (B, N, embed_dim) -- already flattened
            hint = hint.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return frozen_tokens + hint
