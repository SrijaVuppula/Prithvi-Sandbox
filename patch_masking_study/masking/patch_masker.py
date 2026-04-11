"""
patch_masking_study/masking/patch_masker.py

Builds a noise tensor that randomly masks a given fraction of spatial patches
across the entire sequence. Unlike temporal_masker.py which hides a full frame,
this hides a percentage of randomly selected patches — mimicking cloud cover
occlusion scattered across the image.
"""

import torch


def build_noise_for_mask_ratio(
    x: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, int, int]:
    """
    Build a noise tensor that causes the model to mask exactly `mask_ratio`
    fraction of all tokens across the full sequence.

    Prithvi's random_masking sorts tokens by noise value and masks the top
    mask_ratio fraction. We exploit this by:
      - assigning noise=1.0 to tokens we want masked
      - assigning noise=0.0 to tokens we want kept

    Args:
        x:           input tensor (1, C, T, H, W)
        mask_ratio:  fraction of tokens to mask e.g. 0.10, 0.50
        patch_size:  spatial patch size (16 for tiny/100M/300M, 14 for 600M)
        device:      torch device
        seed:        random seed for reproducibility

    Returns:
        noise:             (1, seq_length) float32 tensor
        tokens_per_frame:  number of tokens per time step
        seq_length:        total number of tokens in sequence
    """
    torch.manual_seed(seed)

    T = x.shape[2]
    H = x.shape[-2]
    W = x.shape[-1]

    tokens_per_frame = (H // patch_size) * (W // patch_size)
    seq_length = T * tokens_per_frame
    n_masked = int(seq_length * mask_ratio)

    # Start with all zeros (keep everything)
    noise = torch.zeros((1, seq_length), dtype=torch.float32, device=device)

    # Randomly pick which tokens to mask
    perm = torch.randperm(seq_length, device=device)
    noise[0, perm[:n_masked]] = 1.0

    return noise, tokens_per_frame, seq_length


def get_masked_token_indices(
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Return the indices of tokens that were masked (noise == 1.0).
    Useful for computing masked-region-only metrics later.

    Args:
        noise: (1, seq_length) noise tensor from build_noise_for_mask_ratio

    Returns:
        1D tensor of masked token indices
    """
    return (noise[0] == 1.0).nonzero(as_tuple=True)[0]
