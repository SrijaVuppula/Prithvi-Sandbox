import torch


def get_masked_frame_index(mask_position: str, n_frames: int) -> int:
    """
    Resolve a named mask position to a concrete frame index.

    Args:
        mask_position: "middle" or "endpoint"
        n_frames:      total frames in the sequence

    Returns:
        Integer frame index to mask (0-based).
    """
    if mask_position == "middle":
        # For n=3 → idx 1; n=4 → idx 1; n=6 → idx 2
        # Always the frame with roughly equal context on both sides
        return n_frames // 2 - (1 if n_frames % 2 == 0 else 0)
    elif mask_position == "endpoint":
        return n_frames - 1
    else:
        raise ValueError(
            f"Unknown mask_position '{mask_position}'. Use 'middle' or 'endpoint'."
        )


def build_noise_from_frame_idx(
    x: torch.Tensor,
    frame_idx: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """
    Build the noise tensor that forces a single frame to be 100% masked.

    This is the same logic as your build_noise() in the inference script,
    refactored to take a single frame_idx and be importable.

    Returns:
        noise:             (1, seq_length) float32 — 0=keep, 1=mask
        tokens_per_frame:  int
        seq_length:        int
    """
    tokens_per_frame = (x.shape[-2] // patch_size) * (x.shape[-1] // patch_size)
    seq_length = x.shape[2] * tokens_per_frame

    noise = torch.zeros((1, seq_length), dtype=torch.float32, device=device)
    start = frame_idx * tokens_per_frame
    end   = start + tokens_per_frame
    noise[:, start:end] = 1.0

    return noise, tokens_per_frame, seq_length