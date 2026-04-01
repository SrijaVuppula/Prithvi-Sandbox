import numpy as np
import torch
from pathlib import Path
import sys


def select_files_for_condition(
    all_files: list[str],
    n_frames: int,
    gap_type_cfg: dict,
) -> list[str]:
    """
    Pick the subset of files that realise a given (n_frames, gap_type) condition.

    Args:
        all_files:     ordered list of all available HLS filenames for this tile
        n_frames:      how many frames to use (sequence_length)
        gap_type_cfg:  the gap_type dict from config, e.g.:
                       {"name": "regular", "index_maps": {3:[0,2,5], 4:[0,1,3,5], 6:[...]}}

    Returns:
        List of n_frames filenames in temporal order.
    """
    index_map = gap_type_cfg["index_maps"]
    if n_frames not in index_map:
        raise ValueError(
            f"No index_map entry for n_frames={n_frames} "
            f"in gap_type '{gap_type_cfg['name']}'. "
            f"Available: {list(index_map.keys())}"
        )
    indices = index_map[n_frames]
    if max(indices) >= len(all_files):
        raise ValueError(
            f"Index {max(indices)} out of range for tile with {len(all_files)} files."
        )
    return [all_files[i] for i in indices]


def load_sample_from_inference_module(
    base_dir: Path,
    file_paths: list[str],
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Thin wrapper around Prithvi's own load_example() function.

    Inserts base_dir into sys.path so the local inference.py is importable,
    then delegates to it. This keeps your patching logic in one place.

    Returns:
        x:               (1, C, T, H, W) float32 on device
        temporal_coords: (1, T, 3) float32 on device
        location_coords: (1, 2) float32 on device
        meta_data:       dict from load_example
    """
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from inference import load_example  # type: ignore

    input_data, temporal_coords, location_coords, meta_data = load_example(
        file_paths=[str(Path(base_dir) / "examples" / f) for f in file_paths],
        mean=mean,
        std=std,
    )

    x = torch.tensor(input_data, dtype=torch.float32, device=device)
    temporal_coords = torch.tensor(
        temporal_coords, dtype=torch.float32, device=device
    ).unsqueeze(0)
    location_coords = torch.tensor(
        location_coords[0], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return x, temporal_coords, location_coords, meta_data