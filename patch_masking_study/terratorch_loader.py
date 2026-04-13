"""
patch_masking_study/terratorch_loader.py

Clean model loader using TerraTorch's PrithviMAE directly.
Replaces the hacky importlib + source patching approach from baseline_study/inference/runner.py.

Since TerraTorch's PrithviViT.forward() does not expose the noise parameter,
we replicate the encoder forward pass manually, injecting noise into random_masking
directly. This gives us controlled patch masking at any ratio.
"""

import json
import torch
from pathlib import Path
from terratorch.models.backbones.prithvi_mae import PrithviMAE


def load_backbone_config(config_path: Path) -> dict:
    """
    Read model architecture + normalization stats from config.json.
    Works for all Prithvi backbone sizes.
    """
    with open(config_path) as f:
        raw = json.load(f)

    cfg = raw.get("pretrained_cfg", raw)

    return {
        "embed_dim":         cfg["embed_dim"],
        "depth":             cfg["depth"],
        "num_heads":         cfg["num_heads"],
        "patch_size":        tuple(cfg["patch_size"]),
        "mean":              cfg["mean"],
        "std":               cfg["std"],
        "bands":             cfg["bands"],
        "decoder_embed_dim": 512,
        "decoder_depth":     8,
        "decoder_num_heads": 16,
        "mlp_ratio":         4.0,
        "coords_encoding":   ["time", "location"],
    }


def load_prithvi_from_terratorch(
    backbone_name: str,
    base_dir: Path,
    checkpoint_filename: str,
    num_frames: int,
    device: torch.device,
) -> tuple:
    """
    Load a Prithvi MAE model cleanly using TerraTorch's PrithviMAE class.
    No importlib, no source patching required.

    Returns:
        model, bands, mean, std, patch_size (int)
    """
    base_dir = Path(base_dir)
    cfg      = load_backbone_config(base_dir / "config.json")

    mean       = cfg["mean"]
    std        = cfg["std"]
    bands      = cfg["bands"]
    patch_size = cfg["patch_size"]

    model = PrithviMAE(
        img_size=224,
        patch_size=patch_size,
        num_frames=num_frames,
        in_chans=len(bands),
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        coords_encoding=cfg["coords_encoding"],
    ).to(device)

    checkpoint_path = base_dir / checkpoint_filename
    try:
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    for key in list(state_dict.keys()):
        if key == "encoder.pos_embed":
            state_dict[key] = model.encoder.pos_embed
        elif key == "decoder.decoder_pos_embed":
            state_dict[key] = model.decoder.decoder_pos_embed

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    spatial_patch = patch_size[1]
    print(f"  [TerraTorch] Loaded {backbone_name} ({num_frames} frames) patch={spatial_patch} on {device}")

    return model, bands, mean, std, spatial_patch


@torch.no_grad()
def _encode_with_noise(
    model: PrithviMAE,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    mask_ratio: float,
    noise: torch.Tensor,
) -> tuple:
    """
    Manually replicate PrithviViT.forward() but inject our noise tensor
    into random_masking, giving us controlled patch masking.
    """
    enc = model.encoder

    sample_shape = x.shape[-3:]

    # Patch embedding
    x_enc = enc.patch_embed(x)
    pos_embed = enc.interpolate_pos_encoding(sample_shape)
    x_enc = x_enc + pos_embed[:, 1:, :]

    # Temporal encoding
    if enc.temporal_encoding and temporal_coords is not None:
        num_tokens_per_frame = x_enc.shape[1] // enc.num_frames
        temporal_encoding = enc.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
        x_enc = x_enc + temporal_encoding

    # Location encoding
    if enc.location_encoding and location_coords is not None:
        location_encoding = enc.location_embed_enc(location_coords)
        x_enc = x_enc + location_encoding

    # Controlled masking — inject our noise here
    x_enc, mask, ids_restore = enc.random_masking(x_enc, mask_ratio, noise=noise)

    # CLS token
    cls_token  = enc.cls_token + pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_enc.shape[0], -1, -1)
    x_enc      = torch.cat((cls_tokens, x_enc), dim=1)

    # Transformer blocks
    for block in enc.blocks:
        x_enc = block(x_enc)
    x_enc = enc.norm(x_enc)

    return x_enc, mask, ids_restore


@torch.no_grad()
def run_masked_forward(
    model: PrithviMAE,
    x: torch.Tensor,
    temporal_coords: torch.Tensor,
    location_coords: torch.Tensor,
    mask_ratio: float,
    noise: torch.Tensor,
) -> tuple:
    """
    Full forward pass with controlled noise masking.

    Returns:
        loss, pred_img, rec_img, mask_img, x_cpu
    """
    orig_h, orig_w = x.shape[-2], x.shape[-1]

    # Encode with controlled noise
    latent, mask, ids_restore = _encode_with_noise(
        model, x, temporal_coords, location_coords, mask_ratio, noise
    )

    # Decode
    pred = model.decoder(
        latent, ids_restore, temporal_coords, location_coords,
        input_size=x.shape,
    )

    # Loss
    loss = model.forward_loss(x, pred, mask)

    # Unpatchify
    pred_img = model.unpatchify(pred.detach().cpu(), image_size=(orig_h, orig_w))
    mask_img = model.unpatchify(
        mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]).detach().cpu(),
        image_size=(orig_h, orig_w),
    )

    x_cpu   = x.detach().cpu()
    rec_img = x_cpu.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]

    return float(loss), pred_img, rec_img, mask_img, x_cpu
