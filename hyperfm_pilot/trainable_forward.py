"""
Gradient-enabled replica of terratorch_loader.run_masked_forward /
_encode_with_noise, both of which are @torch.no_grad()-decorated (confirmed
Aug 31 via inspect.getsource -- this whole pipeline predates any training,
built only for zero-shot evaluation). The body below is copied verbatim
from _encode_with_noise's real source, minus the decorator; nothing here is
guessed or reimplemented from scratch.
"""

import torch


def encode_with_noise_trainable(model, x, temporal_coords, location_coords, mask_ratio, noise):
    """
    Exact copy of terratorch_loader._encode_with_noise's body, minus
    @torch.no_grad(). See that function's real source (confirmed via
    inspect.getsource, Aug 31) for what this replicates -- this is not a
    reimplementation from memory or documentation, it's a direct copy.
    """
    enc = model.encoder
    sample_shape = x.shape[-3:]
    x_enc = enc.patch_embed(x)  # ControlNetEncoderAdapter's hook fires here
    pos_embed = enc.interpolate_pos_encoding(sample_shape)
    x_enc = x_enc + pos_embed[:, 1:, :]
    if enc.temporal_encoding and temporal_coords is not None:
        num_tokens_per_frame = x_enc.shape[1] // enc.num_frames
        temporal_encoding = enc.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
        x_enc = x_enc + temporal_encoding
    if enc.location_encoding and location_coords is not None:
        location_encoding = enc.location_embed_enc(location_coords)
        x_enc = x_enc + location_encoding
    x_enc, mask, ids_restore = enc.random_masking(x_enc, mask_ratio, noise=noise)
    cls_token = enc.cls_token + pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_enc.shape[0], -1, -1)
    x_enc = torch.cat((cls_tokens, x_enc), dim=1)
    for block in enc.blocks:
        x_enc = block(x_enc)
    x_enc = enc.norm(x_enc)
    return x_enc, mask, ids_restore


def run_masked_forward_trainable(model, x, temporal_coords, location_coords, mask_ratio, noise):
    """
    Gradient-enabled twin of terratorch_loader.run_masked_forward. Same two
    steps as that function's confirmed real source -- encode, then decode.
    No loss/unpatchify here: training uses the adapters' captured 291-band
    outputs (dec_adapter.last_output) and a separately-computed 291-band
    loss instead of the native model.forward_loss (which only knows about
    6 bands).
    """
    latent, mask, ids_restore = encode_with_noise_trainable(
        model, x, temporal_coords, location_coords, mask_ratio, noise
    )
    pred = model.decoder(
        latent, ids_restore, temporal_coords, location_coords,
        input_size=x.shape,
    )
    return latent, mask, ids_restore, pred
