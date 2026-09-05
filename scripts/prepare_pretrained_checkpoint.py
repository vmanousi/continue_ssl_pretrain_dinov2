"""
Prepare Meta's official DINOv2 ViT-S/14+reg4 checkpoint for use as
`student.pretrained_weights` in continued SSL pretraining.

Two things need fixing:

1. dinov2/train/ssl_meta_arch.py does `torch.load(path)["model"]`, but the
   released .pth is a flat state_dict (no "model" key) -> wrap it.

2. The released pos_embed is sized for 518x518 input (1 + 37*37 = 1370
   positions). Our config trains at global_crops_size=224 (1 + 16*16 = 257).
   load_state_dict(strict=False) would silently SKIP the mismatched pos_embed
   and leave it randomly initialized -- losing the pretrained positional
   information. So we bicubic-interpolate it down to the 224 grid here (the
   same operation DinoVisionTransformer.interpolate_pos_encoding does at
   runtime), baked into the checkpoint so it loads cleanly.

Register tokens carry no positional embedding (position-free by design), so
pos_embed covers only [cls] + patches.

Usage:
    python scripts/prepare_pretrained_checkpoint.py \
        --output checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth \
        --target-crop-size 224
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HUB_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
PATCH_SIZE = 14


def interpolate_pos_embed(pos_embed: torch.Tensor, target_grid: int) -> torch.Tensor:
    """pos_embed: (1, 1 + old_grid^2, dim) -> (1, 1 + target_grid^2, dim)."""
    dim = pos_embed.shape[-1]
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    old_grid = int(round(patch_pos.shape[1] ** 0.5))
    assert old_grid * old_grid == patch_pos.shape[1], "pos_embed patch count is not a perfect square"

    if old_grid == target_grid:
        return pos_embed

    patch_pos = (
        patch_pos.reshape(1, old_grid, old_grid, dim)
        .permute(0, 3, 1, 2)  # (1, dim, old_grid, old_grid)
        .float()
    )
    patch_pos = F.interpolate(
        patch_pos, size=(target_grid, target_grid), mode="bicubic", align_corners=False
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
    return torch.cat([cls_pos, patch_pos.to(cls_pos.dtype)], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="raw Meta checkpoint; if omitted, downloads via torch.hub cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-crop-size", type=int, default=224)
    args = parser.parse_args()

    if args.input is not None:
        state_dict = torch.load(args.input, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(HUB_URL, map_location="cpu")

    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]  # already-wrapped input, unwrap first

    target_grid = args.target_crop_size // PATCH_SIZE
    assert args.target_crop_size % PATCH_SIZE == 0, f"{args.target_crop_size} not divisible by patch {PATCH_SIZE}"

    old_shape = tuple(state_dict["pos_embed"].shape)
    state_dict["pos_embed"] = interpolate_pos_embed(state_dict["pos_embed"], target_grid)
    new_shape = tuple(state_dict["pos_embed"].shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, args.output)

    print(f"pos_embed: {old_shape} -> {new_shape} (grid {target_grid}x{target_grid} for {args.target_crop_size}px)")
    print(f"parameters: {len(state_dict)}")
    print(f"wrapped checkpoint written to: {args.output}")


if __name__ == "__main__":
    main()
