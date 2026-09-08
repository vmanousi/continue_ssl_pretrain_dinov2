"""DINOv2 ViT-S/14 (+4 registers) backbone + linear head for GastroHUN.

Both experiment arms use the SAME architecture and the SAME code path; the
only difference is which weights go into the backbone:

  kind="wrapped"  -> checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth
                     ({"model": state_dict}) = the GENERIC Meta checkpoint,
                     i.e. the exact initialisation point of our SSL run.
  kind="teacher"  -> outputs/full_run/eval/training_<it>/teacher_checkpoint.pth
                     ({"teacher": state_dict}); we keep the `backbone.*` keys.

`set_finetune_mode` is a port of finetuning_models.frozen_dino from
`gastrohun-dino`: unfreeze the last `pct`% of the 12 transformer blocks (plus
the patch-embed stem only if pct > 95). cls/pos/register tokens, the final
norm and the head stay trainable, exactly as in the original.
"""
import torch
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small

EMBED_DIM = 384
DEPTH = 12

_VIT_KWARGS = dict(
    patch_size=14,
    num_register_tokens=4,
    img_size=224,
    block_chunks=0,
    ffn_layer="mlp",
    init_values=1.0e-05,
)


def _load_backbone_state(weights_path, kind):
    ckpt = torch.load(weights_path, map_location="cpu")
    if kind == "wrapped":
        sd = ckpt["model"]
    elif kind == "teacher":
        sd = {k[len("backbone."):]: v for k, v in ckpt["teacher"].items()
              if k.startswith("backbone.")}
    else:
        raise ValueError(f"unknown kind: {kind}")
    return sd


def build_backbone(weights_path, kind):
    model = vit_small(**_VIT_KWARGS)
    sd = _load_backbone_state(weights_path, kind)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("mask_token")]  # unused at inference
    if missing or unexpected:
        raise RuntimeError(f"backbone load mismatch: missing={missing} unexpected={unexpected}")
    return model


class GastroHunClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))  # backbone(x) -> normalised CLS token (384-d)


def set_frozen_mode(model):
    """Linear probe / warm-up: everything frozen except the linear head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def set_finetune_mode(model, pct):
    """Port of frozen_dino(model_ft, pct)."""
    for p in model.parameters():
        p.requires_grad = True

    n_frozen = int((100 - pct) * DEPTH / 100)
    model.backbone.patch_embed.requires_grad_(pct > 95)
    for i, block in enumerate(model.backbone.blocks):
        block.requires_grad_(i >= n_frozen)
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
