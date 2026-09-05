# continue_ssl_pretrain_dinov2

Continued / domain-adaptive DINOv2 SSL pretraining, initialized from Meta's
official checkpoint, on HyperKvasir GI endoscopy images -- followed by
supervised downstream fine-tuning on the GastroHUN labeled dataset.

Self-contained: does not depend on or modify `gastrodino_thesis` or
`gastrohun-dino`. Has its own `.venv`.

## Pipeline

```
data/hyperkvasir_raw_{labeled,unlabeled}/   (symlinks to ~/Downloads, read-only source)
        |
        v  preprocessing/00_build_manifest.py
data/manifest_00.csv
        |
        v  preprocessing/audit_contact_sheets.py   (read-only visual audit, informed the steps below)
        v  preprocessing/01_process_images.py
data/hyperkvasir_ssl/
   images/*.jpg                 -- border-cropped (unlabeled only), luminance-floor filtered, resized
   preprocessing_log.csv        -- per-image audit trail (kept/dropped + why)
   exclusion_boxes.csv          -- detected overlay-panel boxes in final pixel coords (metadata only,
                                    pixels untouched -- consumed by the DINOv2 crop sampler to avoid them)
   train.txt                    -- manifest for the DINOv2 dataset loader
        |
        v  ssl_configs/ + vendored dinov2/          (continued SSL pretraining, backbone: ViT-S/14 + registers)
        |
        v  downstream/                              (standalone supervised fine-tune on GastroHUN, mirrors
                                                       the Gastrohun_official recipe for a fair comparison)
```

## Decisions on record

- Backbone: `vit_small_patch14_reg4_dinov2.lvd142m` (ViT-S/14 + registers).
- SSL trainer: official `facebookresearch/dinov2` (full recipe), not a simplified DINOv1-style reimplementation.
- HyperKvasir: the full corpus (labeled + unlabeled, no upper-GI-only filtering).
- Preprocessing steps kept (evidence-based, see `data/audit/` and PIPELINE notes):
  border-crop (unlabeled), overlay-panel exclusion-box metadata (both subsets -- confirmed present
  in ~18% unlabeled / ~46% labeled), a conservative luminance floor. No blur filter, no bright-end
  cutoff, no dedup yet (audit did not support them / not yet evaluated).
- Baseline for comparison: the `Gastrohun_official` benchmark repo's `dinov2_vits14` results
  (not `gastrodino_thesis`'s), still to be finalized -- see project memory.
