"""
Deep pre-flight check for the continued-SSL run -- runs on the LOGIN NODE
(no GPU). Goes well past `import`: builds the real model, loads the
pretrained checkpoint, builds the dataset + exclusion-aware augmentation,
pulls two samples through, and runs the exact collate the training loop
uses. If this passes, the only things left untested are FSDP wrapping and
actual CUDA kernels (that's what the 10-iteration GPU smoke test covers).

Usage (from the repo root, with the venv active and PYTHONPATH=$PWD/dinov2):
    python scripts/preflight_check.py
"""
import sys
from functools import partial

import torch
from omegaconf import OmegaConf

from dinov2.configs import load_and_merge_config
from dinov2.data import MaskingGenerator, collate_data_and_cast, make_dataset
from dinov2.data.hyperkvasir_augmentations import HyperKvasirDataAugmentationDINO
from dinov2.train.ssl_meta_arch import SSLMetaArch

CONFIG = "train/vits14_reg4_hyperkvasir_continued"


def section(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def main():
    section("1. key package versions")
    import torchvision, xformers, omegaconf, fvcore
    print("python       ", sys.version.split()[0])
    print("torch        ", torch.__version__)
    print("torchvision  ", torchvision.__version__)
    print("xformers     ", xformers.__version__)
    print("omegaconf    ", omegaconf.__version__)
    from importlib.metadata import version
    for p in ("fvcore", "iopath", "torchmetrics", "submitit", "numpy", "pillow"):
        try:
            print(f"{p:13}", version(p))
        except Exception as e:  # noqa: BLE001
            print(f"{p:13} MISSING ({e})")

    section("2. config resolves")
    cfg = load_and_merge_config(CONFIG)
    cr = OmegaConf.to_container(cfg, resolve=True)
    dataset_str = cr["train"]["dataset_path"]
    weights = cr["student"]["pretrained_weights"]
    print("dataset_str:", dataset_str[:110], "...")
    print("pretrained_weights:", weights)
    import os
    assert os.path.isfile(weights), "pretrained checkpoint missing!"
    root = dataset_str.split("root=")[1].split(":")[0]
    assert os.path.isdir(root + "/images"), "corpus images/ missing!"
    print("centering:", cr["train"]["centering"], "| dino K:", cr["dino"]["head_n_prototypes"],
          "| arch:", cr["student"]["arch"], "| registers:", cr["student"]["num_register_tokens"])

    section("3. build model + load pretrained checkpoint (CPU)")
    # torch.load default weights_only=True on torch>=2.6 -- our checkpoint is
    # {"model": <tensor state_dict>}, so this should be fine; assert it here.
    _probe = torch.load(weights, map_location="cpu")
    assert "model" in _probe and isinstance(_probe["model"], dict)
    print(f"checkpoint OK: {len(_probe['model'])} tensors, top keys "
          f"{list(_probe['model'])[:3]}")
    del _probe
    model = SSLMetaArch(cfg)  # builds student+teacher, loads weights into student backbone
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SSLMetaArch built OK -- {n_params/1e6:.1f}M params, embed_dim {model.embed_dim}")

    section("4. dataset + exclusion-aware augmentation, 2 samples")
    transform = HyperKvasirDataAugmentationDINO(
        cr["crops"]["global_crops_scale"], cr["crops"]["local_crops_scale"],
        cr["crops"]["local_crops_number"],
        global_crops_size=cr["crops"]["global_crops_size"],
        local_crops_size=cr["crops"]["local_crops_size"],
    )
    ds = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=lambda _: ())
    print("dataset samples:", len(ds))
    # find one index WITH exclusion boxes and one without, to exercise both paths
    idx_with = next(i for i in range(500) if ds.get_exclusion_boxes(i))
    for label, i in (("no boxes", 0), (f"with boxes {ds.get_exclusion_boxes(idx_with)}", idx_with)):
        out, tgt = ds[i]
        print(f"  [{label}] globals {len(out['global_crops'])}x{tuple(out['global_crops'][0].shape)}"
              f"  locals {len(out['local_crops'])}x{tuple(out['local_crops'][0].shape)}  target={tgt}")

    section("5. collate exactly as the training loop does")
    img_size = cr["crops"]["global_crops_size"]
    patch = cr["student"]["patch_size"]
    n_tokens = (img_size // patch) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch, img_size // patch),
        max_num_patches=int(0.5 * (img_size // patch) ** 2),
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cr["ibot"]["mask_ratio_min_max"],
        mask_probability=cr["ibot"]["mask_sample_probability"],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.float32,
    )
    batch = collate_fn([ds[0], ds[idx_with], ds[1], ds[2]])
    print("collated keys:", list(batch))
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:28} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:28} {type(v).__name__} len={len(v) if hasattr(v,'__len__') else '-'}")

    section("PRE-FLIGHT PASSED -- ready for the GPU smoke test")


if __name__ == "__main__":
    main()
