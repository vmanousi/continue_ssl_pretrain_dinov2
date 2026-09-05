# Added for continued DINOv2 SSL pretraining on HyperKvasir (GI endoscopy).
#
# Identical to the official DataAugmentationDINO (dinov2/data/augmentations.py),
# except the geometric crop step is exclusion-box-aware: some HyperKvasir
# images have a burned-in overlay panel (ScopeGuide-style icon box + patient
# info text, see preprocessing/common.py) that is real pixel content, not
# black margin -- a random crop landing mostly on it is a degenerate view for
# DINO's teacher/student objective and pollutes the batch-level statistics
# SK-centering/KoLeo rely on. Rather than editing the pixels (risking a new
# degenerate all-black/inpainted-texture crop instead), ExclusionAwareRandomResizedCrop
# resamples the crop region (bounded retries, same spirit as
# RandomResizedCrop's own internal retry-until-valid-aspect-ratio loop) when
# the candidate overlaps an exclusion box too much.

import logging

from torchvision import transforms
from torchvision.transforms import functional as F

from .transforms import GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov2")


def _box_overlap_fraction(crop_left, crop_top, crop_w, crop_h, boxes):
    """Max fraction of the crop's area covered by any single exclusion box."""
    if not boxes:
        return 0.0
    crop_area = crop_w * crop_h
    worst = 0.0
    for (bx, by, bw, bh) in boxes:
        ix1, iy1 = max(crop_left, bx), max(crop_top, by)
        ix2, iy2 = min(crop_left + crop_w, bx + bw), min(crop_top + crop_h, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        overlap = (ix2 - ix1) * (iy2 - iy1) / crop_area
        worst = max(worst, overlap)
    return worst


class ExclusionAwareRandomResizedCrop(transforms.RandomResizedCrop):
    """
    Same as torchvision's RandomResizedCrop, but re-samples the crop region
    (up to max_attempts times) when it overlaps `self.current_boxes` by more
    than max_overlap_frac. `current_boxes` is set by the caller (see
    HyperKvasirDataAugmentationDINO.__call__) immediately before each call --
    safe because a DataLoader worker processes one sample at a time
    (each worker holds its own copy of this object; no cross-sample race).
    """

    def __init__(self, *args, max_overlap_frac=0.3, max_attempts=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_overlap_frac = max_overlap_frac
        self.max_attempts = max_attempts
        self.current_boxes = []

    def forward(self, img):
        if not self.current_boxes:
            return super().forward(img)

        best_params, best_overlap = None, float("inf")
        for _ in range(self.max_attempts):
            top, left, height, width = self.get_params(img, self.scale, self.ratio)
            overlap = _box_overlap_fraction(left, top, width, height, self.current_boxes)
            if overlap <= self.max_overlap_frac:
                return F.resized_crop(img, top, left, height, width, self.size, self.interpolation)
            if overlap < best_overlap:
                best_params, best_overlap = (top, left, height, width), overlap

        # Exhausted retries (rare -- only when exclusion boxes cover most of the
        # image): use the least-bad candidate seen rather than looping forever.
        top, left, height, width = best_params
        return F.resized_crop(img, top, left, height, width, self.size, self.interpolation)


class HyperKvasirDataAugmentationDINO:
    """Drop-in replacement for DataAugmentationDINO with exclusion-box-aware crops."""

    accepts_exclusion_boxes = True

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        max_overlap_frac=0.3,
    ):
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using HyperKvasir data augmentation parameters (exclusion-aware crops):")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"exclusion max_overlap_frac: {max_overlap_frac}")
        logger.info("###################################")

        self.crop_global = ExclusionAwareRandomResizedCrop(
            global_crops_size, scale=global_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC, max_overlap_frac=max_overlap_frac,
        )
        self.crop_local = ExclusionAwareRandomResizedCrop(
            local_crops_size, scale=local_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC, max_overlap_frac=max_overlap_frac,
        )
        self.flip = transforms.RandomHorizontalFlip(p=0.5)

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose([transforms.ToTensor(), make_normalize_transform()])

        self.global_transfo1 = transforms.Compose([color_jittering, GaussianBlur(p=1.0), normalize])
        self.global_transfo2 = transforms.Compose(
            [color_jittering, GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2), normalize]
        )
        self.local_transfo = transforms.Compose([color_jittering, GaussianBlur(p=0.5), normalize])

    def _geometric(self, crop_module, image, boxes):
        crop_module.current_boxes = boxes
        return self.flip(crop_module(image))

    def __call__(self, image, exclusion_boxes=None):
        boxes = exclusion_boxes or []
        output = {}

        im1_base = self._geometric(self.crop_global, image, boxes)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self._geometric(self.crop_global, image, boxes)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        output["local_crops"] = [
            self.local_transfo(self._geometric(self.crop_local, image, boxes))
            for _ in range(self.local_crops_number)
        ]
        output["offsets"] = ()

        return output
