# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Added for continued DINOv2 SSL pretraining on HyperKvasir (GI endoscopy).
# Modeled on the official HPAFoV example (cell_dino/hpafov.py): reads a flat
# manifest of already-preprocessed images, no labels needed for SSL.

import csv
import os
from typing import Any, Callable, Optional, Tuple

from .extended import ExtendedVisionDataset


class HyperKvasir(ExtendedVisionDataset):
    """
    root: directory containing images/<image_id>.jpg (the output of this
          project's preprocessing/01_process_images.py + 02_dedup.py)
    extra: path to train_deduped.txt (one "images/<id>.jpg" relative path
           per line -- the final, deduplicated corpus)

    Exclusion boxes (overlay-panel regions to keep the multi-crop sampler
    away from -- see preprocessing/common.py) are exposed via
    get_exclusion_boxes(index) rather than folded into get_target(), so the
    crop-aware transform (added separately) can read them without disturbing
    the plain (image, target) contract the rest of dinov2 expects.
    """

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        exclusion_boxes_csv: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform, **kwargs)

        with open(extra) as f:
            self._image_relpaths = [line.strip() for line in f if line.strip()]

        self._exclusion_boxes = {}
        if exclusion_boxes_csv is not None and os.path.isfile(exclusion_boxes_csv):
            with open(exclusion_boxes_csv) as f:
                for row in csv.DictReader(f):
                    self._exclusion_boxes.setdefault(row["image_id"], []).append(
                        (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
                    )

    def _image_id(self, index: int) -> str:
        # "images/<id>.jpg" -> "<id>"
        return os.path.splitext(os.path.basename(self._image_relpaths[index]))[0]

    def get_image_data(self, index: int) -> bytes:
        full_path = os.path.join(self.root, self._image_relpaths[index])
        with open(full_path, mode="rb") as f:
            return f.read()

    def get_exclusion_boxes(self, index: int):
        """List of (x, y, w, h) overlay-panel boxes for this image, or [] if none detected."""
        return self._exclusion_boxes.get(self._image_id(index), [])

    def get_target(self, index: int) -> Any:
        # SSL pretraining does not use labels.
        return 0

    def get_targets(self):
        return None

    def __len__(self) -> int:
        return len(self._image_relpaths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Overridden (rather than relying on ExtendedVisionDataset.__getitem__) so the
        # crop-aware transform can receive this image's exclusion boxes directly, instead
        # of us having to smuggle them through the (image, target) tuple that the rest of
        # dinov2 assumes is just (PIL.Image, label).
        #
        # Note: make_dataset() (dinov2/data/loaders.py) constructs this dataset with
        # separate `transform=`/`target_transform=` kwargs, not a combined `transforms=`.
        # torchvision's VisionDataset.__init__ stores the raw image-only callable as
        # self.transform (singular) and only wraps the combined StandardTransform as
        # self.transforms (plural) -- we want the former so we can pass exclusion_boxes
        # through directly instead of going through StandardTransform's (image, target)
        # calling convention, which has no such hook.
        image_data = self.get_image_data(index)
        image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        boxes = self.get_exclusion_boxes(index)

        target = self.get_target(index)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            if getattr(self.transform, "accepts_exclusion_boxes", False):
                image = self.transform(image, exclusion_boxes=boxes)
            else:
                image = self.transform(image)

        return image, target
