"""GastroHUN image-classification data — a faithful port of the
`gastrohun-dino` recipe (utils/dataset_module_image.py + the dataframe handling
in image_classification/scripts/train_image_classification.py).

Kept identical to the original on purpose:
  - the 23-class label map
  - Resize((224,224), LANCZOS) -> ToTensor -> Normalize with GastroHUN's own
    channel stats (NOT ImageNet stats)
  - rows whose `label` column is blank / not in the map are dropped
  - class weights = sklearn 'balanced' on the Train split
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MAP_CATEGORIES = {
    "A1": 0, "L1": 1, "P1": 2, "G1": 3,
    "A2": 4, "L2": 5, "P2": 6, "G2": 7,
    "A3": 8, "L3": 9, "P3": 10, "G3": 11,
    "A4": 12, "L4": 13, "P4": 14, "G4": 15,
    "A5": 16, "L5": 17, "P5": 18,
    "A6": 19, "L6": 20, "P6": 21,
    "OTHERCLASS": 22,
}
NUM_CLASSES = 23

# GastroHUN channel stats from the original recipe (train_image_classification.py)
_MEAN = [0.5990, 0.3664, 0.2769]
_STD = [0.2847, 0.2190, 0.1772]


def build_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def load_split_df(official_split_csv, label="Complete agreement"):
    """Read the official split CSV, map `label` to 0..22, drop unlabeled rows."""
    df = pd.read_csv(official_split_csv, index_col=0)
    df[label] = df[label].map(MAP_CATEGORIES)          # unknown / blank -> NaN
    df = df[df[label].notna()].copy()
    df[label] = df[label].astype(np.int64)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_class_weights(df, label="Complete agreement", device="cuda"):
    from sklearn.utils.class_weight import compute_class_weight

    y = df[df["set_type"] == "Train"][label].values
    w = compute_class_weight(class_weight="balanced",
                             classes=np.arange(NUM_CLASSES), y=y)
    return torch.tensor(w, dtype=torch.float32, device=device)


class GastroHunDataset(Dataset):
    def __init__(self, df, data_path, transform, label="Complete agreement",
                 return_index=False):
        self.transform = transform
        self.return_index = return_index
        self.samples = []  # (path, label, row_index)
        missing = 0
        for row_idx in df.index:
            patient = str(df.loc[row_idx, "num patient"])
            fname = df.loc[row_idx, "filename"]
            path = os.path.join(data_path, patient, fname)
            if not os.path.exists(path):
                missing += 1
                continue
            self.samples.append((path, int(df.loc[row_idx, label]), int(row_idx)))
        if missing:
            print(f"[GastroHunDataset] {missing} files not found under {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, row_idx = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB"))
        if self.return_index:
            return img, label, row_idx
        return img, label
