from pathlib import Path


import pandas as pd

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(image_size: int = 224, train: bool = False):
    """
    Returns image transformations.

    For training:
        - resize
        - light augmentation
        - normalise using ImageNet statistics

    For validation/testing:
        - resize only
        - normalise
    """

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class XrayDataset(Dataset):
    """
    Dataset expects a CSV file with two columns:

        image_path,label

    label:
        0 = normal
        1 = abnormal/fracture
    """

    def __init__(self, csv_path: str, transform=None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)

        if "image_path" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("CSV must contain columns: image_path,label")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)