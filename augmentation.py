import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn


def get_transforms(*, data, CFG):
    """
    get_transforms functions to return the augmentations during training
    """

    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size2, p=1),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5
                ),
               
                # ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size2, p=1),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )