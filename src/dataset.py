import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, RandCropByPosNegLabeld,
    EnsureTyped
)
from monai.data import Dataset

import yaml
with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)


def get_datasets(cfg):
    data_dir = cfg["data"]["data_dir"]
    roi_size = cfg["data"]["roi_size"]
    images = sorted(glob.glob(f"{data_dir}/imagesTr/*.nii*"))
    labels = sorted(glob.glob(f"{data_dir}/labelsTr/*.nii*"))

    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]

    train_files = data_dicts[:80]
    val_files = data_dicts[80:85] 

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=4, neg=1,
            num_samples=4
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    return train_ds, val_ds
