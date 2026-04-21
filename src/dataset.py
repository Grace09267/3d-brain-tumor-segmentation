import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    NormalizeIntensityd, RandCropByPosNegLabeld, RandCropByLabelClassesd, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd, RandShiftIntensityd, EnsureTyped
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
    val_files = data_dicts[80:100] 

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # spacing(모든 데이터를 동일한 실제 크기로 맞춤) / orientation(좌우, 위아래 방향 통일)
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), # mean/std기반
        # tumor sampling        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=8, neg=1,
            num_samples=4
        ),
        #RandCropByLabelClassesd( # class 3을 계속 crop하지 못함
        #    keys=["image", "label"],
        #    label_key="label",
        #    spatial_size=roi_size,
        #    num_classes=4,
        #    ratios=[1, 1, 1, 2],  # 🔥 핵심
        #    num_samples=4
        #),
        # ===== augmentation =====
        # flip (강력 추천)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # rotation
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        # affine (shape 변형)
        #RandAffined(
        #    keys=["image", "label"],
        #    prob=0.3,
        #    rotate_range=(0.1, 0.1, 0.1),
        #    scale_range=(0.1, 0.1, 0.1),
        #    mode=("bilinear", "nearest")
        #),
        # noise
        #RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        # blur
        #RandGaussianSmoothd(keys=["image"], prob=0.2),
        # intensity shift
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    return train_ds, val_ds
