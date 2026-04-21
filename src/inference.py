import os
import yaml
import torch
import numpy as np
import nibabel as nib
import glob

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    NormalizeIntensityd,
    EnsureTyped,
    Invertd,
)
from monai.data import Dataset, DataLoader, decollate_batch, MetaTensor

from model import get_model


def inference():
    # config
    with open("configs.yaml") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = cfg["data"]["data_dir"]
    output_dir = os.path.join(BASE_DIR, cfg["paths"]["output_dir"])
    pred_dir = os.path.join(output_dir, "preds_dyn")

    os.makedirs(pred_dir, exist_ok=True)

    # 데이터
    images = sorted(glob.glob(f"{data_dir}/imagesTr/*.nii*"))
    labels = sorted(glob.glob(f"{data_dir}/labelsTr/*.nii*"))

    val_files = [
        {"image": images[i], "label": labels[i]}
        for i in range(20, 30)  # 👉 원하는 범위
    ]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_ds = Dataset(val_files, transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    # 모델
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model_dynunet.pth")))
    model.eval()

    print("✅ Model loaded")

    # ======================
    # post transform (inverse)
    # ======================
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])    

    # inference
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            # forward
            outputs = sliding_window_inference(
                images,
                roi_size=(128,128,128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.75
            )

            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            # 🔥 batch 분리 & decollate # decollate_batch가 batch를 개별 sample로 분리해 각각 inverse transform를 적용시킴
            batch_list = decollate_batch(batch)
            preds_list = decollate_batch(preds)

            # 🔥 각각 meta 붙이기(pred를 sample에 넣기)
            for i in range(len(batch_list)):
                batch_list[i]["pred"] = MetaTensor(
                    preds_list[i],
                    meta=batch_list[i]["image"].meta
                )

            # inverse
            batch = [post_transforms(i) for i in batch_list]

            # ======================
            # save
            # ======================
            for j, item in enumerate(batch):
                pred_np = item["pred"].cpu().numpy()[0]

                meta = item["image"].meta
                affine = item["image"].meta["affine"] # 원래 spacing, 원래 방향, 원래 위치, voxel좌표를 실제 공간 좌표로 변환하는게 affine

                img_path = val_files[i]["image"]
                filename = os.path.basename(img_path).replace(".nii.gz", "")

                save_path = os.path.join(pred_dir, f"{filename}_pred.nii.gz")

                nib.save(
                    nib.Nifti1Image(pred_np.astype(np.uint8), affine),
                    save_path
                )

                print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    inference()