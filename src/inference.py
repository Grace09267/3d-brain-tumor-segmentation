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
    ScaleIntensityd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader

from model import get_model


def inference():
    # config
    with open("configs.yaml") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = cfg["data"]["data_dir"]
    output_dir = os.path.join(BASE_DIR, cfg["paths"]["output_dir"])
    pred_dir = os.path.join(output_dir, "preds_swin")

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
        EnsureTyped(keys=["image", "label"]),
    ])

    val_ds = Dataset(val_files, transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    # 모델
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model_swin.pth")))
    model.eval()

    print("✅ Model loaded")

    # inference
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)

            outputs = sliding_window_inference(
                images,
                roi_size=(128,128,128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.75
            )

            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            pred_np = preds.cpu().numpy()[0]

            # affine (원본 기준)
            affine = np.eye(4)

            img_path = val_files[i]["image"]
            filename = os.path.basename(img_path).replace(".nii.gz", "")
            save_path = os.path.join(pred_dir, f"{filename}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_np.astype(np.uint8), affine), save_path)

            print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    inference()