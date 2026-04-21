import torch, yaml, os
import numpy as np
#from torch.utils.data import DataLoader # 기본적 batch만 처리 list 잘 못다룸
from monai.data import DataLoader, list_data_collate # 의료 영상 전용 batch 처리, MetaTensor 지원, Tensform 결과 안정적으로 처리
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import get_datasets
from model import get_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def tta_inference(model, x):
    preds = []

    # original
    preds.append(sliding_window_inference(x, (96,96,96), 1, model))
    # flip x
    preds.append(torch.flip(
        sliding_window_inference(torch.flip(x, dims=[2]), (96,96,96), 1, model),
        dims=[2]
    ))
    # flip y
    preds.append(torch.flip(
        sliding_window_inference(torch.flip(x, dims=[3]), (96,96,96), 1, model),
        dims=[3]
    ))
    return torch.mean(torch.stack(preds), dim=0)


def train():
    with open("configs.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, cfg["paths"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_ds, val_ds = get_datasets(cfg)

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=list_data_collate, 
        pin_memory=True, 
        persistent_workers=True
        )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = get_model(cfg).to(cfg["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealing(optimizer, T_max=cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='max', # dice 기준이면 max, loss기준이면 min
                                                           factor=0.5, 
                                                           patience=3, 
                                                           verbose=True)
    #loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0) # Dice는 영역 맞춤인데, 영역보다 class 구분이 잘안되서 class 구분을 할 수 있는 CE로 변경해보기
    ce_weight = torch.tensor([0.2, 3.0, 2.5, 2.5]).to(cfg["device"])
    loss_fn = DiceCELoss(
        to_onehot_y=True, 
        softmax=True, 
        lambda_dice=0.7, 
        lambda_ce=0.3,
        weight=ce_weight
        )
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    writer = SummaryWriter(log_dir=f"{output_dir}/runs")

    best_dice = 0
    patience = 0

# -------------------------------------------------------------------
#    resume_path = f"{output_dir}/checkpoints/epoch_97_6405_unet.pth"

#    start_epoch = 0

#    if os.path.exists(resume_path):
#        print("🔄 Resume from checkpoint")

#        checkpoint = torch.load(resume_path)

#        model.load_state_dict(checkpoint["model"])
#        optimizer.load_state_dict(checkpoint["optimizer"])
#        scheduler.load_state_dict(checkpoint["scheduler"])

#        start_epoch = checkpoint["epoch"] + 1

#        print(f"Resume from epoch {start_epoch}")

# ----------------------------------------------------------------

    torch.cuda.empty_cache()# 캐시 초기화
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            images = batch["image"].to(cfg["device"])
            labels = batch["label"].to(cfg["device"])

            optimizer.zero_grad()
            #probs = torch.softmax(outputs, dim=1)
            #print(probs.max())
            #print(torch.unique(batch["label"]))
            #print("class 3 voxel:", (batch["label"] == 3).sum())

            scaler = GradScaler() # 메모리 30~40% 감소, 속도 증가
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        #scheduler.step()
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        
        #print("output mean:", outputs.mean().item())
        #print("output std:", outputs.std().item())

        # validation
        model.eval()
        dice_metric.reset()
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                val_images = val_batch["image"].to(cfg["device"])
                val_labels = val_batch["label"].to(cfg["device"])

                val_outputs = sliding_window_inference(
                    val_images,
                    roi_size=cfg["data"]["roi_size"],
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.75

                ) # TTA
                #val_outputs = tta_inference(model, val_images)

                # SwinUNTR은 확률기반이라 argmax를 쓰면 확률이 퍼져 있어 background로 쏠림
                probs = torch.softmax(val_outputs, dim=1)
                #print("max prob:", probs.max())
                #print("tumor prob mean:", probs[:,1:].mean())
                val_outputs = torch.argmax(probs, dim=1)

                #print("BEFORE POST:", torch.unique(val_outputs))  # 🔥 여기

                # argmax를 threshold로 수정
                #tumor_prob = probs[:,1:,:,:,:].sum(dim=1)
                #pred = (tumor_prob > 0.4).long()

                # connected component 추가
                #from monai.transforms import KeepLargestConnectedComponent
                #post = KeepLargestConnectedComponent(applied_labels=[1,2,3], is_onehot=False)
                #val_outputs = post(val_outputs)

                #print("AFTER POST:", torch.unique(val_outputs))

                # Dice 계산용 ont-hot
                val_outputs = torch.nn.functional.one_hot(val_outputs, num_classes=cfg["model"]["out_channels"])
                val_outputs = val_outputs.permute(0,4,1,2,3).float()

                dice_metric(y_pred=val_outputs, y=val_labels)

        dice = dice_metric.aggregate().item()
        # ReduceLROnPlateau는 반드시 metric이 필요함
        scheduler.step(dice) # max라서 dice 기준이라 dice을 넣음. min이면 loss 넣을 자리가서 loss를 넣야함
        writer.add_scalar("Dice/val", dice, epoch)

        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Dice: {dice:.4f}")

        # checkpoint
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, f"{output_dir}/checkpoints/epoch_{epoch}.pth")

        # best model
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), f"{output_dir}/best_model_dynunet.pth")
            print("saved the best model")
            patience = 0
        else:
            patience += 1

        # early stopping
        if patience > cfg["training"]["patience"]:
            print("Early stopping triggered")
            break
    
    torch.cuda.empty_cache()
    writer.close()


if __name__ == "__main__":
    train()