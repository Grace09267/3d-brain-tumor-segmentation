from monai.networks.nets import UNet, AttentionUnet, DynUNet, SwinUNETR
import yaml
with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)


def get_model(cfg):
    model_name = cfg["model"]["name"]
    in_channels = cfg["model"]["in_channels"]
    out_channels = cfg["model"]["out_channels"]

    if model_name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2, # ResUNet 효과
        )

    elif model_name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
        )

    elif model_name == "dynunet":
        model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3,3,3,3],
            strides=[1,2,2,2],
            upsample_kernel_size=[2,2,2],
        )
    
    elif model_name == "swinunetr":
        model = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,
        )        

    else:
        raise ValueError(f"❌ Unknown model: {model_name}")

    print(f"✅ Using model: {model_name}")
    return model