import os
import torch
import yaml

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from src.data.dataset import DeepFashionAttrDataset
from src.data.preprocess import train_transform, val_transform
from src.models.resnet50_multilabel import ResNet50Attr
from src.models.train import training_loop


def load_config(path="train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ------------------------------------------------
    # 1) Load Config
    # ------------------------------------------------
    config = load_config()

    TRAIN_CSV = config["data"]["train_csv"]
    VAL_CSV = config["data"]["val_csv"]
    IMAGE_ROOT = config["data"]["image_root"]
    SAVE_PATH = config["training"]["save_path"]

    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["lr"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ------------------------------------------------
    # 2) Datasets & Loaders
    # ------------------------------------------------
    train_ds = DeepFashionAttrDataset(TRAIN_CSV, IMAGE_ROOT, train_transform)
    val_ds = DeepFashionAttrDataset(VAL_CSV, IMAGE_ROOT, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")

    # ------------------------------------------------
    # 3) Model
    # ------------------------------------------------
    model = ResNet50Attr(num_outputs=18).to(device)

    # Partial fine-tuning: freeze layer1~3
    for name, param in model.base.named_parameters():
        if name.startswith(("layer1", "layer2", "layer3")):
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ------------------------------------------------
    # 4) Optimizer, Scheduler
    # ------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(trainable_params, lr=LR)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # ------------------------------------------------
    # 5) Training Loop
    # ------------------------------------------------
    print("Start Training...")
    history = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion,
        epochs=EPOCHS,
        save_path=SAVE_PATH
    )


if __name__ == "__main__":
    main()
