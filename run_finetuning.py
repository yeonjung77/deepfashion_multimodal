# run_finetuning.py

import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.data.dataset import DeepFashionAttrDataset
from src.data.preprocess import train_transform, val_transform
from src.models.resnet50_multilabel import ResNet50Attr
from src.models.train import training_loop


# ---------------------------------------------------------
# 1) Paths
# ---------------------------------------------------------
TRAIN_CSV = "data/processed/train.csv"
VAL_CSV   = "data/processed/val.csv"
IMAGE_ROOT = "data/raw/images"

SAVE_PATH = "outputs/checkpoints/resnet50_finetune2.pth"
HISTORY_PATH = "outputs/logs/resnet50_finetune_history2.pt"


# ---------------------------------------------------------
# 2) Device
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ---------------------------------------------------------
# 3) Dataset / DataLoader
# ---------------------------------------------------------
train_ds = DeepFashionAttrDataset(TRAIN_CSV, IMAGE_ROOT, train_transform)
val_ds   = DeepFashionAttrDataset(VAL_CSV, IMAGE_ROOT, val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)


# ---------------------------------------------------------
# 4) Fine-tuning model
# ---------------------------------------------------------
model = ResNet50Attr(num_outputs=18).to(device)

# Partial fine-tuning: layer1~3 freeze
for name, param in model.base.named_parameters():
    if name.startswith(("layer1", "layer2", "layer3")):
        param.requires_grad = False

trainable_params = [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------
# 5) Optimizer, Loss, Scheduler
# ---------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(trainable_params, lr=1e-3)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)


# ---------------------------------------------------------
# 6) Run training
# ---------------------------------------------------------
history = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    criterion=criterion,
    epochs=10,
    save_path=SAVE_PATH
)

os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
torch.save(history, HISTORY_PATH)
print("History saved:", HISTORY_PATH)
