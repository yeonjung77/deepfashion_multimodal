# run_feature_extraction.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from src.data.dataset import DeepFashionAttrDataset
from src.data.preprocess import train_transform, val_transform
from src.models.resnet50_multilabel import ResNet50Attr
from src.models.train import training_loop

import os


# ---------------------------------------------------------
# 1) 경로 설정
# ---------------------------------------------------------
TRAIN_CSV = "data/processed/train.csv"
VAL_CSV   = "data/processed/val.csv"
IMAGE_ROOT = "data/raw/images"

SAVE_PATH = "outputs/checkpoints/resnet50_feature_extraction2.pth"
HISTORY_PATH = "outputs/logs/resnet50_feature_history2.pt"


# ---------------------------------------------------------
# 2) Device
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ---------------------------------------------------------
# 3) Dataset / Loader
# ---------------------------------------------------------
train_ds = DeepFashionAttrDataset(TRAIN_CSV, IMAGE_ROOT, train_transform)
val_ds   = DeepFashionAttrDataset(VAL_CSV, IMAGE_ROOT, val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

print(f"Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")


# ---------------------------------------------------------
# 4) Feature Extraction용 모델 세팅 (전체 freeze)
# ---------------------------------------------------------
model = ResNet50Attr(num_outputs=18).to(device)

# 전체 backbone freeze
for param in model.base.parameters():
    param.requires_grad = False

# FC layer만 학습
for param in model.base.fc.parameters():
    param.requires_grad = True

trainable_params = model.base.fc.parameters()


# ---------------------------------------------------------
# 5) Optimizer, Loss, Scheduler
# ---------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(trainable_params, lr=1e-3) 
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)


# ---------------------------------------------------------
# 6) Training 실행
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
