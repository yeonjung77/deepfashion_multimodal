import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from src.datasets.deepfashion_dataset import DeepFashionDataset
from src.models.resnet_multilabel import ResNetMultiLabel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# 1. Dataset 준비
train_csv = "data/processed/train.csv"
val_csv = "data/processed/val.csv"
image_root = "data/raw/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = DeepFashionDataset(train_csv, image_root, transform)
val_dataset = DeepFashionDataset(val_csv, image_root, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


# 2. 모델 & Loss & Optimizer
model = ResNetMultiLabel(num_classes=18).to(device)
criterion = nn.CrossEntropyLoss()      # shape/fabric/pattern 각각이 "단일 label → 다중 분류"라서 Softmax 사용
optimizer = Adam(model.parameters(), lr=1e-4)


# 3. 학습 루프
def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)      # shape 12 + fabric 3 + pattern 3 = 총 18차원

        optimizer.zero_grad()
        outputs = model(imgs)           # (B, 18)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Train Loss: {total_loss / len(train_loader):.4f}")


def validate(epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    print(f"Epoch {epoch} | Val Loss: {total_loss / len(val_loader):.4f}")


# 4. Train
for epoch in range(5):
    train_one_epoch(epoch)
    validate(epoch)

# 모델 저장
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet50_multilabel.pth")
print("모델 저장 완료!")
