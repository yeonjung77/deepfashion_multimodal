import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from src.datasets.deepfashion_dataset import DeepFashionDataset  # noqa: E402
from src.models.resnet_multilabel import ResNetMultiLabel  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# 1. Dataset 준비 ------------------------------------------------------------
# train_csv/val_csv 는 "이미지 파일 이름 + 18개의 라벨 컬럼"을 담고 있다.
# image_root 아래에는 실제 JPG 파일이 존재한다.
train_csv = "data/processed/train.csv"
val_csv = "data/processed/val.csv"
image_root = "data/raw/images"
num_epochs = 20
batch_size = 128

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모든 이미지를 224x224 로 통일
    transforms.ToTensor(),          # 픽셀을 [0,1] 범위의 Tensor 로 변환
])

# Dataset → DataLoader 순으로 아주 정석적인 PyTorch 파이프라인 구성
# (데이터셋 구현이 궁금하다면 src/datasets/deepfashion_dataset.py 를 열어보면 된다.)
train_dataset = DeepFashionDataset(train_csv, image_root, transform)
val_dataset = DeepFashionDataset(val_csv, image_root, transform)
print(f"[INFO] Train samples: {len(train_dataset):,}")
print(f"[INFO] Val samples:   {len(val_dataset):,}")

# 튜토리얼 이해를 위해 batch_size 변수를 한 곳에서 조절한다.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
print(f"[INFO] Batch size: {train_loader.batch_size} | Num attributes: {len(train_dataset[0][1])}")


# 2. 모델 & Loss & Optimizer -------------------------------------------------
# 모델 내부 구조(속성별 Linear head)는 src/models/resnet_multilabel.py 에 자세히 적혀 있다.
model = ResNetMultiLabel().to(device)
criterion = nn.CrossEntropyLoss()                     # 각 속성마다 CrossEntropyLoss 사용
optimizer = Adam(model.parameters(), lr=1e-4)         # 아주 기본적인 Adam
attribute_names = [name for name, _ in model.attribute_specs]  # ["shape_0", ..., "pattern_2"]
num_attributes = len(attribute_names)                 # 항상 18


def compute_multiclass_multilabel_loss(outputs, labels):
    """
    모델 출력(dict)과 라벨(길이 18 Tensor)을 받아 평균 CrossEntropyLoss 를 구한다.
    예) outputs["shape_0"].shape == (batch, 6), labels[:, 0] == 정답 클래스 인덱스
    """
    losses = []                                     # attribute별 loss를 임시 저장
    for idx, name in enumerate(attribute_names):    # 18개 attribute를 순회
        logits = outputs[name]                      # (B, num_classes_for_attr) 모델 출력
        targets = labels[:, idx]                    # 해당 attribute의 정답 (B,)
        losses.append(criterion(logits, targets))   # CrossEntropyLoss 계산
    return torch.stack(losses).mean()               # attribute 평균 loss 반환


# 3. 학습/평가 유틸리티 ------------------------------------------------------
def _compute_attribute_metrics(attr_targets, attr_preds, attr_probs):
    auroc_scores = []   # attribute별 AUROC 저장
    f1_scores = []      # attribute별 F1 저장

    for idx, name in enumerate(attribute_names):        # 각 attribute 반복
        if not attr_targets[name]:                      # 값이 없으면 skip
            continue

        targets = torch.cat(attr_targets[name], dim=0).numpy()  # 정답 Tensor -> numpy
        preds = torch.cat(attr_preds[name], dim=0).numpy()      # 예측 Tensor -> numpy
        probs = torch.cat(attr_probs[name], dim=0).numpy()      # 확률 Tensor -> numpy
        num_classes = model.attribute_specs[idx][1]             # 클래스 개수

        one_hot = np.zeros((len(targets), num_classes), dtype=np.float32)  # AUROC용 one-hot
        one_hot[np.arange(len(targets)), targets] = 1.0
        auroc = roc_auc_score(one_hot, probs, multi_class="ovr", average="macro")  # AUROC 계산

        auroc_scores.append(auroc)

        f1 = f1_score(targets, preds, average="macro", zero_division=0)    # F1 계산
        f1_scores.append(f1)

    mean_auroc = float(np.nanmean(auroc_scores)) if auroc_scores else float("nan")  # 평균 AUROC
    mean_f1 = float(np.nanmean(f1_scores)) if f1_scores else float("nan")           # 평균 F1
    return mean_auroc, mean_f1


def _init_metric_storage():
    return {
        "targets": {name: [] for name in attribute_names},  # 정답 인덱스 저장 리스트
        "preds": {name: [] for name in attribute_names},    # 예측 인덱스 저장 리스트
        "probs": {name: [] for name in attribute_names},    # 소프트맥스 확률 저장 리스트
    }

breakpoint()
# 4. Train
for epoch in range(num_epochs):
    # ---- Training phase ----------------------------------------------------
    # 모델을 학습 모드로 두고, 각 배치마다 optimizer로 가중치를 갱신한다.
    # 동시에 AUROC/F1 계산에 필요한 정보(확률/정답/예측)를 저장해 둔다.
    model.train()
    train_total_loss = 0.0
    train_metrics = _init_metric_storage()

    for imgs, labels in tqdm(
        train_loader,
        desc=f"[Train] Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
        ascii=True,
        position=0,
    ):
        imgs = imgs.to(device)                      # 1) 배치 이미지를 GPU/CPU device 로 이동
        labels = labels.to(device)                  # 2) 라벨도 동일한 device 로 이동

        optimizer.zero_grad()                       # 3) 직전 배치에서 누적된 gradient 초기화
        outputs = model(imgs)                       # 4) ResNet-101 + head 로 forward 수행
        loss = compute_multiclass_multilabel_loss(outputs, labels)  # 5) attribute별 CrossEntropy 평균
        loss.backward()                             # 6) 역전파로 gradient 계산
        optimizer.step()                            # 7) optimizer가 파라미터 업데이트

        train_total_loss += loss.item()             # 8) epoch 평균을 위해 loss 누적
        for idx, name in enumerate(attribute_names):
            logits = outputs[name]                  # 9) 해당 attribute 의 로짓
            probs = torch.softmax(logits, dim=1)    # 10) 소프트맥스로 확률 변환
            preds = torch.argmax(logits, dim=1)     # 11) argmax 로 예측 클래스 산출
            targets = labels[:, idx]                # 12) 해당 attribute 의 정답 클래스

            train_metrics["probs"][name].append(probs.detach().cpu())     # 13) CPU 로 옮겨 metric 저장
            train_metrics["targets"][name].append(targets.detach().cpu())
            train_metrics["preds"][name].append(preds.detach().cpu())
       

    train_avg_loss = train_total_loss / len(train_loader)
    train_auroc, train_f1 = _compute_attribute_metrics(
        train_metrics["targets"], train_metrics["preds"], train_metrics["probs"]
        )
    print(f"[Train] Epoch {epoch} | Loss {train_avg_loss:.4f} | AUROC {train_auroc:.4f} | F1 {train_f1:.4f}")

    # ---- Validation phase --------------------------------------------------
    # 검증 데이터는 torch.no_grad() 아래에서 forward 만 수행한다.
    # 학습과 동일하게 속성별 확률/정답/예측을 모아 AUROC/F1 을 계산한다.
    model.eval()
    val_total_loss = 0.0
    val_metrics = _init_metric_storage()

    with torch.no_grad():
        for imgs, labels in tqdm(
            val_loader,
            desc=f"[Val]   Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
            position=0,
        ):
            imgs = imgs.to(device)                      # 1) 검증 배치 이미지 device 이동
            labels = labels.to(device)                  # 2) 검증 라벨 device 이동

            outputs = model(imgs)                       # 3) forward만 수행 (no grad)
            loss = compute_multiclass_multilabel_loss(outputs, labels)  # 4) 검증 loss 계산
            val_total_loss += loss.item()               # 5) 추후 평균 산출 위해 누적
            for idx, name in enumerate(attribute_names):
                logits = outputs[name]                               # 6) attribute 로짓
                probs = torch.softmax(logits, dim=1).detach().cpu()   # 7) 확률로 변환 후 CPU 저장
                preds = torch.argmax(logits, dim=1).detach().cpu()    # 8) 예측 클래스
                targets = labels[:, idx].detach().cpu()               # 9) 정답 클래스

                val_metrics["probs"][name].append(probs)              # 10) metric 계산을 위한 저장
                val_metrics["targets"][name].append(targets)
                val_metrics["preds"][name].append(preds)

    val_avg_loss = val_total_loss / len(val_loader)
    val_auroc, val_f1 = _compute_attribute_metrics(
        val_metrics["targets"], val_metrics["preds"], val_metrics["probs"]
        )
    print(f"[Val]   Epoch {epoch} | Loss {val_avg_loss:.4f} | AUROC {val_auroc:.4f} | F1 {val_f1:.4f}")

# 모델 저장 ---------------------------------------------------------------
# 학습이 끝나면 checkpoints/resnet50_multilabel.pth 로 가중치를 저장한다.
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet50_multilabel.pth")
print("모델 저장 완료!")
