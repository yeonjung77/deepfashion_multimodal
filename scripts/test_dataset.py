import sys, os
sys.path.append(os.path.abspath("."))

import torch
from torchvision import transforms
from src.datasets.deepfashion_dataset import DeepFashionDataset

csv_path = "data/processed/labels.csv"
image_root = "data/raw/images" 


# transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = DeepFashionDataset(csv_path, image_root, transform)

print("총 샘플 수:", len(dataset))

img, label = dataset[0]

print("이미지 텐서 shape:", img.shape)
print("라벨 벡터:", label)
print("라벨 길이:", len(label))
