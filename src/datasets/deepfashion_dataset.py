import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class DeepFashionDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        """
        csv_path: labels.csv 경로
        image_root: 이미지가 저장된 data/raw/images/
        transform: torchvision transforms
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["image_name"]  # 예: MEN-Denim-id_00000089...
        img_path = os.path.join(self.image_root, img_name)

        # 이미지 로딩
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # shape: 문자열 → 리스트 변환
        shape = eval(row["shape"])       # ex: [3,4,2,...] → python list
        fabric = eval(row["fabric"])
        pattern = eval(row["pattern"])

        # 하나의 벡터로 합치기
        labels = shape + fabric + pattern     # [12 + 3 + 3 = 18]

        # 텐서로 변환
        labels = torch.tensor(labels, dtype=torch.long)

        return img, labels
