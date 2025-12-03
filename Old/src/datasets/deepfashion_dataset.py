import os
from typing import Callable, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 1) 라벨 컬럼 이름을 미리 적어 둔다.
#    - CSV 안에는 항상 shape_0 ~ shape_11, fabric_0 ~ fabric_2, pattern_0 ~ pattern_2
#      총 18개의 열이 있다.
#    - 초보자라면 "그냥 저 순서대로 값을 읽는다" 라고 이해하면 된다.
# ---------------------------------------------------------------------------
SHAPE_COLS: List[str] = [f"shape_{i}" for i in range(12)]
FABRIC_COLS: List[str] = [f"fabric_{i}" for i in range(3)]
PATTERN_COLS: List[str] = [f"pattern_{i}" for i in range(3)]
ATTR_COLS: List[str] = SHAPE_COLS + FABRIC_COLS + PATTERN_COLS


def _load_image(image_root: str, image_name: str, transform: Callable | None):
    """
    간단한 이미지 로더.
    예) image_root="data/raw/images", image_name="ABC.jpg" 라면
        data/raw/images/ABC.jpg 를 열고 RGB 로 바꾼 뒤 transform 을 적용한다.
    """
    image_path = os.path.join(image_root, image_name)
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    return image


class DeepFashionDataset(Dataset):
    """
    멀티 클래스 라벨 18개(shape 12 + fabric 3 + pattern 3)를 그대로 이어 붙여
    길이 18짜리 벡터로 반환하는 기본 Dataset.
    - __getitem__은 항상 (이미지, torch.LongTensor(18)) 을 돌려준다.
    """

    def __init__(self, csv_path: str, image_root: str, transform: Callable | None = None):
        # csv_path 예시: data/processed/train.csv
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]  # pandas Series 한 줄
        image = _load_image(self.image_root, row["image_name"], self.transform)

        # 예) row["shape_0"] == 3 이라면 labels 리스트에도 3이 들어간다.
        labels = [int(row[col]) for col in ATTR_COLS]
        labels = torch.tensor(labels, dtype=torch.long)
        return image, labels
