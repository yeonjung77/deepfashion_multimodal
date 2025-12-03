import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------------
# DeepFashion 라벨은 속성(attribute)마다 클래스 개수가 다르다.
# 아래 리스트는 "shape_0 은 6개 클래스, fabric_1 은 8개 클래스" 처럼
# 딱 정해진 숫자를 하드코딩한 것이다.
# 초보자라면 "각 속성마다 따로 Softmax 분기를 만든다" 라고 이해하면 된다.
# ---------------------------------------------------------------------------
ATTRIBUTE_SPECS = [
    ("shape_0", 6),
    ("shape_1", 5),
    ("shape_2", 4),
    ("shape_3", 3),
    ("shape_4", 5),
    ("shape_5", 3),
    ("shape_6", 3),
    ("shape_7", 3),
    ("shape_8", 5),
    ("shape_9", 7),
    ("shape_10", 3),
    ("shape_11", 3),
    ("fabric_0", 8),
    ("fabric_1", 8),
    ("fabric_2", 8),
    ("pattern_0", 8),
    ("pattern_1", 8),
    ("pattern_2", 8),
]


class ResNetMultiLabel(nn.Module):
    """
    ResNet-101 백본 + 속성별 FC Layer.
    - forward(x)는 {"shape_0": Tensor(B,6), ..., "pattern_2": Tensor(B,8)} 같은 dict 를 돌려준다.
    - 예) outputs["fabric_1"][0] 은 첫 번째 이미지의 fabric_1 로짓 벡터(길이 8).
    """

    def __init__(self):
        super().__init__()

        self.attribute_specs = ATTRIBUTE_SPECS

        # torchvision 에서 제공하는 ResNet-101 을 백본으로 사용.
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 원래 있던 FC(1000-way)를 제거하고 특징만 받는다.

        # 속성별로 Linear layer 하나씩 만들기.
        self.heads = nn.ModuleDict()
        for name, num_classes in self.attribute_specs:
            self.heads[name] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # shape: (Batch, 2048)
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(features)
        return outputs
