import torch
import torch.nn as nn
import torchvision.models as models

class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 마지막 FC 교체
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)
