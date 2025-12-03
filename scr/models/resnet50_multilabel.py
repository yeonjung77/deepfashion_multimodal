import torch.nn as nn
from torchvision import models

class ResNet50Attr(nn.Module):
    def __init__(self, num_outputs=18):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.base(x)
