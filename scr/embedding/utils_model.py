import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from scr.models.resnet50_multilabel import ResNet50Attr


class ResNet50FeatureExtractor(nn.Module):
    """Pulls 2048-d pooled features from the fine-tuned ResNet-50."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        super().__init__()
        self.device = device
        backbone = ResNet50Attr(num_outputs=18)

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        backbone.load_state_dict(state_dict)
        backbone.eval()

        modules = list(backbone.base.children())[:-1]
        self.feature_net = nn.Sequential(*modules)
        self.feature_dim = backbone.base.fc.in_features
        self.to(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_net(images.to(self.device))
        return feats.view(feats.size(0), -1)

    @torch.no_grad()
    def encode_single(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image.unsqueeze(0)).squeeze(0)


def load_feature_extractor(
    checkpoint_path: str = "outputs/checkpoints/resnet50_finetune2.pth",
    device: Optional[str] = None,
) -> ResNet50FeatureExtractor:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ResNet50FeatureExtractor(ckpt, resolved_device)
