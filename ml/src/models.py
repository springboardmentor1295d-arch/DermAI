from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)


class BaselineCNN(nn.Module):
    """
    Lightweight 4-layer Convolutional Neural Network baseline.
    Serves as an empirical baseline to compare transfer learning models against.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 x 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 x 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 x 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


class MobileNetV3Transfer(nn.Module):
    """
    Pretrained MobileNetV3-Large transfer learning classifier.
    Optimized for high accuracy and low CPU inference latency.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        
        # Replace classifier head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_gradcam(self):
        # Target the last inverted residual feature block
        return self.backbone.features[-1]


class EfficientNetB0Transfer(nn.Module):
    """
    Pretrained EfficientNet-B0 transfer learning classifier.
    Compound scaling CNN balancing parameter count, memory footprint and representational capacity.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_gradcam(self):
        # Target the final convolutional stage (Conv2dNormActivation block)
        return self.backbone.features[-1]


class ResNet50Transfer(nn.Module):
    """
    Pretrained ResNet50 transfer learning classifier for benchmark comparison.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_gradcam(self):
        return self.backbone.layer4[-1]


class ResNet18Transfer(nn.Module):
    """
    Pretrained ResNet18 transfer learning classifier for empirical architecture comparison.
    Standard residual backbone with fast CPU training and inference.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_gradcam(self):
        return self.backbone.layer4[-1]


def get_model(architecture: str = "mobilenetv3_large", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    arch = architecture.lower()
    if arch in ["mobilenetv3", "mobilenetv3_large", "mobilenet_v3_large"]:
        return MobileNetV3Transfer(num_classes=num_classes, pretrained=pretrained)
    elif arch in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return EfficientNetB0Transfer(num_classes=num_classes, pretrained=pretrained)
    elif arch in ["baseline", "baseline_cnn"]:
        return BaselineCNN(num_classes=num_classes)
    elif arch in ["resnet18", "resnet_18"]:
        return ResNet18Transfer(num_classes=num_classes, pretrained=pretrained)
    elif arch in ["resnet50", "resnet_50"]:
        return ResNet50Transfer(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
