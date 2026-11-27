"""Model factory: CNNs and Transformers via timm, with optional clinical-metadata fusion."""
import timm
import torch
import torch.nn as nn

ARCHS = ["resnet50", "densenet121", "efficientnet_b0", "mobilenetv3_large_100",
         "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
         "deit_small_patch16_224"]


class FusionNet(nn.Module):
    """Image embedding (+) metadata MLP embedding -> classifier (feature fusion)."""

    def __init__(self, backbone, feat_dim, meta_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim + 64, num_classes))

    def forward(self, x, meta):
        return self.head(torch.cat([self.backbone(x), self.meta(meta)], dim=1))


def build_model(arch, num_classes, pretrained=True, meta_dim=0):
    if meta_dim:
        bb = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        return FusionNet(bb, bb.num_features, meta_dim, num_classes)
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=0.2)


def is_transformer(arch):
    return any(k in arch for k in ("vit", "swin", "deit"))
