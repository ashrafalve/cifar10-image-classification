"""Model module for CIFAR-10 image classification."""

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """Basic convolution block: Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CIFAR10Baseline(nn.Module):
    """Custom CNN baseline for CIFAR-10 classification.

    Architecture:
        - 3 convolution blocks with increasing channels
        - MaxPooling after each block
        - 2 fully connected layers with dropout
        - Output: 10 class logits
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ResNet18Transfer(nn.Module):
    """ResNet18 with pretrained weights and modified classifier head for CIFAR-10."""

    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        nn.init.kaiming_normal_(self.backbone.fc[1].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.backbone.fc[1].bias, 0)
        nn.init.kaiming_normal_(self.backbone.fc[3].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.backbone.fc[3].bias, 0)

    def forward(self, x):
        return self.backbone(x)


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name="baseline", num_classes=10, dropout=0.3, pretrained=True, freeze_backbone=False):
    """Factory function to create a model.

    Args:
        model_name: 'baseline' or 'resnet18'
        num_classes: number of output classes
        dropout: dropout for baseline model
        pretrained: use ImageNet pretrained weights for ResNet18
        freeze_backbone: freeze ResNet18 backbone layers
    """
    if model_name == "baseline":
        model = CIFAR10Baseline(num_classes=num_classes, dropout=dropout)
        print(f"Created CIFAR10Baseline with {count_parameters(model):,} trainable parameters.")
    elif model_name == "resnet18":
        model = ResNet18Transfer(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
        print(f"Created ResNet18Transfer with {count_parameters(model):,} trainable parameters.")
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose 'baseline' or 'resnet18'.")
    return model
