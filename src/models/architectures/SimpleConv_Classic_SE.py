import torch
import torch.nn as nn

from src.models.components import Classic_SE_Block


class SimpleConv_Classic_SE(nn.Module):
    """Сверточная сеть с классическим SE-вниманием"""

    def __init__(self, num_classes, reduction=4):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.se2 = Classic_SE_Block(channels=32, reduction=reduction)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.se3 = Classic_SE_Block(channels=64, reduction=reduction)

        self.head = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = torch.flatten(x, 1)
        return self.head(x)
