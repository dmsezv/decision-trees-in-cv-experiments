import torch
import torch.nn as nn
from torchvision import models

from external.NODE.arch import DenseBlock
from src.models.components.poolings import SpatialAttentionPooling


class ResNet18_Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class ResNet18_NODE(nn.Module):
    def __init__(self, num_classes, layer_dim, num_layers, depth):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()

        in_f = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.bn1 = nn.BatchNorm1d(in_f)

        self.node_head = DenseBlock(
            input_dim=in_f,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=num_classes,
            depth=depth,
            flatten_output=False,
        )

    def forward(self, x):
        features = self.resnet18(x)
        features = torch.flatten(features, 1)
        features = self.bn1(features)
        out = self.node_head(features)
        return out.mean(dim=1)

    def initial_weights(self, loader, device):
        self.eval()
        init_inputs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                f = self.resnet18(x)
                f = torch.flatten(f, 1)
                f = self.bn1(f)
                init_inputs.append(f)
                if len(torch.cat(init_inputs, dim=0)) >= 1000:
                    break
            full_inputs = torch.cat(init_inputs, dim=0)
            _ = self.node_head(full_inputs)


class ResNet18_MultiScaleAttention_NODE(nn.Module):
    def __init__(self, num_classes, layer_dim, num_layers, depth):
        super().__init__()
        backbone = models.resnet18(weights=None)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), backbone.bn1, backbone.relu
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.attn_pool1 = SpatialAttentionPooling(64)
        self.attn_pool2 = SpatialAttentionPooling(128)
        self.attn_pool3 = SpatialAttentionPooling(256)
        self.attn_pool4 = SpatialAttentionPooling(512)

        self.bn1 = nn.BatchNorm1d(960)

        self.node_head = DenseBlock(
            input_dim=960,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=num_classes,
            depth=depth,
            flatten_output=False,
        )

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        p1 = self.attn_pool1(f1)
        p2 = self.attn_pool2(f2)
        p3 = self.attn_pool3(f3)
        p4 = self.attn_pool4(f4)

        multi_scale_features = torch.cat((p1, p2, p3, p4), dim=1)
        multi_scale_features = self.bn1(multi_scale_features)

        out = self.node_head(multi_scale_features)
        return out.mean(dim=1)

    def initial_weights(self, loader, device):
        self.eval()
        init_inputs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                x = self.stem(x)
                f1 = self.layer1(x)
                f2 = self.layer2(f1)
                f3 = self.layer3(f2)
                f4 = self.layer4(f3)

                p1 = self.attn_pool1(f1)
                p2 = self.attn_pool2(f2)
                p3 = self.attn_pool3(f3)
                p4 = self.attn_pool4(f4)

                features = torch.cat((p1, p2, p3, p4), dim=1)
                features = self.bn1(features)
                init_inputs.append(features)

                if len(torch.cat(init_inputs, dim=0)) >= 1000:
                    break

            full_inputs = torch.cat(init_inputs, dim=0)
            _ = self.node_head(full_inputs)
