import torch
import torch.nn as nn

from external.NODE.arch import DenseBlock


class NODE_SE_Block(nn.Module):
    def __init__(self, channels, layer_dim, num_layers, depth):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(channels)
        
        self.node = DenseBlock(
            input_dim=channels,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=channels,
            depth=depth,
            flatten_output=False,
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        v = self.pool(x).view(b, c)
        v = self.bn(v)
        tree_logits = self.node(v).mean(dim=1)
        weights = torch.sigmoid(tree_logits).view(b, c, 1, 1)
        return x * weights