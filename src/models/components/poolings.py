import torch
import torch.nn as nn


class SpatialAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        attn_weights = self.attention(x).view(b, 1, h * w)
        attn_weights = self.softmax(attn_weights)
        features = x.view(b, c, h * w)
        pooled = torch.bmm(features, attn_weights.transpose(1, 2)).squeeze(-1)  # [b, c]
        return pooled
