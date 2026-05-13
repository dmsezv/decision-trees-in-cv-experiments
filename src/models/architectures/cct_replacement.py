import torch
import torch.nn as nn
from torchvision import models

from external.CT.src import cct_4_3x2_32
from external.NODE.arch import DenseBlock

import torch.nn.functional as F

class CCT_4_Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cct = cct_4_3x2_32(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.cct(x)
    

class NODE_FFN(nn.Module):
    def __init__(self, d_model, layer_dim, num_layers, depth):
        super().__init__()
        self.node = DenseBlock(
            input_dim=d_model,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=d_model,
            depth=depth,
            flatten_output=False,
        )

    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b * n, d)
        x = self.node(x).mean(dim=1)
        x = x.reshape(b, n, d)
        return x


class TransformerEncoderLayer_NODE_FFN(nn.Module):
    def __init__(self, original_block, d_model, layer_dim, num_layers, depth):
        super().__init__()
        self.pre_norm = original_block.pre_norm
        self.self_attn = original_block.self_attn
        self.norm1 = original_block.norm1
        self.drop_path = original_block.drop_path
        self.node_ffn = NODE_FFN(d_model, layer_dim, num_layers, depth)

    def forward(self, src):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src = src + self.drop_path(self.node_ffn(src))
        return src
    

class CCT_4_NODE_FFN(nn.Module):
    def __init__(self, num_classes, layer_dim, num_layers, depth):
        super().__init__()
        self.cct = cct_4_3x2_32(pretrained=False, num_classes=num_classes)

        d_model = self.cct.classifier.fc.in_features
        total_blocks = len(self.cct.classifier.blocks)

        for i in range(total_blocks):
            original = self.cct.classifier.blocks[i]
            self.cct.classifier.blocks[i] = TransformerEncoderLayer_NODE_FFN(
                original_block=original,
                d_model=d_model,
                layer_dim=layer_dim,
                num_layers=num_layers,
                depth=depth,
            )
    
    def forward(self, x):
        return self.cct(x)

    def initial_weights(self, loader, device):
        self.train()
        init_inputs = []
        with torch.no_grad():
            for x, _ in loader:
                init_inputs.append(x.to(device))
                if sum(t.shape[0] for t in init_inputs) >= 1000:
                    break
        
        full_batch = torch.cat(init_inputs, dim=0)[:1000]
        _ = self.forward(full_batch)
        

class CCT_4_NODE_Head(nn.Module):
    def __init__(self, num_classes, layer_dim, num_layers, depth):
        super().__init__()
        self.cct = cct_4_3x2_32(pretrained=False, num_classes=num_classes)

        in_f = self.cct.classifier.fc.in_features
        self.cct.classifier.fc = nn.Identity()

        self.bn = nn.BatchNorm1d(in_f)
        self.node_head = DenseBlock(
            input_dim=in_f,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=num_classes,
            depth=depth,
            flatten_output=False,
        )

    def forward(self, x):
        features = self.cct(x)
        features = self.bn(features)
        out = self.node_head(features)
        return out.mean(dim=1)

    def initial_weights(self, loader, device):
        self.eval()
        init_inputs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                f = self.cct(x)
                f = self.bn(f)
                init_inputs.append(f)
                if len(torch.cat(init_inputs, dim=0)) >= 1000:
                    break
            full_inputs = torch.cat(init_inputs, dim=0)
            _ = self.node_head(full_inputs)