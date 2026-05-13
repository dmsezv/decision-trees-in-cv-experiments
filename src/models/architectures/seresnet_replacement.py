import timm
import torch
import torch.nn as nn
from src.models.components.se_blocks import NODE_SE_Block

class SEResnet18_Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.seresnet = timm.create_model(
            "seresnet18", 
            pretrained=False,
            num_classes=num_classes,
        )
        self.seresnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.seresnet.maxpool = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.seresnet(x)
    

class SEResnet18_NODE_SE(nn.Module):
    def __init__(self, num_classes, layer_dim, num_layers, depth):
        super().__init__()
        self.seresnet = timm.create_model(
            "seresnet18", 
            pretrained=False,
            num_classes=num_classes,
        )
        self.seresnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.seresnet.maxpool = nn.Identity()

        target_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        for layer_name in target_layers:
            layer = getattr(self.seresnet, layer_name)
            for block in layer.children():
                if hasattr(block, 'se'):
                    in_channels = block.se.fc1.in_channels
                    block.se = NODE_SE_Block(
                        channels=in_channels,
                        layer_dim=layer_dim,
                        num_layers=num_layers,
                        depth=depth,
                    )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def forward(self, x):
        return self.seresnet(x)
    
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
        