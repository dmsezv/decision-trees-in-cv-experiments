import torch
import torch.nn as nn

from external.NODE.arch import DenseBlock


class Classic_SE_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        Классический SE-блок на базе полносвязных слоев (MLP).
        reduction: коэффициент сжатия бутылочного горлышка.
        Для 32 каналов при reduction=4 внутренний слой будет иметь размер 8.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1. Squeeze: усредняем пространственные размеры
        v = self.pool(x).view(b, c)

        # 2. Excitation: пропускаем через полносвязную сеть с узким горлышком
        weights = self.fc(v).view(b, c, 1, 1)

        # 3. Перевзвешивание каналов
        return x * weights


class NODE_SE_Block(nn.Module):
    """
    Описание:
    Блок внимания в концепции Squeeze-and-Excitation (SE) в виде деревьев
    Механика работы (Soft Attention):
    Деревья принимают на вход глобально усредненные карты признаков и
    генерируют логиты важности для каждого канала.
    В качестве функции активации используется Sigmoid, которая сжимает логиты в диапазон от 0 до 1.
    """

    def __init__(self, channels, layer_dim=8, num_layers=2, depth=4):
        super().__init__()
        # Сжиматель: превращает (B, C, H, W) в (B, C, 1, 1)
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

        # 1. Squeeze: получаем "табличный" вектор из каналов
        v = self.pool(x).view(b, c)
        v = self.bn(v)

        # 2. Excitation: получаем веса от деревьев
        # Выход node: (Batch, layer_dim, channels). Усредняем голоса всех деревьев.
        tree_logits = self.node(v).mean(dim=1)

        # # Превращаем логиты в вероятности [0, 1]
        weights = torch.sigmoid(tree_logits).view(b, c, 1, 1)

        # # 3. Умножаем исходные карты признаков на веса внимания
        return x * weights
