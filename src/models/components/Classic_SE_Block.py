import torch.nn as nn


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
