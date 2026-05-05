import torch
import torch.nn as nn

from src.models.components.se_blocks import Classic_SE_Block, NODE_SE_Block


class SimpleConv_Baseline(nn.Module):
    def __init__(self, num_classes):
        """
        Описание:
        Классическая сверточная нейронная сеть (CNN), выступающая в роли контрольной модели.
        Архитектура состоит из простого последовательного извлекателя признаков (backbone)
        и линейного классификатора (head).
        Backbone включает в себя три сверточных блока, каждый из которых состоит из операции свертки (Conv2d),
        пакетной нормализации (BatchNorm2d), функции активации (ReLU) и операции подвыборки (MaxPool2d).
        Размерность каналов увеличивается последовательно: 16 → 32 → 64.

        Механика работы:
        Модель применяет статическую фильтрацию.
        Матрицы весов внутри Conv2d одинаково обрабатывают каждое входящее изображение,
        извлекая универсальный набор признаков независимо от контекста и класса объекта.

        Роль в исследовании:
        Демонстрирует проблему раннего переобучения (overfitting) на малых выборках и задает базовую планку
        (baseline) точности и вычислительной сложности, с которой сравниваются гибридные архитектуры.
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.head = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        return self.head(x)


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


class SimpleConv_Node_SE(nn.Module):
    """
    Как работает:
    Плавно сжимает любое число в диапазон от 0 до 1.

    Особенность:
      Она асимптотическая. Она никогда не выдает строгий математический ноль.
      Даже если дерево считает канал абсолютно бесполезным, сигмоида выдаст что-то вроде 0.00001.
      Сигнал становится микроскопическим, но физически канал остается "живым",
      и через него течет микро-градиент.

    Описание:
        Гибридная архитектура, в которой механизм полносвязных слоев в концепции Squeeze-and-Excitation (SE)
        заменен на ансамбль дифференцируемых решающих деревьев (NODE).
        Блоки внимания NODE_SE_Block интегрированы после второго и третьего сверточных слоев.

    Механика работы (Soft Attention):
        Деревья принимают на вход глобально усредненные карты признаков и
        генерируют логиты важности для каждого канала. В качестве функции активации используется Sigmoid,
        которая сжимает логиты в диапазон от 0 до 1.

    Сигмоида обеспечивает мягкое перевзвешивание:
        Полезные признаки получают веса, близкие к 1, а шумовые — близкие к 0,
        но никогда не обнуляются полностью. Это сохраняет поток микро-градиентов во время
        обратного распространения ошибки, позволяя сверточным слоям плавно дообучаться.

    Args:
        layer_dim (int): Количество деревьев в одном слое. При передаче в ODST этот
            параметр маппится на аргумент num_trees.
        num_layers (int): Количество слоев из деревьев (блоков ODST),
            которые будут идти друг за другом в этом DenseBlock
        tree_dim (int): Размерность выхода каждого отдельного дерева.
        max_features (int, optional): Ограничение на максимальное количество признаков,
            которые могут быть переданы на следующий слой.
        input_dropout (float): Вероятность зануления входных признаков. Служит для регуляризации,
            чтобы модель не переобучалась на конкретных фичах.
        depth (int): Глубина каждого дерева.
            Количество листьев в таком дереве всегда равно 2^depth. Глубокие деревья могут ловить
            сложные паттерны, но требуют больше памяти. Обычно используют от 4 до 8.
    """

    def __init__(self, num_classes, layer_dim=4, num_layers=2, depth=4):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.se2 = NODE_SE_Block(channels=32, layer_dim=layer_dim, num_layers=num_layers, depth=depth)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.se3 = NODE_SE_Block(channels=64, layer_dim=layer_dim, num_layers=num_layers, depth=depth)

        self.head = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = torch.flatten(x, 1)
        return self.head(x)

    def initial_weights(self, loader, device):
        self.eval()
        init_inputs = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                init_inputs.append(x)

                if len(torch.cat(init_inputs, dim=0)) >= 1000:
                    break

            full_inputs = torch.cat(init_inputs, dim=0)
            self.forward(full_inputs)


class SEBasicBlock(nn.Module):
    """
    BasicBlock из ResNet18 с SE-блоком внутри (канонический SE-ResNet, Hu et al. 2018).
    SE применяется к residual-ветке ПЕРЕД сложением со skip-connection.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = Classic_SE_Block(channels=planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18_Classic_SE(nn.Module):
    """
    Канонический SE-ResNet18, адаптированный под CIFAR (32x32):
    - Уменьшенный stem (3x3 conv, stride=1, без MaxPool)
    - SE-блок внутри каждого BasicBlock (всего 8 SE-блоков на сеть)

    Соответствует оригинальной реализации из Hu, Shen, Sun (2018).
    """

    def __init__(self, num_classes: int, reduction: int = 16):
        super().__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, reduction):
        downsample = None
        if stride != 1 or self.inplanes != planes * SEBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * SEBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * SEBasicBlock.expansion),
            )

        layers = [SEBasicBlock(self.inplanes, planes, stride, downsample, reduction)]
        self.inplanes = planes * SEBasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(SEBasicBlock(self.inplanes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class NODE_SEBasicBlock(nn.Module):
    """
    BasicBlock из ResNet18 с NODE-SE-блоком внутри.
    Структурно полностью аналогичен каноническому SEBasicBlock (Hu et al. 2018),
    но excitation-слои MLP заменены на ансамбль дифференцируемых деревьев NODE.

    NODE-SE применяется к residual-ветке ПЕРЕД сложением со skip-connection.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_dim=8, num_layers=2, depth=4):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = NODE_SE_Block(channels=planes, layer_dim=layer_dim, num_layers=num_layers, depth=depth)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18_NODE_SE(nn.Module):
    """
    Канонический NODE-SE-ResNet18, адаптированный под CIFAR (32x32):
    - Уменьшенный stem (3x3 conv, stride=1, без MaxPool)
    - NODE-SE-блок внутри каждого BasicBlock (всего 8 NODE-SE-блоков на сеть)

    По аналогии с SE-ResNet (Hu et al. 2018), но с дифференцируемым ансамблем
    решающих деревьев (NODE) вместо MLP в excitation-части.
    """

    def __init__(self, num_classes: int, layer_dim: int = 8, num_layers: int = 2, depth: int = 4):
        super().__init__()
        self.inplanes = 64
        self._block_kwargs = {"layer_dim": layer_dim, "num_layers": num_layers, "depth": depth}

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * NODE_SEBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * NODE_SEBasicBlock.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * NODE_SEBasicBlock.expansion),
            )

        layers = [NODE_SEBasicBlock(self.inplanes, planes, stride, downsample, **self._block_kwargs)]
        self.inplanes = planes * NODE_SEBasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(NODE_SEBasicBlock(self.inplanes, planes, **self._block_kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def initial_weights(self, loader, device):
        self.eval()
        init_inputs = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                init_inputs.append(x)

                if len(torch.cat(init_inputs, dim=0)) >= 1000:
                    break

            full_inputs = torch.cat(init_inputs, dim=0)
            self.forward(full_inputs)
