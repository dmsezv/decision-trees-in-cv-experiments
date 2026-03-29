from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    train_size: int | None = None,
    test_size: int | None = None,
):
    name = dataset_name.lower()

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = datasets.CIFAR10
    elif name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"[ERROR] Unknown dataset: {dataset_name}")

    ds_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = dataset_class(root=data_dir, train=True, download=True, transform=ds_transform)
    test_ds = dataset_class(root=data_dir, train=False, download=True, transform=ds_transform)

    if train_size is not None:
        train_ds = Subset(train_ds, list(range(train_size)))
        print(f"[DATA] Обучающая выборка ограничена до {train_size} семплов.")

    if test_size is not None:
        test_ds = Subset(test_ds, list(range(test_size)))
        print(f"[DATA] Тестовая выборка ограничена до {test_size} семплов.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
