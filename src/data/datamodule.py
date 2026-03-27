from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    train_size: int | None = None,
    test_size: int | None = None,
):
    ds_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=ds_transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=ds_transform)

    if train_size is not None:
        train_ds = Subset(train_ds, list(range(train_size)))
        print(f"[DATA] Обучающая выборка ограничена до {train_size} семплов.")

    if test_size is not None:
        test_ds = Subset(test_ds, list(range(test_size)))
        print(f"[DATA] Тестовая выборка ограничена до {test_size} семплов.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
