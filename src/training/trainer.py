import time

import mlflow
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    it_count = len(loader)

    print("[INFO] training")
    with tqdm(total=it_count, desc="Train") as pbar:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            pbar.update()

    return total_loss / it_count, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    it_count = len(loader)

    print("[INFO] validating")
    with tqdm(total=it_count, desc="Valid") as pqbar:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

                pqbar.update()

    return total_loss / it_count, 100.0 * correct / total


def train_validate_loop(
    model, train_loader, test_loader, opt, crt, device, num_epochs, early_stopping_patience=10
):

    test_accuracy = []
    test_loss = []
    train_accuracy = []
    train_loss = []

    start_time = time.time()
    train_epoch_times = []
    val_epoch_times = []

    sheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5)
    best_val_l = float("inf")
    epochs_no_improve = 0

    epochs_count = num_epochs

    for epoch in range(num_epochs):
        print(f"\n[INFO] epoch: {epoch + 1}/{num_epochs}")

        start_train_time = time.time()
        t_l, t_a = train_one_epoch(model, train_loader, opt, crt, device)
        train_epoch_times.append(time.time() - start_train_time)

        train_loss.append(t_l)
        train_accuracy.append(t_a)

        start_val_time = time.time()
        val_l, val_a = validate(model, test_loader, crt, device)
        val_epoch_times.append(time.time() - start_val_time)

        test_loss.append(val_l)
        test_accuracy.append(val_a)

        print(f"[INFO] Train Acc {t_a:.2f}% | Val Acc {val_a:.2f}%")

        sheduler.step(val_l)
        curr_lr = opt.param_groups[0]["lr"]


        mlflow.log_metrics(
            {
                "train_accuracy": t_a,
                "val_loss": val_l,
                "train_loss": t_l,
                "val_accuracy": val_a,
                "lr": curr_lr,
                "overfitting_gap_acc": t_a - val_a
            },
            step=epoch,
        )

        if val_l < best_val_l:
            best_val_l = val_l
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"[INFO] Early stopping after {epoch + 1} epochs")
            epochs_count = epoch + 1
            break

    total_time = time.time() - start_time
    avg_train_time = sum(train_epoch_times) / len(train_epoch_times)
    avg_val_time = sum(val_epoch_times) / len(val_epoch_times)
    
    mlflow.log_metrics({
        "best_val_accuracy": max(test_accuracy),
        "avg_time_train_epoch_s": avg_train_time,
        "avg_time_val_epoch_s": avg_val_time,
        "epochs": epochs_count,
        "total_time_s": total_time
    })

    return train_accuracy, train_loss, test_accuracy, test_loss, total_time
