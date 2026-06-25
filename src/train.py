"""Training module for CIFAR-10 image classification."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / total, correct / total


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
        },
        checkpoint_path,
    )


def save_best_model(model, best_acc, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc,
        },
        best_model_path,
    )


def train(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.01,
    weight_decay=5e-4,
    checkpoint_dir="checkpoints",
    log_dir="results/logs",
    patience=15,
    scheduler_type="step",
    step_size=30,
    gamma=0.1,
    min_lr=1e-5,
    warmup_epochs=5,
):
    device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) if scheduler_type == "step" else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    early_stopping = EarlyStopping(patience=patience, mode="max")
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (epoch / warmup_epochs)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        if scheduler_type == "step":
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        tqdm.write(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {100.0 * train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {100.0 * val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_best_model(model, best_acc, checkpoint_dir)

        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, checkpoint_dir)

        if early_stopping(val_acc):
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()
    tqdm.write(f"Training complete. Best validation accuracy: {100.0 * best_acc:.2f}%")
