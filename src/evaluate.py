"""Evaluation module for CIFAR-10 image classification."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import torch
from tqdm import tqdm

try:
    from tensorboard.backend.event_processing import event_accumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_model(model, dataloader, device=None, class_names=None):
    if device is None:
        device = get_device()
    if class_names is None:
        class_names = CIFAR10_CLASSES

    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    running_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        total += targets.size(0)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))

    metrics = {
        "loss": running_loss / total,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "classification_report": report,
    }

    return metrics, cm, all_targets, all_preds


def save_confusion_matrix(cm, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_classification_report(report, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)
    for cls_name, vals in report.items():
        if cls_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        lines.append(
            f"{cls_name:<15} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
            f"{vals['f1-score']:>10.4f} {vals['support']:>10.0f}"
        )
    for avg_name in ("macro avg", "weighted avg"):
        vals = report[avg_name]
        lines.append("-" * 60)
        lines.append(
            f"{avg_name:<15} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
            f"{vals['f1-score']:>10.4f} {vals['support']:>10.0f}"
        )
    lines.append("-" * 60)
    lines.append(f"{'Accuracy':<15} {report['accuracy']:>10.4f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def plot_curves_from_events(log_dir, save_dir):
    if not HAS_TENSORBOARD:
        raise ImportError("tensorboard is required to plot curves from event logs.")
    os.makedirs(save_dir, exist_ok=True)
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    def _get_scalar(tag):
        if tag not in ea.Tags()["scalars"]:
            return [], []
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        return steps, vals

    def _plot(metric, train_tag, val_tag, filename, ylabel):
        train_steps, train_vals = _get_scalar(train_tag)
        val_steps, val_vals = _get_scalar(val_tag)
        if not train_steps and not val_steps:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        if train_steps:
            ax.plot(train_steps, train_vals, label="Train")
        if val_steps:
            ax.plot(val_steps, val_vals, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close(fig)

    _plot("Loss Curve", "Loss/train", "Loss/val", "loss_curve.png", "Loss")
    _plot("Accuracy Curve", "Accuracy/train", "Accuracy/val", "accuracy_curve.png", "Accuracy")


def run_evaluation(
    model,
    dataloader,
    checkpoint_path=None,
    class_names=None,
    results_dir="results",
    log_dir="results/logs",
):
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    metrics, cm, all_targets, all_preds = evaluate_model(model, dataloader, device, class_names)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(cm, class_names or CIFAR10_CLASSES, os.path.join(results_dir, "confusion_matrix.png"))
    save_classification_report(metrics["classification_report"], os.path.join(results_dir, "classification_report.txt"))

    if HAS_TENSORBOARD and os.path.isdir(log_dir):
        plot_curves_from_events(log_dir, results_dir)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Results saved to {results_dir}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-dir", type=str, default="results/logs")
    args = parser.parse_args()

    from src.model import get_model
    from src.dataset import get_cifar10_datasets, get_dataloaders

    train_ds, val_ds, test_ds = get_cifar10_datasets(root=args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
    model = get_model()
    run_evaluation(model, test_loader, checkpoint_path=args.checkpoint, results_dir=args.results_dir, log_dir=args.log_dir)
