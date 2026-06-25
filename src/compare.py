"""Experiment comparison script for CIFAR-10 baseline vs ResNet18."""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from src.model import get_model, count_parameters
from src.dataset import get_cifar10_datasets, get_dataloaders
from src.train import train
from src.evaluate import run_evaluation, CIFAR10_CLASSES


def run_experiment(model_name, train_loader, val_loader, test_loader, epochs, lr, results_dir, log_dir):
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name}")
    print(f"{'='*60}")

    exp_results_dir = os.path.join(results_dir, model_name)
    exp_log_dir = os.path.join(log_dir, model_name)
    exp_ckpt_dir = os.path.join("checkpoints", model_name)

    pretrained = model_name == "resnet18"
    freeze_backbone = model_name == "resnet18"

    model = get_model(model_name=model_name, num_classes=10, pretrained=pretrained, freeze_backbone=freeze_backbone)

    best_val_acc = train(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=exp_ckpt_dir,
        log_dir=exp_log_dir,
        patience=15,
        scheduler_type="cosine",
        min_lr=1e-5,
        warmup_epochs=5,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
    )

    checkpoint_path = os.path.join(exp_ckpt_dir, "best_model.pth")
    metrics = run_evaluation(
        model,
        test_loader,
        checkpoint_path=checkpoint_path,
        results_dir=exp_results_dir,
        log_dir=exp_log_dir,
    )

    return {
        "model": model_name,
        "trainable_params": count_parameters(model),
        "best_val_acc": float(best_val_acc),
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision_macro"],
        "test_recall": metrics["recall_macro"],
        "test_f1": metrics["f1_macro"],
        "test_loss": metrics["loss"],
    }


def print_comparison_table(results):
    header = f"{'Model':<15} {'Params':>12} {'Val Acc':>10} {'Test Acc':>10} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Loss':>10}"
    sep = "-" * len(header)
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model']:<15} {r['trainable_params']:>12,} "
            f"{100.0*r['best_val_acc']:>9.2f}% {100.0*r['test_accuracy']:>9.2f}% "
            f"{r['test_precision']:>8.4f} {r['test_recall']:>8.4f} {r['test_f1']:>8.4f} "
            f"{r['test_loss']:>10.4f}"
        )
    print(sep)


def save_comparison_table(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    lines.append("| Model | Trainable Params | Best Val Acc | Test Acc | Precision | Recall | F1 | Loss |")
    lines.append("|-------|-----------------|--------------|----------|-----------|--------|----|------|")
    for r in results:
        lines.append(
            f"| {r['model']} | {r['trainable_params']:,} | "
            f"{100.0*r['best_val_acc']:.2f}% | {100.0*r['test_accuracy']:.2f}% | "
            f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} | "
            f"{r['test_loss']:.4f} |"
        )
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CIFAR-10 models")
    parser.add_argument("--models", nargs="+", default=["baseline", "resnet18"], choices=["baseline", "resnet18"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-dir", type=str, default="results/logs")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = get_cifar10_datasets(root=args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)

    all_results = []
    for model_name in args.models:
        if model_name == "resnet18":
            lr = 0.01
            epochs = args.epochs
        else:
            lr = args.lr
            epochs = args.epochs
        result = run_experiment(model_name, train_loader, val_loader, test_loader, epochs, lr, args.results_dir, args.log_dir)
        all_results.append(result)

    print_comparison_table(all_results)
    save_comparison_table(all_results, os.path.join(args.results_dir, "comparison_table.md"))

    with open(os.path.join(args.results_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nComparison table saved to {os.path.join(args.results_dir, 'comparison_table.md')}")
