"""Dataset module for CIFAR-10 image classification."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_cifar10_datasets(root="data", val_split=0.1, seed=42):
    train_dataset_full = datasets.CIFAR10(
        root=root, train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=get_val_transforms()
    )

    train_size = int((1 - val_split) * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size], generator=generator
    )
    val_dataset.dataset.transform = get_val_transforms()

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader, test_loader
