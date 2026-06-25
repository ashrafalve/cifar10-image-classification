# CIFAR-10 Image Classification

Professional PyTorch implementation for CIFAR-10 image classification.

## Project Structure

```
.
├── data/            # Dataset storage
├── checkpoints/     # Model checkpoints
├── results/         # Training results and logs
├── src/             # Source code
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── app.py
│   └── utils.py
├── notebooks/       # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.dataset import get_cifar10_datasets, get_dataloaders

train_ds, val_ds, test_ds = get_cifar10_datasets()
train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)
```
