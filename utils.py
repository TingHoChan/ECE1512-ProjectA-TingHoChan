"""Utilities intentionally provided to students.

The project focuses on the PyTorch model/training pipeline. Dataset engineering,
metric boilerplate, and reproducibility helpers are provided so students can
concentrate on Dataset/DataLoader usage, nn.Module, forward passes, losses,
backpropagation, optimization, validation, and model selection.
"""
from __future__ import annotations

import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Struct({items})"


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return Struct(**(yaml.safe_load(f) or {}))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_feature_dim(dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Cannot infer feature dimension from an empty dataset.")
    if hasattr(dataset, "feature_dim"):
        return int(dataset.feature_dim)
    return int(dataset[0]["input"].shape[-1])


def classification_metrics(probabilities: torch.Tensor,
                           labels: torch.Tensor,
                           n_class: int):
    """Return macro AUROC, accuracy (%), and macro F1.

    Metric implementation is provided because writing AUROC/F1 from scratch is
    not a learning objective of this project.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    labels = labels.long()
    p = probabilities.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    if len(y) == 0 or p.shape[1] != n_class or not np.isfinite(p).all():
        raise ValueError('Invalid probabilities or empty labels')
    if labels.ndim != 1 or len(y) != len(p):
        raise ValueError('labels must be a length-N vector')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1.0, atol=1e-5):
        raise ValueError('Class probabilities must be nonnegative and sum to one')
    if (y < 0).any() or (y >= n_class).any():
        raise ValueError('Class labels out of range')
    # A full macro one-vs-rest AUROC is undefined if any class is absent.
    # Do not silently turn undefined classes into AUC=0 or 0.5.
    aucs = [roc_auc_score(y == c, p[:, c]) if 0 < (y == c).sum() < len(y)
            else float('nan') for c in range(n_class)]
    auc = float(np.mean(aucs))
    f1 = float(f1_score(y, p.argmax(1), labels=list(range(n_class)),
                        average='macro', zero_division=0))
    acc = 100.0 * int((probabilities.argmax(dim=1) == labels).sum()) / len(labels)
    return {"auc": auc, "acc": acc, "f1": f1}
