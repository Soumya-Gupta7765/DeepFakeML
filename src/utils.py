import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def seed_everything(seed: int = 42):
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes classification performance metrics:
    - Primary: AUC-ROC
    - Secondary: Accuracy, Precision, Recall, F1-score
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred_probs = np.asarray(y_pred_probs, dtype=float).ravel()
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5  # fallback if only one class in batch

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold)
    }


def find_best_threshold(y_true: np.ndarray, y_pred_probs: np.ndarray) -> tuple[float, float]:
    """
    Grid search to find the probability threshold maximizing F1 score on validation set.
    Returns: (best_threshold, best_f1)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred_probs = np.asarray(y_pred_probs, dtype=float).ravel()

    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_pred_probs >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh), float(best_f1)

