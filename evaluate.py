"""Metrics, confusion matrices, error analysis."""
import numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from config import CLASSES

def full_report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    report = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "per_class": {CLASSES[i]: dict(precision=p[i], recall=r[i],
                                       f1=f[i], support=int(sup[i]))
                      for i in range(len(CLASSES))},
        "lowest_recall_class": CLASSES[int(np.argmin(r))],
        "lowest_precision_class": CLASSES[int(np.argmin(p))],
    }
    return report

def plot_confusion(y_true, y_pred, normalize=False, path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    if path: fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return cm

def rank_confused_pairs(cm_norm):
    """pair_confusion(i,j) = C_norm[i,j] + C_norm[j,i]"""
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            pairs.append((CLASSES[i], CLASSES[j], cm_norm[i, j] + cm_norm[j, i]))
    return sorted(pairs, key=lambda t: -t[2])

def low_confidence_analysis(logits, labels, threshold=0.7):
    """Return coverage & accuracy when we route low-confidence samples to review."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    routed = conf < threshold
    accepted = ~routed
    acc_accepted = ((pred == labels) & accepted).sum().item() / max(1, accepted.sum().item())
    return {
        "coverage": accepted.float().mean().item(),
        "accuracy_on_accepted": acc_accepted,
        "review_rate": routed.float().mean().item(),
    }