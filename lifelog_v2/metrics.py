"""Subject-level endpoints and a training-OOF-only decision rule."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, log_loss

NAMES = ["CN", "MCI", "DEM"]


def measure(y, scores, prediction=None):
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    prediction = scores.argmax(axis=1) if prediction is None else np.asarray(prediction, int)
    recalls = recall_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    result = {
        "roc_auc_macro_ovr": float(roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=[0, 1, 2])),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1, 2])),
    }
    for k, name in enumerate(NAMES):
        result[f"recall_{name}"] = float(recalls[k])
        result[f"auc_{name}_ovr"] = float(roc_auc_score(y == k, scores[:, k]))
        result[f"n_{name}"] = int((y == k).sum())
    return result


def objective(metrics):
    """Prioritize the weaker of the two requested endpoints; tiny tie-breaks."""
    a, r = metrics["roc_auc_macro_ovr"], metrics["macro_recall"]
    return min(a, r) + 1e-4 * (a + r) / 2 + 1e-6 * metrics["macro_f1"]


def select_policy(y, scores, *, enabled=True):
    """Class log-score offsets chosen ONLY from development/inner OOF labels.

    ROC-AUC is always computed from original normalized scores. Offsets affect
    decisions (and recall) only; the exported scores are not relabeled as
    calibrated posterior probabilities.
    """
    if not enabled:
        return np.zeros(3), measure(y, scores)
    grid = np.linspace(-1.5, 1.5, 13)
    offsets = np.array([(0., a, b) for a, b in itertools.product(grid, grid)])
    offsets -= offsets.mean(axis=1, keepdims=True)
    pred = (np.log(np.clip(scores, 1e-12, 1))[None, :, :] + offsets[:, None, :]).argmax(axis=2)
    y = np.asarray(y)
    actual = np.bincount(y, minlength=3)
    true_positive = np.column_stack([((pred == k) & (y[None, :] == k)).sum(axis=1) for k in range(3)])
    predicted = np.column_stack([(pred == k).sum(axis=1) for k in range(3)])
    recalls = (true_positive / actual[None, :]).mean(axis=1)
    f1 = (2 * true_positive / (predicted + actual[None, :])).mean(axis=1)
    norm = np.linalg.norm(offsets, axis=1)
    best = np.lexsort((-norm, f1, recalls))[-1]
    return offsets[best], measure(y, scores, pred[best])


def intervals(y, scores, prediction, *, repeats=2000, seed=2026):
    """Conditional stratified SUBJECT bootstrap; not training/tuning uncertainty."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == k) for k in range(3)]
    keys = ["roc_auc_macro_ovr", "macro_recall", "macro_f1", "accuracy", "recall_CN", "recall_MCI", "recall_DEM"]
    samples = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in classes])
        m = measure(y[idx], scores[idx], prediction[idx])
        samples.append([m[k] for k in keys])
    samples = np.array(samples)
    point = measure(y, scores, prediction)
    return pd.DataFrame({"metric": keys, "estimate": [point[k] for k in keys],
                         "lower_95": np.quantile(samples, .025, axis=0),
                         "upper_95": np.quantile(samples, .975, axis=0)})


def target_status(metrics, threshold=.8):
    return {"macro_target_met": bool(metrics["roc_auc_macro_ovr"] >= threshold and metrics["macro_recall"] >= threshold),
            "all_classes_target_met": bool(metrics["roc_auc_macro_ovr"] >= threshold and
                                            all(metrics[f"recall_{n}"] >= threshold for n in NAMES))}
