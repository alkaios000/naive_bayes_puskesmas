from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASSES = ["Naik", "Stabil", "Turun"]

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi lengkap per algoritma:
    Accuracy, precision per class, recall per class, f1 per class, F1 macro, dan confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )

    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    precision_dict = {cls: float(precision[i]) for i, cls in enumerate(CLASSES)}
    recall_dict = {cls: float(recall[i]) for i, cls in enumerate(CLASSES)}
    f1_dict = {cls: float(f1[i]) for i, cls in enumerate(CLASSES)}
    cm_list = [[int(cell) for cell in row] for row in cm]

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision": {k: round(v, 4) for k, v in precision_dict.items()},
        "recall": {k: round(v, 4) for k, v in recall_dict.items()},
        "f1": {k: round(v, 4) for k, v in f1_dict.items()},
        "confusion_matrix": cm_list
    }
