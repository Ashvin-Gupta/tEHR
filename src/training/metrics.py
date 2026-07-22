"""
Evaluation metrics for classification tasks.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(eval_pred):
    """
    Compute classification metrics for evaluation.
    
    Calculates accuracy, precision, recall, F1, AUROC, and AUPR for binary classification.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels.
    
    Returns:
        Dict of metric names to values.
    """
    predictions, labels = eval_pred
    
    # Extract logits if predictions is tuple/dict
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    elif isinstance(predictions, dict):
        predictions = predictions['logits']
    
    t = torch.tensor(predictions, dtype=torch.float32)

    # Logistic head produces shape (N, 1); linear/mlp produce (N, C).
    if t.ndim == 2 and t.shape[1] == 1:
        probs = torch.sigmoid(t).squeeze(1).numpy()
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.argmax(predictions, axis=1)
        probs = torch.softmax(t, dim=1).numpy()[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.0  # Only one class present

    try:
        aupr = average_precision_score(labels, probs)
    except ValueError:
        aupr = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'aupr': aupr,
    }
