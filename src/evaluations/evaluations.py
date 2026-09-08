"""
Binary classification metrics and plots (AUROC, AUPR, ECE).
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import torch


def compute_classification_metrics(eval_pred):
    """HF Trainer callback: convert logits to probabilities, then score."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    elif isinstance(predictions, dict):
        predictions = predictions["logits"]

    t = torch.tensor(predictions, dtype=torch.float32)
    if t.ndim == 2 and t.shape[1] == 1:
        probs = torch.sigmoid(t).squeeze(1).numpy()
    else:
        probs = torch.softmax(t, dim=1).numpy()[:, 1]
    return compute_binary_metrics(labels, probs)


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute binary classification metrics from predicted probabilities.

    Returns AUROC / AUPR / ECE plus threshold-0.5 accuracy / precision / recall / F1
    (same set used by the logistic classifier eval).
    """
    # Ensure arrays are numpy
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    # Validate inputs
    if len(labels) != len(probs):
        raise ValueError(f"Labels and probabilities must have same length. Got {len(labels)} and {len(probs)}")

    preds = (probs >= 0.5).astype(int)
    accuracy = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. Setting ranking metrics to 0.0")
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": 0.0,
            "aupr": 0.0,
            "ece": 0.0,
        }
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError as e:
        print(f"Warning: Could not compute AUROC: {e}. Setting to 0.0")
        auroc = 0.0
    
    # Compute AUPR (Average Precision)
    try:
        aupr = average_precision_score(labels, probs)
    except ValueError as e:
        print(f"Warning: Could not compute AUPR: {e}. Setting to 0.0")
        aupr = 0.0
    
    # Compute Expected Calibration Error (ECE)
    ece = compute_ece(labels, probs)
    
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "aupr": float(aupr),
        "ece": float(ece),
    }


def mean_t_ci(values, ci: float = 0.95) -> dict:
    """Mean, sample SD, and Student-t CI for a list of seed-level metrics.

    ``n=1`` returns the point with empty ``lo``/``hi``. This is the
    across-run interval (training luck), not the test-set bootstrap.
    """
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        raise ValueError("mean_t_ci got no finite values")
    mean = float(np.mean(x))
    out = {
        "n": n,
        "mean": mean,
        "sd": float("nan") if n < 2 else float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "lo": None,
        "hi": None,
        "ci": float(ci),
        "t_crit": None,
    }
    if n < 2:
        return out
    from scipy.stats import t as student_t

    se = out["sd"] / np.sqrt(n)
    t_crit = float(student_t.ppf(0.5 + ci / 2.0, df=n - 1))
    out["t_crit"] = t_crit
    out["lo"] = mean - t_crit * se
    out["hi"] = mean + t_crit * se
    return out


def bootstrap_binary_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    stratified: bool = True,
    clusters: np.ndarray | None = None,
) -> dict:
    """Percentile bootstrap CIs for AUROC and AUPR on a fixed test set.

    Resamples the already-scored ``(label, prob)`` pairs — no retraining.
    ``stratified=True`` keeps the positive rate fixed in each replicate
    (appropriate for rare-event clinical tasks). If ``clusters`` is given
    (e.g. patient IDs), whole clusters are drawn with replacement instead.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels and probs must have the same length")
    if labels.size == 0:
        raise ValueError("empty arrays")

    alpha = (1.0 - ci) / 2.0
    rng = np.random.default_rng(seed)
    n = labels.shape[0]

    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.shape[0] != n:
            raise ValueError("clusters must align with labels")
        uniq, inverse = np.unique(clusters, return_inverse=True)
        groups = [np.where(inverse == i)[0] for i in range(len(uniq))]
        n_groups = len(groups)

    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)

    aurocs = np.empty(n_boot, dtype=np.float64)
    auprs = np.empty(n_boot, dtype=np.float64)
    kept = 0
    for _ in range(n_boot):
        if clusters is not None:
            chosen = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([groups[i] for i in chosen])
        elif stratified and pos.size and neg.size:
            idx = np.concatenate(
                [
                    rng.choice(pos, size=pos.size, replace=True),
                    rng.choice(neg, size=neg.size, replace=True),
                ]
            )
        else:
            idx = rng.integers(0, n, size=n)
        y_b, p_b = labels[idx], probs[idx]
        if np.unique(y_b).size < 2:
            continue
        try:
            aurocs[kept] = roc_auc_score(y_b, p_b)
            auprs[kept] = average_precision_score(y_b, p_b)
        except ValueError:
            continue
        kept += 1

    if kept == 0:
        raise ValueError("every bootstrap replicate had a single class")
    aurocs = aurocs[:kept]
    auprs = auprs[:kept]

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
        return float(np.mean(arr)), float(lo), float(hi)

    auroc_mean, auroc_lo, auroc_hi = _ci(aurocs)
    aupr_mean, aupr_lo, aupr_hi = _ci(auprs)
    point = compute_binary_metrics(labels, probs)
    return {
        "n": int(n),
        "n_pos": int(pos.size),
        "n_boot": int(kept),
        "ci": float(ci),
        "stratified": bool(stratified and clusters is None),
        "clustered": clusters is not None,
        "auroc": float(point["auroc"]),
        "auroc_boot_mean": auroc_mean,
        "auroc_lo": auroc_lo,
        "auroc_hi": auroc_hi,
        "aupr": float(point["aupr"]),
        "aupr_boot_mean": aupr_mean,
        "aupr_lo": aupr_lo,
        "aupr_hi": aupr_hi,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and true frequencies.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
    
    Returns:
        Expected Calibration Error (float)
    """
    try:
        # Use quantile strategy to ensure we have data in all bins
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='quantile'
        )
        
        # Get bin counts using the same binning strategy
        # For quantile strategy, bins are based on quantiles of y_prob
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_counts, _ = np.histogram(y_prob, bins=bin_edges)
        
        # Only compute ECE for bins that have samples
        valid_bins = bin_counts > 0
        if not np.any(valid_bins):
            return 0.0
        
        # Ensure arrays have same length (should match number of valid bins)
        min_len = min(len(fraction_of_positives), len(mean_predicted_value), np.sum(valid_bins))
        if min_len == 0:
            return 0.0
        
        # Truncate arrays to same length
        fraction_of_positives = fraction_of_positives[:min_len]
        mean_predicted_value = mean_predicted_value[:min_len]
        bin_counts_valid = bin_counts[valid_bins][:min_len]
        
        # Compute ECE
        bin_weights = bin_counts_valid / len(y_prob)
        ece = np.sum(bin_weights * np.abs(fraction_of_positives - mean_predicted_value))
        return float(ece)
    except Exception as e:
        print(f"Warning: Could not compute ECE: {e}. Returning 0.0")
        return 0.0


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, output_dir: str):
    """
    Plot and save ROC curve.
    
    Args:
        labels: True binary labels
        probs: Predicted probabilities
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - ROC curve saved to {output_dir}/roc_curve.png")
    except Exception as e:
        print(f"Warning: Could not generate ROC plot: {e}")


def plot_pr_curve(labels: np.ndarray, probs: np.ndarray, output_dir: str):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        labels: True binary labels
        probs: Predicted probabilities
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        precision, recall, _ = precision_recall_curve(labels, probs)
        avg_precision = average_precision_score(labels, probs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Precision-Recall curve saved to {output_dir}/precision_recall_curve.png")
    except Exception as e:
        print(f"Warning: Could not generate PR plot: {e}")


def plot_calibration_curve(labels: np.ndarray, probs: np.ndarray, output_dir: str, n_bins: int = 10):
    """
    Plot and save calibration curve.
    
    Args:
        labels: True binary labels
        probs: Predicted probabilities
        output_dir: Directory to save plot
        n_bins: Number of bins for calibration curve
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=n_bins, strategy='uniform'
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add ECE to plot
        ece = compute_ece(labels, probs, n_bins=n_bins)
        ax.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Calibration curve saved to {output_dir}/calibration_curve.png")
    except Exception as e:
        print(f"Warning: Could not generate calibration plot: {e}")


def plot_all_curves(labels: np.ndarray, probs: np.ndarray, output_dir: str, n_bins: int = 10):
    """
    Plot and save all evaluation curves (ROC, PR, and Calibration).
    
    Args:
        labels: True binary labels
        probs: Predicted probabilities
        output_dir: Directory to save plots
        n_bins: Number of bins for calibration curve
    """
    plot_roc_curve(labels, probs, output_dir)
    plot_pr_curve(labels, probs, output_dir)
    plot_calibration_curve(labels, probs, output_dir, n_bins=n_bins)


def save_results(results_dict: dict, output_dir: str, filename: str = 'results.json'):
    """
    Save evaluation results to JSON file.
    
    Args:
        results_dict: Dictionary of results to save
        output_dir: Directory to save results
        filename: Name of output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"  - Results saved to {filepath}")


def print_results(results_dict: dict, split_name: str = ""):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results_dict: Dictionary of results
        split_name: Name of the split (e.g., 'validation', 'test')
    """
    prefix = f"{split_name.upper()} " if split_name else ""
    print(f"\n{prefix}Results:")
    print("-" * 50)
    for key, value in results_dict.items():
        if isinstance(value, float):
            print(f"  {key.upper()}: {value:.4f}")
        else:
            print(f"  {key.upper()}: {value}")
    print("-" * 50)


def write_split_artifacts(
    output_dir: str,
    split_name: str,
    labels,
    probs,
    *,
    plots: bool = True,
    extra: dict | None = None,
) -> dict:
    """Score a split, print metrics, and write plots + ``{split}_results.json``."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    metrics = compute_binary_metrics(labels, probs)
    print_results(metrics, split_name)
    if plots:
        plot_all_curves(labels, probs, os.path.join(output_dir, "plots", split_name))
        payload = {
            "metrics": metrics,
            "labels": labels.tolist(),
            "probs": probs.tolist(),
        }
        if extra:
            payload.update(extra)
        save_results(
            payload,
            os.path.join(output_dir, "results"),
            f"{split_name}_results.json",
        )
    return metrics

