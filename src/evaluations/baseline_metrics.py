"""
Unified evaluation metrics for baseline models.

Computes AUROC, AUPR, and Calibration (ECE) for pancreatic cancer prediction.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve


def compute_baseline_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute AUROC, AUPR, and Expected Calibration Error (ECE) for binary classification.
    
    Args:
        labels: True binary labels (shape: [n_samples])
        probs: Predicted probabilities for positive class (shape: [n_samples])
    
    Returns:
        Dictionary with 'auroc', 'aupr', and 'ece' metrics
    """
    # Ensure arrays are numpy
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    # Validate inputs
    if len(labels) != len(probs):
        raise ValueError(f"Labels and probabilities must have same length. Got {len(labels)} and {len(probs)}")
    
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. Setting metrics to 0.0")
        return {'auroc': 0.0, 'aupr': 0.0, 'ece': 0.0}
    
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
        'auroc': float(auroc),
        'aupr': float(aupr),
        'ece': float(ece)
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

