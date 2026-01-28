"""
Visualization utilities for classification performance.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


def plot_classification_performance(labels: np.ndarray, probs: np.ndarray, output_dir: str):
    """
    Create and save comprehensive classification performance plots.
    
    Generates:
    - Confusion matrix
    - ROC curves (per-class and micro/macro averaged)
    - Precision-Recall curves
    
    Args:
        labels: True labels (shape: [n_samples])
        probs: Predicted probabilities (shape: [n_samples, n_classes])
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions and number of classes
    preds = np.argmax(probs, axis=1)
    n_classes = probs.shape[1]
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Confusion Matrix
    print("  - Generating confusion matrix...")
    plot_confusion_matrix(labels, preds, n_classes, output_dir)
    
    # 2. ROC Curves
    if n_classes == 2:
        # Binary classification
        print("  - Generating ROC curve (binary)...")
        plot_roc_binary(labels, probs[:, 1], output_dir)
    else:
        # Multi-class classification
        print("  - Generating ROC curves (multi-class)...")
        plot_roc_multiclass(labels, probs, n_classes, output_dir)
    
    # 3. Precision-Recall Curves
    if n_classes == 2:
        print("  - Generating Precision-Recall curve (binary)...")
        plot_pr_binary(labels, probs[:, 1], output_dir)
    else:
        print("  - Generating Precision-Recall curves (multi-class)...")
        plot_pr_multiclass(labels, probs, n_classes, output_dir)
    
    # 4. Probability distribution
    print("  - Generating probability distributions...")
    plot_probability_distribution(labels, probs, n_classes, output_dir)
    
    print(f"  ✓ All plots saved to {output_dir}")


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int, output_dir: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=range(n_classes), yticklabels=range(n_classes))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save raw counts
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(n_classes), yticklabels=range(n_classes))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_binary(labels: np.ndarray, probs: np.ndarray, output_dir: str):
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
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


def plot_roc_multiclass(labels: np.ndarray, probs: np.ndarray, n_classes: int, output_dir: str):
    """Plot ROC curves for multi-class classification."""
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot micro-average
    ax.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    # Plot per-class curves
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Multi-Class ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_multiclass.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_binary(labels: np.ndarray, probs: np.ndarray, output_dir: str):
    """Plot Precision-Recall curve for binary classification."""
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


def plot_pr_multiclass(labels: np.ndarray, probs: np.ndarray, n_classes: int, output_dir: str):
    """Plot Precision-Recall curves for multi-class classification."""
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    
    # Compute PR curve and AP for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        avg_precision[i] = average_precision_score(labels_bin[:, i], probs[:, i])
    
    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        labels_bin.ravel(), probs.ravel()
    )
    avg_precision["micro"] = average_precision_score(labels_bin, probs, average="micro")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot micro-average
    ax.plot(recall["micro"], precision["micro"],
            label=f'Micro-average (AP = {avg_precision["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    # Plot per-class curves
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'Class {i} (AP = {avg_precision[i]:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Multi-Class Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves_multiclass.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distribution(labels: np.ndarray, probs: np.ndarray, n_classes: int, output_dir: str):
    """Plot distribution of predicted probabilities per class."""
    fig, axes = plt.subplots(1, min(n_classes, 3), figsize=(15, 4))
    
    if n_classes == 1:
        axes = [axes]
    
    for i in range(min(n_classes, 3)):  # Plot first 3 classes
        ax = axes[i] if n_classes > 1 else axes[0]
        
        # Get probabilities for this class
        class_probs = probs[:, i]
        
        # Separate by true label
        correct = class_probs[labels == i]
        incorrect = class_probs[labels != i]
        
        # Plot histograms
        ax.hist(correct, bins=30, alpha=0.6, label=f'True Class {i}', color='green', edgecolor='black')
        ax.hist(incorrect, bins=30, alpha=0.6, label=f'Other Classes', color='red', edgecolor='black')
        
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Class {i} Probability Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


