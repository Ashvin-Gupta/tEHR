"""
Longitudinal UMAP Alignment & Disease Trajectory Visualization.

Pipeline:
    1. Load the frozen 2D UMAP model and 1-month cluster labels.
    2. Load raw embeddings for 12, 6, 3, and 1 month pre-diagnosis.
    3. Project historical embeddings onto the frozen 2D map using .transform().
    4. Plot a 1x4 geographic grid, coloring all patients by their 1-month outcome.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config: Update these to your exact paths
# ---------------------------------------------------------------------------
BASE_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results"

# Where your 1-month models and labels were just saved
ANCHOR_DIR = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/journal_clustering_output"

# Paths to the raw embeddings for each time point
PATH_12M = f"{BASE_DIR}/lora-12-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_6M  = f"{BASE_DIR}/lora-6-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_3M  = f"{BASE_DIR}/lora-3-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_1M  = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"

OUTPUT_DIR = f"{BASE_DIR}/longitudinal_trajectory_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: Safe Loading & Alignment
# ---------------------------------------------------------------------------
def load_and_align_embeddings(emb_dir: str, reference_ids: list):
    """
    Loads embeddings and ensures they are in the EXACT SAME order as the 1-month 
    reference data. This is critical so the colors map to the correct patients.
    """
    print(f"Loading data from: {emb_dir}")
    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
    patient_ids = np.load(os.path.join(emb_dir, "patient_ids.npy"), allow_pickle=True).tolist()
    
    # Create a mapping of patient_id -> embedding vector
    emb_dict = {pid: emb for pid, emb in zip(patient_ids, embeddings)}
    
    # Rebuild the embedding array in the exact order of the reference IDs
    aligned_embeddings = []
    for pid in reference_ids:
        if pid in emb_dict:
            aligned_embeddings.append(emb_dict[pid])
        else:
            raise ValueError(f"Patient {pid} is missing from {emb_dir}!")
            
    return np.array(aligned_embeddings)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("1. Loading 1-Month Anchor Models & Labels...")
    # Load the freeze-dried 2D UMAP map
    umap_2d_model = joblib.load(os.path.join(ANCHOR_DIR, "umap_2d.joblib"))
    
    # Load the labels and reference patient IDs
    df_anchor = pd.read_csv(os.path.join(ANCHOR_DIR, "patient_clusters.csv"))
    reference_ids = df_anchor['patient_id'].tolist()
    labels_1m = df_anchor['cluster'].values
    
    print("2. Loading and Aligning Historical Embeddings...")
    # We load and align the older data to match the 1-month patient order
    emb_12m = load_and_align_embeddings(PATH_12M, reference_ids)
    emb_6m  = load_and_align_embeddings(PATH_6M, reference_ids)
    emb_3m  = load_and_align_embeddings(PATH_3M, reference_ids)
    emb_1m  = load_and_align_embeddings(PATH_1M, reference_ids)

    print("3. Projecting Historical Data onto Anchor Map...")
    # CRITICAL: We use .transform() to apply the existing map coordinates
    z_12m = umap_2d_model.transform(emb_12m)
    z_6m  = umap_2d_model.transform(emb_6m)
    z_3m  = umap_2d_model.transform(emb_3m)
    z_1m  = umap_2d_model.transform(emb_1m) # Should exactly match your 1-month plot

    print("4. Generating 1x4 Trajectory Plot...")
    sns.set(style="white", context="paper")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    # Unified color palette based on your 1-month labels
    unique_labels = np.unique(labels_1m)
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    color_dict = {label: palette[i] if label != -1 else "#e0e0e0" for i, label in enumerate(unique_labels)}

    timepoints = [
        ("12 Months Pre-Diagnosis", z_12m),
        ("6 Months Pre-Diagnosis", z_6m),
        ("3 Months Pre-Diagnosis", z_3m),
        ("1 Month Pre-Diagnosis (Anchor)", z_1m)
    ]

    for ax, (title, z_data) in zip(axes, timepoints):
        sns.scatterplot(
            x=z_data[:, 0], 
            y=z_data[:, 1], 
            hue=labels_1m,       # Color by their ultimate 1-month destination
            palette=color_dict,
            s=12, 
            alpha=0.6, 
            linewidth=0,
            ax=ax, 
            legend=False
        )
        ax.set_title(title, weight='bold', fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig5_longitudinal_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Masterpiece saved to: {save_path}")

if __name__ == "__main__":
    main()