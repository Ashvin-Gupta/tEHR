"""
Joint Longitudinal UMAP Visualization.

Pipeline:
    1. Load and L2-normalize embeddings for 12, 6, 3, and 1 month.
    2. Load your finalized 1-month cluster labels (Ground Truth).
    3. Stack all time points into one giant array and run a JOINT UMAP.
    4. Split the 2D results back apart and plot the 1x4 trajectory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Config: Update to your exact paths
# ---------------------------------------------------------------------------
BASE_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results"

# The directory where you saved your final 1-month CSV
JOURNAL_DIR = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output"
ANCHOR_DIR = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"

PATH_12M = f"{BASE_DIR}/lora-12-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_6M  = f"{BASE_DIR}/lora-6-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_3M  = f"{BASE_DIR}/lora-3-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_1M  = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"

OUTPUT_DIR = f"{BASE_DIR}/longitudinal_trajectory_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def load_and_align_embeddings(emb_dir: str, reference_ids: list):
    """Loads, aligns, and L2 normalizes embeddings."""
    print(f"Loading data from: {emb_dir}")
    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
    patient_ids = np.load(os.path.join(emb_dir, "patient_ids.npy"), allow_pickle=True).tolist()

    emb_dict = {pid: emb for pid, emb in zip(patient_ids, embeddings)}
    
    aligned_embeddings = []
    for pid in reference_ids:
        if pid in emb_dict:
            aligned_embeddings.append(emb_dict[pid])
        else:
            raise ValueError(f"Patient {pid} missing from {emb_dir}!")
            
    # L2 Normalize the array
    return normalize(np.array(aligned_embeddings), norm='l2')

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("1. Loading 1-Month Ground Truth Labels...")
    df_anchor = pd.read_csv(os.path.join(ANCHOR_DIR, "patient_clusters.csv"))
    reference_ids = df_anchor['patient_id'].tolist()
    labels_1m = df_anchor['cluster'].values
    
    print("2. Loading and Aligning All Time Points...")
    emb_12m = load_and_align_embeddings(PATH_12M, reference_ids)
    emb_6m  = load_and_align_embeddings(PATH_6M, reference_ids)
    emb_3m  = load_and_align_embeddings(PATH_3M, reference_ids)
    emb_1m  = load_and_align_embeddings(PATH_1M, reference_ids)

    # Number of patients
    N = len(reference_ids)

    print("3. Stacking Data and Running JOINT UMAP (This may take a minute)...")
    # Stack all 4 datasets on top of each other (Shape: 4N x 4096)
    combined_embeddings = np.vstack([emb_12m, emb_6m, emb_3m, emb_1m])
    
    # Run UMAP on everything at once so it learns the universal map
    # Using cosine metric as it pairs perfectly with L2 normalization
    joint_umap = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42, n_jobs=1)
    z_combined = joint_umap.fit_transform(combined_embeddings)

    print("4. Splitting the map back into time points...")
    # Slice the array back into the 4 distinct months
    z_12m = z_combined[0:N]
    z_6m  = z_combined[N:2*N]
    z_3m  = z_combined[2*N:3*N]
    z_1m  = z_combined[3*N:4*N]

    print("5. Generating 1x4 Trajectory Plot...")
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
            hue=labels_1m,       # Color EVERY plot by their 1-month outcome!
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
    save_path = os.path.join(OUTPUT_DIR, "fig5_joint_longitudinal_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Masterpiece saved to: {save_path}")

if __name__ == "__main__":
    main()