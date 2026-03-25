"""
Longitudinal UMAP Alignment & Disease Trajectory Visualization.

Pipeline:
    1. Load embeddings for 1, 3, 6, and 12 months pre-diagnosis.
    2. Train an 'Anchor' UMAP model and HDBSCAN on the 1-month data.
    3. Project the 3, 6, and 12-month embeddings into the 1-month UMAP space.
    4. Plot a 1x4 geographic grid, coloring all patients by their 1-month outcome.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan

# ---------------------------------------------------------------------------
# Config: Update these paths to point to your specific output folders
# ---------------------------------------------------------------------------
BASE_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results"

# Example paths - please adjust to match your actual directory names!
PATH_1M  = f"{BASE_DIR}/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_3M  = f"{BASE_DIR}/lora-3-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_6M  = f"{BASE_DIR}/lora-6-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
PATH_12M = f"{BASE_DIR}/lora-12-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"

OUTPUT_DIR = f"{BASE_DIR}/longitudinal_trajectory_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed Anchor Parameters (The ones that worked perfectly for 1-month)
N_NEIGHBORS = 41
MIN_CLUSTER_SIZE = 67
MIN_SAMPLES = 5

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def load_embeddings(output_dir: str):
    """Loads embeddings from a specified directory."""
    print(f"Loading from: {output_dir}")
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
    # We don't strictly need patient_ids for the plot, but good to have
    return embeddings

def train_anchor_space(embeddings_1m: np.ndarray):
    """
    Trains the master coordinate system and defines the 'Ground Truth' phenotypes.
    """
    print("Training Anchor UMAP on 1-month data...")
    # Notice we keep the umap_model object so we can use it later!
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    z_1m = umap_model.fit_transform(embeddings_1m)

    print("Defining Anchor Phenotypes with HDBSCAN...")
    # We do a 10D reduction just for HDBSCAN clustering accuracy
    reducer_10d = umap.UMAP(n_components=10, n_neighbors=N_NEIGHBORS, min_dist=0.0, random_state=42)
    emb_10d = reducer_10d.fit_transform(embeddings_1m)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method='eom'
    )
    labels_1m = clusterer.fit_predict(emb_10d)
    
    return umap_model, labels_1m, z_1m

def project_historical_data(umap_model, embeddings_past: np.ndarray):
    """
    Takes past data and drops it into the 1-month coordinate space.
    """
    # CRITICAL: We use .transform(), NOT .fit_transform()
    z_past = umap_model.transform(embeddings_past)
    return z_past

# ---------------------------------------------------------------------------
# Journal Visualization
# ---------------------------------------------------------------------------
def plot_disease_trajectory(z_12m, z_6m, z_3m, z_1m, labels_1m, output_dir):
    """Plots the 1x4 grid showing cluster evolution."""
    sns.set(style="white", context="paper")
    
    # Set up the 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
    # Create a unified color palette based on the 1-month labels
    unique_labels = np.unique(labels_1m)
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    # Make the noise cluster (-1) light grey so it doesn't distract
    color_dict = {label: palette[i] if label != -1 else "#e0e0e0" 
                  for i, label in enumerate(unique_labels)}

    timepoints = [
        ("12 Months Pre-Diagnosis", z_12m),
        ("6 Months Pre-Diagnosis", z_6m),
        ("3 Months Pre-Diagnosis", z_3m),
        ("1 Month Pre-Diagnosis (Anchor)", z_1m)
    ]

    for ax, (title, z_data) in zip(axes, timepoints):
        # We color EVERY plot using the labels_1m (their destiny)
        sns.scatterplot(
            x=z_data[:, 0], 
            y=z_data[:, 1], 
            hue=labels_1m, 
            palette=color_dict,
            s=10, 
            alpha=0.6, 
            linewidth=0,
            ax=ax, 
            legend=False
        )
        
        ax.set_title(title, weight='bold', fontsize=14)
        ax.axis('off') # Remove axes for a clean, modern look

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig_longitudinal_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    # 1. Load Data
    emb_1m = load_embeddings(PATH_1M)
    emb_3m = load_embeddings(PATH_3M)
    emb_6m = load_embeddings(PATH_6M)
    emb_12m = load_embeddings(PATH_12M)
    
    # Quick sanity check - make sure the patient counts match!
    assert emb_1m.shape[0] == emb_12m.shape[0], "Error: The number of patients varies across timepoints!"

    # 2. Build the Anchor Space
    umap_model, labels_1m, z_1m = train_anchor_space(emb_1m)
    
    # 3. Project the Past into the Anchor Space
    print("Projecting historical embeddings...")
    z_3m = project_historical_data(umap_model, emb_3m)
    z_6m = project_historical_data(umap_model, emb_6m)
    z_12m = project_historical_data(umap_model, emb_12m)
    
    # 4. Generate the Plot
    print("Generating longitudinal trajectory plot...")
    plot_disease_trajectory(z_12m, z_6m, z_3m, z_1m, labels_1m, OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()