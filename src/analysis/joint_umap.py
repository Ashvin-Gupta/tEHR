"""
Joint 2D UMAP visualisation across 4 time-horizon models.

For each of the 4 month configurations (1, 3, 6, 12) this script:
    1. Loads the raw model-dimension embeddings.
    2. Projects them directly to 2D with UMAP.
    3. Colours each point by its HDBSCAN cluster label (produced by hdbscan.py).
    4. Annotates cluster centroids with their ID.

All 4 subplots are arranged in a 2x2 grid and saved as a single high-res figure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "/gpfs/scratch/qc25022/pancreas/experiments/Results"
MONTHS = [1, 3, 6, 12]
EXPERIMENT_TEMPLATE = "lora-{m}-month-logistic-refactored-new-time-mistral-7B"
EMBEDDINGS_SUBDIR = "patient_clustering_heldout"
CLUSTERS_SUBDIR = "patient_clustering_heldout/journal_clustering_output"
OUTPUT_PATH = os.path.join(BASE, "joint_umap_4months.png")

RANDOM_SEED = 42
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

NOISE_COLOUR = "lightgrey"
CMAP_NAME = "tab10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_month_data(month: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        embeddings   : (N, D) float array — raw model-dimension vectors
        patient_ids  : (N,) array
        df_clusters  : DataFrame with at least columns [patient_id, cluster]
    """
    exp_dir = os.path.join(BASE, EXPERIMENT_TEMPLATE.format(m=month))
    emb_dir = os.path.join(exp_dir, EMBEDDINGS_SUBDIR)
    clus_dir = os.path.join(exp_dir, CLUSTERS_SUBDIR)

    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))

    pid_path = os.path.join(emb_dir, "patient_ids.npy")
    if os.path.exists(pid_path):
        patient_ids = np.load(pid_path, allow_pickle=True)
    else:
        patient_ids = np.arange(len(embeddings))

    clusters_path = os.path.join(clus_dir, "patient_clusters.csv")
    df_clusters = pd.read_csv(clusters_path, dtype={"patient_id": str})

    return embeddings, patient_ids, df_clusters


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project raw embeddings directly to 2D with UMAP."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )
    return reducer.fit_transform(embeddings)


def plot_month(ax: plt.Axes, z_2d: np.ndarray, cluster_labels: np.ndarray, month: int) -> None:
    """Draw a single subplot coloured by cluster, with centroid annotations."""
    unique_clusters = sorted(set(cluster_labels))
    real_clusters = [c for c in unique_clusters if c != -1]
    n_clusters = len(real_clusters)
    noise_count = int((cluster_labels == -1).sum())
    noise_pct = noise_count / len(cluster_labels) * 100

    cmap = plt.cm.get_cmap(CMAP_NAME, max(n_clusters, 1))

    # Plot noise first so real clusters render on top
    if -1 in unique_clusters:
        mask = cluster_labels == -1
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            s=6, alpha=0.3, color=NOISE_COLOUR, label=f"Noise ({noise_count})",
            rasterized=True,
        )

    for i, cid in enumerate(real_clusters):
        mask = cluster_labels == cid
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            s=8, alpha=0.6, color=cmap(i), label=f"Cluster {cid} ({mask.sum()})",
            rasterized=True,
        )
        # Centroid annotation
        cx, cy = z_2d[mask, 0].median(), z_2d[mask, 1].median()
        ax.text(
            cx, cy, str(cid),
            fontsize=10, weight="bold", ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    ax.set_title(f"{month}-Month Embeddings", weight="bold", fontsize=13)
    ax.set_xlabel(
        f"{n_clusters} clusters  |  noise {noise_pct:.1f}%",
        fontsize=9, color="grey",
    )
    ax.axis("off")
    ax.legend(
        loc="lower left", fontsize=7, markerscale=1.5,
        framealpha=0.6, borderpad=0.4,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    for ax, month in zip(axes_flat, MONTHS):
        print(f"Processing {month}-month model...")
        try:
            embeddings, patient_ids, df_clusters = load_month_data(month)
        except FileNotFoundError as e:
            ax.set_title(f"{month}-Month  [missing data]", color="red", weight="bold")
            ax.axis("off")
            print(f"  Skipped — {e}")
            continue

        # Align cluster labels to the embedding order
        pid_series = pd.Series(patient_ids.astype(str), name="patient_id")
        merged = pid_series.to_frame().merge(
            df_clusters[["patient_id", "cluster"]].astype({"patient_id": str}),
            on="patient_id", how="left",
        )
        cluster_labels = merged["cluster"].fillna(-1).astype(int).to_numpy()

        print(f"  Projecting {len(embeddings)} × {embeddings.shape[1]}D embeddings to 2D...")
        z_2d = project_2d(embeddings)

        plot_month(ax, z_2d, cluster_labels, month)
        print(f"  Done.")

    fig.suptitle(
        "Patient Embeddings — 2D UMAP Coloured by HDBSCAN Cluster\n"
        "(1, 3, 6, 12-Month Time Horizons)",
        fontsize=15, weight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
