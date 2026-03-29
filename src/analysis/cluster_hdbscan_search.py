"""
HDBSCAN Hyperparameter Search.

Pipeline:
    1. Load held-out embeddings and derive binary case/control labels.
    2. Run a random search over HDBSCAN hyperparameters (n_neighbors for the
       10D UMAP reduction, min_cluster_size, min_samples,
       cluster_selection_method), scoring each configuration on cluster purity.
    3. Save the full search results and the best parameter set to disk.
"""

import json
import os
import random

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.data.dataset_factory import get_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "src/configs/classification_config.yaml"
EMBEDDINGS_DIR = (
    "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-3-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
)
OUTPUT_DIR = EMBEDDINGS_DIR + "/hdbscan_search"
N_TRIALS = 30
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(output_dir: str):
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
    embeddings = normalize(embeddings, norm='l2')
    patient_ids_path = os.path.join(output_dir, "patient_ids.npy")
    if os.path.exists(patient_ids_path):
        patient_ids = np.load(patient_ids_path, allow_pickle=True).tolist()
    else:
        patient_ids = list(range(embeddings.shape[0]))
    return embeddings, patient_ids


def load_labels(config_path: str, split: str, patient_ids: list) -> np.ndarray:
    """Return a binary label array (1 = case, 0 = control) aligned to patient_ids."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    dataset = get_dataset(data_config.get("dataset", "CPRD"), data_config, split)
    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    return np.array(
        [1 if int(subject_to_label.get(str(pid), 0)) > 0 else 0 for pid in patient_ids]
    )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------
def reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Constraints & scoring
# ---------------------------------------------------------------------------
MIN_NOISE_FRACTION = 0.005  # must have *some* noise (pure assignments are suspect)
MAX_NOISE_FRACTION = 0.05   # more than 5% noise → rejected outright
MIN_CLUSTERS = 4            # outside [MIN_CLUSTERS, MAX_CLUSTERS] → rejected outright
MAX_CLUSTERS = 12


def passes_constraints(cluster_labels: np.ndarray) -> bool:
    """Return True only if the clustering satisfies the hard noise/count constraints."""
    noise_ratio = float((cluster_labels == -1).mean())
    if noise_ratio <= MIN_NOISE_FRACTION or noise_ratio >= MAX_NOISE_FRACTION:
        return False
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    return MIN_CLUSTERS <= len(unique_clusters) <= MAX_CLUSTERS


def calculate_purity(cluster_labels: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Diagnostic metric: avg |case_rate - 0.5| * 2 across clusters.
    1.0 = all clusters entirely case or control; 0.0 = every cluster 50/50.
    Always computed (even for failing trials) so it can be inspected.
    """
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    if len(unique_clusters) == 0:
        return float("nan")
    purity_per_cluster = []
    for cid in unique_clusters:
        idx = np.where(cluster_labels == cid)[0]
        case_rate = binary_labels[idx].mean()
        purity_per_cluster.append(abs(case_rate - 0.5) * 2)
    return round(float(np.mean(purity_per_cluster)), 4)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def save_trial_umap(
    z_2d: np.ndarray,
    cluster_labels: np.ndarray,
    binary_labels: np.ndarray,
    params: dict,
    output_dir: str,
) -> None:
    """Save a 1x2 UMAP figure: left = clusters, right = case/control."""
    fig, (ax_cl, ax_cc) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Trial {int(params['trial'])}  —  silhouette={params['silhouette_score']:.4f}",
        weight="bold", fontsize=12,
    )

    # --- Left: coloured by cluster ---
    unique = sorted(set(cluster_labels))
    real_clusters = [c for c in unique if c != -1]
    cmap = plt.cm.get_cmap("tab10", max(len(real_clusters), 1))

    if -1 in unique:
        mask = cluster_labels == -1
        ax_cl.scatter(z_2d[mask, 0], z_2d[mask, 1], s=6, alpha=0.3,
                      color="lightgrey", label=f"Noise ({mask.sum()})", rasterized=True)
    for i, cid in enumerate(real_clusters):
        mask = cluster_labels == cid
        ax_cl.scatter(z_2d[mask, 0], z_2d[mask, 1], s=8, alpha=0.6,
                      color=cmap(i), label=f"Cluster {cid} ({mask.sum()})", rasterized=True)
        cx, cy = float(np.median(z_2d[mask, 0])), float(np.median(z_2d[mask, 1]))
        ax_cl.text(cx, cy, str(cid), fontsize=9, weight="bold", ha="center", va="center",
                   bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))

    ax_cl.set_title("Clusters", fontsize=11)
    ax_cl.axis("off")
    ax_cl.legend(title="Cluster", loc="lower right", fontsize=7, markerscale=1.5, framealpha=0.6)

    # --- Right: coloured by case / control ---
    case_mask = binary_labels == 1
    ctrl_mask = binary_labels == 0
    ax_cc.scatter(z_2d[ctrl_mask, 0], z_2d[ctrl_mask, 1], s=6, alpha=0.35,
                  color="lightgrey", label=f"Control ({ctrl_mask.sum()})", rasterized=True)
    ax_cc.scatter(z_2d[case_mask, 0], z_2d[case_mask, 1], s=8, alpha=0.7,
                  color="#e74c3c", label=f"Case ({case_mask.sum()})", rasterized=True)

    ax_cc.set_title("Case vs Control", fontsize=11)
    ax_cc.axis("off")
    ax_cc.legend(loc="lower right", fontsize=7, markerscale=1.5, framealpha=0.6)

    # --- Parameter box: top-left corner, small ---
    param_text = (
        f"nn={int(params['n_neighbors'])}  "
        f"mcs={int(params['min_cluster_size'])}  "
        f"ms={int(params['min_samples'])}\n"
        f"clusters={int(params['n_clusters'])}  "
        f"noise={params['noise_pct']:.1f}%\n"
        f"sil={params['silhouette_score']:.4f}  "
        f"dbcv={params['dbcv_validity']:.4f}  "
        f"purity={params['purity_score']:.4f}"
    )
    fig.text(
        0.01, 0.99, param_text,
        fontsize=6.5, family="monospace",
        va="top", ha="left",
        bbox=dict(facecolor="lightyellow", edgecolor="grey", boxstyle="round,pad=0.3", alpha=0.8),
    )

    plt.tight_layout()
    fname = f"trial_{int(params['trial']):03d}_sil{params['silhouette_score']:.4f}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Random hyperparameter search
# ---------------------------------------------------------------------------
def run_random_search(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    output_dir: str,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly sample HDBSCAN hyperparameter combinations and score each one.

    Search space:
        n_neighbors              : int in [10, 100]  (for the 10D UMAP reduction)
        min_cluster_size         : int in [5, 40]
        min_samples              : int in [1, 15]
        cluster_selection_method : 'eom'

    10D UMAP reductions are cached by n_neighbors to avoid redundant computation.
    A fixed 2D UMAP (n_neighbors=15) is computed once for visualisation.
    For every trial that passes all hard constraints a scatter plot is saved.

    Returns a DataFrame of all trial results sorted by score descending.
    """
    rng = random.Random(seed)
    umap_cache: dict = {}
    results = []

    plots_dir = os.path.join(output_dir, "trial_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("  Pre-computing 2D UMAP for visualisation (fixed, n_neighbors=15)...")
    z_2d = reduce_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1)

    for trial in range(1, n_trials + 1):
        n_neighbors = rng.randint(5, 30)
        min_cluster_size = rng.randint(5, 40)
        min_samples = rng.randint(1, 15)
        method = "eom"

        if n_neighbors not in umap_cache:
            print(f"  [Trial {trial}/{n_trials}] UMAP 10D (n_neighbors={n_neighbors})...")
            umap_cache[n_neighbors] = reduce_umap(
                embeddings, n_components=10, n_neighbors=n_neighbors
            )

        emb_10d = umap_cache[n_neighbors]

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
            prediction_data=False,
            gen_min_span_tree=True,
        ).fit(emb_10d)
        cluster_labels = clusterer.labels_

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_pct = float((cluster_labels == -1).mean() * 100)
        valid = passes_constraints(cluster_labels)

        # Purity: supervised diagnostic (reported but not used for ranking)
        purity = calculate_purity(cluster_labels, binary_labels)

        # Silhouette: primary optimisation target — unsupervised geometry score.
        # Only meaningful (and only computed) for trials passing hard constraints.
        non_noise = cluster_labels != -1
        if valid and non_noise.sum() > 0 and len(np.unique(cluster_labels[non_noise])) >= 2:
            sil = round(float(silhouette_score(emb_10d[non_noise], cluster_labels[non_noise])), 4)
        else:
            sil = float("nan")

        # DBCV relative validity: HDBSCAN's own density-based internal score
        try:
            dbcv = round(float(clusterer.relative_validity_), 4)
        except Exception:
            dbcv = float("nan")

        row = {
            "trial": trial,
            "n_neighbors": n_neighbors,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "silhouette_score": sil,
            "dbcv_validity": dbcv,
            "purity_score": purity,
            "n_clusters": n_clusters,
            "noise_pct": round(noise_pct, 2),
        }
        results.append(row)

        status = "✓ VALID" if valid else "✗"
        print(
            f"  [Trial {trial}/{n_trials}] {status} "
            f"nn={n_neighbors}, mcs={min_cluster_size}, ms={min_samples} "
            f"→ sil={sil:.3f}, dbcv={dbcv:.3f}, purity={purity:.3f}, "
            f"clusters={n_clusters}, noise={noise_pct:.1f}%"
        )

        if valid:
            save_trial_umap(z_2d, cluster_labels, binary_labels, row, plots_dir)

    return (
        pd.DataFrame(results)
        .sort_values("silhouette_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings...")
    print(EMBEDDINGS_DIR)
    embeddings, patient_ids = load_embeddings(EMBEDDINGS_DIR)

    print("Loading labels...")
    binary_labels = load_labels(CONFIG_PATH, "held_out", patient_ids)

    print(f"\nStarting {N_TRIALS}-trial random search...")
    df_search = run_random_search(embeddings, binary_labels, OUTPUT_DIR, n_trials=N_TRIALS)

    search_path = os.path.join(OUTPUT_DIR, "hdbscan_search_results.csv")
    df_search.to_csv(search_path, index=False)
    print(f"\nSearch complete. Results saved to {search_path}")

    print("\nTop 5 configurations:")
    print(df_search.head(5).to_string(index=False))

    valid_results = df_search.dropna(subset=["silhouette_score"])
    if valid_results.empty:
        print("\nNo trials passed the hard constraints. Consider relaxing them.")
        return
    best = valid_results.iloc[0].to_dict()
    best_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest parameters saved to {best_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
