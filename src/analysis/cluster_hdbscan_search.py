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

from src.data.dataset_factory import get_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "src/configs/classification_config.yaml"
EMBEDDINGS_DIR = (
    "/gpfs/scratch/qc25022/pancreas/experiments/Results/"
    "lora-1-month-logistic-refactored-new-time-mistral-7B/"
    "patient_clustering_heldout"
)
OUTPUT_DIR = EMBEDDINGS_DIR + "/hdbscan_search"
N_TRIALS = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(output_dir: str):
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
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
# Scoring
# ---------------------------------------------------------------------------
MIN_NOISE_FRACTION = 0.001  # must have *some* noise (pure assignments are suspect)
MAX_NOISE_FRACTION = 0.05   # more than 5% noise → rejected outright
MIN_CLUSTERS = 4            # outside [MIN_CLUSTERS, MAX_CLUSTERS] → rejected outright
MAX_CLUSTERS = 6


def calculate_purity_score(cluster_labels: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Score how well clusters separate cases from controls.

    Hard constraints (score = 0 immediately if violated):
        - noise fraction must be in (MIN_NOISE_FRACTION, MAX_NOISE_FRACTION).
        - cluster count must be in [MIN_CLUSTERS, MAX_CLUSTERS].

    Purity per cluster = |case_rate - 0.5| * 2
        → 1.0 means entirely case or entirely control.
        → 0.0 means a perfect 50/50 mix.

    Final score = avg_purity * (1 - noise_fraction)^2.
    """
    noise_ratio = float((cluster_labels == -1).mean())

    if noise_ratio <= MIN_NOISE_FRACTION or noise_ratio >= MAX_NOISE_FRACTION:
        return 0.0

    mask = cluster_labels != -1
    unique_clusters = np.unique(cluster_labels[mask])

    if not (MIN_CLUSTERS <= len(unique_clusters) <= MAX_CLUSTERS):
        return 0.0

    purity_per_cluster = []
    for cid in unique_clusters:
        idx = np.where(cluster_labels == cid)[0]
        case_rate = binary_labels[idx].mean()
        purity_per_cluster.append(abs(case_rate - 0.5) * 2)

    avg_purity = float(np.mean(purity_per_cluster))
    noise_penalty = (1.0 - noise_ratio) ** 2

    return avg_purity * noise_penalty


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def save_trial_umap(
    z_2d: np.ndarray,
    cluster_labels: np.ndarray,
    params: dict,
    output_dir: str,
) -> None:
    """Save a 2D UMAP scatter for a valid trial with params annotated on the side."""
    fig, ax = plt.subplots(figsize=(9, 6))

    unique = sorted(set(cluster_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique))
    for i, cid in enumerate(unique):
        mask = cluster_labels == cid
        label = f"Noise ({mask.sum()})" if cid == -1 else f"Cluster {cid} ({mask.sum()})"
        color = "lightgrey" if cid == -1 else cmap(i)
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=8, alpha=0.5, color=color, label=label)

    ax.set_title(f"Trial {int(params['trial'])}  —  score={params['score']:.4f}", weight="bold")
    ax.axis("off")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    param_text = (
        f"n_neighbors : {int(params['n_neighbors'])}\n"
        f"min_cluster_size : {int(params['min_cluster_size'])}\n"
        f"min_samples : {int(params['min_samples'])}\n"
        f"method : {params['cluster_selection_method']}\n"
        f"n_clusters : {int(params['n_clusters'])}\n"
        f"noise : {params['noise_pct']:.2f}%"
    )
    fig.text(
        0.78, 0.35, param_text,
        fontsize=8, family="monospace",
        va="top", ha="left",
        bbox=dict(facecolor="lightyellow", edgecolor="grey", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()
    fname = f"trial_{int(params['trial']):03d}_score{params['score']:.4f}.png"
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
        n_neighbors              : int in [10, 50]  (for the 10D UMAP reduction)
        min_cluster_size         : int in [20, 80]
        min_samples              : int in [5, 30]
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
        n_neighbors = rng.randint(10, 50)
        min_cluster_size = rng.randint(20, 80)
        min_samples = rng.randint(5, 30)
        method = "eom"

        if n_neighbors not in umap_cache:
            print(f"  [Trial {trial}/{n_trials}] UMAP 10D (n_neighbors={n_neighbors})...")
            umap_cache[n_neighbors] = reduce_umap(
                embeddings, n_components=10, n_neighbors=n_neighbors
            )

        emb_10d = umap_cache[n_neighbors]

        cluster_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
            prediction_data=False,
        ).fit_predict(emb_10d)

        score = calculate_purity_score(cluster_labels, binary_labels)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_pct = float((cluster_labels == -1).mean() * 100)

        row = {
            "trial": trial,
            "n_neighbors": n_neighbors,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "score": round(score, 4),
            "n_clusters": n_clusters,
            "noise_pct": round(noise_pct, 2),
        }
        results.append(row)

        status = "✓ VALID" if score > 0 else "✗"
        print(
            f"  [Trial {trial}/{n_trials}] {status} "
            f"nn={n_neighbors}, mcs={min_cluster_size}, ms={min_samples} "
            f"→ score={score:.3f}, clusters={n_clusters}, noise={noise_pct:.1f}%"
        )

        if score > 0:
            save_trial_umap(z_2d, cluster_labels, row, plots_dir)

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading embeddings...")
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

    best = df_search.iloc[0].to_dict()
    best_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest parameters saved to {best_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
