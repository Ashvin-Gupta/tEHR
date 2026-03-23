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
    "lora-12-month-logistic-refactored-new-time-mistral-7B/"
    "patient_clustering_heldout"
)
OUTPUT_DIR = EMBEDDINGS_DIR + "/hdbscan_search"
N_TRIALS = 100
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
MAX_NOISE_FRACTION = 0.15   # configs with more noise than this are rejected outright


def calculate_validity_score(clusterer, cluster_labels: np.ndarray) -> float:
    """
    Score clusters based on intrinsic density and separation (DBCV),
    while still penalizing massive noise or weird cluster counts.
    """
    noise_ratio = float((cluster_labels == -1).mean())
    if noise_ratio > 0.30:  # Be a bit more lenient with noise at 12 months
        return 0.0

    mask = cluster_labels != -1
    unique_clusters = np.unique(cluster_labels[mask])

    if len(unique_clusters) < 3: # We want at least a few distinct phenotypes
        return 0.0

    # 1. HDBSCAN's native relative validity score (usually between 0 and 1)
    # Higher means clusters are dense and well-separated.
    try:
        validity = clusterer.relative_validity_
    except ValueError:
        return 0.0 # Catches edge cases where validity can't be computed

    # 2. Apply the same common-sense penalties
    noise_penalty = (1.0 - noise_ratio) ** 2
    cluster_penalty = 1.0 if 5 <= len(unique_clusters) <= 30 else 0.5

    return validity * noise_penalty * cluster_penalty


# ---------------------------------------------------------------------------
# Random hyperparameter search
# ---------------------------------------------------------------------------
def run_random_search(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly sample HDBSCAN hyperparameter combinations and score each one.

    Search space:
        n_neighbors              : int in [10, 100]  (for the 10D UMAP reduction)
        min_cluster_size         : int in [10, 150]
        min_samples              : int in [1, 30]
        cluster_selection_method : 'eom' or 'leaf' | changed to only eom

    10D UMAP reductions are cached by n_neighbors to avoid redundant computation.
    Returns a DataFrame of all trial results sorted by score descending.
    """
    rng = random.Random(seed)
    umap_cache: dict = {}
    results = []

    for trial in range(1, n_trials + 1):
        n_neighbors = rng.randint(10, 100)
        min_cluster_size = rng.randint(10, 150)
        min_samples = rng.randint(1, 30)
        method = rng.choice(["eom"])

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
            gen_min_span_tree=True # Required for relative_validity_
        )
        
        clusterer.fit(emb_10d)
        cluster_labels = clusterer.labels_

        # Use the new unsupervised score
        score = calculate_validity_score(clusterer, cluster_labels)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_pct = float((cluster_labels == -1).mean() * 100)

        results.append({
            "trial": trial,
            "n_neighbors": n_neighbors,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "score": round(score, 4),
            "n_clusters": n_clusters,
            "noise_pct": round(noise_pct, 2),
        })
        print(
            f"  [Trial {trial}/{n_trials}] "
            f"nn={n_neighbors}, mcs={min_cluster_size}, ms={min_samples}, "
            f"method={method} → score={score:.3f}, clusters={n_clusters}, "
            f"noise={noise_pct:.1f}%"
        )

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
    df_search = run_random_search(embeddings, binary_labels, n_trials=N_TRIALS)

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
