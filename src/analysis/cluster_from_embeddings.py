"""
Cluster analysis starting from saved embeddings (no GPU required).

Pipeline:
    - Load embeddings.npy and patient_ids.npy from a clustering run.
    - Run UMAP on original embedding space for visualization only.
    - Run k-means on the original embedding space.
    - Visualize clusters in 2D.
    - Analyse events and measurements per cluster, including continuous lab values.
"""

import os
import yaml
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from src.data.dataset_factory import get_dataset


# --- CONFIG: update these paths for your run ---
CONFIG_PATH = "src/configs/classification_config.yaml"   # same config you used for training
OUTPUT_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"  # directory containing embeddings.npy
SPLIT = "held_out"  # "train", "tuning", or "held_out"
K = 5               # number of clusters for k-means


def load_embeddings(output_dir: str):
    emb_path = os.path.join(output_dir, "embeddings.npy")
    embeddings = np.load(emb_path)

    patient_ids_path = os.path.join(output_dir, "patient_ids.npy")
    if os.path.exists(patient_ids_path):
        patient_ids = np.load(patient_ids_path, allow_pickle=True).tolist()
    else:
        patient_ids = list(range(embeddings.shape[0]))

    return embeddings, patient_ids


def run_umap(embeddings: np.ndarray):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        # random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_kmeans(embeddings: np.ndarray, k: int):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    print(f"Silhouette score (original space, k={k}): {sil:.3f}")
    return labels


def load_dataset(config_path: str, split: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    dataset_name = data_config.get("dataset", "CPRD")

    dataset = get_dataset(dataset_name, data_config, split)

    # Map subject_id -> label_id and label_id -> human-readable label
    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    label_to_name = {v: k for k, v in dataset.label_to_id.items()}

    # Map subject_id -> record index (string keys to be consistent)
    subject_to_record_idx = {}
    for idx, record in enumerate(dataset.patient_records):
        sid = record.get("subject_id")
        if sid is not None:
            subject_to_record_idx[str(sid)] = idx

    return dataset, subject_to_label, label_to_name, subject_to_record_idx


def visualize_clusters(df_clusters: pd.DataFrame, output_dir: str):
    sns.set(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: UMAP colored by cluster
    sns.scatterplot(
        data=df_clusters,
        x="dim1",
        y="dim2",
        hue="cluster",
        palette="tab10",
        s=10,
        alpha=0.7,
        ax=axes[0],
    )
    axes[0].set_title("UMAP: colored by cluster")
    axes[0].set_xlabel("dim1")
    axes[0].set_ylabel("dim2")
    axes[0].legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Right: UMAP colored by case/control (if available)
    if "case_label" in df_clusters.columns:
        sns.scatterplot(
            data=df_clusters,
            x="dim1",
            y="dim2",
            hue="case_label",
            palette={"Control": "lightgrey", "Case": "red"},
            s=10,
            alpha=0.7,
            ax=axes[1],
        )
        axes[1].set_title("UMAP: cases (red) vs controls (grey)")
        axes[1].legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        axes[1].axis("off")

    axes[1].set_xlabel("dim1")
    axes[1].set_ylabel("dim2")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "umap_clusters_from_embeddings.png"), dpi=300)
    plt.close()


def analyse_events_and_measurements(
    dataset,
    df_clusters: pd.DataFrame,
    subject_to_record_idx: dict,
    output_dir: str,
):
    """
    Analyse which events and measurements are more frequent / have different values per cluster.

    For measurement-like events (LAB//, MEASUREMENT//, MEDICAL//BMI, MEDICAL//bp_),
    we:
        - Count how often the test is done.
        - Parse the following numeric token as the value when possible (e.g. '91.5').
        - Compute simple summary stats (mean, median, quartiles) across clusters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # cluster -> event_key -> count
    cluster_event_counts = defaultdict(lambda: defaultdict(int))
    # cluster -> event_key -> list of numeric values
    cluster_event_values = defaultdict(lambda: defaultdict(list))

    for _, row in df_clusters.iterrows():
        pid = row["patient_id"]
        cluster = int(row["cluster"])
        pid_key = str(pid)

        rec_idx = subject_to_record_idx.get(pid_key)
        if rec_idx is None:
            continue

        record = dataset.patient_records[rec_idx]
        token_ids = get_filtered_token_ids(dataset, record)
        string_codes = [str(dataset.id_to_token.get(tid, "")) for tid in token_ids]

        i = 0
        while i < len(string_codes):
            code = string_codes[i]

            is_measurement = (
                code.startswith("LAB//")
                or code.startswith("MEASUREMENT//")
                or code.startswith("MEDICAL//BMI")
                or code.startswith("MEDICAL//bp_")
            )

            event_key = None
            if is_measurement:
                event_key = code  # keep raw code; you can later map to human-readable

            # Generic medical events where frequency alone may be interesting
            elif code.startswith("MEDICAL//"):
                event_key = code

            if event_key is not None:
                cluster_event_counts[cluster][event_key] += 1

                # Try to parse the next token as a numeric value (continuous lab)
                if is_measurement and i + 1 < len(string_codes):
                    next_code = string_codes[i + 1]
                    try:
                        value = float(next_code)
                        cluster_event_values[cluster][event_key].append(value)
                    except ValueError:
                        # Non-numeric (e.g. a bin like 'low/normal/high'); skip for numeric stats
                        pass

            i += 1

    # Build per-cluster CSVs and a global summary
    summary_rows = []
    for cluster_id, event_counts in cluster_event_counts.items():
        rows = []
        for event_key, count in event_counts.items():
            values = cluster_event_values[cluster_id].get(event_key, [])
            n_values = len(values)

            if n_values > 0:
                arr = np.array(values, dtype=float)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                median = float(np.median(arr))
                q25 = float(np.percentile(arr, 25))
                q75 = float(np.percentile(arr, 75))
            else:
                mean = std = median = q25 = q75 = np.nan

            # Optionally, translate event_key to something human readable
            human_readable = event_key
            try:
                human_readable = dataset.translator.translate(event_key).rstrip("; ")
            except Exception:
                pass

            row = {
                "cluster": cluster_id,
                "event_code": event_key,
                "event_human": human_readable,
                "count": count,
                "n_values": n_values,
                "mean_value": mean,
                "std_value": std,
                "median_value": median,
                "q25_value": q25,
                "q75_value": q75,
            }
            rows.append(row)
            summary_rows.append(row)

        df_cluster = pd.DataFrame(rows)
        df_cluster.sort_values("count", ascending=False, inplace=True)
        df_cluster.to_csv(
            os.path.join(output_dir, f"cluster_{cluster_id}_events_measurements.csv"),
            index=False,
        )

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            os.path.join(output_dir, "cluster_events_measurements_summary.csv"),
            index=False,
        )

def calculate_cluster_composition(df_clusters: pd.DataFrame, output_dir: str):
    """Calculates the ratio of Cases vs Controls per cluster."""
    composition = df_clusters.groupby('cluster')['case_label'].value_counts(normalize=False).unstack(fill_value=0)
    composition['Total'] = composition.sum(axis=1)
    composition['Case_%'] = (composition.get('Case', 0) / composition['Total'] * 100).round(2)
    
    print("\n--- Cluster Composition ---")
    print(composition)
    
    composition.to_csv(os.path.join(output_dir, "cluster_case_control_composition.csv"))
    return composition

def analyse_differential_features(
    dataset,
    df_clusters: pd.DataFrame,
    subject_to_record_idx: dict,
    output_dir: str,
):
    """
    Analyzes which events/measurements are unique or statistically different 
    in each cluster compared to the rest of the population.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect all data into a flat structure for global comparison
    all_data = []
    
    for _, row in df_clusters.iterrows():
        pid = str(row["patient_id"])
        cluster = int(row["cluster"])
        rec_idx = subject_to_record_idx.get(pid)
        if rec_idx is None: continue

        record = dataset.patient_records[rec_idx]
        token_ids = get_filtered_token_ids(dataset, record)
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in token_ids]

        for i, code in enumerate(tokens):
            is_measurement = any(code.startswith(prefix) for prefix in ["LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"])
            is_medical = code.startswith("MEDICAL//")

            if is_measurement or is_medical:
                val = np.nan
                if is_measurement and i + 1 < len(tokens):
                    try: val = float(tokens[i+1])
                    except ValueError: pass
                
                all_data.append({
                    'cluster': cluster,
                    'event_code': code,
                    'value': val
                })

    df_raw = pd.DataFrame(all_data)
    
    # 2. Compute Global Stats (for comparison)
    global_counts = df_raw['event_code'].value_counts()
    global_medians = df_raw.groupby('event_code')['value'].median()
    total_patients = df_clusters['patient_id'].nunique()

    # 3. Compute Per-Cluster Stats and compare to Global
    summary_rows = []
    
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        cluster_df = df_raw[df_raw['cluster'] == cluster_id]
        cluster_size = df_clusters[df_clusters['cluster'] == cluster_id]['patient_id'].nunique()
        
        counts = cluster_df['event_code'].value_counts()
        
        for event_code, count in counts.items():
            vals = cluster_df[cluster_df['event_code'] == event_code]['value'].dropna()
            
            # Frequency Analysis
            # How much more common is this event here than in the general population?
            global_freq = global_counts[event_code] / total_patients
            cluster_freq = count / cluster_size
            enrichment_factor = cluster_freq / global_freq if global_freq > 0 else 0

            # Value Analysis
            mean_v = vals.mean() if not vals.empty else np.nan
            median_v = vals.median() if not vals.empty else np.nan
            
            # Delta from global median (instead of mean)
            diff_from_global_median = median_v - global_medians[event_code] if not np.isnan(median_v) else 0

            try:
                human = dataset.translator.translate(event_code).rstrip("; ")
            except Exception:
                human = event_code

            summary_rows.append({
                "cluster": cluster_id,
                "event_code": event_code,
                "event_human": human,
                "count": count,
                "enrichment_score": round(enrichment_factor, 3), # >1 means more common in this cluster
                "mean_value": mean_v,
                "median_value": median_v,
                "diff_from_global_median": diff_from_global_median,
                "q25_value": vals.quantile(0.25) if not vals.empty else np.nan,
                "q75_value": vals.quantile(0.75) if not vals.empty else np.nan,
            })

    df_final = pd.DataFrame(summary_rows)
    
    # Sort by cluster and then by enrichment to see "Top characteristic features"
    df_final = df_final.sort_values(['cluster', 'enrichment_score'], ascending=[True, False])
    
    # Save the full summary
    df_final.to_csv(os.path.join(output_dir, "differential_cluster_analysis.csv"), index=False)
    
    # Save individual files for each cluster for easy reading
    for cid in df_clusters['cluster'].unique():
        cluster_subset = df_final[df_final['cluster'] == cid]
        cluster_subset.to_csv(os.path.join(output_dir, f"cluster_{cid}_analysis.csv"), index=False)

    print(f"Differential analysis complete. Files saved to {output_dir}")

def get_filtered_token_ids(dataset, record: dict) -> list:
    """Return token IDs for a record after applying time cutoff and removing cancer tokens."""
    subject_id = record.get("subject_id")
    token_ids = list(record.get("tokens", []))
    timestamps = record.get("timestamps", [])

    if subject_id is None:
        return token_ids

    label = dataset.subject_to_label.get(subject_id)
    token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, label)

    return [
        tid for tid in token_ids
        if "cancer" not in str(dataset.id_to_token.get(tid, "")).lower()
    ]


def run_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 15, min_samples: int = 5):
    """
    Runs HDBSCAN on the high-dimensional embedding space.
    
    Args:
        min_cluster_size: The smallest size a group can be to be considered a cluster.
        min_samples: How 'conservative' the clustering is. Higher = more points labeled as noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean', # or 'cosine' if your embeddings are normalized
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # Calculate some stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points out of {len(labels)} patients.")
    
    return labels, clusterer

# def main():
#     # 1) Load embeddings and run clustering
#     embeddings, patient_ids = load_embeddings(OUTPUT_DIR)
#     Z = run_umap(embeddings)
#     cluster_labels = run_kmeans(embeddings, K)

#     # 2) Build cluster DataFrame
#     df_clusters = pd.DataFrame(
#         {
#             "patient_id": patient_ids,
#             "cluster": cluster_labels,
#             "dim1": Z[:, 0],
#             "dim2": Z[:, 1],
#         }
#     )

#     # 3) Load dataset to attach labels and access raw tokens
#     dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(
#         CONFIG_PATH, SPLIT
#     )

#     # Map labels
#     df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
#     df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
#     df_clusters["label_name"] = df_clusters["label_id"].map(label_to_name)

#     # Binary case vs control
#     df_clusters["is_case"] = df_clusters["label_id"].fillna(0).astype(int) > 0
#     df_clusters["case_label"] = np.where(df_clusters["is_case"], "Case", "Control")

#     # Save clusters table
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     df_clusters.to_csv(
#         os.path.join(OUTPUT_DIR, "patient_clusters_from_embeddings.csv"), index=False
#     )

#     # # 4) Visualise clusters
#     # visualize_clusters(df_clusters, OUTPUT_DIR)

#     # # 5) Analyse events and measurements per cluster
#     # analyse_events_and_measurements(
#     #     dataset=dataset,
#     #     df_clusters=df_clusters,
#     #     subject_to_record_idx=subject_to_record_idx,
#     #     output_dir=OUTPUT_DIR,
#     # )
#     # 4) Calculate Case/Control mix
#     calculate_cluster_composition(df_clusters, OUTPUT_DIR)

#     # 5) Run the new differential analysis
#     analyse_differential_features(
#         dataset=dataset,
#         df_clusters=df_clusters,
#         subject_to_record_idx=subject_to_record_idx,
#         output_dir=OUTPUT_DIR
#     )

#     # 6) Visualise (as before)
#     visualize_clusters(df_clusters, OUTPUT_DIR)

#     print("Cluster analysis from embeddings complete.")

def main():
    # 1) Load dataset FIRST (required for mapping labels later)
    dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(
        CONFIG_PATH, SPLIT
    )

    # 2) Load embeddings
    embeddings, patient_ids = load_embeddings(OUTPUT_DIR)
    
    # 3) Run HDBSCAN
    # cluster_labels contains -1 for outliers (noise)
    cluster_labels, clusterer = run_hdbscan(embeddings, min_cluster_size=20)
    
    # 4) Build cluster DataFrame
    Z = run_umap(embeddings) # Dimensionality reduction for viz
    df_clusters = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster": cluster_labels,
        "dim1": Z[:, 0],
        "dim2": Z[:, 1],
    })

    # 5) Map labels (Now variables like subject_to_label exist)
    df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
    df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
    df_clusters["label_name"] = df_clusters["label_id"].map(label_to_name)

    # Binary case vs control logic
    df_clusters["is_case"] = df_clusters["label_id"].fillna(0).astype(int) > 0
    df_clusters["case_label"] = np.where(df_clusters["is_case"], "Case", "Control")

    # 6) Save clusters table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_clusters.to_csv(
        os.path.join(OUTPUT_DIR, "patient_clusters_from_embeddings.csv"), index=False
    )

    # 7) Analysis & Visualization
    calculate_cluster_composition(df_clusters, OUTPUT_DIR)

    analyse_differential_features(
        dataset=dataset,
        df_clusters=df_clusters,
        subject_to_record_idx=subject_to_record_idx,
        output_dir=OUTPUT_DIR
    )

    visualize_clusters(df_clusters, OUTPUT_DIR)

    print("Cluster analysis from embeddings complete.")

if __name__ == "__main__":
    main()
