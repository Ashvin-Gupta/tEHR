"""
Patient clustering analysis using LLM + LoRA classifier representations.

This script:
    - Loads a trained classification model (base LLM + LoRA + head).
    - Extracts the patient-level representation that feeds into the classifier head.
    - Runs dimensionality reduction (UMAP by default).
    - Runs k-means clustering on the reduced (or original) space.
    - Links clusters back to raw EHR events for simple enrichment summaries.
"""

import argparse
import os
import json
import math
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipelines.evaluate_model import ModelEvaluator
from src.baselines.utils import load_baseline_config, load_datasets
from src.data.classification_collator import ClassificationCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster patients based on LLM + LoRA classifier representations."
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file (same as used for training).",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help=(
            "Path to saved model checkpoint directory "
            "(e.g., training output_dir/final_model)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="held_out",
        choices=["train", "tuning", "held_out"],
        help="Dataset split to use for clustering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write embeddings, projections, clusters, and summaries.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional maximum number of patients to process (for quick tests).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'cuda:0', or 'cpu'). "
             "Defaults to 'cuda' if available, else 'cpu'.",
    )
    # UMAP / dimensionality reduction
    parser.add_argument(
        "--dim_reducer",
        type=str,
        default="umap",
        choices=["umap", "tsne", "none"],
        help="Dimensionality reduction method before clustering.",
    )
    parser.add_argument(
        "--umap_n_components",
        type=int,
        default=2,
        help="UMAP number of components.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--umap_metric",
        type=str,
        default="euclidean",
        help="UMAP distance metric.",
    )
    # Clustering
    parser.add_argument(
        "--cluster_space",
        type=str,
        default="umap",
        choices=["umap", "original"],
        help="Space to run k-means on ('umap' projection or original embeddings).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters for k-means.",
    )
    parser.add_argument(
        "--k_range",
        type=str,
        default=None,
        help="Optional comma-separated list of k values (e.g. '2,3,4,5') "
             "to compute inertia/silhouette and help choose k.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate simple 2D UMAP scatter plots colored by cluster.",
    )

    return parser.parse_args()


class ClusterCollator:
    """
    Wraps ClassificationCollator to also return patient IDs.

    - Uses the existing collator to build input_ids/attention_mask/labels.
    - Additionally collects 'subject_id' from each item (if present) and
      returns it as a list in the batch dict under 'patient_ids'.
    """

    def __init__(self, base_collator: ClassificationCollator):
        self.base_collator = base_collator

    def __call__(self, batch):
        patient_ids: List = []
        for idx, item in enumerate(batch):
            pid = item.get("subject_id", idx)
            patient_ids.append(pid)

        collated = self.base_collator(batch)
        collated["patient_ids"] = patient_ids
        return collated


def resolve_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    config_path: str,
    model_checkpoint: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, any, dict, dict]:
    """
    Reuse the evaluation pipeline's ModelEvaluator to load the trained model.

    Returns:
        model: LLMClassifier instance with base LLM + LoRA + classifier head.
        tokenizer: Associated tokenizer.
        data_config: Data section of the config.
        training_config: Training section of the config.
    """
    config = load_baseline_config(config_path)
    evaluator = ModelEvaluator(config, model_checkpoint)
    evaluator.load_model()
    evaluator.wrap_with_classifier()

    model = evaluator.model.to(device)
    model.eval()

    return model, evaluator.tokenizer, evaluator.data_config, evaluator.training_config


def build_dataloader(
    data_config: dict,
    tokenizer,
    split: str,
    batch_size: int,
) -> Tuple[DataLoader, any]:
    """
    Build a DataLoader over the UnifiedEHRDataset for the given split.

    Returns:
        dataloader, dataset
    """
    datasets = load_datasets(
        data_config=data_config,
        splits=[split],
        format="text",
        tokenizer=tokenizer,
    )
    dataset = datasets[split]

    collator = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 12000),
    )
    cluster_collator = ClusterCollator(collator)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("dataloader_num_workers", 4),
        collate_fn=cluster_collator,
    )

    return dataloader, dataset


def extract_patient_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_patients: int | None = None,
) -> Tuple[np.ndarray, List]:
    """
    Run the model over the dataset and collect patient-level representations.

    Returns:
        embeddings: (N, hidden_dim) numpy array.
        patient_ids: list of patient identifiers aligned with embeddings.
    """
    all_embeddings: List[np.ndarray] = []
    all_patient_ids: List = []

    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            patient_ids = batch["patient_ids"]

            reps = model.get_patient_representation(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Ensure tensor is in a NumPy-compatible dtype (e.g. float32 instead of bfloat16).
            reps_np = reps.detach().to(torch.float32).cpu().numpy()

            if max_patients is not None:
                remaining = max_patients - processed
                if remaining <= 0:
                    break
                reps_np = reps_np[:remaining]
                patient_ids = patient_ids[:remaining]

            all_embeddings.append(reps_np)
            all_patient_ids.extend(patient_ids)
            processed += len(patient_ids)

            if max_patients is not None and processed >= max_patients:
                break

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted. Check dataset and filters.")

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_patient_ids


def run_dimensionality_reduction(
    embeddings: np.ndarray,
    args: argparse.Namespace,
    output_dir: str,
) -> np.ndarray | None:
    """
    Apply dimensionality reduction to embeddings.

    Returns:
        Z: reduced embeddings (N, n_components) or None if dim_reducer='none'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if args.dim_reducer == "none":
        return None

    if args.dim_reducer == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError(
                "UMAP is not installed. Install via 'pip install umap-learn'."
            ) from e

        reducer = umap.UMAP(
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=42,
        )
        Z = reducer.fit_transform(embeddings)
        np.save(os.path.join(output_dir, "umap_embeddings.npy"), Z)
        return Z

    if args.dim_reducer == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for t-SNE. "
                "Install via 'pip install scikit-learn'."
            ) from e

        tsne = TSNE(
            n_components=args.umap_n_components,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        Z = tsne.fit_transform(embeddings)
        np.save(os.path.join(output_dir, "tsne_embeddings.npy"), Z)
        return Z

    raise ValueError(f"Unknown dim_reducer: {args.dim_reducer}")


def run_kmeans_clustering(
    embeddings: np.ndarray,
    Z: np.ndarray | None,
    patient_ids: List,
    args: argparse.Namespace,
    output_dir: str,
) -> np.ndarray:
    """
    Run k-means clustering on either the reduced or original space.

    Returns:
        cluster_labels: numpy array of shape (N,) with cluster assignments.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for k-means clustering. "
            "Install via 'pip install scikit-learn'."
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    if args.cluster_space == "umap":
        if Z is None:
            raise ValueError(
                "cluster_space='umap' requires a non-None reduced embedding Z."
            )
        X = Z
    else:
        X = embeddings

    n_samples = X.shape[0]

    metrics = {}
    if args.k_range is not None:
        ks = [int(k.strip()) for k in args.k_range.split(",") if k.strip()]
        inertia_vals = {}
        silhouette_vals = {}
        for k in ks:
            if k <= 1 or k >= n_samples:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels_k = km.fit_predict(X)
            inertia_vals[k] = float(km.inertia_)
            try:
                sil = silhouette_score(X, labels_k)
            except Exception:
                sil = float("nan")
            silhouette_vals[k] = sil

        metrics["inertia"] = inertia_vals
        metrics["silhouette"] = silhouette_vals

    k = args.k
    if k <= 1 or k > n_samples:
        raise ValueError(f"Invalid k={k} for n_samples={n_samples}")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)

    metrics["chosen_k"] = k
    metrics["cluster_sizes"] = {
        int(c): int((cluster_labels == c).sum()) for c in range(k)
    }

    with open(os.path.join(output_dir, "clustering_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save patient-level assignments
    import pandas as pd

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "cluster": cluster_labels,
        }
    )
    if Z is not None and Z.shape[1] >= 2:
        df["dim1"] = Z[:, 0]
        df["dim2"] = Z[:, 1]
    if Z is not None and Z.shape[1] >= 3:
        df["dim3"] = Z[:, 2]

    df.to_csv(os.path.join(output_dir, "patient_clusters.csv"), index=False)

    return cluster_labels


def summarize_clusters_with_ehr(
    dataset,
    patient_ids: List,
    cluster_labels: np.ndarray,
    output_dir: str,
    top_k_events: int = 30,
) -> None:
    """
    Simple enrichment: for each cluster, count frequent medical/lab events.

    Uses the dataset's underlying patient_records and lookup tables to map
    raw token codes to human-readable terms where possible.
    In addition to within-cluster frequency, also computes how specific an
    event is to a cluster compared to all other clusters.
    """
    from collections import Counter, defaultdict
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    subject_to_record_idx = getattr(dataset, "subject_to_record_idx", None)
    if subject_to_record_idx is None:
        subject_to_record_idx = {}
        for idx, record in enumerate(dataset.patient_records):
            sid = record.get("subject_id")
            if sid is not None and sid not in subject_to_record_idx:
                subject_to_record_idx[sid] = idx

    cluster_event_counts = defaultdict(Counter)

    for pid, cluster in zip(patient_ids, cluster_labels):
        rec_idx = subject_to_record_idx.get(pid)
        if rec_idx is None:
            continue
        record = dataset.patient_records[rec_idx]
        token_ids = record.get("tokens", [])

        for tid in token_ids:
            token_str = str(dataset.id_to_token_map.get(tid, ""))
            if not token_str:
                continue

            event_key = None

            if token_str.startswith("MEDICAL//"):
                code = token_str.split("//", 1)[1].upper()
                term = dataset.medical_lookup.get(code, code) if hasattr(dataset, "medical_lookup") else code
                event_key = f"MEDICAL::{term}"
            elif token_str.startswith("LAB//"):
                code = token_str.split("//", 1)[1].upper()
                term = dataset.lab_lookup.get(code, code) if hasattr(dataset, "lab_lookup") else code
                event_key = f"LAB::{term}"
            elif token_str.startswith("MEASUREMENT//"):
                code = token_str.split("//", 1)[1].upper()
                term = dataset.medical_lookup.get(code, code) if hasattr(dataset, "medical_lookup") else code
                event_key = f"MEASUREMENT::{term}"

            if event_key is not None:
                cluster_event_counts[int(cluster)][event_key] += 1

    # Write per-cluster CSVs and overall summary, including cross-cluster enrichment stats
    summary_rows = []

    # Global event statistics across all clusters
    global_event_counts = Counter()
    cluster_total_events: dict[int, int] = {}
    for cluster_id, counter in cluster_event_counts.items():
        total_events = sum(counter.values())
        cluster_total_events[cluster_id] = total_events
        global_event_counts.update(counter)

    total_events_all = sum(global_event_counts.values())

    for cluster_id, counter in cluster_event_counts.items():
        total_events = cluster_total_events[cluster_id]
        most_common = counter.most_common(top_k_events)

        rows = []
        for event, count in most_common:
            frac = count / total_events if total_events > 0 else 0.0

            # Global stats for this event across all clusters
            global_count = global_event_counts[event]
            global_frac = global_count / total_events_all if total_events_all > 0 else 0.0

            # Enrichment of this event in this cluster vs all other clusters
            count_in_cluster = count
            events_in_cluster_total = total_events
            count_outside = global_count - count_in_cluster
            events_outside_total = total_events_all - events_in_cluster_total

            if (
                events_in_cluster_total > 0
                and events_outside_total > 0
                and count_in_cluster > 0
                and count_outside > 0
            ):
                freq_in = count_in_cluster / events_in_cluster_total
                freq_out = count_outside / events_outside_total
                enrichment = freq_in / freq_out if freq_out > 0 else float("inf")
                log2_enrichment = (
                    math.log2(enrichment) if enrichment > 0 and np.isfinite(enrichment) else float("nan")
                )
            else:
                enrichment = float("nan")
                log2_enrichment = float("nan")

            rows.append(
                {
                    "cluster": cluster_id,
                    "event": event,
                    "count": count,
                    "fraction": frac,
                    "total_events_in_cluster": total_events,
                    "global_count": global_count,
                    "global_fraction": global_frac,
                    # Of all occurrences of this event, what fraction are in this cluster?
                    "cluster_fraction_of_event": (
                        count_in_cluster / global_count if global_count > 0 else 0.0
                    ),
                    # How much more frequent (per event) is this event in this cluster
                    # vs all other clusters combined?
                    "enrichment_ratio": enrichment,
                    "log2_enrichment": log2_enrichment,
                }
            )
            summary_rows.append(rows[-1])

        df_cluster = pd.DataFrame(rows)
        df_cluster.to_csv(
            os.path.join(output_dir, f"cluster_{cluster_id}_events.csv"),
            index=False,
        )

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            os.path.join(output_dir, "cluster_event_enrichment.csv"),
            index=False,
        )


def maybe_plot(
    Z: np.ndarray | None,
    cluster_labels: np.ndarray,
    output_dir: str,
) -> None:
    """
    Optional 2D scatter plot of reduced embeddings colored by cluster.
    """
    if Z is None or Z.shape[1] < 2:
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=cluster_labels, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("Patient Clusters (Reduced Embedding Space)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model, tokenizer, data_config, _ = load_model_and_tokenizer(
        config_path=args.config_filepath,
        model_checkpoint=args.model_checkpoint,
        device=device,
    )

    print(f"Building dataloader for split='{args.split}'...")
    dataloader, dataset = build_dataloader(
        data_config=data_config,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
    )

    print("Extracting patient embeddings...")
    embeddings, patient_ids = extract_patient_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        max_patients=args.max_patients,
    )

    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(args.output_dir, "patient_ids.npy"), np.array(patient_ids, dtype=object))

    print("Running dimensionality reduction...")
    Z = run_dimensionality_reduction(
        embeddings=embeddings,
        args=args,
        output_dir=args.output_dir,
    )

    print("Running k-means clustering...")
    cluster_labels = run_kmeans_clustering(
        embeddings=embeddings,
        Z=Z,
        patient_ids=patient_ids,
        args=args,
        output_dir=args.output_dir,
    )

    print("Summarizing clusters with EHR events...")
    summarize_clusters_with_ehr(
        dataset=dataset,
        patient_ids=patient_ids,
        cluster_labels=cluster_labels,
        output_dir=args.output_dir,
    )

    if args.plot:
        print("Generating cluster plot...")
        maybe_plot(Z, cluster_labels, output_dir=args.output_dir)

    print("Clustering analysis complete.")


if __name__ == "__main__":
    main()

