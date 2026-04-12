"""
HDBSCAN Clustering & Clinical Phenotyping Pipeline for Medical Journals.

Features:
    - Uses fixed, predefined hyperparameters for reproducibility.
    - Separates Categorical (Enrichment) and Continuous (Median/IQR) clinical markers.
    - Generates publication-ready figures (Annotated UMAP, Z-Score Lab Heatmaps, Composition Bars).
"""

import argparse
import json
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, normalize
import joblib

from src.data.dataset_factory import get_dataset
from src.resources.my_terms import MY_TERMS
from src.resources.lab_terms import LAB_TERMS

# ---------------------------------------------------------------------------
# Configuration & Hyperparameters (Fixed for Reproducibility)
# ---------------------------------------------------------------------------
CONFIG_PATH = "src/configs/classification_config.yaml"
EMBEDDINGS_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
OUTPUT_DIR = EMBEDDINGS_DIR + "/journal_clustering_output"
# Path to the per-patient prediction JSON produced by evaluate_model.py.
# Matches training.output_dir in classification_config.yaml.
EVAL_RESULTS_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B"
PREDICTIONS_JSON = os.path.join(EVAL_RESULTS_DIR, "results", "test_results.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# Fixed HDBSCAN parameters from previous search
N_NEIGHBORS = 22
MIN_CLUSTER_SIZE = 127
MIN_SAMPLES = 12
CLUSTER_SELECTION_METHOD = "eom"

# ---------------------------------------------------------------------------
# Data Loading & Base Functions
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

def load_dataset(config_path: str, split: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    dataset = get_dataset(config["data"].get("dataset", "CPRD"), config["data"], split)
    
    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    label_to_name = {v: k for k, v in dataset.label_to_id.items()}
    subject_to_record_idx = {str(r.get("subject_id")): idx for idx, r in enumerate(dataset.patient_records) if r.get("subject_id")}
    return dataset, subject_to_label, label_to_name, subject_to_record_idx

def get_filtered_token_ids(dataset, record: dict) -> list:
    subject_id = record.get("subject_id")
    token_ids = list(record.get("tokens", []))
    timestamps = record.get("timestamps", [])
    if subject_id is None: return token_ids
    
    label = dataset.subject_to_label.get(subject_id)
    token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, label)
    return [tid for tid in token_ids if "cancer" not in str(dataset.id_to_token.get(tid, "")).lower()]

# ---------------------------------------------------------------------------
# Clustering Pipeline
# ---------------------------------------------------------------------------
def run_fixed_clustering(embeddings: np.ndarray) -> tuple:
    print(f"Running clustering with fixed params: nn={N_NEIGHBORS}, mcs={MIN_CLUSTER_SIZE}, ms={MIN_SAMPLES}")
    
    # 1. 10D Reduction
    reducer_10d = umap.UMAP(n_components=10, n_neighbors=N_NEIGHBORS, min_dist=0.0, random_state=RANDOM_SEED, n_jobs=1)
    emb_10d = reducer_10d.fit_transform(embeddings)

    # 2. HDBSCAN
    cluster_labels = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    ).fit_predict(emb_10d)

    # 3. 2D Reduction for visualization
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED, n_jobs=1)
    z_2d = reducer_2d.fit_transform(embeddings)

    return cluster_labels,z_2d, reducer_10d, reducer_2d

# ---------------------------------------------------------------------------
# Data Analysis
# ---------------------------------------------------------------------------
def analyse_differential_features(dataset, df_clusters: pd.DataFrame, subject_to_record_idx: dict, output_dir: str):
    all_data = []
    
    for _, row in df_clusters.iterrows():
        pid, cluster = str(row["patient_id"]), int(row["cluster"])
        if (rec_idx := subject_to_record_idx.get(pid)) is None: continue

        record = dataset.patient_records[rec_idx]
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in get_filtered_token_ids(dataset, record)]

        for i, code in enumerate(tokens):
            is_measurement = any(code.startswith(p) for p in ["LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"])
            is_medical = code.startswith("MEDICAL//")

            if is_measurement or is_medical:
                val = np.nan
                if is_measurement and i + 1 < len(tokens):
                    try: val = float(tokens[i + 1])
                    except ValueError: pass
                all_data.append({"cluster": cluster, "event_code": code, "value": val})

    df_raw = pd.DataFrame(all_data)
    global_counts = df_raw["event_code"].value_counts()
    global_medians = df_raw.groupby("event_code")["value"].median()
    total_patients = df_clusters["patient_id"].nunique()

    summary_rows = []
    for cluster_id in sorted(df_clusters["cluster"].unique()):
        cluster_df = df_raw[df_raw["cluster"] == cluster_id]
        cluster_size = df_clusters[df_clusters["cluster"] == cluster_id]["patient_id"].nunique()

        for event_code, count in cluster_df["event_code"].value_counts().items():
            vals = cluster_df[cluster_df["event_code"] == event_code]["value"].dropna()
            
            # Translated Human Name
            try: human = dataset.translator.translate(event_code).rstrip("; ")
            except Exception: human = event_code

            summary_rows.append({
                "cluster": cluster_id,
                "event_code": event_code,
                "event_human": human,
                "count": count,
                "enrichment_score": round((count / cluster_size) / (global_counts[event_code] / total_patients), 3),
                "median_value": vals.median() if not vals.empty else np.nan,
                "q25_value": vals.quantile(0.25) if not vals.empty else np.nan,
                "q75_value": vals.quantile(0.75) if not vals.empty else np.nan,
                "global_median_diff": vals.median() - global_medians[event_code] if not vals.empty else np.nan,
            })

    df_final = pd.DataFrame(summary_rows)
    df_final.to_csv(os.path.join(output_dir, "full_differential_analysis.csv"), index=False)
    return df_final

def generate_journal_reports(df_summary: pd.DataFrame, output_dir: str):
    """Separates and formats reports for Categorical vs Continuous terms."""
    
    # 1. Categorical (Diseases/Conditions) - Report Fold Enrichment
    cat_rows = []
    for term in list(dict.fromkeys(MY_TERMS)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if not match.empty:
            match = match.sort_values("count", ascending=False).drop_duplicates(subset=["cluster"])
            cat_rows.append(match)
            
    if cat_rows:
        df_cat = pd.concat(cat_rows).drop_duplicates(subset=["event_human", "cluster"])
        df_cat["display"] = df_cat["enrichment_score"].apply(lambda x: f"x{x:.1f}")

        heatmap_cat = df_cat.pivot(index="event_human", columns="cluster", values="enrichment_score").fillna(0)
        # Drop noise cluster before ordering
        if -1 in heatmap_cat.columns:
            heatmap_cat = heatmap_cat.drop(columns=[-1])
        row_order = cluster_priority_row_order(heatmap_cat)
        heatmap_cat = heatmap_cat.loc[row_order]

        cat_pivot = df_cat.pivot(index="event_human", columns="cluster", values="display").fillna("-").loc[row_order]
        cat_pivot.to_csv(os.path.join(output_dir, "table_categorical_enrichment.csv"))
    else:
        heatmap_cat = None

    # 2. Continuous (Labs) - Report Medians & IQR
    lab_rows = []
    for term in list(dict.fromkeys(LAB_TERMS)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if not match.empty:
            match = match.sort_values("count", ascending=False).drop_duplicates(subset=["cluster"])
            lab_rows.append(match)
            
    if lab_rows:
        df_lab = pd.concat(lab_rows).drop_duplicates(subset=["event_human", "cluster"])
        df_lab["display"] = df_lab.apply(
            lambda r: f"{r['median_value']:.1f} [{r['q25_value']:.1f}-{r['q75_value']:.1f}]"
            if pd.notnull(r['median_value']) else "-", axis=1
        )

        # Raw median pivot for Z-score plotting; order rows by cluster priority
        heatmap_lab = df_lab.pivot(index="event_human", columns="cluster", values="median_value")
        if -1 in heatmap_lab.columns:
            heatmap_lab = heatmap_lab.drop(columns=[-1])
        heatmap_lab = heatmap_lab.dropna(how='all')
        if not heatmap_lab.empty:
            # Use filled version only for ordering, keep NaNs for plotting
            filled = heatmap_lab.fillna(0)
            row_order_lab = cluster_priority_row_order(filled)
            heatmap_lab = heatmap_lab.loc[row_order_lab]

        lab_pivot = df_lab.pivot(index="event_human", columns="cluster", values="display").fillna("-")
        if -1 in lab_pivot.columns:
            lab_pivot = lab_pivot.drop(columns=[-1])
        lab_pivot.to_csv(os.path.join(output_dir, "table_continuous_labs_median_iqr.csv"))
    else:
        heatmap_lab = None

    return heatmap_cat, heatmap_lab

# ---------------------------------------------------------------------------
# Row ordering helper
# ---------------------------------------------------------------------------
def cluster_priority_row_order(heatmap_data: pd.DataFrame) -> list:
    """
    Order rows cluster-by-cluster.

    For each cluster (ascending ID order), find the rows where that cluster
    has the highest enrichment value and they haven't already been assigned
    to a prior cluster.  Within each cluster's group, rows are sorted by
    that cluster's value descending.  Any rows not dominated by any single
    cluster are appended at the end.
    """
    ordered = []
    seen = set()

    for col in sorted(heatmap_data.columns):
        remaining_idx = [r for r in heatmap_data.index if r not in seen]
        if not remaining_idx:
            break
        remaining = heatmap_data.loc[remaining_idx]
        dominated = remaining[remaining.idxmax(axis=1) == col]
        dominated = dominated.sort_values(col, ascending=False)
        ordered.extend(dominated.index.tolist())
        seen.update(dominated.index.tolist())

    # Any rows not dominated by any cluster (all-zero or tied)
    ordered.extend([r for r in heatmap_data.index if r not in seen])
    return ordered


# ---------------------------------------------------------------------------
# Journal-Ready Visualizations
# ---------------------------------------------------------------------------
def plot_annotated_umap(df_clusters: pd.DataFrame, output_dir: str):
    """UMAP with Cluster Centroid labels."""
    sns.set(style="white", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    df_clusters["plot_label"] = df_clusters["case_label"].apply(lambda x: "Control" if str(x).lower() == "control" else "Case")

    # Plot 1: Clusters
    sns.scatterplot(data=df_clusters, x="dim1", y="dim2", hue="cluster", palette="viridis", s=15, alpha=0.5, ax=axes[0], legend=False)
    
    # Annotate Centroids
    for cluster_id in df_clusters['cluster'].unique():
        subset = df_clusters[df_clusters['cluster'] == cluster_id]
        axes[0].text(subset['dim1'].median(), subset['dim2'].median(), str(cluster_id), 
                     fontsize=12, weight='bold', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                     
    axes[0].set_title("A. UMAP Cluster Embeddings", weight='bold')
    axes[0].axis('off')

    # Plot 2: Status
    sns.scatterplot(data=df_clusters, x="dim1", y="dim2", hue="plot_label", palette={"Control": "lightgrey", "Case": "red"}, s=15, alpha=0.6, ax=axes[1])
    axes[1].set_title("B. Case vs Control Distribution", weight='bold')
    axes[1].axis('off')
    axes[1].legend(title="", loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig12_annotated_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_composition(df_clusters: pd.DataFrame, output_dir: str):
    """100% Stacked Bar Chart for Cluster Purity."""
    sns.set(style="whitegrid", context="paper")
    
    df_clusters["plot_label"] = df_clusters["case_label"].apply(lambda x: "Control" if str(x).lower() == "control" else "Case")
    composition = df_clusters.groupby('cluster')['plot_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    ax = composition[['Case', 'Control']].plot(kind='bar', stacked=True, color=['#e74c3c', '#bdc3c7'], figsize=(8, 5))
    plt.title("Cluster Case/Control Composition", weight='bold')
    plt.ylabel("Percentage (%)")
    plt.xlabel("Cluster ID")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    sns.despine(left=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig11_cluster_composition.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lab_zscore_heatmap(heatmap_lab: pd.DataFrame, output_dir: str):
    """Row-standardized heatmap of median lab values."""
    if heatmap_lab is None or heatmap_lab.empty: return
    if -1 in heatmap_lab.columns:
        heatmap_lab = heatmap_lab.drop(columns=[-1])

    heatmap_lab = heatmap_lab.dropna(how='all')
    if heatmap_lab.empty: return

    # Rename columns to "Cluster N" for readability
    heatmap_lab = heatmap_lab.rename(columns={c: f"Cluster {c}" for c in heatmap_lab.columns})

    # Z-Score scale by row (Lab type)
    scaler = StandardScaler()
    z_scored_data = scaler.fit_transform(heatmap_lab.T).T
    z_scored_df = pd.DataFrame(z_scored_data, index=heatmap_lab.index, columns=heatmap_lab.columns)

    plt.figure(figsize=(8, max(6, len(z_scored_df) * 0.3)))
    sns.heatmap(z_scored_df, cmap="coolwarm", center=0, annot=heatmap_lab, fmt=".1f",
                cbar_kws={'label': 'Row Z-Score (Std Dev from Mean)'}, annot_kws={"size": 9})
    plt.title("Lab Value Signatures (Median)", weight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig10_lab_zscore_heatmap.png"), dpi=300)
    plt.close()

def plot_clinical_heatmap(heatmap_cat: pd.DataFrame, output_dir: str):
    """Heatmap of Categorical Enrichment Scores."""
    if heatmap_cat is None or heatmap_cat.empty: return
    if -1 in heatmap_cat.columns:
        heatmap_cat = heatmap_cat.drop(columns=[-1])

    # Filter to only show highly enriched terms to save space in the paper
    top_heatmap = heatmap_cat[heatmap_cat.max(axis=1) >= 1.5]
    if top_heatmap.empty: return

    # Rename columns to "Cluster N" for readability
    top_heatmap = top_heatmap.rename(columns={c: f"Cluster {c}" for c in top_heatmap.columns})

    plt.figure(figsize=(8, max(8, len(top_heatmap) * 0.25)))
    sns.heatmap(top_heatmap, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'Enrichment Score (x-fold)'})
    plt.title("Clinical Phenotypes (Enrichment > 1.5x)", weight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig9_clinical_fingerprint_heatmap.png"), dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Prediction-correctness helpers
# ---------------------------------------------------------------------------
def load_predictions(results_json_path: str) -> pd.DataFrame | None:
    """
    Load per-patient predictions saved by evaluate_model.py.

    Returns a DataFrame with columns:
        patient_id, true_label, pred_label, prob, predicted_correctly
    Returns None (with a warning) if the file is missing or has no patient_ids.
    """
    if not os.path.exists(results_json_path):
        print(f"  [pred plot] Results JSON not found: {results_json_path}. Skipping.")
        return None

    with open(results_json_path) as f:
        data = json.load(f)

    if "patient_ids" not in data:
        print("  [pred plot] No patient_ids in results JSON — re-run evaluate_model.py first.")
        return None

    df = pd.DataFrame({
        "patient_id": [str(p) for p in data["patient_ids"]],
        "true_label": data["labels"],
        "prob": data["probs"],
    })
    df["pred_label"] = (df["prob"] >= 0.5).astype(int)
    df["predicted_correctly"] = (df["pred_label"] == df["true_label"])
    return df


def plot_prediction_correctness_umap(
    df_clusters: pd.DataFrame,
    df_preds: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    UMAP scatter coloured by prediction outcome (TP / TN / FP / FN / Noise).

    df_clusters must have columns: patient_id_str, dim1, dim2, cluster.
    df_preds    must have columns: patient_id, true_label, pred_label, predicted_correctly.
    """
    # CSV reload can make patient_id_str int64; JSON preds use str — align dtypes for merge.
    merged = (
        df_clusters.assign(patient_id_str=df_clusters["patient_id_str"].astype(str))
        .merge(
            df_preds.rename(columns={"patient_id": "patient_id_str"}).assign(
                patient_id_str=lambda d: d["patient_id_str"].astype(str)
            ),
            on="patient_id_str",
            how="left",
        )
    )

    OUTCOME_COLOURS = {
        "TP": "#e74c3c",   # red    — case correctly identified
        "TN": "#3498db",   # blue   — control correctly dismissed
        "FN": "#e67e22",   # orange — case missed by model
        "FP": "#9b59b6",   # purple — control wrongly flagged
        "Noise": "#cccccc",
    }

    def outcome(row):
        if row["cluster"] == -1:
            return "Noise"
        if pd.isna(row.get("true_label")):
            return "Noise"
        tl, pl = int(row["true_label"]), int(row["pred_label"])
        if tl == 1 and pl == 1:
            return "TP"
        if tl == 0 and pl == 0:
            return "TN"
        if tl == 1 and pl == 0:
            return "FN"
        return "FP"

    merged["outcome"] = merged.apply(outcome, axis=1)

    sns.set(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 7))

    order = ["TN", "TP", "FN", "FP", "Noise"]
    for cat in order:
        sub = merged[merged["outcome"] == cat]
        if sub.empty:
            continue
        alpha = 0.25 if cat == "Noise" else 0.65
        size = 6 if cat == "Noise" else 10
        ax.scatter(
            sub["dim1"], sub["dim2"],
            s=size, alpha=alpha,
            color=OUTCOME_COLOURS[cat],
            label=f"{cat} ({len(sub)})",
            rasterized=True,
        )

    ax.set_title("UMAP: Prediction Outcomes", weight="bold")
    ax.axis("off")
    ax.legend(title="Outcome", loc="upper right", fontsize=9, markerscale=1.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_prediction_correctness_umap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved fig8_prediction_correctness_umap.png")


# ---------------------------------------------------------------------------
# Cluster + probability stacked figure
# ---------------------------------------------------------------------------
def _scatter_predicted_probability_umap(
    ax: plt.Axes,
    fig: plt.Figure,
    df_clusters: pd.DataFrame,
    df_preds: pd.DataFrame,
) -> None:
    """Scatter UMAP coloured by predicted P(cancer); grey if no match in df_preds."""
    merged = (
        df_clusters.assign(patient_id_str=df_clusters["patient_id_str"].astype(str))
        .merge(
            df_preds[["patient_id", "prob"]].assign(
                patient_id=lambda d: d["patient_id"].astype(str)
            ),
            left_on="patient_id_str",
            right_on="patient_id",
            how="left",
        )
    )
    no_pred = merged[merged["prob"].isna()]
    has_pred = merged[merged["prob"].notna()].copy()
    if not no_pred.empty:
        ax.scatter(
            no_pred["dim1"], no_pred["dim2"],
            s=6, alpha=0.2, color="lightgrey", rasterized=True,
        )
    sc = ax.scatter(
        has_pred["dim1"], has_pred["dim2"],
        c=has_pred["prob"],
        cmap="RdBu_r",
        vmin=0.0, vmax=1.0,
        s=10, alpha=0.7,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(Cancer)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def plot_predicted_probability_umap_case_control_markers(
    df_clusters: pd.DataFrame,
    df_preds: pd.DataFrame | None,
    output_dir: str,
) -> None:
    """
    Single UMAP: point colour = predicted P(cancer); marker shape encodes true case vs control
    (○ control, × case). HDBSCAN cluster IDs at spatial medians (noise cluster omitted). No second panel.
    """
    if df_preds is None:
        print("  [prob + marker shape] No predictions JSON — skipping.")
        return

    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    df = df_clusters.copy()
    df["_is_case"] = df["case_label"].apply(lambda x: str(x).lower() != "control")
    merged = (
        df.assign(patient_id_str=df["patient_id_str"].astype(str))
        .merge(
            df_preds[["patient_id", "prob"]].assign(
                patient_id=lambda d: d["patient_id"].astype(str)
            ),
            left_on="patient_id_str",
            right_on="patient_id",
            how="left",
        )
    )
    no_pred = merged[merged["prob"].isna()]
    has_pred = merged[merged["prob"].notna()]

    fig, ax = plt.subplots(figsize=(9, 8))
    if not no_pred.empty:
        ax.scatter(
            no_pred["dim1"], no_pred["dim2"],
            s=6, alpha=0.2, color="lightgrey", rasterized=True,
        )

    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    controls = has_pred[~has_pred["_is_case"]]
    cases = has_pred[has_pred["_is_case"]]

    if not controls.empty:
        ax.scatter(
            controls["dim1"], controls["dim2"],
            c=controls["prob"],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=14,
            alpha=0.78,
            edgecolors="face",
            linewidths=0.35,
            rasterized=True,
        )
    if not cases.empty:
        ax.scatter(
            cases["dim1"], cases["dim2"],
            c=cases["prob"],
            cmap=cmap,
            norm=norm,
            marker="x",
            s=45,
            alpha=0.88,
            linewidths=1.0,
            rasterized=True,
        )

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        subset = df[df["cluster"] == cluster_id]
        ax.text(
            float(subset["dim1"].median()),
            float(subset["dim2"].median()),
            str(cluster_id),
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(Cancer)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Predicted probability (○ = control, × = case; cluster ID at centroid)",
        weight="bold",
        fontsize=16,
        pad=6,
    )
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="grey", linestyle="None",
                markersize=7, label="Control",
            ),
            Line2D(
                [0], [0], marker="x", color="grey", linestyle="None",
                markersize=9, markeredgewidth=1.1, label="Case",
            ),
        ],
        loc="upper right",
        fontsize=10,
        title="True label",
        title_fontsize=9,
    )

    fig.subplots_adjust(left=0.03, right=0.92, top=0.94, bottom=0.03)

    out_name = "fig14_predicted_probability_umap_case_control_markers.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_name}")


def plot_case_control_cluster_annotated_and_probability_umap(
    df_clusters: pd.DataFrame,
    df_preds: pd.DataFrame | None,
    output_dir: str,
) -> None:
    """
    Like fig7 (stacked UMAP + probability) but top panel is case vs control only,
    with HDBSCAN cluster IDs annotated at per-cluster spatial medians.
    Panel titles use a larger font than fig7.
    """
    has_probs = df_preds is not None
    title_fs = 18
    title_pad = 4

    if has_probs:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(9, 15),
            gridspec_kw={"hspace": 0.02, "height_ratios": [1, 1]},
        )
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(9, 8))
        ax_bot = None

    df = df_clusters.copy()
    df["_is_case"] = df["case_label"].apply(lambda x: str(x).lower() != "control")

    controls = df[~df["_is_case"]]
    cases = df[df["_is_case"]]
    if not controls.empty:
        ax_top.scatter(
            controls["dim1"], controls["dim2"],
            s=12, alpha=0.55, color="#bdc3c7", label="Control", rasterized=True,
        )
    if not cases.empty:
        ax_top.scatter(
            cases["dim1"], cases["dim2"],
            s=18, alpha=0.65, color="#e74c3c", label="Case", rasterized=True,
        )

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        subset = df[df["cluster"] == cluster_id]
        ax_top.text(
            float(subset["dim1"].median()),
            float(subset["dim2"].median()),
            str(cluster_id),
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    ax_top.set_title(
        "Case vs. control (cluster ID at centroid)",
        weight="bold",
        fontsize=title_fs,
        pad=title_pad,
    )
    ax_top.axis("off")
    ax_top.legend(loc="upper right", fontsize=10)

    if has_probs and ax_bot is not None:
        _scatter_predicted_probability_umap(ax_bot, fig, df_clusters, df_preds)
        ax_bot.set_title("Predicted probability", weight="bold", fontsize=title_fs, pad=title_pad)
        ax_bot.axis("off")
        span_x = float(df_clusters["dim1"].max() - df_clusters["dim1"].min())
        span_y = float(df_clusters["dim2"].max() - df_clusters["dim2"].min())
        px = 0.02 * span_x if span_x > 0 else 0.1
        py = 0.02 * span_y if span_y > 0 else 0.1
        xlim = (float(df_clusters["dim1"].min()) - px, float(df_clusters["dim1"].max()) + px)
        ylim = (float(df_clusters["dim2"].min()) - py, float(df_clusters["dim2"].max()) + py)
        ax_top.set_xlim(xlim)
        ax_top.set_ylim(ylim)
        ax_bot.set_xlim(xlim)
        ax_bot.set_ylim(ylim)

    fig.subplots_adjust(left=0.03, right=0.91, top=0.97, bottom=0.02)

    out_name = "fig13_case_control_cluster_annotated_and_probability_umap.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_name}")


def plot_cluster_and_probability_umap(
    df_clusters: pd.DataFrame,
    df_preds: pd.DataFrame | None,
    output_dir: str,
) -> None:
    """
    Two-panel stacked figure sharing the same 2D UMAP space.

    Top panel   — coloured by HDBSCAN cluster; cases shown as crosses (×),
                  controls as circles (○).
    Bottom panel — coloured by model-predicted probability (continuous heatmap).
                  Requires df_preds; skipped with a warning if None.
    """
    has_probs = df_preds is not None
    title_pad = 4

    if has_probs:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(9, 15),
            gridspec_kw={"hspace": 0.02, "height_ratios": [1, 1]},
        )
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(9, 8))
        ax_bot = None

    # Derive is_case flag
    df_clusters = df_clusters.copy()
    df_clusters["_is_case"] = df_clusters["case_label"].apply(
        lambda x: str(x).lower() != "control"
    )

    real_clusters = sorted(c for c in df_clusters["cluster"].unique() if c != -1)
    cmap = plt.cm.get_cmap("tab10", max(len(real_clusters), 1))

    # ── Top panel: cluster colour, marker by case/control ──────────────────
    # Noise first so real clusters render on top
    noise = df_clusters[df_clusters["cluster"] == -1]
    if not noise.empty:
        ax_top.scatter(
            noise["dim1"], noise["dim2"],
            s=6, alpha=0.25, color="lightgrey",
            marker="o", label="_nolegend_", rasterized=True,
        )

    for i, cid in enumerate(real_clusters):
        sub = df_clusters[df_clusters["cluster"] == cid]
        colour = cmap(i)
        controls = sub[~sub["_is_case"]]
        cases    = sub[sub["_is_case"]]

        if not controls.empty:
            ax_top.scatter(
                controls["dim1"], controls["dim2"],
                s=10, alpha=0.55, color=colour,
                marker="o", label=f"Cluster {cid} – control" if i == 0 else f"Cluster {cid}",
                rasterized=True,
            )
        if not cases.empty:
            ax_top.scatter(
                cases["dim1"], cases["dim2"],
                s=35, alpha=0.8, color=colour,
                marker="x", linewidths=0.9,
                label=f"Cluster {cid} – case" if i == 0 else "_nolegend_",
                rasterized=True,
            )
        # Centroid annotation
        cx = float(sub["dim1"].median())
        cy = float(sub["dim2"].median())
        ax_top.text(
            cx, cy, str(cid),
            fontsize=10, weight="bold", ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    ax_top.set_title(
        "HDBSCAN Clusters\n(× = case, ○ = control)",
        weight="bold",
        fontsize=18,
        pad=title_pad,
    )
    ax_top.axis("off")

    # Compact legend: one entry per cluster + a marker-shape key
    handles, labels_ = ax_top.get_legend_handles_labels()
    # Add shape-only legend entries
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=6, label="Control"),
        Line2D([0], [0], marker="x", color="grey", linestyle="None", markersize=8,
               markeredgewidth=1.2, label="Case"),
    ]
    ax_top.legend(
        handles=handles,
        loc="lower right", fontsize=7, markerscale=1.2,
        framealpha=0.6, ncol=2,
    )

    # ── Bottom panel: predicted probability heatmap ─────────────────────────
    if has_probs and ax_bot is not None:
        _scatter_predicted_probability_umap(ax_bot, fig, df_clusters, df_preds)
        ax_bot.set_title("Predicted Probability", weight="bold", fontsize=18, pad=title_pad)
        ax_bot.axis("off")
        span_x = float(df_clusters["dim1"].max() - df_clusters["dim1"].min())
        span_y = float(df_clusters["dim2"].max() - df_clusters["dim2"].min())
        px = 0.02 * span_x if span_x > 0 else 0.1
        py = 0.02 * span_y if span_y > 0 else 0.1
        xlim = (float(df_clusters["dim1"].min()) - px, float(df_clusters["dim1"].max()) + px)
        ylim = (float(df_clusters["dim2"].min()) - py, float(df_clusters["dim2"].max()) + py)
        ax_top.set_xlim(xlim)
        ax_top.set_ylim(ylim)
        ax_bot.set_xlim(xlim)
        ax_bot.set_ylim(ylim)

    fig.subplots_adjust(left=0.03, right=0.91, top=0.97, bottom=0.02)

    plt.savefig(
        os.path.join(output_dir, "fig7_cluster_and_probability_umap.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print("  Saved fig7_cluster_and_probability_umap.png")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def generate_plots_from_saved_outputs(output_dir: str, clusters_csv: str = None, summary_csv: str = None):
    """Generate plots from precomputed CSV outputs (no clustering run)."""
    clusters_csv = clusters_csv or os.path.join(output_dir, "patient_clusters.csv")
    summary_csv = summary_csv or os.path.join(output_dir, "full_differential_analysis.csv")

    if not os.path.exists(clusters_csv):
        raise FileNotFoundError(f"Missing clusters CSV: {clusters_csv}")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    print(f"Loading clusters from: {clusters_csv}")
    df_clusters = pd.read_csv(clusters_csv)
    print(f"Loading differential summary from: {summary_csv}")
    df_summary = pd.read_csv(summary_csv)

    # INSERT_YOUR_CODE
    # Make the output directory output_dir + "/testing"
    output_dir = os.path.join(output_dir, "testing")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating journal tables from loaded summary...")
    heatmap_cat, heatmap_lab = generate_journal_reports(df_summary, output_dir)

    print("Generating figures...")
    plot_annotated_umap(df_clusters, output_dir)
    plot_stacked_composition(df_clusters, output_dir)
    plot_clinical_heatmap(heatmap_cat, output_dir)
    plot_lab_zscore_heatmap(heatmap_lab, output_dir)

    df_preds = load_predictions(PREDICTIONS_JSON)
    if df_preds is not None:
        plot_prediction_correctness_umap(df_clusters, df_preds, output_dir)
        plot_predicted_probability_umap_case_control_markers(df_clusters, df_preds, output_dir)
    plot_cluster_and_probability_umap(df_clusters, df_preds, output_dir)
    plot_case_control_cluster_annotated_and_probability_umap(df_clusters, df_preds, output_dir)

    print(f"Plot-only run complete. Check {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN pipeline + plotting utilities.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip clustering/analysis and only generate figures from saved CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for generated artifacts (default: pipeline OUTPUT_DIR).",
    )
    parser.add_argument(
        "--clusters-csv",
        default=None,
        help="Optional path to patient_clusters.csv (used with --plot-only).",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path to full_differential_analysis.csv (used with --plot-only).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.plot_only:
        generate_plots_from_saved_outputs(
            output_dir=output_dir,
            clusters_csv=args.clusters_csv,
            summary_csv=args.summary_csv,
        )
        return

    print("1. Loading dataset and embeddings...")
    dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(CONFIG_PATH, "held_out")
    embeddings, patient_ids = load_embeddings(EMBEDDINGS_DIR)

    print("2. Running Fixed HDBSCAN Clustering...")
    cluster_labels, z_2d, reducer_10d, reducer_2d = run_fixed_clustering(embeddings)

    joblib.dump(reducer_10d, os.path.join(output_dir, "umap_10d.joblib"))
    joblib.dump(reducer_2d, os.path.join(output_dir, "umap_2d.joblib"))

    df_clusters = pd.DataFrame({"patient_id": patient_ids, "cluster": cluster_labels, "dim1": z_2d[:, 0], "dim2": z_2d[:, 1]})
    df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
    df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
    df_clusters["case_label"] = df_clusters["label_id"].map(label_to_name).fillna("Control")
    df_clusters.to_csv(os.path.join(output_dir, "patient_clusters.csv"), index=False)

    print("3. Analyzing Differential Features...")
    df_summary = analyse_differential_features(dataset, df_clusters, subject_to_record_idx, output_dir)

    print("4. Generating Journal Tables (Categorical vs Continuous)...")
    heatmap_cat, heatmap_lab = generate_journal_reports(df_summary, output_dir)

    print("5. Generating High-Res Journal Plots...")
    plot_annotated_umap(df_clusters, output_dir)
    plot_stacked_composition(df_clusters, output_dir)
    plot_clinical_heatmap(heatmap_cat, output_dir)
    plot_lab_zscore_heatmap(heatmap_lab, output_dir)

    print("6. Generating prediction figures (if evaluation results exist)...")
    df_preds = load_predictions(PREDICTIONS_JSON)
    if df_preds is not None:
        plot_prediction_correctness_umap(df_clusters, df_preds, output_dir)
        plot_predicted_probability_umap_case_control_markers(df_clusters, df_preds, output_dir)
    plot_cluster_and_probability_umap(df_clusters, df_preds, output_dir)
    plot_case_control_cluster_annotated_and_probability_umap(df_clusters, df_preds, output_dir)

    print(f"Pipeline Complete! Check {output_dir} for publication-ready figures.")

if __name__ == "__main__":
    main()