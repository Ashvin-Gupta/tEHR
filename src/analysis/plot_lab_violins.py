"""
Standalone lab violin plotter from existing cluster assignments.

Reads:
  - patient clusters CSV (from previous HDBSCAN run)
  - dataset records via classification config

Outputs:
  - violin plot for selected labs (no re-clustering)
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from src.data.dataset_factory import get_dataset
from src.resources.lab_reference_ranges import LAB_REFERENCE_RANGES


TARGET_LABS = [
    "Lab Ferritin",
    "Lab ALP",
    "Lab HbA1c",
    "Lab GGT",
    "Lab Glucose",
    "Lab Amylase",
    "Lab ALT",
]

LAB_DISPLAY_NAMES = {
    "Lab Ferritin": "Ferritin",
    "Lab ALP": "ALP",
    "Lab HbA1c": "HbA1c",
    "Lab GGT": "GGT",
    "Lab Glucose": "Glucose",
    "Lab Amylase": "Amylase",
    "Lab ALT": "ALT",
}


def load_dataset(config_path: str, split: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset = get_dataset(config["data"].get("dataset", "CPRD"), config["data"], split)
    subject_to_record_idx = {
        str(r.get("subject_id")): idx
        for idx, r in enumerate(dataset.patient_records)
        if r.get("subject_id")
    }
    return dataset, subject_to_record_idx


def get_filtered_token_ids(dataset, record: dict):
    subject_id = record.get("subject_id")
    token_ids = list(record.get("tokens", []))
    timestamps = record.get("timestamps", [])
    if subject_id is None:
        return token_ids

    label = dataset.subject_to_label.get(subject_id)
    token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, label)
    return [tid for tid in token_ids if "cancer" not in str(dataset.id_to_token.get(tid, "")).lower()]


def extract_lab_values(dataset, subject_to_record_idx, clusters_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_set = set(TARGET_LABS)

    for _, row in clusters_df.iterrows():
        patient_id = str(row["patient_id"])
        cluster = int(row["cluster"])
        rec_idx = subject_to_record_idx.get(patient_id)
        if rec_idx is None:
            continue

        record = dataset.patient_records[rec_idx]
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in get_filtered_token_ids(dataset, record)]

        for i, code in enumerate(tokens[:-1]):
            if not (code.startswith("LAB//") or code.startswith("MEASUREMENT//")):
                continue

            try:
                value = float(tokens[i + 1])
            except ValueError:
                continue

            try:
                human_name = dataset.translator.translate(code).rstrip("; ")
            except Exception:
                continue

            if human_name in target_set:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "cluster": cluster,
                        "lab": human_name,
                        "lab_display": LAB_DISPLAY_NAMES[human_name],
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


def filter_extreme_high_values(df: pd.DataFrame, upper_threshold_multiplier: float) -> pd.DataFrame:
    """
    Remove extreme high values using lab-specific normal ranges.

    A value is kept when:
      value <= normal_upper * upper_threshold_multiplier
    """
    if df.empty:
        return df

    keep_mask = pd.Series(True, index=df.index)
    for idx, row in df.iterrows():
        ref = LAB_REFERENCE_RANGES.get(row["lab"])
        if ref is None:
            continue
        _, upper = ref
        upper_threshold = upper * upper_threshold_multiplier
        if row["value"] > upper_threshold:
            keep_mask.at[idx] = False
    return df.loc[keep_mask].copy()


def plot_violin(df: pd.DataFrame, output_path: str, highlight_cluster: int):
    if df.empty:
        raise ValueError("No matching lab values were found for the requested labs.")

    cluster_order = sorted(df["cluster"].unique())

    # Bold color for highlight cluster (e.g., cluster 1), muted colors for others.
    palette = {
        c: ("#d62728" if c == highlight_cluster else "#b8b8b8")
        for c in cluster_order
    }

    lab_order = TARGET_LABS
    display_order = [LAB_DISPLAY_NAMES[l] for l in lab_order]
    df = df.copy()
    df["lab"] = pd.Categorical(df["lab"], categories=lab_order, ordered=True)
    df["lab_display"] = pd.Categorical(df["lab_display"], categories=display_order, ordered=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for i, lab in enumerate(lab_order):
        ax = axes_flat[i]
        lab_df = df[df["lab"] == lab].copy()

        if lab_df.empty:
            ax.set_title(f"{LAB_DISPLAY_NAMES[lab]} (no data)")
            ax.axis("off")
            continue

        sns.violinplot(
            data=lab_df,
            x="cluster",
            y="value",
            hue="cluster",
            hue_order=cluster_order,
            order=cluster_order,
            palette=palette,
            cut=0,
            dodge=False,
            inner="quartile",
            linewidth=0.9,
            saturation=1.0,
            ax=ax,
            legend=False,
        )

        if lab in LAB_REFERENCE_RANGES:
            low, high = LAB_REFERENCE_RANGES[lab]
            ax.axhline(low, color="red", linestyle="dotted", linewidth=1.4)
            ax.axhline(high, color="red", linestyle="dotted", linewidth=1.4)

        ax.set_title(LAB_DISPLAY_NAMES[lab], weight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Value")

    # Hide the 8th subplot since we currently have 7 labs.
    for j in range(len(lab_order), len(axes_flat)):
        axes_flat[j].axis("off")

    handles = []
    labels = []
    for c in cluster_order:
        line = plt.Line2D([0], [0], color=palette[c], lw=8)
        handles.append(line)
        labels.append(f"Cluster {c}")
    fig.legend(handles, labels, title="Cluster", loc="upper right")

    fig.suptitle("Lab Value Distributions by Cluster", weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot lab violins using existing patient clusters.")
    parser.add_argument(
        "--clusters-csv",
        required=True,
        help="Path to patient_clusters.csv",
    )
    parser.add_argument(
        "--config-path",
        default="src/configs/classification_config.yaml",
        help="Path to classification config yaml",
    )
    parser.add_argument(
        "--split",
        default="held_out",
        help="Dataset split used to generate clusters (default: held_out)",
    )
    parser.add_argument(
        "--output-path",
        default="lab_violin_clusters.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--highlight-cluster",
        type=int,
        default=1,
        help="Cluster to highlight (bold color), default: 1",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Exclude HDBSCAN noise cluster (-1) before plotting.",
    )
    parser.add_argument(
        "--upper-threshold-multiplier",
        type=float,
        default=3.0,
        help="Drop values above (normal_upper * multiplier). Default: 3.0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    clusters_df = pd.read_csv(args.clusters_csv)
    required_cols = {"patient_id", "cluster"}
    if not required_cols.issubset(clusters_df.columns):
        raise ValueError(f"{args.clusters_csv} must contain columns: {sorted(required_cols)}")

    if args.exclude_noise:
        clusters_df = clusters_df[clusters_df["cluster"] != -1].copy()

    dataset, subject_to_record_idx = load_dataset(args.config_path, args.split)
    lab_df = extract_lab_values(dataset, subject_to_record_idx, clusters_df)
    original_n = len(lab_df)
    lab_df = filter_extreme_high_values(lab_df, args.upper_threshold_multiplier)
    after_high_filter_n = len(lab_df)
    lab_df = lab_df[lab_df["value"] != 0].copy()
    removed_n = original_n - len(lab_df)
    removed_high_n = original_n - after_high_filter_n
    removed_zero_n = after_high_filter_n - len(lab_df)
    plot_violin(lab_df, args.output_path, highlight_cluster=args.highlight_cluster)
    print(f"Filtered {removed_high_n} extreme high values using multiplier={args.upper_threshold_multiplier}")
    print(f"Filtered {removed_zero_n} zero values")
    print(f"Total filtered values: {removed_n}")
    print(f"Saved violin plot to: {args.output_path}")


if __name__ == "__main__":
    main()
