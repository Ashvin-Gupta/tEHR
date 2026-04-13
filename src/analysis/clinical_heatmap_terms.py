import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def cluster_priority_row_order(heatmap_data: pd.DataFrame) -> List[str]:
    """Order rows by dominant cluster, then by enrichment descending."""
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

    ordered.extend([r for r in heatmap_data.index if r not in seen])
    return ordered


def plot_clinical_heatmap_terms(
    df_summary: pd.DataFrame,
    terms_of_interest: List[str],
    output_dir: str,
    output_name: str = "fig7_specific_clinical_fingerprint_heatmap_terms.png",
) -> None:
    """Recreate the clinical enrichment heatmap for only selected terms."""
    if df_summary is None or df_summary.empty:
        return

    if not terms_of_interest:
        raise ValueError("terms_of_interest is empty.")

    selected_rows = []
    for term in list(dict.fromkeys(terms_of_interest)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if not match.empty:
            match = match.sort_values("count", ascending=False).drop_duplicates(subset=["cluster"])
            selected_rows.append(match)

    if not selected_rows:
        print("No terms of interest found in df_summary. Nothing to plot.")
        return

    df_cat = pd.concat(selected_rows).drop_duplicates(subset=["event_human", "cluster"])
    heatmap_cat = df_cat.pivot(index="event_human", columns="cluster", values="enrichment_score").fillna(0)

    if -1 in heatmap_cat.columns:
        heatmap_cat = heatmap_cat.drop(columns=[-1])
    if heatmap_cat.empty:
        print("Only noise cluster found after filtering. Nothing to plot.")
        return

    row_order = cluster_priority_row_order(heatmap_cat)
    heatmap_cat = heatmap_cat.loc[row_order]
    heatmap_cat = heatmap_cat.rename(columns={c: f"Cluster {c}" for c in heatmap_cat.columns})

    plt.figure(figsize=(8, max(4, len(heatmap_cat) * 0.45)))
    sns.heatmap(
        heatmap_cat,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Relative Risk (x-fold)"},
    )
    plt.title("Clinical Phenotypes (Selected Terms)", weight="bold")
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clinical enrichment heatmap for a custom list of terms."
    )
    parser.add_argument(
        "--summary-csv",
        required=True,
        help="Path to full_differential_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the heatmap image",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=['Gastrectomy', 'Steatorrhea', 'Ankylosing spondylitus', 'Pancreatitis', 'Jaundice', 'Abdominal mass', 'Obesity', 'Lupus erythematosus', 'Liver failure', 'Non-Alcoholic fatty liver disease', 'GIT bleed', 'Dysmenorrhoea', 'Male infertility', 'Leiomyoma'],
        help="Term of interest (repeat this flag for multiple terms)",
    )
    args = parser.parse_args()

    df_summary = pd.read_csv(args.summary_csv)
    plot_clinical_heatmap_terms(
        df_summary=df_summary,
        terms_of_interest=args.term,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
