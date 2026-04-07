import argparse

from src.analysis.hdbscan import generate_plots_from_saved_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HDBSCAN journal figures from existing CSV outputs."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures/tables are written.",
    )
    parser.add_argument(
        "--clusters-csv",
        default=None,
        help="Optional path to patient_clusters.csv (defaults to <output-dir>/patient_clusters.csv).",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path to full_differential_analysis.csv (defaults to <output-dir>/full_differential_analysis.csv).",
    )
    args = parser.parse_args()

    generate_plots_from_saved_outputs(
        output_dir=args.output_dir,
        clusters_csv=args.clusters_csv,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()
