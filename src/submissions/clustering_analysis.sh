#!/bin/bash
#SBATCH -n 4
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=1G
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_clustering/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_clustering/loge/%x.e%j

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
# module load python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate
echo "Environment setup complete."
# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# python -m src.analysis.cluster_hdbscan_search
# python -m src.analysis.hdbscan
# python -m src.analysis.plot_hdbscan_figures \
#     --output-dir /gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output \
#     --clusters-csv /gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output/patient_clusters.csv \
#     --summary-csv /gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output/full_differential_analysis.csv

python src/analysis/plot_lab_violins.py \
  --clusters-csv "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output/patient_clusters.csv" \
  --config-path "src/configs/classification_config.yaml" \
  --split "held_out" \
  --output-path "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout/journal_clustering_output/fig5_lab_violins.png" \
  --highlight-cluster 1 \
  --exclude-noise \
  --upper-threshold-multiplier 3



echo "Pipeline finished."
deactivate