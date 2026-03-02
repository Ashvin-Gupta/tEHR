#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.analysis.cluster_patients \
  --config_filepath src/configs/classification_config.yaml \
  --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/lora-1-month-logistic-refactored-new-time-mistral-7B/final_model \
  --split held_out \
  --output_dir /data/scratch/qc25022/pancreas_MEDS/experiments/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout \
  --batch_size 4 \
  --dim_reducer umap \
  --cluster_space umap \
  --k 5 \
  --plot
echo "Pipeline finished."
deactivate