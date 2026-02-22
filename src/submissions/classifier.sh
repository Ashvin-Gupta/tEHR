#!/bin/bash
#$ -cwd                 
#$ -pe smp 24
#$ -l h_rt=240:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Classifier/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Classifier/loge/

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

# python -m src.pipelines.finetune_llm_classifier --config_filepath src/configs/classification_config.yaml 
torchrun --nproc_per_node=2 src/pipelines/finetune_llm_classifier.py --config_filepath src/configs/classification_config2.yaml

echo "Pipeline finished."
deactivate

