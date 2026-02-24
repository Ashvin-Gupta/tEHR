#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/

set -e 

BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

cd "${BASE_DIR}"

echo "Starting Qwen3-Embedding baseline from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.pipelines.run_baseline_qwen3_embedding --config_filepath src/configs/baseline_qwen3_embedding_config.yaml 
# torchrun --nproc_per_node=2 src/pipelines/run_baseline_qwen3_embedding.py --config_filepath src/configs/baseline_qwen3_embedding_config.yaml


echo "Pipeline finished."
deactivate
