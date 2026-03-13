#!/bin/bash
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/%x.o%j   # STDOUT
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/%x.e%j   # STDERR
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:1                    # Request 1 GPU of any type
#SBATCH --cpus-per-gpu=12              # 12 CPU cores 
#SBATCH -t 240:0:0                        # 1 hour runtime
#SBATCH --mem-per-cpu=7500M             # Total memory 90G (approx 12 * 7500M)

set -e 

BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

cd "${BASE_DIR}"

echo "Starting Qwen3-Embedding baseline from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.pipelines.run_baseline_qwen3_embedding --config_filepath src/configs/baseline_qwen3_embedding_config.yaml 
# torchrun --nproc_per_node=2 src/pipelines/run_baseline_qwen3_embedding.py --config_filepath src/configs/baseline_qwen3_embedding_config.yaml


echo "Pipeline finished."
deactivate
