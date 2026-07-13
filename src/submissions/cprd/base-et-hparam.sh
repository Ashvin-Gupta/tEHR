#!/bin/bash
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/%x.e%j

set -e

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"


# --- Environment Setup ---
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

cd "${BASE_DIR}"
echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.pipelines.run_baseline \
  --baseline encoder_transformer_hparam_search \
  --config_filepath src/configs/baseline_encoder_transformer_hparam_search_config.yaml
echo "Pipeline finished."
deactivate