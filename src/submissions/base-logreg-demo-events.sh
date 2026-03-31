#!/bin/bash
#SBATCH -n 4
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=1G
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/%x.e%j

set -e

BASE_DIR="/data/home/qc25022/TextCancEHR2"
export WANDB_API_KEY="REDACTED"

module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

cd "${BASE_DIR}"
echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.pipelines.run_baseline \
  --baseline logreg_demo_events \
  --config_filepath src/configs/baseline_logreg_demo_events_config.yaml

echo "Pipeline finished."
deactivate
