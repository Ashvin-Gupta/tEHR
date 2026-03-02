#!/bin/bash
# 1. Environment Setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate meds_train_clean

# 2. Pathing and Auth
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_API_KEY="REDACTED"
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=1,2

# 3. Execution with Logging
echo "🚀 Starting Pretraining on GPUs..."
# torchrun --nproc_per_node=1 src/pipelines/llm_pretrain.py \
#     --config_filepath src/configs/mimic_pretrain.yaml \
#     2>&1 | tee -a last_run.log
torchrun \
  --standalone \
  --nproc_per_node=3 \
  src/pipelines/llm_pretrain.py \
  --config_filepath src/configs/mimic_pretrain.yaml \
  2>&1 | tee -a last_run.log