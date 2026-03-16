#!/bin/bash
# 1. Environment Setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate meds_train_clean

# 2. Pathing and Auth
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_API_KEY="REDACTED"
export CUDA_VISIBLE_DEVICES=3
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

# MIMIC Pretrain
# torchrun \
#   --standalone \
#   --nproc_per_node=3 \
#   src/pipelines/llm_pretrain.py \
#   --config_filepath src/configs/mimic_pretrain.yaml \
#   2>&1 | tee -a last_run.log

# MIMIC Resume Pretrain
# torchrun \
#   --standalone \
#   --nproc_per_node=3 \
#   src/pipelines/resume_pretrain.py \
#   --config_filepath src/configs/mimic_resume_pretrain_config.yaml \
#   --checkpoint_path /data/private/ag619/experiments/Pretrain-Mistral-7B-Mimic/checkpoint-3000 \
#   2>&1 | tee -a continue_run.log

# MIMIC Classifier
# torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   src/pipelines/finetune_llm_classifier.py \
#   --config_filepath src/configs/mimic_classifier.yaml \
#   2>&1 | tee -a mimic_classifier.log

# MIMIC Pure LLM Baseline
    # python -m src.pipelines.run_baseline_pure_llm \
    #   --config_filepath src/configs/baseline_pure_llm_config.yaml \
    #   2>&1 | tee -a pure_llm_baseline.log
  
# # MIMIC Trajectory Generation
python -m src.inference.trajectory_generator \
  --config src/configs/generate_trajectories.yaml \
  --split tuning \
  --benchmark \
  2>&1 | tee -a trajectory_generation.log