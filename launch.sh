#!/bin/bash
# 1. Environment Setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate meds_train_clean

# 2. Pathing and Auth
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_API_KEY="REDACTED"
export CUDA_VISIBLE_DEVICES=0,1
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
#   --nproc_per_node=1 \
#   src/pipelines/llm_pretrain.py \
#   --config_filepath src/configs/mimic_pretrain.yaml \
#   2>&1 | tee -a logs/last_run.log

# CPRD Pretrain
# torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   src/pipelines/llm_pretrain.py \
#   --config_filepath src/configs/llm_pretrain.yaml \
#   2>&1 | tee -a logs/cprd_pretrain.log

# MIMIC Resume Pretrain
# torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   src/pipelines/resume_pretrain.py \
#   --config_filepath src/configs/mimic_resume_pretrain_config.yaml \
#   --checkpoint_path /data/private/ag619/experiments/Pretrain-Mistral-7B-Mimic/checkpoint-2000 \
#   2>&1 | tee -a logs/continue_run.log

# MIMIC Classifier
# torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   src/pipelines/finetune_llm_classifier.py \
#   --config_filepath src/configs/mimic_classifier.yaml \
#   2>&1 | tee -a logs/mimic_classifier.log

# MIMICPRDC Classifier
# torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   src/pipelines/finetune_llm_classifier.py \
#   --config_filepath src/configs/classification_config2.yaml \
#   2>&1 | tee -a logs/cprd_classifier.log

# MIMIC Pure LLM Baseline
# python -m src.pipelines.run_baseline --baseline pure_llm --config_filepath /home/ag619/TextCancEHR2/src/configs/baseline_pure_llm_config.yaml
  
# # MIMIC Trajectory Generation
# python -m src.inference.trajectory_generator \
#   --config src/configs/generate_trajectories.yaml \
#   --split tuning \
#   --benchmark \
#   2>&1 | tee -a trajectory_generation.log

# CPRD Baseline ClinicalBERT
# python -m src.pipelines.run_baseline --baseline clinicalbert --config_filepath src/configs/baseline_clinicalbert_config.yaml 2>&1 | tee -a logs/cprd_clinicalbert.log

# CPRD Baseline Encoder Transformer
# python -m src.pipelines.run_baseline --baseline encoder_transformer --config_filepath src/configs/baseline_encoder_transformer_config.yaml 2>&1 | tee -a logs/cprd_encoder_transformer.log

# CPRD Baseline Frozen LLM Linear
# python -m src.pipelines.run_baseline --baseline frozen_llm_linear --config_filepath src/configs/baseline_frozen_llm_linear_config.yaml 2>&1 | tee -a logs/cprd_frozen_llm_linear.log

# CPRD Baseline Lora LLM Linear
# python -m src.pipelines.run_baseline --baseline lora_llm_linear --config_filepath src/configs/baseline_lora_llm_linear_config.yaml 2>&1 | tee -a logs/cprd_lora_llm_linear.log

# CPRD Baseline Pretrained LLM Linear
# python -m src.pipelines.run_baseline --baseline pretrained_llm_linear --config_filepath src/configs/baseline_pretrained_llm_linear_config.yaml 2>&1 | tee -a logs/cprd_pretrained_llm_linear.log

# CPRD Baseline XGBoost BOW
# python -m src.pipelines.run_baseline --baseline xgboost_bow --config_filepath src/configs/baseline_xgboost_bow_config.yaml 2>&1 | tee -a logs/cprd_xgboost_bow.log

# CPRD Baseline Qwen3 Embedding
# python -m src.pipelines.run_baseline --baseline qwen3_embedding --config_filepath src/configs/baseline_qwen3_embedding_config.yaml 2>&1 | tee -a logs/cprd_qwen3_embedding.log

# EHRSHOT Classifier
# python -m src.evaluations.evaluate_model \
#   --config_filepath src/configs/ehrshot_evaluate_config.yaml \
#   --model_checkpoint /data/private/ag619/experiments/Results/mimic-ICU-mortality-classification/seed_405745800/checkpoint-63

torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/evaluations/evaluate_model.py \
  --config_filepath src/configs/ehrshot_evaluate_config.yaml \
  --model_checkpoint /data/private/ag619/experiments/Results/mimic-ICU-mortality-classification/seed_405745800/checkpoint-63