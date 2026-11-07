# tEHR

Clinical prediction from textualised EHR timelines. Stage 1 is continued pretraining on unlabelled patient text. Stage 2 adds a LoRA adapter and a linear head.

Evaluated on MIMIC-IV and a CPRD pancreatic-cancer cohort. Both are expected already tokenised. Point `configs/local.yaml` at the timelines, vocab, labels, stage-1 checkpoints, and results directory.

## Setup

Install into a CUDA PyTorch environment:

```bash
pip install -r requirements.txt
```

Copy [`configs/local.yaml.example`](configs/local.yaml.example) to `configs/local.yaml` and fill in the paths.

HPC submission scripts belong in `scripts/`. A cluster wrapper there builds a resolved config and starts `src/pipelines/run.py`. Submit from the repository root:

```bash
sbatch scripts/<cluster>/run.sh <dataset> <experiment> [model] [task]
```

CPU-only jobs use `run_cpu.sh` in the same place. Without Slurm, call the same entry point directly:

```bash
python -m src.training.config_utils build \
  --dataset mimic --experiment tehr_stage2 --model mistral --task icu_mortality \
  --out configs/_resolved/mimic-tehr_stage2-mistral-icu_mortality.yaml
python -m src.pipelines.run --config_filepath configs/_resolved/mimic-tehr_stage2-mistral-icu_mortality.yaml
```

`tehr_stage1` does not take a task. Model and task can follow the experiment name in either order. Stage 2 reads the stage-1 checkpoint listed for that dataset and model in `configs/local.yaml`.

```bash
sbatch scripts/<cluster>/run.sh mimic tehr_stage1 mistral
sbatch scripts/<cluster>/run.sh mimic tehr_stage2 mistral icu_mortality
sbatch scripts/<cluster>/run.sh cprd tehr_stage2 qwen8 6m
```

## Data and models

| | |
|---|---|
| **dataset** | `mimic` · `cprd` |
| **model** | `mistral` (Mistral-7B-v0.3) · `qwen8` (Qwen3-8B) · `qwen0.6` (Qwen3-0.6B) |
| **MIMIC-IV tasks** | `icu_mortality` · `icu_readmission` · `hospital_mortality` · `hospital_readmission` |
| **CPRD tasks** | pancreatic cancer, history cut at `1m` · `3m` · `6m` · `12m` (default `6m`) |

MIMIC-IV is version 3.1 (ICD and NDC). CPRD events are Read Codes. The CPRD cohort is a matched pancreatic-cancer case-control set, not the full database. Encoder-style runs do not take a model.

## Experiments

Names match files in `configs/experiments/`. Routing is in `src/experiment_map.py`.

### tEHR

`tehr_stage1`, then `tehr_stage2`. The same two-stage run is repeated with Mistral-7B-v0.3, Qwen3-8B, and Qwen3-0.6B.

### Baselines

| | Experiment |
|---|---|
| XGBoost on bag-of-codes counts | `xgboost_bow` |
| Transformer encoder, masked-language pretraining then a supervised head | `encoder` |
| Frozen CLMBR-T plus a linear head (MIMIC-IV only) | `clmbr` |
| Fine-tuned CLMBR-T (MIMIC-IV only) | `clmbr_ft` |
| BioClinical ModernBERT on the same text | `bioclinical_modernbert` |
| Zero-shot LLM prompting | `zero_shot` |
| Prompt supervised fine-tuning (MIMIC-IV only) | `sft` |

### Ablations

Run on MIMIC-IV. CPRD was not swept.

**Training.** `ablation_supervised_only` drops stage 1 and adapts the base LLM with LoRA and a linear head. `ablation_cpt_frozen` keeps stage 1 and trains only the head. `ablation_nopretrain_frozen` freezes the base LLM and trains only the head.

**Text.** `ablation_hash` replaces each concept with a meaningless id. `ablation_scrambled` permutes concept descriptions within event type. `ablation_no_time` drops elapsed-time text and keeps event order. `ablation_no_value` drops numeric values and keeps the lab or vital-sign identity.

**Atomic codes.** `event_token_stage1` then `event_token_stage2` run the same two-stage procedure on Mistral-7B, with clinical-code tokens in place of natural-language event descriptions.

**Other uses of pretrained language.** `encoder_phrase_init` is a code transformer whose embeddings are initialised from LLM descriptions of the codes. `text_embedding` feeds frozen event embeddings to a trainable transformer. `llm_phrase` projects those embeddings into an LLM. The last two are supervised only, with no continued pretraining.

## Score a run

Append `evaluate` to score the newest checkpoint instead of training. Set `CHECKPOINT` to pick a file. Stage 1 and `zero_shot` have nothing to score.

```bash
sbatch scripts/<cluster>/run.sh mimic tehr_stage2 mistral hospital_readmission evaluate
```

Extra seeds write `seed_<n>/` (default 42):

```bash
SEED=43 sbatch scripts/<cluster>/run.sh mimic tehr_stage2 mistral icu_mortality
```

Runs land under `results_root` from `configs/local.yaml`. Bootstrap intervals, across-seed intervals, paired tests, and ablation plots are CPU jobs: `bootstrap_ci`, `seed_ci`, `paired_compare`, `plot_ablation`.
