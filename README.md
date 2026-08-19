# tEHR

EHR text modelling for clinical prediction on MIMIC-IV and CPRD.

## Start here

1. Copy [`configs/local.yaml.example`](configs/local.yaml.example) to `configs/local.yaml` (data and results paths).
2. Copy `scripts/site_a/local.env.example` or `scripts/site_b/local.env.example` to `local.env` (conda, tokens).
3. From the repository root, submit a job. That is the only way most people need to run experiments.

```bash
sbatch scripts/site_a/run.sh <dataset> <experiment> [model] [task]
sbatch scripts/site_b/run.sh <dataset> <experiment> [model] [task]
```

`run.sh` builds a resolved YAML, then calls **`src/pipelines/run.py`**. That one Python file looks at `config['_meta']['experiment']` and starts the matching trainer. You do not pick a trainer on the command line.

```text
sbatch scripts/<site>/run.sh
        → configs/_resolved/<tag>.yaml
        → src/pipelines/run.py
        → trainer or comparison method (table below)
```

Append `evaluate` to score a checkpoint instead of training (any experiment that saves weights). Pass `CHECKPOINT=/path/to/ckpt` if you do not want the newest file under the run directory. `tehr_stage1` and `zero_shot` have nothing to score.

## Arguments

| Kind | Values |
|---|---|
| **dataset** | `mimic` \| `cprd` |
| **model** (LLM experiments) | `mistral` \| `qwen0.6` \| `qwen8` |
| **mimic task** | `icu_mortality` \| `hospital_mortality` \| `hospital_readmission` \| `icu_readmission` |
| **cprd task** | pancreatic cancer cutoffs `1m` \| `3m` \| `6m` \| `12m` (default `6m`) |

`tehr_stage1` does not take a task. Model/task args can be in any order after the experiment name.

## Experiment → file

YAML lives in `configs/experiments/<name>.yaml`. How it runs is listed once in `src/experiment_map.py` (Python routing + shell arrays). To add a method: write the YAML, add a `Spec` there, implement `run()` and `run_eval_only()` on the class.

### Both datasets

| Experiment | Code | What it runs |
|---|---|---|
| `tehr_stage1` | `src/training/pretrainer_trainer.py` | Continued pretraining (LoRA) |
| `tehr_stage2` | `src/training/classification_trainer.py` | Pretrained LLM → LoRA + linear head |
| `ablation_no_time` | `src/training/classification_trainer.py` | Stage 2 text without timestamps |
| `ablation_no_value` | `src/training/classification_trainer.py` | Stage 2 text without values/units |
| `sft` | `src/baselines/sft.py` | Prompt SFT to emit 0/1 |
| `zero_shot` | `src/baselines/zero_shot.py` | Zero-shot LLM prompting |
| `xgboost_bow` | `src/baselines/xgboost_bow.py` | XGBoost bag-of-words (also `run_cpu.sh`) |
| `clinicalbert` | `src/baselines/clinicalbert.py` | Bio_ClinicalBERT fine-tune |
| `encoder` | `src/baselines/encoder.py` | Transformer encoder, MLM + supervised |
| `encoder_supervised` | `src/baselines/encoder.py` | Encoder + head from scratch |
| `encoder_frozen` | `src/baselines/encoder.py` | Freeze MLM encoder; train head |
| `text_embedding` | `src/baselines/text_embedding.py` | Phrase embeddings → transformer |

### MIMIC-only ablations

| Experiment | Code | What it runs |
|---|---|---|
| `ablation_supervised_only` | `src/training/classification_trainer.py` | No EHR CPT; LoRA + head |
| `ablation_cpt_frozen` | `src/training/classification_trainer.py` | CPT then frozen backbone; train head |
| `ablation_nopretrain_frozen` | `src/baselines/ablation_nopretrain_frozen.py` | No CPT, frozen backbone; train head |
| `ablation_hash` | `src/training/classification_trainer.py` | Lexical `Concept_<hash>` titles |
| `ablation_scrambled` | `src/training/classification_trainer.py` | Shuffled clinical titles |

Shared pieces (not experiments): `src/baselines/base.py`, `src/baselines/hf_classifier.py`, `src/models/llm_classifier.py`.

## Where things live

| You want to… | Look here |
|---|---|
| Submit a job | `scripts/site_a/` or `scripts/site_b/` (`run.sh`, `run_cpu.sh`) |
| Change hyperparameters | `configs/experiments/<experiment>.yaml` (plus `configs/models/`, `configs/tasks/`) |
| Change a method | the Python file in the table above |
| Merge dataset + experiment + model + paths | `src/training/config_utils.py` |
| Metrics / ROC / PR / ECE **plots** | `src/evaluations/evaluations.py` |
| Re-score a checkpoint | sbatch `… evaluate` → `run.py --eval_only` (not `src/evaluations/`) |

`src/training/` is the tEHR pipeline (CPT + linear head). `src/baselines/` is every comparison method, including prompt SFT; **file names match experiment names**. Every job writes under `output_dir/seed_<n>/`. `src/evaluations/` only scores arrays and writes plots — it does not launch jobs. Prompt SFT still uses `torchrun` because it trains a full LLM with LoRA.

## Examples

```bash
sbatch scripts/site_b/run.sh mimic tehr_stage2 mistral icu_mortality
sbatch scripts/site_b/run.sh mimic ablation_no_time mistral icu_mortality
sbatch scripts/site_b/run.sh mimic ablation_no_value mistral icu_mortality
sbatch scripts/site_b/run.sh cprd tehr_stage1 qwen8
sbatch scripts/site_b/run.sh cprd sft mistral 6m
sbatch scripts/site_b/run.sh mimic text_embedding icu_mortality
sbatch scripts/site_a/run_cpu.sh mimic xgboost_bow hospital_readmission
sbatch scripts/site_b/run.sh mimic ablation_hash mistral icu_mortality
sbatch scripts/site_b/run.sh mimic tehr_stage2 mistral hospital_readmission evaluate
CHECKPOINT=/path/to/models/best_model.pt sbatch scripts/site_b/run.sh mimic encoder icu_mortality evaluate
```

Without Slurm, build a config then call the same entry point:

```bash
python -m src.training.config_utils build \
  --dataset mimic --experiment tehr_stage2 --model mistral --task icu_mortality \
  --out configs/_resolved/mimic-tehr_stage2-mistral-icu_mortality.yaml
python -m src.pipelines.run --config_filepath configs/_resolved/mimic-tehr_stage2-mistral-icu_mortality.yaml
```

LLM jobs that need DDP use `torchrun … src/pipelines/run.py` (the dispatcher does this).

## Analysis (CPU)

```bash
sbatch scripts/site_b/run_cpu.sh plot_ablation
EMBEDDINGS_DIR=... CONFIG_PATH=... sbatch scripts/site_b/run_cpu.sh hdbscan
CLUSTERS_CSV=... CONFIG_PATH=... sbatch scripts/site_b/run_cpu.sh plot_labs
SUMMARY_CSV=... sbatch scripts/site_b/run_cpu.sh plot_rr
```

`plot_ablation` writes `analysis_out/ablations/auroc_dot_1x4.png` from `analysis_out/ablations/results.csv`. Re-plot HDBSCAN figures from saved CSVs with `python -m src.analysis.hdbscan --plot-only`.

```bash
python -m src.analysis.plot_ablation_auroc --style dot_1x4
python -m src.analysis.hdbscan --config-path ... --embeddings-dir ... --output-dir ...
python -m src.analysis.plot_lab_violins --clusters-csv ... --config-path ... --output-path ...
python -m src.analysis.clinical_heatmap_terms --summary-csv ... --output-dir ...
```

Resolved configs are written to `configs/_resolved/` (gitignored). `configs/local.yaml` and `scripts/*/local.env` stay gitignored.
