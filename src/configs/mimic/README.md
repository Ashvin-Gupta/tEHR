# MIMIC experiments

One dispatcher runs the full experiment matrix:

```bash
sbatch src/submissions/mimic/run.sh <experiment> [model] [task]
```

On Isambard, use the same args with path remaps:

```bash
sbatch src/submissions/isambard/run.sh <experiment> [model] [task]
```

See `src/submissions/isambard/README.md` and `src/configs/shared/sites/isambard.yaml`.

Model and task args can be in **any order**.

## Arguments

| Kind | Values |
|---|---|
| **experiment** | see table below |
| **model** | `mistral` \| `medgemma` \| `qwen0.6` \| `qwen8` |
| **task** | `icu_mortality` \| `hospital_mortality` \| `hospital_readmission` \| `icu_readmission` (default: `icu_mortality`) |

## Experiments

| Experiment | Model | Task | What it runs |
|---|---|---|---|
| `pretrain` | required | ignored | Continued pretraining (LoRA) |
| `classifier` | required | yes | **Exp 1** — pretrained → LoRA + linear head |
| `prompt` | required | yes | **Exp 2** — zero-shot logit prompting |
| `sft` | required | yes | **Exp 3** — SFT to emit 0/1 |
| `ablation_no_pretrain_lora` | required | yes | **5a** |
| `ablation_frozen_pretrained` | required | yes | **5b** |
| `ablation_frozen_nopretrain` | required | yes | **5c** |
| `surface_raw` | required | yes | Lexical ablation: raw codes (no CPT, like 5a) |
| `surface_scramble_hash` | required | yes | Lexical ablation: `Concept_<hash>` titles |
| `surface_scramble_shuffle` | required | yes | Lexical ablation: shuffled titles |
| `explain` | required | yes | Post-hoc grounded explanations, base LM (GPU) |
| `xgboost` | — | yes | XGBoost bag-of-words (CPU: `run_cpu.sh`) |
| `clinicalbert` | — | yes | Bio_ClinicalBERT fine-tune (`run.sh` or `run_cpu.sh`) |
| `encoder` | — | yes | Custom transformer encoder (MLM pretrain + supervised) |
| `encoder_frozen_head` | — | yes | Freeze MLM encoder; train head only (≈ LLM 5b) |
| `encoder_supervised` | — | yes | Encoder + head from scratch; supervised only (no MLM) |
| `text_embedding` | — | yes | Phrase-level Qwen3 embeddings → transformer (matched to encoder_supervised) |

## Examples

```bash
sbatch src/submissions/mimic/run.sh classifier mistral icu_mortality
sbatch src/submissions/mimic/run.sh surface_raw mistral icu_mortality
sbatch src/submissions/mimic/run.sh surface_scramble_hash mistral hospital_mortality
sbatch src/submissions/mimic/run.sh sft medgemma hospital_readmission
sbatch src/submissions/mimic/run.sh prompt mistral hospital_mortality
sbatch src/submissions/mimic/run_cpu.sh xgboost hospital_readmission
sbatch src/submissions/mimic/run.sh clinicalbert icu_mortality
sbatch src/submissions/mimic/run_cpu.sh clinicalbert icu_mortality
sbatch src/submissions/mimic/run.sh encoder   # defaults to icu_mortality
sbatch src/submissions/mimic/run.sh encoder_supervised icu_readmission
sbatch --gres=gpu:2 src/submissions/mimic/run.sh pretrain mistral
sbatch src/submissions/isambard/run.sh pretrain qwen8
```

## Lexical / analysis helpers

Surface-form ablations (`surface_raw`, `surface_scramble_*`) match **5a** (no EHR CPT;
fresh LLM + LoRA + head) so the comparison isolates lexical grounding.
Fair text control: `ablation_no_pretrain_lora`.

```bash
# Paired subgroup analysis (rare-code + length strata, prevalence %, Δ vs encoder)
sbatch src/submissions/isambard/run_cpu.sh subgroup_full icu_mortality
sbatch src/submissions/isambard/run_cpu.sh subgroup_full hospital_mortality
sbatch src/submissions/isambard/run_cpu.sh subgroup_frozen icu_mortality   # after encoder_frozen_head
sbatch src/submissions/isambard/run_cpu.sh preview_surface icu_mortality
sbatch src/submissions/isambard/run_cpu.sh qualitative_dump icu_mortality
# Stratified case pick (CPU) then grounded explanations (GPU):
sbatch src/submissions/isambard/run_cpu.sh explain_select icu_mortality
sbatch --gpus=1 src/submissions/isambard/run.sh explain mistral icu_mortality
```

## Config layout

```
src/configs/
  shared/                 # data paths, LoRA, classifier training defaults
  mimic/
    models/               # mistral.yaml, medgemma.yaml, qwen0.6.yaml, qwen8.yaml
    tasks/                # icu_mortality, hospital_mortality, hospital_readmission, icu_readmission
    <experiment>.yaml
```

`run.sh` merges shared + experiment + model + task into
`src/configs/mimic/_resolved/<experiment>-<model>-<task>.yaml`.

## Seed & resume

Defaults live in `src/configs/shared/reproducibility.yaml` (merged into every
resolved MIMIC config):

```yaml
seed: 42
training:
  seed: 42
  resume_from_last: false
```

- Change `seed` there (or override in an experiment YAML / `--seed`) for controlled runs.
- Set `training.resume_from_last: true` to auto-continue from the newest
  `checkpoint-*` under the run's `output_dir` after a 24h walltime kill.
- CLI overrides (no YAML edit needed):
  - `--resume_from_last` / `--no-resume_from_last` on pipeline scripts
  - or `RESUME_FROM_LAST=true|false sbatch ... run.sh ...`
- Classifier / SFT write under `<output_dir>/seed_<seed>/`.
- On Isambard (`max_runtime_hours: 22`), training stops after 22h from
  train start so the usual final val/test eval can finish before the 24h
  walltime kill. Override per experiment or set `null` to disable.
- Supported for HF Trainer jobs: `pretrain`, `classifier`, `sft`, ablations,
  `clinicalbert`, frozen-LLM baselines. Custom loops (`encoder`, `xgboost`)
  use the seed but do not auto-resume yet.

## Length-sorted data cache

Length sorting caches **index lists** (not text). Exact token lengths differ
slightly by tokenizer, but the order is close enough to share one cache per
task for packing efficiency:

```text
/home/ag619/experiments/labels_cache/{task}/
  MIMIC_train_sorted.pt
  MIMIC_tuning_sorted.pt
  MIMIC_held_out_sorted.pt
```

`build_mimic_config` sets `training.data_cache_dir` to that path automatically.

## After MedGemma pretrain

Edit `src/configs/mimic/models/medgemma.yaml` and set
`model.pretrained_checkpoint` to the real checkpoint path.

Set `HF_TOKEN` before MedGemma jobs.
