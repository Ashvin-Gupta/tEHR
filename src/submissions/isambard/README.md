# Isambard MIMIC experiments

Same experiment matrix as `src/submissions/mimic/`, with Isambard filesystem
paths applied automatically via `--site isambard`.

**Always submit with `sbatch`** (do not run the script directly on the login node).

```bash
# Default: 4 GPUs
sbatch src/submissions/isambard/run.sh <experiment> [model] [task]

# Override GPU count
sbatch --gpus=1 src/submissions/isambard/run.sh <experiment> [model] [task]
sbatch --gpus=4 src/submissions/isambard/run.sh <experiment> [model] [task]

# CPU jobs
sbatch src/submissions/isambard/run_cpu.sh <experiment> [task]
```

`torchrun --nproc_per_node` follows the allocated GPU count automatically.

## Examples

```bash
sbatch src/submissions/isambard/run.sh classifier mistral icu_mortality
sbatch src/submissions/isambard/run.sh sft mistral hospital_mortality
sbatch src/submissions/isambard/run.sh pretrain mistral
sbatch --gpus=1 src/submissions/isambard/run.sh classifier mistral icu_mortality
sbatch src/submissions/isambard/run_cpu.sh xgboost icu_mortality
```

## How paths work

Experiment YAMLs under `src/configs/mimic/` keep the ag619 layout.
`run.sh` builds a resolved config with:

```bash
python -m src.training.config_utils build --site isambard ...
```

which applies remaps from `src/configs/shared/sites/isambard.yaml`:

| ag619 | Isambard |
|---|---|
| `/home/ag619/MIMIC-IV-tokenisation` | `$HOME/MIMIC-IV_tokenisation` |
| `/home/ag619/MIMIC_data` | `$HOME/MIMIC_data` |
| `/home/ag619/experiments/Results/Pretrain-Mistral-7B-Mimic` | `$HOME/Pretrain-Mistral-7B-Mimic` |
| `/home/ag619/experiments/Results` | `/projects/u6ps/TextCancEHR2/experiments/Results` |
| `/home/ag619/experiments/labels_cache` | `/projects/u6ps/TextCancEHR2/experiments/labels_cache` |

Resolved configs are written to `src/configs/mimic/_resolved_isambard/`.

## Env

- Conda: `meds_train_clean_311`
- Repo: `$HOME/TextCancEHR2`
- Default GPUs: **4** (`#SBATCH --gpus=4`); override with `sbatch --gpus=N`
- `NPROC` defaults to allocated GPUs; override with `NPROC=2 sbatch --export=ALL,NPROC=2 ...`
