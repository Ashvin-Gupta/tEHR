# TextCancEHR2

EHR text modelling for clinical prediction (MIMIC-IV focus).

## Quick start (MIMIC)

```bash
sbatch src/submissions/mimic/run.sh <experiment> [model] [task]
```

Examples:

```bash
sbatch src/submissions/mimic/run.sh classifier mistral icu_mortality
sbatch src/submissions/mimic/run.sh sft medgemma hospital_readmission
sbatch src/submissions/mimic/run.sh xgboost hospital_mortality
```

See [`src/configs/mimic/README.md`](src/configs/mimic/README.md) for the full matrix.

## Layout

| Path | Purpose |
|---|---|
| `src/configs/shared/` | Shared MIMIC data / LoRA / training defaults |
| `src/configs/mimic/` | Experiment configs |
| `src/configs/mimic/models/` | Model overlays (`mistral`, `medgemma`) |
| `src/configs/mimic/tasks/` | Task overlays (`icu_mortality`, …) |
| `src/configs/cprd/` | Archived CPRD configs |
| `src/configs/ehrshot/` | Archived EHRSHOT configs |
| `src/submissions/mimic/run.sh` | Single MIMIC job dispatcher |
