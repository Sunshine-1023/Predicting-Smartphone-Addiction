# Checkpoint A — Engineering foundation & official data

Status: **APPROVED**
Date prepared: 2026-08-07
Approved by: user
Approved at (local): 2026-08-07 10:55 (CST)

## Checklist (plan)

- [x] Reproducible environment (`environment.yml`, `pyproject.toml`, `make test`)
- [x] Safe ignore rules for raw data / artifacts / credentials
- [x] Typed YAML config merge (`configs/`)
- [x] Official data contract + secure download (`data/download.py`)
- [x] Validated local official CSVs under `data/raw/`
- [x] Deterministic features + `features.groups` wiring
- [x] EDA facts and figures from the package API
- [x] Unit/integration tests (non-slow) green
- [x] `make lint` green (Ruff format + lint; notebooks excluded from Ruff, still checked for outputs in CI)
- [x] **User approval to proceed to modeling / Checkpoint B**

## Evidence reviewed

1. `reports/data_validation.md` — fingerprints, scale, target rate, missingness
2. `reports/figures/` — EDA charts
3. Feature groups wired for later ablation
4. Scope: CatBoost + LightGBM only (no Logistic / Dummy)

## Next (Checkpoint B path)

1. Smoke train (small sample / few folds) to prove the full OOF pipeline
2. CatBoost + LightGBM baselines with completed artifacts under `artifacts/runs/`
3. Build submission CSV and **manually** upload to Kaggle
4. Record Public LB in `reports/submissions.csv`
5. Request Checkpoint B review before Optuna / full final runs

Suggested first command:

```bash
make train
# equivalent:
# smartphone-addiction train \
#   --profile configs/profiles/smoke.yaml \
#   --model-config configs/models/catboost.yaml
```
