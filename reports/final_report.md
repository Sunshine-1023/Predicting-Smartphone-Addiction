# Final report: Predicting Smartphone Addiction

Report date (UTC): **2026-08-10**

Numerical results below come from completed local runs and manual Kaggle uploads.
Public Leaderboard scores were **not** used for tuning or blend weight search.

## Data facts

See `reports/data_validation.md` (validated UTC `2026-08-07T02:36:07Z`).

- Official files: `train.csv`, `test.csv`, `sample_submission.csv` (read-only)
- Train rows: **691,369**; test rows: **296,302**
- Target positive rate (train): **≈ 0.709**
- File SHA-256 fingerprints recorded in the validation report
- Processed features: `data/processed/*.parquet` + `feature_manifest.json` (45 features)
- EDA figures: `reports/figures/`

## Validation design

Stratified K-fold OOF on local labels only.

| Profile | Config | Folds | Seeds | Purpose |
|---|---|---:|---|---|
| Smoke | `configs/profiles/smoke.yaml` | 2 | 42 | Pipeline wiring (5k rows) |
| Dev | `configs/profiles/dev.yaml` | 5 | 42 | Full-data single-seed baseline |
| Final | `configs/profiles/final.yaml` | 5 | 42, 2026, 3407 | Multi-seed stability check |

Feature groups (all production runs): `raw`, `missingness`, `behavioral_totals`,
`behavioral_ratios`, `behavioral_deltas`, `log_counts`, `categorical_interactions`.

Model configs: `configs/models/catboost.yaml`, `configs/models/lightgbm.yaml`.

## Baselines and training runs

All completed runs are listed in `reports/experiment_summary.csv`.

| Run ID | Model | Profile | OOF AUC | Notes |
|---|---|---|---:|---|
| `20260807T070004Z-catboost-smoke-983bd13` | CatBoost | smoke | 0.92738 | 5k rows, 2-fold |
| `20260807T070027Z-lightgbm-smoke-983bd13` | LightGBM | smoke | 0.92689 | 5k rows, 2-fold |
| `20260807T071924Z-catboost-dev-6532ed6` | CatBoost | dev | 0.96169 | Full data; all folds hit 2000 iterations |
| `20260809T021753Z-lightgbm-dev-6532ed6` | LightGBM | dev | 0.96179 | Full data |
| `20260809T022155Z-lightgbm-final-6532ed6` | LightGBM | final | 0.96239 | seed std ≈ 0.00011 |
| `20260809T024020Z-catboost-final-6532ed6` | CatBoost | final | 0.96184 | seed std ≈ 0.00004 |

**Takeaways**

- Full-data OOF (~0.962) is far above smoke OOF (~0.927); smoke validates plumbing only.
- LightGBM slightly beats CatBoost on OOF at default hyperparameters.
- CatBoost early-stopping did not trigger (iterations capped at 2000); LightGBM stopped earlier (~500–1250).

## Feature ablations

Feature groups are wired for ablation, but **group-wise ablation runs were not executed**
in this project phase. All production runs used the full domain feature set above.

## Tuning (Optuna)

**Not performed.** Archived candidate configs under `configs/experiments/archive/`
(`catboost_tune_v1.yaml`, `lightgbm_tune_v1.yaml`) were not used after Checkpoint B.

## Multi-seed stability

Final profile results (5-fold OOF per seed):

| Model | Seed 42 | Seed 2026 | Seed 3407 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.96179 | 0.96158 | 0.96184 | 0.96174 | 0.00011 |
| CatBoost | 0.96169 | 0.96163 | 0.96158 | 0.96163 | 0.00004 |

Both models are stable across seeds; LightGBM mean OOF is slightly higher.

## Model correlation

OOF Pearson correlation (full train, domain features):

| Pair | Pearson | Spearman |
|---|---:|---:|
| CatBoost-dev vs LightGBM-dev | 0.988 | 0.985 |
| CatBoost-dev vs LightGBM-final | 0.990 | 0.986 |
| LightGBM-dev vs LightGBM-final | 0.998 | 0.998 |

High correlation limits blend upside but still yielded a small LB gain.

## Blend decision

Blend searched on OOF only (`smartphone-addiction blend --runs ...`, step 0.05).

**Selected finalist blend**

| Field | Value |
|---|---|
| CatBoost run | `20260807T071924Z-catboost-dev-6532ed6` |
| LightGBM run | `20260809T022155Z-lightgbm-final-6532ed6` |
| Method | **rank** (beat probability: 0.962801 vs 0.962798 OOF) |
| Weights | CatBoost **0.4**, LightGBM **0.6** |
| OOF AUC | **0.96280** |
| Artifact dir | `artifacts/blends/20260807T071924Z-catboost-dev-6532ed6__20260809T022155Z-lightgbm-final-6532ed6` |

CatBoost-final + LightGBM-final blend was estimated at OOF ≈ 0.96282 (+0.00002) and
**was not submitted** (diminishing returns; user chose to stop experimentation).

## Submission history

Manual uploads via `make submit` / Kaggle CLI. See `reports/submissions.csv`.

| UTC time | Run / artifact | Local OOF | Public LB | Notes |
|---|---|---:|---:|---|
| 2026-08-10T06:50:32Z | LightGBM final | 0.96239 | 0.96336 | First submission |
| 2026-08-10T06:58:08Z | Blend rank 0.4/0.6 | 0.96280 | **0.96368** | **Best public score** |

Local OOF tracks Public LB directionally (blend OOF +0.00041 → LB +0.00032 vs single model).

Submission files (not in Git; default path is `submissions/<run_dir_name>.csv`):

- `submissions/submission.csv` — LightGBM final (first upload; historical local name)
- `submissions/blend_cb_dev_lgb_final.csv` — selected blend

## Limitations

- No SHAP; permutation importance is sampled AUC drop (CLI available, not run for this report)
- Public LB was not used for tuning or blend weight search
- Feature ablations and Optuna were scoped but deferred
- Logistic / Dummy intentionally out of scope
- CatBoost `iterations=2000` may be under-converged; not changed after LB success

## Checkpoint status

- **Checkpoint A:** approved (2026-08-07)
- **Checkpoint B:** complete — smoke/dev/final runs, first Kaggle submission, LB recorded
- **Checkpoint C:** complete for chosen scope — multi-seed finals, blend selection,
  submission history, and this report filled; Optuna / ablation / extra blends not pursued

## Reproduce best submission

```bash
conda activate smartphone-addiction
make features   # if feature code changed

# Prerequisite runs (already completed in artifacts/runs/)
# - artifacts/runs/20260807T071924Z-catboost-dev-6532ed6
# - artifacts/runs/20260809T022155Z-lightgbm-final-6532ed6

smartphone-addiction blend \
  --runs artifacts/runs/20260807T071924Z-catboost-dev-6532ed6 \
  --runs artifacts/runs/20260809T022155Z-lightgbm-final-6532ed6

make submit \
  RUN_DIR=artifacts/blends/20260807T071924Z-catboost-dev-6532ed6__20260809T022155Z-lightgbm-final-6532ed6 \
  SUBMISSION_CSV=submissions/blend_cb_dev_lgb_final.csv \
  MSG="blend rank w=0.4 cb-dev + 0.6 lgb-final oof=0.96280"
```
