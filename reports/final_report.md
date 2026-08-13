# Final report: Predicting Smartphone Addiction

Report date (UTC): **2026-08-13**

Numerical results below come from completed local runs and manual Kaggle uploads.
Public Leaderboard scores were **not** used for tuning, masking fraction, or
feature-group decisions.

## Data facts

See `reports/data_validation.md` (validated UTC `2026-08-07T02:36:07Z`).

- Official files: `train.csv`, `test.csv`, `sample_submission.csv` (read-only)
- Train rows: **691,369**; test rows: **296,302**
- Target positive rate (train): **≈ 0.709**
- File SHA-256 fingerprints recorded in the validation report
- Processed parquet still stores the full domain set; the winning train run
  selects **34** columns (`lightgbm_masked_v2.yaml`)
- EDA figures: `reports/figures/`

## Validation design

Stratified K-fold OOF on local labels only.

| Profile | Config | Folds | Seeds | Purpose |
|---|---|---:|---|---|
| Smoke | `configs/profiles/smoke.yaml` | 2 | 42 | Pipeline wiring (5k rows) |
| Dev | `configs/profiles/dev.yaml` | 5 | 42 | Full-data single-seed baseline |
| Final | `configs/profiles/final.yaml` | 5 | 42, 2026, 3407 | Multi-seed stability check |

Winning feature groups: `raw`, `missingness`, `behavioral_totals`,
`behavioral_deltas`, `log_counts`, plus `exclude_columns: [missing_pattern]`.
Train-fold masking is enabled at fraction **0.20** (valid/test unmasked).

Earlier production runs used the full domain set (`raw` + `missingness` +
`behavioral_totals` + `behavioral_ratios` + `behavioral_deltas` + `log_counts`
+ `categorical_interactions`; 45 columns).

Model configs: `configs/models/catboost.yaml`, `configs/models/lightgbm.yaml`.

## Baselines and training runs

All completed runs are listed in `reports/experiment_summary.csv`.

| Run ID | Model | Profile | OOF AUC | Notes |
|---|---|---|---:|---|
| `20260807T070004Z-catboost-smoke-983bd13` | CatBoost | smoke | 0.92738 | 5k rows, 2-fold |
| `20260807T070027Z-lightgbm-smoke-983bd13` | LightGBM | smoke | 0.92689 | 5k rows, 2-fold |
| `20260807T071924Z-catboost-dev-6532ed6` | CatBoost | dev | 0.96169 | Full domain; all folds hit 2000 iterations |
| `20260809T021753Z-lightgbm-dev-6532ed6` | LightGBM | dev | 0.96179 | Full domain |
| `20260809T022155Z-lightgbm-final-6532ed6` | LightGBM | final | 0.96239 | Full domain; seed std ≈ 0.00011 |
| `20260809T024020Z-catboost-final-6532ed6` | CatBoost | final | 0.96184 | Full domain; seed std ≈ 0.00004 |
| `20260813T022331Z-lightgbm-dev-dcaf81b` | LightGBM | dev | 0.96371 | Winning 34-col recipe + masking 0.20 |
| `20260813T063020Z-lightgbm-final-dcaf81b` | LightGBM | final | **0.96425** | Winning recipe; 3 seeds; Public LB **0.96543** |

**Takeaways**

- Full-data OOF (~0.962 domain, **0.964** masked LightGBM) is far above smoke
  OOF (~0.927); smoke validates plumbing only.
- LightGBM beats CatBoost on OOF at default hyperparameters.
- Train-fold core-pattern masking (0.20) plus dropping `missing_pattern`,
  `categorical_interactions`, and `behavioral_ratios` is the current best recipe.

## Feature ablations

LightGBM single-seed (dev) ablations on top of masking 0.20 selected the
current 34-column set:

- **Kept:** `raw`, `missingness` flags/summaries, `behavioral_totals`,
  `behavioral_deltas`, `log_counts`
- **Dropped:** `missing_pattern`, `categorical_interactions`, `behavioral_ratios`
- Masking 0.20 on the train fold only; valid/test stay unmasked

Recipe file: `configs/experiments/lightgbm_masked_v2.yaml`.

## Tuning (Optuna)

**Not performed.** Archived candidate configs under `configs/experiments/archive/`
(`catboost_tune_v1.yaml`, `lightgbm_tune_v1.yaml`) were not used after Checkpoint B.

## Multi-seed stability

Full-domain final profile (5-fold OOF per seed):

| Model | Seed 42 | Seed 2026 | Seed 3407 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.96179 | 0.96158 | 0.96184 | 0.96174 | 0.00011 |
| CatBoost | 0.96169 | 0.96163 | 0.96158 | 0.96163 | 0.00004 |

Winning LightGBM final (`20260813T063020Z-lightgbm-final-dcaf81b`): OOF
**0.96425**, seed-mean 0.96361, seed std ≈ 0.00008.

## Model correlation

OOF Pearson correlation (full train, domain features):

| Pair | Pearson | Spearman |
|---|---:|---:|
| CatBoost-dev vs LightGBM-dev | 0.988 | 0.985 |
| CatBoost-dev vs LightGBM-final | 0.990 | 0.986 |
| LightGBM-dev vs LightGBM-final | 0.998 | 0.998 |

High correlation limited blend upside on the original domain models. After the
masked LightGBM recipe, a single model beat the earlier blend on Public LB.

## Blend decision

Blend searched on OOF only (`smartphone-addiction blend --runs ...`, step 0.05).

**Previous public-best blend (superseded)**

| Field | Value |
|---|---|
| CatBoost run | `20260807T071924Z-catboost-dev-6532ed6` |
| LightGBM run | `20260809T022155Z-lightgbm-final-6532ed6` |
| Method | **rank** (beat probability: 0.962801 vs 0.962798 OOF) |
| Weights | CatBoost **0.4**, LightGBM **0.6** |
| OOF AUC | **0.96280** |
| Public LB | 0.96368 |
| Artifact dir | `artifacts/blends/20260807T071924Z-catboost-dev-6532ed6__20260809T022155Z-lightgbm-final-6532ed6` |

That blend is kept as history. The current best submission is a **single**
LightGBM final run, not a blend.

## Submission history

Manual uploads via `make submit` / Kaggle CLI. See `reports/submissions.csv`.

| UTC time | Run / artifact | Local OOF | Public LB | Notes |
|---|---|---:|---:|---|
| 2026-08-10T06:50:32Z | LightGBM final (domain) | 0.96239 | 0.96336 | First submission |
| 2026-08-10T06:58:08Z | Blend rank 0.4/0.6 | 0.96280 | 0.96368 | Previous public best |
| 2026-08-13T07:18:09Z | LightGBM masked final | 0.96425 | **0.96543** | **Best public score** (+0.00175 vs blend) |

Local OOF tracks Public LB directionally (masked final OOF +0.00145 vs domain
LightGBM final → LB +0.00207).

Submission files (not in Git; default path is `submissions/<run_dir_name>.csv`):

- `submissions/20260813T063020Z-lightgbm-final-dcaf81b.csv` — **current best**
- `submissions/blend_cb_dev_lgb_final.csv` — previous blend
- `submissions/submission.csv` — first LightGBM final upload (historical local name)

## Limitations

- No SHAP; permutation importance is sampled AUC drop (CLI available)
- Public LB was not used for tuning, masking fraction, or group selection
- Optuna was scoped but not run
- Logistic / Dummy intentionally out of scope
- CatBoost was not retrained on the winning 34-column + masking recipe
- Makefile `train-lgbm*` still points at `lightgbm_domain_v1.yaml` (45 feats, no mask)

## Checkpoint status

- **Checkpoint A:** approved (2026-08-07)
- **Checkpoint B:** complete — smoke/dev/final runs, first Kaggle submission, LB recorded
- **Checkpoint C:** complete for original scope — multi-seed finals, blend selection
- **2026-08-13:** LightGBM masking + group ablations; Public LB **0.96543**

## Reproduce best submission

```bash
conda activate smartphone-addiction
make features   # if feature code changed

smartphone-addiction train \
  --profile configs/profiles/final.yaml \
  --model-config configs/models/lightgbm.yaml \
  --experiment configs/experiments/lightgbm_masked_v2.yaml \
  --allow-dirty

make submit \
  RUN_DIR=artifacts/runs/20260813T063020Z-lightgbm-final-dcaf81b \
  SUBMISSION_CSV=submissions/20260813T063020Z-lightgbm-final-dcaf81b.csv \
  MSG="lgbm masked_v2 final 3-seed oof=0.96425 lb=0.96543"
```

The already-submitted run is `artifacts/runs/20260813T063020Z-lightgbm-final-dcaf81b`.
A new train creates a new run id; do not overwrite that directory.
