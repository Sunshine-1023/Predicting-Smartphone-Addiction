# Final report: Predicting Smartphone Addiction

Numerical results must be filled only after source runs complete.
Do not tune on Public Leaderboard feedback.

## Data facts

See `reports/data_validation.md` (validated UTC `2026-08-07T02:36:07Z`).

- Official files: `train.csv`, `test.csv`, `sample_submission.csv` (read-only)
- Train rows: **691,369**; test rows: **296,302**
- Target positive rate (train): **≈ 0.709**
- File SHA-256 fingerprints recorded in the validation report
- Processed features: `data/processed/*.parquet` + `feature_manifest.json`
- EDA figures: `reports/figures/`

## Validation design

Stratified K-fold OOF on local labels only.

- Smoke: `configs/profiles/smoke.yaml`
- Dev: `configs/profiles/dev.yaml`
- Final: `configs/profiles/final.yaml` (5 folds × seeds `42, 2026, 3407`)
- Optuna budget: 50% stratified sample, 3 folds, seed 42, ≤20 trials;
  top candidates must be re-evaluated on full 5-fold CV before any submission decision

## Baselines

_Pending CatBoost / LightGBM smoke and full-data baselines._

Record selected run IDs in `reports/experiment_summary.csv`.

## Feature ablations

Feature groups are wired in `build_features` / `transform_competition_frames` /
`run_training` / CLI:

`raw`, `missingness`, `behavioral_totals`, `behavioral_ratios`,
`behavioral_deltas`, `log_counts`, `categorical_interactions`.

_Pending group-wise ablation runs after Checkpoint A approval._

## Tuning candidates

Configs: `configs/experiments/catboost_tune_v1.yaml`,
`configs/experiments/lightgbm_tune_v1.yaml`.

_Pending Optuna top-3 exports under `artifacts/tuning/`._

## Multi-seed stability

Finalist placeholders:

- `configs/experiments/catboost_final_v1.yaml`
- `configs/experiments/lightgbm_final_v1.yaml`

_Pending seeds 42 / 2026 / 3407 on selected finalist params._

## Model correlation

_Pending pairwise OOF correlation between CatBoost and LightGBM._

## Blend decision

Probability vs rank blend with step 0.05 on OOF only; selected weights applied
unchanged to test predictions (`smartphone-addiction blend --runs ...`).

_Pending after two completed OOF runs exist._

## Submission history

See `reports/submissions.csv` (manual uploads only; CLI never auto-uploads).

## Limitations

- No SHAP; permutation importance is sampled AUC drop
- Public LB is not used for tuning or weight search
- Logistic / Dummy intentionally out of scope

## Next steps

1. User approves Checkpoint A (`reports/checkpoint_a.md`)
2. Smoke + baseline training → first manual Kaggle submission (Checkpoint B)
3. Ablation / Optuna / final multi-seed / blend (task 14)
4. Fill this report + notebooks `02` / `03` with selected run numbers
5. Package offline Kaggle bundle and keep the public tree free of competition artifacts
