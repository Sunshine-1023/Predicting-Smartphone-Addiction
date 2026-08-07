# Final report: Predicting Smartphone Addiction

Numerical results must be filled only after source runs complete.
Do not tune on Public Leaderboard feedback.

## Data facts

_Pending selected completed runs._

Official files: `train.csv`, `test.csv`, `sample_submission.csv` (read-only).
Processed features land in `data/processed/` with a feature manifest.

## Validation design

Stratified K-fold OOF on local labels. Optuna uses a bounded budget
(50% stratified sample, 3 folds, seed 42, ≤20 trials). Full candidates are
re-evaluated on 5-fold CV before any submission decision.

## Baselines

_Pending CatBoost / LightGBM smoke and full-data baselines._

## Feature ablations

_Pending group-wise ablation after feature-group switches are wired._

## Tuning candidates

_Pending Optuna top-3 exports under `artifacts/tuning/`._

## Multi-seed stability

_Pending seeds 42 / 2026 / 3407 on the finalist configs._

## Model correlation

_Pending pairwise OOF correlation between CatBoost and LightGBM._

## Blend decision

Probability vs rank blend with step 0.05 on OOF only; selected weights applied
unchanged to test predictions.

## Submission history

See `reports/submissions.csv` (manual uploads only; CLI never auto-uploads).

## Limitations

No SHAP; permutation importance is sampled. Public LB is not used for tuning.
Feature-group filtering is configured but not yet fully enforced in transform.

## Next steps

Complete approved full experiments (task 14), package offline Kaggle run, and
publish a clean public repository without competition artifacts.
