# Experiment configuration directory.
#
# Merge order (later overrides earlier):
#   configs/base.yaml
#   -> configs/profiles/{smoke|dev|final}.yaml
#   -> configs/models/{catboost|lightgbm}.yaml
#   -> configs/experiments/{name}.yaml
#   -> CLI dotted overrides (key=value)
#
# Current experiment files:
#   catboost_domain_v1.yaml / lightgbm_domain_v1.yaml  — full feature groups
#   catboost_tune_v1.yaml / lightgbm_tune_v1.yaml      — Optuna stage
#     Budget comes from YAML: cv.n_splits, cv.seeds[0], tuning.sample_fraction,
#     tuning.n_trials, tuning.n_candidates (CLI --n-trials overrides n_trials).
#     Candidate YAMLs contain only model.{name,params}; trial metadata is in
#     candidate_*.meta.json. Re-evaluate with:
#       smartphone-addiction evaluate-candidates -c ...
#     then promote:
#       smartphone-addiction promote --selection ... -o configs/experiments/...
#   catboost_final_v1.yaml / lightgbm_final_v1.yaml    — finalist placeholders
#     (params to be replaced after Optuna + full 5-fold re-evaluation)
#
# Feature groups:
#   raw, missingness, behavioral_totals, behavioral_ratios, behavioral_deltas,
#   log_counts, categorical_interactions
