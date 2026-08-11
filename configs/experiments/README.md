# Experiment configuration directory

Merge order (later overrides earlier):

```text
configs/base.yaml
  -> configs/profiles/{smoke|dev|final}.yaml
  -> configs/models/{catboost|lightgbm}.yaml
  -> configs/experiments/{name}.yaml
  -> CLI dotted overrides (key=value)
```

## Active experiment files

- `catboost_domain_v1.yaml` / `lightgbm_domain_v1.yaml` — full domain feature groups
  (used by all `make train*` targets)

## Archived (not used in final workflow)

See `archive/` for Optuna and finalist placeholder configs.

Feature groups: `raw`, `missingness`, `behavioral_totals`, `behavioral_ratios`,
`behavioral_deltas`, `log_counts`, `categorical_interactions`.
