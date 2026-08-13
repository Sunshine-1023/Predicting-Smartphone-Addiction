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

- `lightgbm_masked_v3.yaml` — current winning LightGBM recipe (v2 features +
  directed search params; Public LB 0.96781)
- `lightgbm_masked_v2.yaml` — previous LightGBM recipe (masking 0.20; drop
  `missing_pattern`, `categorical_interactions`, `behavioral_ratios`)
- `catboost_domain_v1.yaml` / `lightgbm_domain_v1.yaml` — original full domain
  feature groups (used by `make train*` targets)

## Archived (not used in final workflow)

See `archive/` for Optuna and finalist placeholder configs.

Feature groups: `raw`, `missingness`, `behavioral_totals`, `behavioral_ratios`,
`behavioral_deltas`, `log_counts`, `categorical_interactions`.
