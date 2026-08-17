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

- `lightgbm_masked_v3.yaml` — winning LightGBM tree recipe (v2 features +
  directed search params; 3-seed Public LB 0.96781)
- `lightgbm_imputed_v1.yaml` — same tree recipe plus fold-native missing-only
  imputation; used only as the 0.40 arm of C1
- `lightgbm_masked_v2.yaml` — previous LightGBM recipe (masking 0.20; drop
  `missing_pattern`, `categorical_interactions`, `behavioral_ratios`)
- `catboost_domain_v1.yaml` / `lightgbm_domain_v1.yaml` — original full domain
  feature groups (used by `make train*` targets)

Current submitted champion is the C1 probability blend
`0.60 * 3-seed masked-v3 + 0.40 * fold-native imputed` (Public LB 0.96803).

## Archived (not used in final workflow)

See `archive/` for Optuna and finalist placeholder configs.

Feature groups: `raw`, `missingness`, `behavioral_totals`, `behavioral_ratios`,
`behavioral_deltas`, `log_counts`, `categorical_interactions`.
