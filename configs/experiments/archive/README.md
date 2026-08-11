# Archived experiment configs

These YAML files are kept for reference. They were **not used** in the
final competition workflow (training used `*_domain_v1.yaml` instead).

- `catboost_tune_v1.yaml` / `lightgbm_tune_v1.yaml` — Optuna stage (not run)
- `catboost_final_v1.yaml` / `lightgbm_final_v1.yaml` — finalist placeholders

To run Optuna smoke tuning:

```bash
make tune
# uses production model-config + archived experiment YAML:
# smartphone-addiction tune \
#   --profile configs/profiles/smoke.yaml \
#   --model-config configs/models/catboost.yaml \
#   --experiment configs/experiments/archive/catboost_tune_v1.yaml
```
