# Experiment configuration directory.
#
# Merge order (later overrides earlier):
#   configs/base.yaml
#   -> configs/profiles/{smoke|dev|final}.yaml
#   -> configs/models/{catboost|lightgbm}.yaml
#   -> configs/experiments/{name}.yaml
#   -> CLI dotted overrides (key=value)
#
# Experiment YAML files usually set feature groups and any run-specific notes.
