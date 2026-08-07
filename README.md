# Predicting Smartphone Addiction

## English summary

Reproducible tabular binary-classification pipeline for Kaggle Playground Series
Season 6 Episode 8: **Predicting Smartphone Addiction**.

The project is Python-package first (`smartphone_addiction`). Notebooks are for
EDA and reporting only. Training targets **CatBoost** and **LightGBM** with
stratified OOF validation; Logistic/Dummy baselines are intentionally out of
scope.

**Implemented:** official data contract + secure download, deterministic features
with `features.groups`, typed YAML config, OOF runner, Optuna/blend/importance
helpers, submission builder, offline Kaggle packaging, pre-commit + GitHub Actions,
and Checkpoint A evidence under `reports/`.

**Still open (needs real runs):** smoke/baseline training artifacts, first accepted
Kaggle submission, Optuna/ablation/final multi-seed results, and public release
checklist (Checkpoint B/C).

## 中文说明

本仓库用于 Kaggle 比赛 **Predicting Smartphone Addiction** 的可复现工程。

- **目标**：表格二分类，指标 ROC-AUC；以学习可信 ML 工程流程为主。
- **模型范围**：CatBoost + LightGBM（不做 Logistic / Dummy 基线）。
- **数据**：只用官方 `train.csv` / `test.csv` / `sample_submission.csv`；
  原始文件只读，处理后的特征写在 `data/processed/`。
- **验证**：分层 K 折 OOF；正式阶段可多随机种子复核。
- **检查点 A 材料**：见 `reports/checkpoint_a.md` 与 `reports/data_validation.md`。

设计与实施文档见 `plan/`。

## 环境安装（Miniconda + Python 3.11）

```bash
cd "/path/to/Predicting Smartphone Addiction"
conda env create -f environment.yml
conda activate smartphone-addiction
```

若环境已存在：

```bash
conda activate smartphone-addiction
make setup-update
# 可选：pre-commit install
```

验证：

```bash
python -c "import smartphone_addiction; print(smartphone_addiction.__version__)"
smartphone-addiction --help
make test
make lint
```

## Kaggle 凭据（安全）

```bash
mkdir -p ~/.kaggle
# 推荐：新版 access_token，或传统 kaggle.json；权限必须 600
chmod 600 ~/.kaggle/access_token   # 或 ~/.kaggle/kaggle.json

# 接受比赛规则后：
smartphone-addiction data download --output-dir data/raw
smartphone-addiction data validate --data-dir data/raw
make validate-data   # 写入 reports/data_validation.md + figures
```

**不要**把凭据、原始/处理后的大数据、`artifacts/`、`submissions/` 提交到 GitHub。

## 常用命令

```bash
make validate-data
make features
make train
make train-raw
make train-lgbm
make train-lgbm-raw
make train-dev
make train-lgbm-dev
make train-final
make train-lgbm-final
make tune
make package CONFIG=configs/experiments/catboost_domain_v1.yaml
make submission RUN_DIR=artifacts/runs/<run>
make test
make lint
make format
make build
```

CLI 示例：

```bash
smartphone-addiction train -p configs/profiles/smoke.yaml -m configs/models/catboost.yaml -e configs/experiments/catboost_domain_v1.yaml
smartphone-addiction train -p configs/profiles/smoke.yaml -m configs/models/lightgbm.yaml -e configs/experiments/lightgbm_domain_v1.yaml
smartphone-addiction features build -g raw -g missingness
smartphone-addiction blend --runs <run_a> --runs <run_b>
smartphone-addiction submission build --run artifacts/blends/<blend_dir>
smartphone-addiction evaluate-candidates -c artifacts/tuning/catboost/candidate_1.yaml -c artifacts/tuning/catboost/candidate_2.yaml
smartphone-addiction promote --selection artifacts/tuning/evaluation/selection.json -o configs/experiments/catboost_final_v2.yaml
smartphone-addiction importance --run artifacts/runs/<run>
smartphone-addiction package kaggle --config configs/experiments/catboost_final_v1.yaml
```

## 配置合并顺序

```text
configs/base.yaml
  -> configs/profiles/{smoke|dev|final}.yaml
  -> configs/models/{catboost|lightgbm}.yaml
  -> configs/experiments/{name}.yaml
  -> CLI overrides
```

特征组（可消融）：`raw` / `missingness` / `behavioral_totals` /
`behavioral_ratios` / `behavioral_deltas` / `log_counts` /
`categorical_interactions`。

Final 占位实验：`configs/experiments/catboost_final_v1.yaml`、
`lightgbm_final_v1.yaml`（参数待 Optuna + 全量复核后填入）。

## CI

`.github/workflows/ci.yml` 在 push/PR 到 `main` 时运行：Ruff、pytest（排除 slow）、
build、notebook 无输出检查、detect-secrets、禁止跟踪竞赛产物。

## 许可证

MIT License — 仅覆盖本仓库原创代码与文档，**不包含** Kaggle 比赛数据。
见 `LICENSE` 与 `data/README.md`。
