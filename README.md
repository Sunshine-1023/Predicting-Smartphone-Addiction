# Predicting Smartphone Addiction

## English summary

Reproducible tabular binary-classification pipeline for Kaggle Playground Series
Season 6 Episode 8: **Predicting Smartphone Addiction**.

The project is Python-package first (`smartphone_addiction`). Notebooks are for
EDA and reporting only. Training targets **CatBoost** and **LightGBM** with
stratified OOF validation; Logistic/Dummy baselines are intentionally out of
scope.

**Implemented:** official data contract + secure download, deterministic features
with `features.groups`, typed YAML config, OOF runner (smoke/dev/final),
rank/probability blend, submission builder + Kaggle CLI upload (`make submit`),
offline Kaggle packaging, pre-commit + GitHub Actions.

**Completed (2026-08-13):** winning LightGBM recipe `lightgbm_masked_v3.yaml`
(masking 0.20; 34 features; `max_bin=1023`, `num_leaves=95`, `min_child_samples=200`,
`reg_alpha=1`, `reg_lambda=10`, `path_smooth=5`). Public LB **0.96781** (OOF 0.96663).
Previous recipe `lightgbm_masked_v2.yaml` scored Public LB **0.96543**. See
`reports/final_report.md`, `reports/submissions.csv`, and
`reports/experiment_summary.csv`.

**Not pursued:** Optuna tuning. Archived tune/finalist configs live under
`configs/experiments/archive/`.

## 中文说明

本仓库用于 Kaggle 比赛 **Predicting Smartphone Addiction** 的可复现工程。

- **目标**：表格二分类，指标 ROC-AUC；以学习可信 ML 工程流程为主。
- **模型范围**：CatBoost + LightGBM（不做 Logistic / Dummy 基线）。
- **数据**：只用官方 `train.csv` / `test.csv` / `sample_submission.csv`；
  原始文件只读，处理后的特征写在 `data/processed/`（clone 后需 `make features`）。
- **验证**：分层 K 折 OOF；final 阶段 5-fold × seeds `42, 2026, 3407`。
- **最佳提交**：LightGBM `lightgbm_masked_v3`（34 列 + masking 0.20 + 定向搜索参数），
  Public LB **0.96781**（OOF 0.96663）。此前 `lightgbm_masked_v2` 为 0.96543。
- **报告**：`reports/final_report.md`、`reports/data_validation.md`。

设计与实施文档见 `plan/2026-08-06-data-processing-implementation.md`。

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
make package CONFIG=configs/experiments/catboost_domain_v1.yaml
make submission RUN_DIR=artifacts/runs/<run>
make submit RUN_DIR=artifacts/runs/<run> MSG="description"
make test
make lint
make format
make build
```

CLI 示例：

```bash
smartphone-addiction train -p configs/profiles/dev.yaml -m configs/models/lightgbm.yaml -e configs/experiments/lightgbm_masked_v3.yaml
smartphone-addiction train -p configs/profiles/smoke.yaml -m configs/models/catboost.yaml -e configs/experiments/catboost_domain_v1.yaml
smartphone-addiction train -p configs/profiles/smoke.yaml -m configs/models/lightgbm.yaml -e configs/experiments/lightgbm_domain_v1.yaml
smartphone-addiction features build -g raw -g missingness
smartphone-addiction blend --runs <run_a> --runs <run_b>
smartphone-addiction submission build --run artifacts/blends/<blend_dir>
smartphone-addiction importance --run artifacts/runs/<run>
smartphone-addiction package kaggle --config configs/experiments/catboost_domain_v1.yaml
```

Optional Optuna (archived config):

```bash
make tune
```

## 配置合并顺序

```text
configs/base.yaml
  -> configs/profiles/{smoke|dev|final}.yaml
  -> configs/models/{catboost|lightgbm}.yaml
  -> configs/experiments/{name}.yaml
  -> CLI overrides
```

特征组：`raw` / `missingness` / `behavioral_totals` / `behavioral_ratios` /
`behavioral_deltas` / `log_counts` / `categorical_interactions`。

当前胜出配方是 `configs/experiments/lightgbm_masked_v3.yaml`。
`catboost_domain_v1.yaml` / `lightgbm_domain_v1.yaml` 仍是 Makefile `train*`
默认的全组 domain 配置。Optuna / finalist 占位配置见 `configs/experiments/archive/`。

## CI

`.github/workflows/ci.yml` 在 push/PR 到 `main` 时运行：Ruff、pytest（排除 slow）、
build、notebook 无输出检查、detect-secrets、禁止跟踪竞赛产物。

## 许可证

MIT License — 仅覆盖本仓库原创代码与文档，**不包含** Kaggle 比赛数据。
见 `LICENSE` 与 `data/README.md`。
