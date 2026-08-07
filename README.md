# Predicting Smartphone Addiction

## English summary

Reproducible tabular binary-classification pipeline for Kaggle Playground Series
Season 6 Episode 8: **Predicting Smartphone Addiction**.

The project is Python-package first (`smartphone_addiction`). Notebooks are for
EDA and reporting only. Training targets **CatBoost** and **LightGBM** with
stratified OOF validation; Logistic/Dummy baselines are intentionally out of
scope.

Core pieces already implemented:

- Official data load/validation and processed features
- Stratified CV, OOF metrics, ArtifactStore
- CatBoost / LightGBM adapters and OOF runner
- Typed YAML config + CLI (`train` / `tune` / `blend` / `submission` / `report` / `package`)
- Bounded Optuna tuning, OOF blend, permutation importance, public report helpers
- Offline Kaggle bundle helpers, EDA/validation notebooks, pre-commit + GitHub Actions

Still open for a full competition cycle: approved full-data experiments, manual
Kaggle submission scoring, feature-group switches in transform, and git publish.

## 中文说明

本仓库用于 Kaggle 比赛 **Predicting Smartphone Addiction** 的可复现工程。

- **目标**：表格二分类，指标 ROC-AUC；以学习可信 ML 工程流程为主。
- **模型范围**：CatBoost + LightGBM（不做 Logistic / Dummy 基线）。
- **数据**：只用官方 `train.csv` / `test.csv` / `sample_submission.csv`；
  原始文件只读，处理后的特征写在 `data/processed/`。
- **验证**：分层 K 折 OOF；正式阶段可多随机种子复核。

设计与实施文档见 `plan/`：

- `plan/smartphone-addiction-project-design_副本.md`
- `plan/smartphone-addiction-implementation-plan_副本.md`
- `plan/智能手机成瘾比赛-数据处理流程_副本.md`
- `plan/2026-08-06-data-processing-implementation.md`

## 环境安装（Miniconda + Python 3.11）

```bash
cd "/path/to/Predicting Smartphone Addiction"
conda env create -f environment.yml
conda activate smartphone-addiction
```

若环境已存在，只需更新可编辑安装：

```bash
conda activate smartphone-addiction
make setup-update
# 或: pip install -e ".[analysis,dev]"
```

验证：

```bash
python -c "import smartphone_addiction; print(smartphone_addiction.__version__)"
smartphone-addiction --help
make test
```

## Kaggle 凭据（安全）

1. 在 Kaggle Account → API 创建 Token。
2. 推荐任选其一（勿把 Token 提交进 Git）：

```bash
# 新版 CLI：access_token
mkdir -p ~/.kaggle
printf '%s' 'YOUR_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# 或传统 kaggle.json（权限必须 600）
# mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json
```

3. 先在比赛页接受规则，再下载：

```bash
kaggle competitions download -c playground-series-s6e8 -p data/raw --force
cd data/raw && unzip -o *.zip
```

**不要**把 `kaggle.json`、`access_token`、原始/处理后的大数据文件提交到 GitHub。

## 常用命令

```bash
make features      # 从 data/raw 生成 data/processed
make train         # smoke 训练（catboost）
make tune          # 受限 Optuna 调参
make package CONFIG=configs/experiments/catboost_domain_v1.yaml
make submission RUN_DIR=artifacts/runs/<run>
make test          # 运行测试（排除 slow）
make test-model    # CatBoost / LightGBM 适配器测试
make lint          # Ruff 检查
make format        # Ruff 自动修复
make build         # 构建 Wheel
```

CLI 也可直接：

```bash
smartphone-addiction train -p configs/profiles/smoke.yaml -m configs/models/catboost.yaml
smartphone-addiction blend --runs <run_a> --runs <run_b>
smartphone-addiction report experiments --run <run_dir>
smartphone-addiction package kaggle --config configs/experiments/catboost_domain_v1.yaml
```

特征构建也可：

```bash
python scripts/build_features.py --raw-dir data/raw --out-dir data/processed
```

## 配置合并顺序

```text
configs/base.yaml
  -> configs/profiles/{smoke|dev|final}.yaml
  -> configs/models/{catboost|lightgbm}.yaml
  -> configs/experiments/{name}.yaml
  -> CLI overrides  (例如 runtime.threads=2)
```

示例（Python）：

```python
from pathlib import Path
from smartphone_addiction.config import load_config
from smartphone_addiction.paths import project_root

root = project_root()
cfg = load_config([
    root / "configs/base.yaml",
    root / "configs/profiles/dev.yaml",
    root / "configs/models/catboost.yaml",
    root / "configs/experiments/catboost_domain_v1.yaml",
])
print(cfg.model.name, cfg.cv.seeds, cfg.data.directory)
```

默认模型为 **catboost** / **lightgbm**（不做 Logistic）。

## 项目结构（简要）

```text
configs/                     # YAML：base / profiles / models / experiments
data/raw/                    # 官方 CSV（不进 Git）
data/processed/              # parquet + feature_manifest.json
src/smartphone_addiction/    # 核心包（含 config.py / paths.py）
scripts/build_features.py    # 特征流水线入口
tests/                       # pytest
plan/                        # 设计与实施文档
```

## 许可证

MIT License — 仅覆盖本仓库原创代码与文档，**不包含** Kaggle 比赛数据。
见 `LICENSE` 与 `data/README.md`。
