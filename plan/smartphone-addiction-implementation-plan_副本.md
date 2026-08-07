# Predicting Smartphone Addiction 实施计划

> **面向智能体执行者：** 必须使用子技能 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐项执行本计划。步骤使用复选框（- [ ]）跟踪进度。

**目标：** 构建一个可复现、公开且以 Python 包为核心的 Kaggle 项目：训练经过验证的 Logistic、CatBoost 和 LightGBM 模型，追踪 OOF 实验，融合经过验证的预测，并安全生成提交文件。

**架构：** 使用 Typer CLI 组合经过校验的 YAML 配置，并调用职责单一的模块完成数据契约、特征工程、交叉验证、模型适配、产物存储、调参、融合、报告和提交生成。本地 Miniconda 开发环境与 Kaggle CPU 环境使用同一个 Wheel 和 CLI；Notebook 只负责分析与展示。

**技术栈：** Python 3.11、Miniconda、pandas、NumPy、PyArrow、scikit-learn、CatBoost、LightGBM、Optuna、Pydantic、PyYAML、Typer、Matplotlib、Seaborn、pytest、Ruff、pre-commit、GitHub Actions、Kaggle CLI。

---

## 交付检查点

- **检查点 A——工程基础与官方数据（任务 1～4）：** 审核环境、仓库安全、官方数据校验和 EDA 事实。
- **检查点 B——第一次有效提交（任务 5～8）：** 审核 OOF 行为、模型基线和第一次手动提交的 Kaggle 分数。
- **检查点 C——最终候选方案（任务 9～15）：** 审核调参、三随机种子稳定性、融合选择、报告、CI 和公开发布。

在检查点测试通过且用户批准记录结果之前，不得继续执行后续任务。

### 任务 1：仓库基础与可复现环境

**文件：**
- 新建：.gitignore
- 新建：LICENSE
- 新建：README.md
- 新建：environment.yml
- 新建：pyproject.toml
- 新建：Makefile
- 新建：src/smartphone_addiction/__init__.py
- 新建：tests/test_package.py
- 新建：data/README.md

- [ ] **步骤 1：编写预期失败的包测试**

~~~python
# tests/test_package.py
from smartphone_addiction import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
~~~

- [ ] **步骤 2：运行测试并确认出现预期失败**

运行：python -m pytest tests/test_package.py -q

预期：测试失败，并报告找不到 smartphone_addiction 的 ModuleNotFoundError。

- [ ] **步骤 3：创建包元数据**

创建 pyproject.toml，内容如下：

~~~toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "smartphone-addiction"
version = "0.1.0"
description = "Reproducible Kaggle pipeline for Predicting Smartphone Addiction"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = { text = "MIT" }
dependencies = [
  "catboost>=1.2,<2",
  "lightgbm>=4.3,<5",
  "numpy>=1.26,<3",
  "optuna>=4,<5",
  "pandas>=2.2,<3",
  "pyarrow>=16,<22",
  "pydantic>=2.7,<3",
  "pyyaml>=6,<7",
  "scikit-learn>=1.5,<2",
  "typer>=0.12,<1",
]

[project.optional-dependencies]
analysis = ["ipykernel>=6,<7", "jupyterlab>=4,<5", "matplotlib>=3.9,<4", "seaborn>=0.13,<1"]
dev = [
  "build>=1.2,<2",
  "detect-secrets>=1.5,<2",
  "nbstripout>=0.7,<1",
  "pre-commit>=3.7,<5",
  "pytest>=8,<9",
  "pytest-cov>=5,<7",
  "ruff>=0.6,<1",
]

[project.scripts]
smartphone-addiction = "smartphone_addiction.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/smartphone_addiction"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = [
  "model: optional CatBoost or LightGBM adapter test",
  "slow: full-data or long-running test",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
~~~

创建 environment.yml：

~~~yaml
name: smartphone-addiction
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip>=24
  - pip:
      - -e .[analysis,dev]
~~~

创建 src/smartphone_addiction/__init__.py：

~~~python
"""Predicting Smartphone Addiction competition package."""

__version__ = "0.1.0"
~~~

- [ ] **步骤 4：创建公开仓库安全规则**

创建 .gitignore：

~~~gitignore
.DS_Store
.env
.idea/
.pytest_cache/
.ruff_cache/
.coverage
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
data/raw/*
!data/raw/.gitkeep
artifacts/
submissions/
dist/
work/
outputs/
kaggle.json
*.cbm
*.joblib
*.pkl
*.parquet
~~~

创建标准 MIT 许可证，版权信息为 2026 sunshine。创建双语 README 框架，包含英文摘要、中文项目目标、安全提示，以及已批准设计文档和实施计划的链接。创建 data/README.md，说明不得重新分发比赛数据。

- [ ] **步骤 5：创建确定性的 Make 命令**

~~~makefile
.PHONY: setup test lint format build

setup:
	conda env create -f environment.yml

test:
	python -m pytest -m "not slow" -q

lint:
	python -m ruff format --check .
	python -m ruff check .

format:
	python -m ruff format .
	python -m ruff check --fix .

build:
	python -m build
~~~

- [ ] **步骤 6：创建环境并验证工程基础**

运行：

~~~bash
conda env create -f environment.yml
conda activate smartphone-addiction
python -m pytest tests/test_package.py -q
python -m ruff check .
python -m build
~~~

预期：测试通过，Ruff 以退出码 0 结束，dist 中包含 Wheel 和源码归档文件。

- [ ] **步骤 7：提交变更**

~~~bash
git add .gitignore LICENSE README.md environment.yml pyproject.toml Makefile \
  src/smartphone_addiction/__init__.py tests/test_package.py data/README.md
git commit -m "build: initialize reproducible competition project"
~~~

### 任务 2：类型化配置组合

**文件：**
- 新建：src/smartphone_addiction/errors.py
- 新建：src/smartphone_addiction/config.py
- 新建：src/smartphone_addiction/paths.py
- 新建：configs/base.yaml
- 新建：configs/profiles/smoke.yaml
- 新建：configs/profiles/dev.yaml
- 新建：configs/profiles/final.yaml
- 新建：configs/models/logistic.yaml
- 新建：configs/models/catboost.yaml
- 新建：configs/models/lightgbm.yaml
- 新建：configs/experiments/logistic_raw_v1.yaml
- 新建：tests/unit/test_config.py

- [ ] **步骤 1：编写预期失败的配置测试**

~~~python
from pathlib import Path

import pytest

from smartphone_addiction.config import CVConfig, load_config
from smartphone_addiction.errors import ConfigurationError


def test_final_profile_has_expected_seeds() -> None:
    config = load_config(
        [
            Path("configs/base.yaml"),
            Path("configs/profiles/final.yaml"),
            Path("configs/models/catboost.yaml"),
        ]
    )
    assert config.cv == CVConfig(n_splits=5, seeds=[42, 2026, 3407])


def test_cli_override_replaces_nested_value() -> None:
    config = load_config([Path("configs/base.yaml")], ["runtime.threads=2"])
    assert config.runtime.threads == 2


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config([Path("configs/base.yaml")], ["runtime.typo=2"])
~~~

- [ ] **步骤 2：运行测试并确认缺少导入模块**

运行：python -m pytest tests/unit/test_config.py -q

预期：测试失败，因为 smartphone_addiction.config 尚不存在。

- [ ] **步骤 3：实现类型化配置**

定义 Pydantic 模型 DataConfig、CVConfig、FeatureConfig、ModelConfig、RuntimeConfig、ArtifactConfig 和 RunConfig。实现以下精确的公共函数：

~~~python
def deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Return a recursive copy with override values taking precedence."""


def apply_overrides(config: dict[str, object], overrides: list[str]) -> dict[str, object]:
    """Apply validated dotted key=value overrides parsed with yaml.safe_load."""


def load_config(paths: list[Path], overrides: list[str] | None = None) -> RunConfig:
    """Merge YAML files left-to-right, apply overrides, and validate RunConfig."""
~~~

必须满足的行为：

- YAML 文件按从左到右的顺序合并。
- CLI 参数值使用 yaml.safe_load 解析，使数字和布尔值保留正确类型。
- 未知 YAML 键和未知覆盖路径抛出 ConfigurationError。
- n_splits 至少为 2；seeds 非空且不重复；threads 必须为正数。
- 相对路径以 paths.project_root() 返回的项目根目录为基准解析。

使用以下基础 YAML：

~~~yaml
competition: playground-series-s6e8
target: addicted_label
id_column: id
metric: roc_auc
data:
  directory: data/raw
artifacts:
  directory: artifacts/runs
runtime:
  threads: 4
  environment: local
cv:
  n_splits: 5
  seeds: [42]
features:
  groups: [raw]
model:
  name: logistic
  params: {}
~~~

- [ ] **步骤 4：添加运行配置档和模型默认值**

smoke.yaml 设置 sample_rows=5000、2 折和随机种子 42；dev.yaml 使用完整数据、5 折和随机种子 42；final.yaml 使用完整数据、5 折以及随机种子 42、2026、3407。模型 YAML 只包含模型名称和稳定的手工默认参数。

- [ ] **步骤 5：验证并提交**

运行：python -m pytest tests/unit/test_config.py -q

预期：测试通过。

~~~bash
git add src/smartphone_addiction/errors.py src/smartphone_addiction/config.py \
  src/smartphone_addiction/paths.py configs tests/unit/test_config.py
git commit -m "feat: add validated experiment configuration"
~~~

### 任务 3：官方数据契约与安全下载器

**文件：**
- 新建：src/smartphone_addiction/data/__init__.py
- 新建：src/smartphone_addiction/data/schema.py
- 新建：src/smartphone_addiction/data/load.py
- 新建：src/smartphone_addiction/data/validate.py
- 新建：src/smartphone_addiction/data/download.py
- 新建：tests/conftest.py
- 新建：tests/unit/test_data_validation.py
- 新建：tests/unit/test_download.py
- 修改：data/README.md

- [ ] **步骤 1：创建带固定随机种子的合成测试数据**

在 tests/conftest.py 中生成至少 300 行数据，并严格使用以下特征列：

~~~python
FEATURE_COLUMNS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
]
~~~

该 Fixture 返回训练集、测试集和样例提交 DataFrame。数据需包含数值与类别缺失值，并同时包含两个目标类别。

- [ ] **步骤 2：编写预期失败的数据校验测试**

~~~python
import pytest

from smartphone_addiction.data.validate import validate_competition_frames
from smartphone_addiction.errors import DataValidationError


def test_valid_frames_pass(competition_frames) -> None:
    validate_competition_frames(*competition_frames)


def test_duplicate_train_id_fails(competition_frames) -> None:
    train, test, sample = competition_frames
    train.loc[1, "id"] = train.loc[0, "id"]
    with pytest.raises(DataValidationError, match="train id must be unique"):
        validate_competition_frames(train, test, sample)


def test_sample_order_must_match_test(competition_frames) -> None:
    train, test, sample = competition_frames
    reversed_sample = sample.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="sample submission ids"):
        validate_competition_frames(train, test, reversed_sample)
~~~

- [ ] **步骤 3：实现 Schema、加载、校验与文件指纹**

实现：

~~~python
@dataclass(frozen=True)
class CompetitionFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_competition_frames(directory: Path) -> CompetitionFrames:
    """Read the three official CSV files and validate them."""


def validate_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """Validate exact columns, IDs, target, values, and sample alignment."""


def fingerprint_files(directory: Path) -> dict[str, str]:
    """Return SHA-256 for each official competition CSV."""
~~~

校验必须强制检查精确列集合、ID 唯一性、二元目标、训练集与测试集特征一致、样例提交 ID 顺序完全一致、测试集不包含目标列，以及数值中不存在无穷值。缺失值仍然合法。

- [ ] **步骤 4：使用伪命令执行器编写预期失败的下载器测试**

~~~python
from pathlib import Path
from subprocess import CompletedProcess

from smartphone_addiction.data.download import download_competition


def test_download_uses_expected_competition_and_directory(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    download_competition(
        "playground-series-s6e8",
        tmp_path,
        runner=fake_run,
        credential_check=False,
        extract_and_validate=False,
    )
    assert calls == [
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "playground-series-s6e8",
            "-p",
            str(tmp_path),
            "--force",
        ]
    ]
~~~

- [ ] **步骤 5：实现安全下载器**

download_competition 必须：

1. 检查 PATH 中是否存在 kaggle。
2. 检查 ~/.kaggle/kaggle.json 是否存在，并确认组用户和其他用户没有访问权限。
3. 绝不读取或记录凭据内容。
4. 将文件下载到临时目录。
5. 只解压 train.csv、test.csv 和 sample_submission.csv。
6. 校验 DataFrame 后，再将文件原子移动到 data/raw。
7. 无论成功还是失败，都清理临时文件。
8. 将认证失败和未接受比赛规则等错误转换为可操作的 DataValidationError 消息。

- [ ] **步骤 6：记录凭据配置方法**

添加以下内容：

~~~bash
mkdir -p ~/.kaggle
mv /path/to/downloaded/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions files -c playground-series-s6e8
~~~

- [ ] **步骤 7：验证并提交**

运行：python -m pytest tests/unit/test_data_validation.py tests/unit/test_download.py -q

预期：无需网络或真实凭据，测试即可通过。

~~~bash
git add src/smartphone_addiction/data tests/conftest.py tests/unit/test_data_validation.py \
  tests/unit/test_download.py data/README.md
git commit -m "feat: validate and download official competition data"
~~~

### 任务 4：确定性特征工程与 EDA 事实

**文件：**
- 新建：src/smartphone_addiction/features/__init__.py
- 新建：src/smartphone_addiction/features/base.py
- 新建：src/smartphone_addiction/features/domain.py
- 新建：tests/unit/test_features.py
- 新建：notebooks/01_eda.ipynb
- 新建：reports/figures/.gitkeep
- 新建：reports/data_validation.md

- [ ] **步骤 1：编写预期失败的特征测试**

~~~python
import numpy as np
import pandas as pd

from smartphone_addiction.features.domain import build_features, safe_divide


def test_safe_divide_rejects_zero_and_preserves_missing() -> None:
    result = safe_divide(
        pd.Series([4.0, 2.0, np.nan]),
        pd.Series([2.0, 0.0, 1.0]),
    )
    assert result.iloc[0] == 2.0
    assert result.iloc[1:].isna().all()


def test_feature_groups_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0],
            "social_media_hours": [2.0],
            "gaming_hours": [1.0],
            "work_study_hours": [3.0],
            "sleep_hours": [4.0],
            "notifications_per_day": [80.0],
            "app_opens_per_day": [40.0],
            "weekend_screen_time": [10.0],
            "age": [20.0],
            "gender": ["Female"],
            "stress_level": ["Medium"],
            "academic_work_impact": ["No"],
        }
    )
    result = build_features(frame, ["raw", "missingness", "behavioral_ratios"])
    assert result.loc[0, "screen_to_sleep_ratio"] == 2.0
    assert result.loc[0, "missing_count"] == 0
~~~

- [ ] **步骤 2：实现无状态特征组**

build_features 仅支持 raw、missingness、behavioral_totals、behavioral_ratios 和 behavioral_deltas。遇到未知特征组时拒绝执行；保留原索引；绝不修改输入。safe_divide 在分子缺失、分母缺失或分母绝对值小于 1e-12 时返回缺失值。只要任一必要输入缺失，派生总量也保持缺失。

- [ ] **步骤 3：验证特征**

运行：python -m pytest tests/unit/test_features.py -q

预期：测试通过。

- [ ] **步骤 4：通过包 API 构建 EDA**

01_eda.ipynb 通过项目包加载经过校验的 DataFrame，并报告数据规模、列、目标分布、缺失情况、数值分布、类别计数以及训练集/测试集对比。图表以明确名称导出到 reports/figures；Notebook 中不得包含模型实现。

- [ ] **步骤 5：只记录经过验证的官方事实**

使用程序输出编写 reports/data_validation.md，包含文件哈希、数据规模、目标分布、缺失数量和校验时间。如果官方文件可用，不得复制社区 Notebook 中的结论。

- [ ] **步骤 6：清除输出并提交**

~~~bash
nbstripout notebooks/01_eda.ipynb
git add src/smartphone_addiction/features tests/unit/test_features.py notebooks/01_eda.ipynb \
  reports/figures/.gitkeep reports/data_validation.md
git commit -m "feat: add validated domain features and EDA"
~~~

### 检查点 A：审核工程基础与官方数据

- [ ] 运行 make lint 和 make test。
- [ ] 配置 ~/.kaggle/kaggle.json，并将权限设为 600。
- [ ] 下载并校验官方文件。
- [ ] 展示文件哈希、数据规模、目标分布、缺失数量和 EDA 图表。
- [ ] 在开始模型工作前请求用户批准。

### 任务 5：交叉验证、指标与实验产物

**文件：**
- 新建：src/smartphone_addiction/training/__init__.py
- 新建：src/smartphone_addiction/training/cv.py
- 新建：src/smartphone_addiction/evaluation/__init__.py
- 新建：src/smartphone_addiction/evaluation/metrics.py
- 新建：src/smartphone_addiction/artifacts/__init__.py
- 新建：src/smartphone_addiction/artifacts/manifest.py
- 新建：src/smartphone_addiction/artifacts/store.py
- 新建：tests/unit/test_cv_metrics.py
- 新建：tests/unit/test_artifacts.py

- [ ] **步骤 1：编写预期失败的确定性折分测试**

~~~python
import numpy as np

from smartphone_addiction.evaluation.metrics import summarize_oof
from smartphone_addiction.training.cv import make_folds


def test_folds_are_deterministic_and_cover_every_row_once() -> None:
    y = np.array([0, 1] * 50)
    first = make_folds(y, n_splits=5, seed=42)
    second = make_folds(y, n_splits=5, seed=42)
    assert np.array_equal(first, second)
    assert sorted(np.unique(first).tolist()) == [0, 1, 2, 3, 4]


def test_oof_summary_reports_auc_and_coverage() -> None:
    y = np.array([0, 0, 1, 1])
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    summary = summarize_oof(y, predictions)
    assert summary.auc == 1.0
    assert summary.coverage == 1.0
~~~

- [ ] **步骤 2：实现折分和指标数据类**

make_folds 使用启用 shuffle 和 random_state 的 StratifiedKFold。它为每行返回一个整数折编号，并拒绝只有单一类别的目标。summarize_oof 拒绝缺失或无穷预测，并返回 AUC、覆盖率、最小值、最大值、均值和标准差。

- [ ] **步骤 3：编写预期失败的实验产物生命周期测试**

~~~python
from pathlib import Path

from smartphone_addiction.artifacts.store import ArtifactStore


def test_run_lifecycle_is_recorded_atomically(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path, slug="logistic-raw", git_sha="abc1234")
    store.start(
        config={"model": {"name": "logistic"}},
        data_hashes={"train.csv": "hash"},
    )
    assert store.manifest().status == "running"
    store.complete(metrics={"oof_auc": 0.75})
    assert store.manifest().status == "completed"
    assert not list(store.run_dir.glob("*.tmp"))
~~~

- [ ] **步骤 4：实现运行 ID、Manifest 和原子写入**

运行 ID 由 UTC 时间戳、实验短名和 Git SHA 组成。ArtifactStore 提供 create、write_json、write_yaml、write_frame、start、mark_fold_complete、complete、fail 和 resume_missing_folds。恢复运行时比较配置哈希与数据哈希。Manifest 记录 Python 版本、包版本、平台、Git SHA 与工作区状态、数据哈希、耗时、产物和状态，但不得记录环境变量或凭据内容。

- [ ] **步骤 5：验证并提交**

~~~bash
python -m pytest tests/unit/test_cv_metrics.py tests/unit/test_artifacts.py -q
git add src/smartphone_addiction/training src/smartphone_addiction/evaluation \
  src/smartphone_addiction/artifacts tests/unit/test_cv_metrics.py tests/unit/test_artifacts.py
git commit -m "feat: add deterministic CV and experiment artifacts"
~~~

### 任务 6：模型协议与三个基线适配器

**文件：**
- 新建：src/smartphone_addiction/models/__init__.py
- 新建：src/smartphone_addiction/models/base.py
- 新建：src/smartphone_addiction/models/logistic.py
- 新建：src/smartphone_addiction/models/catboost.py
- 新建：src/smartphone_addiction/models/lightgbm.py
- 新建：tests/unit/test_logistic_model.py
- 新建：tests/unit/test_catboost_model.py
- 新建：tests/unit/test_lightgbm_model.py

- [ ] **步骤 1：编写预期失败的 Logistic 和 Dummy 测试**

~~~python
import numpy as np

from smartphone_addiction.models.logistic import build_dummy, build_logistic


def test_logistic_handles_numeric_categorical_and_missing(competition_frames) -> None:
    train, _, _ = competition_frames
    x = train.drop(columns=["id", "addicted_label"])
    y = train["addicted_label"]
    model = build_logistic(random_state=42, max_iter=200)
    model.fit(x.iloc[:200], y.iloc[:200])
    prediction = model.predict_proba(x.iloc[200:])[:, 1]
    assert prediction.shape == (len(x) - 200,)
    assert np.isfinite(prediction).all()


def test_dummy_probability_matches_training_prevalence(competition_frames) -> None:
    train, _, _ = competition_frames
    model = build_dummy()
    model.fit(train[["age"]], train["addicted_label"])
    actual = model.predict_proba(train[["age"]])[:, 1].mean()
    assert actual == train["addicted_label"].mean()
~~~

- [ ] **步骤 2：实现共享模型适配器接口**

定义 fit、predict_proba、save 和 best_iteration。Dummy 使用 prior 策略。Logistic 使用 ColumnTransformer：数值特征进行中位数插补和标准化；类别特征进行众数插补，并使用可忽略未知类别的 OneHotEncoder。

- [ ] **步骤 3：编写并运行带标记的 CatBoost 适配器测试**

使用 20 次迭代、深度 4 和 2 个线程实例化 CatBoostAdapter。在 200 行合成训练数据和独立验证集上拟合，断言概率是一维有限值，并成功保存 .cbm 模型。

- [ ] **步骤 4：实现 CatBoost 预处理与早停**

复制 DataFrame，将类别缺失值填充为 __MISSING__ 并转成字符串，保留数值 NaN，传入类别列名；使用 Logloss、AUC、CPU 任务类型、配置指定的线程数和随机种子、早停以及 allow_writing_files=False。

- [ ] **步骤 5：编写并运行带标记的 LightGBM 适配器测试**

使用 30 个估计器、15 个叶子和 2 个线程。验证集中加入训练时未出现的类别，并断言可以得到有限概率且不报错。

- [ ] **步骤 6：实现只在训练折拟合的 LightGBM 类别映射**

类别水平只在训练折上拟合。缺失值映射为 __MISSING__，未见值映射为 __UNKNOWN__。使用 binary 目标、AUC 指标、配置指定的随机种子、早停回调和 joblib 持久化。

- [ ] **步骤 7：验证并提交**

~~~bash
python -m pytest tests/unit/test_logistic_model.py -q
python -m pytest tests/unit/test_catboost_model.py tests/unit/test_lightgbm_model.py -m model -q
git add src/smartphone_addiction/models tests/unit/test_logistic_model.py \
  tests/unit/test_catboost_model.py tests/unit/test_lightgbm_model.py
git commit -m "feat: add validated model adapters"
~~~

### 任务 7：端到端 OOF 训练运行器

**文件：**
- 新建：src/smartphone_addiction/training/runner.py
- 新建：tests/integration/test_smoke_pipeline.py

- [ ] **步骤 1：编写预期失败的冒烟流水线测试**

~~~python
from pathlib import Path

import pandas as pd

from smartphone_addiction.training.runner import run_training


def test_logistic_smoke_pipeline_creates_complete_oof(
    tmp_path: Path,
    competition_frames,
) -> None:
    result = run_training(
        frames=competition_frames,
        model_name="logistic",
        model_params={"max_iter": 200},
        feature_groups=["raw", "missingness"],
        n_splits=2,
        seeds=[42],
        artifact_root=tmp_path,
    )
    oof = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    assert len(oof) == len(competition_frames[0])
    assert oof["prediction"].notna().all()
    assert result.metrics["oof_coverage"] == 1.0
~~~

- [ ] **步骤 2：实现单个随机种子/折组合的执行逻辑**

对于一个随机种子/折组合：选择索引、构建特征、删除 ID 与目标列、创建全新的模型适配器、只在训练折上拟合、预测验证集和测试集、保存模型与预测、记录折指标，随后释放本折对象。

- [ ] **步骤 3：实现完整聚合**

遍历配置中的随机种子和折。确认每个随机种子下每个训练样本只有一个 OOF 预测。计算各折 AUC、每个随机种子的总体 AUC、跨种子统计，并平均测试集预测。保存解析后的配置、折分、特征名称、OOF、测试集预测、指标、模型和日志。

- [ ] **步骤 4：实现失败处理与显式恢复**

普通异常将状态标记为 failed；KeyboardInterrupt 标记为 interrupted。显式恢复时校验配置哈希和数据哈希，并且只执行缺失的随机种子/折组合。

- [ ] **步骤 5：验证并提交**

~~~bash
python -m pytest tests/integration/test_smoke_pipeline.py -q
git add src/smartphone_addiction/training/runner.py tests/integration/test_smoke_pipeline.py
git commit -m "feat: run reproducible OOF experiments"
~~~

### 任务 8：CLI、安全提交与第一次被接受的提交

**文件：**
- 新建：src/smartphone_addiction/cli.py
- 新建：src/smartphone_addiction/submission.py
- 新建：tests/unit/test_submission.py
- 新建：tests/integration/test_cli.py
- 新建：reports/submissions.csv
- 修改：Makefile

- [ ] **步骤 1：编写预期失败的提交文件测试**

~~~python
import numpy as np
import pytest

from smartphone_addiction.errors import SubmissionValidationError
from smartphone_addiction.submission import build_submission


def test_submission_preserves_sample_ids(competition_frames) -> None:
    _, test, sample = competition_frames
    result = build_submission(sample, test["id"], np.full(len(test), 0.25))
    assert result.columns.tolist() == ["id", "addicted_label"]
    assert result["id"].equals(sample["id"])


def test_submission_rejects_nonfinite_probability(competition_frames) -> None:
    _, test, sample = competition_frames
    predictions = np.full(len(test), 0.25)
    predictions[0] = np.nan
    with pytest.raises(SubmissionValidationError, match="finite"):
        build_submission(sample, test["id"], predictions)
~~~

- [ ] **步骤 2：实现 CSV 与 JSON 旁路元数据**

校验 ID 及顺序、精确表头、行数、有限值和概率范围。原子写入后重新读取并再次校验，同时将来源运行、OOF AUC、权重、时间戳和 SHA-256 写入 JSON。

- [ ] **步骤 3：编写预期失败的 CLI 帮助测试**

~~~python
from typer.testing import CliRunner

from smartphone_addiction.cli import app


def test_cli_help_lists_primary_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "data" in result.stdout
~~~

- [ ] **步骤 4：实现 Typer 命令组**

提供 data download、data validate、train、tune、blend、submission build、report experiments 和 package kaggle。将领域异常转换成简洁的非零 CLI 退出，同时保留文件日志。Makefile 目标只能调用 CLI。

- [ ] **步骤 5：验证 CLI 和提交文件**

~~~bash
python -m pytest tests/unit/test_submission.py tests/integration/test_cli.py -q
smartphone-addiction --help
smartphone-addiction data validate
~~~

- [ ] **步骤 6：在官方数据上运行模型冒烟测试**

~~~bash
smartphone-addiction train --config configs/experiments/logistic_raw_v1.yaml \
  --profile configs/profiles/smoke.yaml
smartphone-addiction train --config configs/models/catboost.yaml \
  --profile configs/profiles/smoke.yaml
smartphone-addiction train --config configs/models/lightgbm.yaml \
  --profile configs/profiles/smoke.yaml
~~~

预期：Manifest 完整，OOF/测试集预测均为有限值，并生成有效的候选提交。

- [ ] **步骤 7：生成并手动提交基线**

生成提交文件，检查 CSV 和旁路 JSON，然后请求用户手动上传。在 reports/submissions.csv 中记录 UTC 时间、运行 ID、本地 OOF AUC、Public LB 分数和备注。不得自动上传。

- [ ] **步骤 8：提交变更**

~~~bash
git add src/smartphone_addiction/cli.py src/smartphone_addiction/submission.py \
  tests/unit/test_submission.py tests/integration/test_cli.py Makefile reports/submissions.csv
git commit -m "feat: expose safe training and submission CLI"
~~~

### 检查点 B：审核完整基线流水线

- [ ] 运行 make lint 和 make test。
- [ ] 展示 Logistic、CatBoost 和 LightGBM 的指标、耗时和预测范围。
- [ ] 仅将第一次被 Kaggle 接受的分数作为合理性检查。
- [ ] 在调参和正式训练前请求用户批准。

### 任务 9：受限的 Optuna 调参

**文件：**
- 新建：src/smartphone_addiction/training/tuning.py
- 新建：configs/experiments/catboost_tune_v1.yaml
- 新建：configs/experiments/lightgbm_tune_v1.yaml
- 新建：tests/unit/test_tuning.py

- [ ] **步骤 1：编写确定性的伪 Study 测试**

使用固定采样器随机种子和确定性伪目标函数创建内存 Study。断言 Trial 数量符合配置，参数和分数已保存，候选项按分数降序排列，并能导出三个解析完成的候选配置。

- [ ] **步骤 2：实现 CatBoost 搜索空间**

搜索深度、学习率、L2 正则化、随机强度、bagging temperature 和配合早停的最大迭代次数。不得搜索随机种子、折分、目标处理或特征组。

- [ ] **步骤 3：实现 LightGBM 搜索空间**

搜索叶子数、学习率、最小子节点样本数、特征采样比例、样本采样比例与频率、L1/L2 正则化，以及配合早停的最大估计器数量。

- [ ] **步骤 4：强制执行调参预算**

使用 50% 分层样本、3 折、随机种子 42；每个模型最多 20 个 Trial；使用 SQLite 持久化；不保留每个 Trial 的模型。导出 trials.csv 和排名前三的候选 YAML。

- [ ] **步骤 5：验证并提交**

~~~bash
python -m pytest tests/unit/test_tuning.py -q
git add src/smartphone_addiction/training/tuning.py configs/experiments tests/unit/test_tuning.py
git commit -m "feat: add bounded Optuna tuning"
~~~

### 任务 10：OOF 融合、特征重要性和公开汇总

**文件：**
- 新建：src/smartphone_addiction/evaluation/blend.py
- 新建：src/smartphone_addiction/evaluation/importance.py
- 新建：src/smartphone_addiction/evaluation/report.py
- 新建：tests/unit/test_blend.py
- 新建：tests/unit/test_importance.py
- 新建：tests/unit/test_report.py
- 新建：reports/experiment_summary.csv
- 新建：reports/final_report.md

- [ ] **步骤 1：编写预期失败的融合测试**

~~~python
import numpy as np

from smartphone_addiction.evaluation.blend import search_two_model_blend


def test_blend_search_returns_valid_weights() -> None:
    y = np.array([0, 0, 1, 1])
    first = np.array([0.1, 0.4, 0.6, 0.9])
    second = np.array([0.2, 0.3, 0.8, 0.7])
    result = search_two_model_blend(y, first, second, step=0.05)
    assert 0.0 <= result.first_weight <= 1.0
    assert result.second_weight == 1.0 - result.first_weight
    assert result.auc >= 0.5
~~~

- [ ] **步骤 2：实现概率融合与排名融合比较**

要求各 OOF 文件的 ID 和目标完全一致。以 0.05 为步长搜索 0～1 的权重。报告各组成模型 AUC、预测相关性、概率融合 AUC、排名融合 AUC 和选定方法。将完全相同的方法应用于测试集预测。

- [ ] **步骤 3：实现抽样排列重要性**

使用可复现的分层样本、已保存的模型适配器、AUC 评分、配置指定的重复次数和明确特征名。保存平均下降量、标准差、样本量、运行 ID 和随机种子。不要加入 SHAP。

- [ ] **步骤 4：实现经过选择的公开实验汇总**

只发布显式选中的已完成运行 ID。固定列为 run_id、model、feature_groups、profile、seeds、folds、oof_auc_mean、oof_auc_std、duration_seconds、git_sha、status 和 notes。不得将模型或预测文件路径复制到公开报告。

- [ ] **步骤 5：构建报告结构**

final_report.md 包含经过验证的数据事实、验证设计理由、基线、特征消融、调参候选、多随机种子稳定性、模型相关性、融合决策、提交历史、限制和后续方向。只有来源运行完成后才能填写数值结果。

- [ ] **步骤 6：验证并提交**

~~~bash
python -m pytest tests/unit/test_blend.py tests/unit/test_importance.py tests/unit/test_report.py -q
git add src/smartphone_addiction/evaluation tests/unit/test_blend.py \
  tests/unit/test_importance.py tests/unit/test_report.py reports
git commit -m "feat: add blending importance and public reports"
~~~

### 任务 11：Kaggle 离线代码包

**文件：**
- 新建：scripts/package_kaggle.py
- 新建：scripts/verify_environment.py
- 新建：kaggle/run_competition.ipynb
- 新建：tests/unit/test_kaggle_bundle.py

- [ ] **步骤 1：编写可复现的代码包测试**

使用固定内容构建两次。断言两个 Manifest 只列出 Wheel、选定配置、版本要求、Git 元数据和启动器；同时断言排除 .git、凭据、数据、实验产物、提交文件、无关 Notebook 和本地绝对路径。

- [ ] **步骤 2：实现环境校验**

检查 Python 3.11 以及 NumPy、pandas、PyArrow、scikit-learn、CatBoost、LightGBM、Pydantic、PyYAML、Typer 和 Optuna 的支持版本。打印紧凑表格；如果缺少必要导入或版本不受支持，则在训练前失败。

- [ ] **步骤 3：实现确定性打包**

构建 Wheel，将 Wheel 和选定 YAML 放入暂存目录，添加包含哈希和 Git SHA 的 Manifest，创建确定性 ZIP，最后原子移动到 dist。

- [ ] **步骤 4：创建最小化 Kaggle Notebook**

Notebook 单元格只包含输入路径、解压、pip install --no-deps、环境校验、一条 CLI 训练命令、产物检查和提交文件下载。不得包含模型代码，也不得自动上传。提交前清除所有输出。

- [ ] **步骤 5：验证并提交**

~~~bash
python -m pytest tests/unit/test_kaggle_bundle.py -q
python scripts/package_kaggle.py --config configs/experiments/logistic_raw_v1.yaml
nbstripout kaggle/run_competition.ipynb
git add scripts kaggle/run_competition.ipynb tests/unit/test_kaggle_bundle.py
git commit -m "feat: package offline Kaggle execution"
~~~

### 任务 12：验证分析与最终报告 Notebook

**文件：**
- 新建：notebooks/02_validation_analysis.ipynb
- 新建：notebooks/03_final_report.ipynb
- 修改：reports/final_report.md

- [ ] **步骤 1：构建验证分析**

读取选定的 OOF 和指标，比较各折与随机种子，展示特征组消融、预测分布和模型相关性，并解释为何不使用 Public LB 作调参依据。

- [ ] **步骤 2：构建最终展示**

只读取选定的公开汇总，创建最终对比表、重要性图、融合结果和提交历史。导出图表并更新 Markdown 报告。

- [ ] **步骤 3：清除输出并强制保证包边界**

~~~bash
nbstripout notebooks/*.ipynb kaggle/run_competition.ipynb
rg -n "CatBoostClassifier|LGBMClassifier|LogisticRegression" notebooks kaggle
~~~

预期：notebooks 或 kaggle 目录下不存在模型实现类。

- [ ] **步骤 4：提交变更**

~~~bash
git add notebooks reports/final_report.md reports/figures
git commit -m "docs: add reproducible validation reports"
~~~

### 任务 13：pre-commit 与 GitHub Actions

**文件：**
- 新建：.pre-commit-config.yaml
- 新建：.secrets.baseline
- 新建：.github/workflows/ci.yml

- [ ] **步骤 1：配置 pre-commit**

添加行末空格修复、文件结尾修复、YAML 检查、大文件检查、Ruff 格式化、Ruff 静态检查、nbstripout 和 detect-secrets。从干净仓库生成基线，逐项检查所有发现，绝不把真实凭据加入基线。

- [ ] **步骤 2：配置 CI**

在向 main 分支 Push 或创建 Pull Request 时：

1. 创建 Python 3.11 Miniconda 环境。
2. 安装 .[analysis,dev]。
3. 运行 Ruff 格式检查和 Ruff 静态检查。
4. 运行 pytest，并排除 slow 测试。
5. 构建 Wheel。
6. 验证 Notebook 已清除输出。
7. 运行 detect-secrets。
8. 如果被跟踪文件匹配数据、凭据、模型、实验产物、预测或提交文件模式，则使 CI 失败。

- [ ] **步骤 3：在本地验证**

~~~bash
pre-commit run --all-files
python -m pytest -m "not slow" -q
python -m build
~~~

预期：所有命令均以退出码 0 结束。

- [ ] **步骤 4：提交变更**

~~~bash
git add .pre-commit-config.yaml .secrets.baseline .github/workflows/ci.yml
git commit -m "ci: enforce tests quality and secret safety"
~~~

### 任务 14：经过批准的完整实验与手动提交

**文件：**
- 新建：configs/experiments/catboost_domain_v1.yaml
- 新建：configs/experiments/lightgbm_domain_v1.yaml
- 新建：configs/experiments/catboost_final_v1.yaml
- 新建：configs/experiments/lightgbm_final_v1.yaml
- 修改：reports/experiment_summary.csv
- 修改：reports/final_report.md
- 修改：reports/submissions.csv

- [ ] **步骤 1：运行仅含原始特征的开发基线**

在完整数据上对 Logistic、CatBoost 和 LightGBM 运行 5 折、随机种子 42 的实验。记录选定运行 ID，并验证 OOF 覆盖率等于 1.0。

- [ ] **步骤 2：运行特征组消融**

依次单独加入 missingness、totals、ratios 和 deltas。只有当平均 AUC 提升约 0.0002，或分数持平但成本更低时，才晋级该特征组。拒绝只在单个折出现的收益。

- [ ] **步骤 3：运行受限 Optuna Study**

使用已批准的 3 折、50% 样本配置，分别对 CatBoost 和 LightGBM 执行最多 20 个 Trial。每个模型导出三个候选方案。

- [ ] **步骤 4：在完整开发交叉验证上重新评估候选方案**

在完整数据上使用 5 折和随机种子 42 运行每个候选方案。根据 AUC、折间一致性、耗时和复杂度，为每个模型选定一套配置。

- [ ] **步骤 5：运行最终 CatBoost**

使用 5 折和随机种子 42、2026、3407。确认 15 个检查点全部完成，每个随机种子的 OOF 覆盖完整，测试集预测均为有限值，Manifest 状态为已完成。

- [ ] **步骤 6：运行最终 LightGBM**

重复相同的 15 次拟合校验。

- [ ] **步骤 7：生成特征重要性与融合结果**

计算抽样排列重要性。比较 OOF 相关性、概率加权和排名平均。在查看新的 Public LB 结果之前冻结融合方案。

- [ ] **步骤 8：生成候选提交并手动上传**

请求用户最多上传计划中的基线、最佳单模型和最终融合。记录时间、运行 ID、OOF 统计、Public LB 分数和备注。不得根据榜单分数修改融合权重。

- [ ] **步骤 9：只提交轻量结果**

~~~bash
git add configs/experiments reports
git commit -m "exp: record validated final model comparison"
~~~

### 任务 15：全新环境验证与公开 GitHub 发布

**文件：**
- 修改：README.md
- 修改：reports/final_report.md

- [ ] **步骤 1：完成经过验证的文档**

README 包含英文摘要、中文安装说明、架构、安全 Kaggle 凭据配置、精确的本地与 Kaggle 命令、选定实验表、最终结果、限制和可复现性声明。每一个数值结论都必须对应一个已完成且选中的运行或提交记录。

- [ ] **步骤 2：重建环境**

~~~bash
conda env remove -n smartphone-addiction --yes
conda env create -f environment.yml
conda activate smartphone-addiction
~~~

预期：能够从全新状态成功安装环境。

- [ ] **步骤 3：运行全部发布门禁检查**

~~~bash
pre-commit run --all-files
python -m pytest -m "not slow" -q
python -m build
smartphone-addiction --help
if git ls-files | rg -q '(^data/raw/.*\.csv$|kaggle\.json$|^artifacts/|^submissions/|\.cbm$|\.parquet$)'; then
  echo "Forbidden competition artifact is tracked"
  exit 1
fi
git status --short
~~~

预期：所有检查通过；被禁止的跟踪文件搜索没有匹配项；只剩有意修改的 README/报告文件。

- [ ] **步骤 4：提交并创建标签**

~~~bash
git add README.md reports/final_report.md
git commit -m "docs: publish reproducible competition results"
git tag -a v0.1.0 -m "First reproducible competition release"
~~~

- [ ] **步骤 5：创建已批准的公开仓库**

使用 GitHub 发布流程创建公开仓库 predicting-smartphone-addiction，添加 origin，推送 main 和 v0.1.0，并等待 GitHub Actions 完成。

- [ ] **步骤 6：审核公开页面**

确认页面中没有数据、kaggle.json、环境密钥、模型、OOF/测试集预测、实验产物或提交文件；确认 README 和 CI 徽章渲染正常。

- [ ] **步骤 7：完成检查点 C**

展示 CI 状态、仓库 URL、选定运行 ID、多随机种子统计、融合决策、提交历史和限制，并请求用户批准最终交付。

## 完成标准

只有满足以下全部条件，实施才算完成：所有任务均已提交；非慢速测试和质量门禁全部通过；官方数据校验成功；Kaggle 离线代码包运行相同 CLI；至少一个提交被 Kaggle 接受；最终候选方案具有完整 OOF 和三随机种子记录；公开仓库不包含受限产物；用户批准检查点 C。
