# 数据处理流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按定稿流程把官方 `train.csv` / `test.csv` 变成同一套特征的 Parquet，供 CatBoost 与 LightGBM 使用，且不读标签、不改原始 CSV。

**Architecture:** 在现有 `src/smartphone_addiction/data/` 与 `features/` 骨架上实现：先 Schema 校验与加载，再确定性特征变换（缺失信息 → 行为组合 → 比例差值 → 对数 → 类别组合），最后写出 `processed/` 与 `feature_manifest.json`。训练集与测试集共用同一 `transform`，无折内学习参数。

**Tech Stack:** Python 3.11、pandas、NumPy、PyArrow、pytest；输出 Parquet + JSON。

## Global Constraints

- 原始 `data/raw/*.csv` 只读，永不覆盖。
- 不因缺失删行；数值缺失保留 `NaN`；类别缺失填 `__MISSING__`。
- 不读取 `addicted_label` 生成任何特征。
- 除法分母为 0 或参与字段缺失 → 结果为 `NaN`，禁止 `±inf`。
- 组合特征任一必要输入缺失 → 结果为 `NaN`（类别组合除外：组合前已填 `__MISSING__`）。
- 不做标准化、SMOTE、目标编码、多项式爆炸、多版本并行处理。
- 最终产物：`processed/train_features.parquet`、`processed/test_features.parquet`、`processed/feature_manifest.json`。

## 定稿流程对照（理解核对）

```text
原始 CSV
  → 检查列名、行数、编号和答案
  → 分离 id、输入特征和 addicted_label
  → 保留数值 NaN，填充类别空值
  → 增加缺失信息、行为组合、比例、对数和类别组合
  → 检查训练集与测试集完全一致
  → 保存 Parquet 和特征清单
```

| 步骤 | 内容 | 对应模块 |
| --- | --- | --- |
| 1 | 读取并检查 | `data/schema.py`、`data/validate.py`、`data/load.py` |
| 2 | 分离 id / X / y | `data/load.py` 或 `features/base.py` |
| 3 | 缺失处理 + 缺失特征 | `features/domain.py` |
| 4 | 行为组合 | `features/domain.py` |
| 5 | 比例与差值 | `features/domain.py` |
| 6 | 对数压缩 | `features/domain.py` |
| 7 | 类别组合 | `features/domain.py` |
| 8 | 一致性检查 + 落盘 | `features/base.py` + 小脚本或 CLI |

### 特征清单（实现时必须全部覆盖）

**原始（12）**  
`age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`, `gender`, `stress_level`, `academic_work_impact`

**类别（3）**  
`gender`, `stress_level`, `academic_work_impact`（空 → `__MISSING__`）

**缺失信息**  
`missing_count`, `missing_ratio`, `missing_pattern`，以及每个原始字段的 `{col}_is_missing`

**行为组合**  
`entertainment_hours = social_media_hours + gaming_hours`  
`work_minus_entertainment = work_study_hours - entertainment_hours`（允许负值）  
`known_usage_hours = social_media_hours + gaming_hours + work_study_hours`  
`unaccounted_screen_time = daily_screen_time_hours - known_usage_hours`（允许负值）

**比例 / 差值**  
`screen_to_sleep_ratio`, `entertainment_to_screen_ratio`, `work_to_screen_ratio`, `weekend_minus_daily`, `weekend_to_daily_ratio`, `notifications_per_screen_hour`, `opens_per_screen_hour`, `opens_per_notification`, `notifications_minus_opens`

**对数**  
`log_notifications = log(1 + notifications_per_day)`, `log_app_opens = log(1 + app_opens_per_day)`（原字段保留）

**类别组合**  
`gender_x_stress`, `gender_x_impact`, `stress_x_impact`

---

### Task 1: Schema 常量与数据校验

**Files:**
- Create: `src/smartphone_addiction/data/schema.py`
- Create: `src/smartphone_addiction/data/validate.py`
- Create: `src/smartphone_addiction/errors.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_data_validation.py`

**Interfaces:**
- Produces: `FEATURE_COLUMNS: list[str]`（12 个，顺序固定）
- Produces: `CATEGORICAL_COLUMNS: list[str]`
- Produces: `NUMERIC_COLUMNS: list[str]`
- Produces: `TARGET_COLUMN = "addicted_label"`, `ID_COLUMN = "id"`
- Produces: `validate_competition_frames(train, test, sample) -> None`
- Produces: `DataValidationError`

- [ ] **Step 1: 写失败测试（合法通过 / 重复 id / 目标非法）**

```python
# tests/unit/test_data_validation.py
import pytest
from smartphone_addiction.data.validate import validate_competition_frames
from smartphone_addiction.errors import DataValidationError


def test_valid_frames_pass(competition_frames) -> None:
    train, test, sample = competition_frames
    validate_competition_frames(train, test, sample)


def test_duplicate_train_id_fails(competition_frames) -> None:
    train, test, sample = competition_frames
    train = train.copy()
    train.loc[1, "id"] = train.loc[0, "id"]
    with pytest.raises(DataValidationError, match="unique"):
        validate_competition_frames(train, test, sample)


def test_target_must_be_binary(competition_frames) -> None:
    train, test, sample = competition_frames
    train = train.copy()
    train.loc[0, "addicted_label"] = 2
    with pytest.raises(DataValidationError, match="addicted_label"):
        validate_competition_frames(train, test, sample)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_data_validation.py -q`  
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 schema + validate + 合成 fixture**

`FEATURE_COLUMNS` 顺序必须与定稿一致。`validate_competition_frames` 检查：

- train 列 = `id` + 12 特征 + `addicted_label`
- test 列 = `id` + 相同 12 特征（无目标）
- train/test 特征名与顺序一致
- `id` 唯一；目标仅为 0/1 且无缺失
- sample 的 `id` 与 test 完全一致且顺序相同
- 数值列无 `±inf`（缺失允许）

`tests/conftest.py` 生成 ≥300 行合成数据，含数值/类别缺失与两类标签。

- [ ] **Step 4: 测试通过**

Run: `python -m pytest tests/unit/test_data_validation.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/smartphone_addiction/errors.py src/smartphone_addiction/data/schema.py \
  src/smartphone_addiction/data/validate.py tests/conftest.py tests/unit/test_data_validation.py
git commit -m "feat: add competition data schema and validation"
```

---

### Task 2: 加载官方 CSV（只读）

**Files:**
- Create: `src/smartphone_addiction/data/load.py`
- Create: `tests/unit/test_load.py`

**Interfaces:**
- Consumes: `validate_competition_frames`
- Produces:

```python
@dataclass(frozen=True)
class CompetitionFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame

def load_competition_frames(directory: Path) -> CompetitionFrames:
    """Read train.csv, test.csv, sample_submission.csv; validate; never mutate files."""
```

- [ ] **Step 1: 写失败测试**

```python
def test_load_reads_three_csvs(tmp_path, competition_frames) -> None:
    train, test, sample = competition_frames
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    frames = load_competition_frames(tmp_path)
    assert len(frames.train) == len(train)
    assert list(frames.test.columns) == list(test.columns)
```

- [ ] **Step 2: 运行确认失败 → 实现 → 再跑通**

Run: `python -m pytest tests/unit/test_load.py -q`

- [ ] **Step 3: 对真实 `data/raw` 做一次人工核对（不写进自动化 CI）**

```bash
python -c "
from pathlib import Path
from smartphone_addiction.data.load import load_competition_frames
f = load_competition_frames(Path('data/raw'))
print(len(f.train), len(f.test), f.train.shape[1], f.test.shape[1])
"
```

Expected: `691369 296302 14 13`

- [ ] **Step 4: Commit**

```bash
git add src/smartphone_addiction/data/load.py tests/unit/test_load.py
git commit -m "feat: load and validate official competition CSVs"
```

---

### Task 3: 安全除法与缺失信息特征

**Files:**
- Modify: `src/smartphone_addiction/features/domain.py`
- Create: `tests/unit/test_features_missing.py`

**Interfaces:**
- Produces:

```python
def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Return NaN when either side missing or |denominator| < eps; never ±inf."""

def add_missingness_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Add missing_count, missing_ratio, missing_pattern, and {col}_is_missing. Copy-in/copy-out."""
```

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
from smartphone_addiction.features.domain import safe_divide, add_missingness_features


def test_safe_divide() -> None:
    out = safe_divide(pd.Series([4.0, 2.0, np.nan]), pd.Series([2.0, 0.0, 1.0]))
    assert out.iloc[0] == 2.0
    assert out.iloc[1:].isna().all()


def test_missingness_features() -> None:
    frame = pd.DataFrame({
        "daily_screen_time_hours": [8.0, np.nan],
        "sleep_hours": [np.nan, 7.0],
        "gender": ["Female", None],
    })
    cols = list(frame.columns)
    out = add_missingness_features(frame, cols)
    assert out.loc[0, "missing_count"] == 1
    assert out.loc[0, "sleep_hours_is_missing"] == 1
    assert out.loc[1, "gender_is_missing"] == 1
    assert "missing_pattern" in out.columns
```

`missing_pattern` 建议实现为：按固定字段顺序，用 `|` 拼接缺失列名；全不缺则为空字符串 `""`（写入 manifest 说明）。实现选定后测试锁定该约定。

- [ ] **Step 2: 实现 → 测试通过 → Commit**

```bash
git add src/smartphone_addiction/features/domain.py tests/unit/test_features_missing.py
git commit -m "feat: add safe_divide and missingness features"
```

---

### Task 4: 类别填充、行为组合、比例差值、对数、类别组合

**Files:**
- Modify: `src/smartphone_addiction/features/domain.py`
- Create: `tests/unit/test_features_domain.py`

**Interfaces:**
- Produces:

```python
MISSING_TOKEN = "__MISSING__"

def fill_categorical_missing(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame: ...
def add_behavioral_totals(frame: pd.DataFrame) -> pd.DataFrame: ...
def add_ratio_and_delta_features(frame: pd.DataFrame) -> pd.DataFrame: ...
def add_log_count_features(frame: pd.DataFrame) -> pd.DataFrame: ...
def add_categorical_interactions(frame: pd.DataFrame) -> pd.DataFrame: ...
```

行为规则（必须测）：

- totals：任一组成字段缺失 → 结果 `NaN`；`unaccounted_screen_time` 可为负。
- ratios：用 `safe_divide`；`weekend_minus_daily` / `notifications_minus_opens` 任一端缺失 → `NaN`。
- log：`np.log1p`；输入缺失 → 输出 `NaN`；保留原 `notifications_per_day` / `app_opens_per_day`。
- 类别组合在 fill 之后做：`gender.astype(str) + "_" + stress_level.astype(str)`（分隔符写入 manifest，建议 `_`）。

- [ ] **Step 1: 写覆盖上述规则的失败测试（至少 5 个断言用例）**
- [ ] **Step 2: 实现 → `pytest tests/unit/test_features_domain.py -q` PASS**
- [ ] **Step 3: Commit**

```bash
git add src/smartphone_addiction/features/domain.py tests/unit/test_features_domain.py
git commit -m "feat: add behavioral, ratio, log, and category interaction features"
```

---

### Task 5: 统一 transform（train/test 同一套）

**Files:**
- Modify: `src/smartphone_addiction/features/base.py`
- Create: `tests/unit/test_transform_pipeline.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class TransformedFrames:
    train: pd.DataFrame   # 含 id + 全部特征 + addicted_label
    test: pd.DataFrame    # 含 id + 全部特征（无 label）
    feature_columns: list[str]  # 模型输入列（不含 id/label），顺序固定
    categorical_columns: list[str]
    numeric_columns: list[str]

def transform_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> TransformedFrames:
    """Apply identical deterministic transforms to train and test. Never use addicted_label."""
```

流水线顺序（写死，与定稿一致）：

1. 复制输入，校验特征列存在  
2. 在「仅特征列」上算 missingness（类别填补前，使 `*_is_missing` 反映真实缺失）  
3. `fill_categorical_missing`  
4. behavioral totals → ratios/deltas → log → category interactions  
5. 组装列顺序；断言 train/test 特征列名、顺序、dtype 一致  
6. 断言无 `±inf`；三类原始类别无真正空值  
7. 断言 `id` 顺序与输入相同；行数不变  

- [ ] **Step 1: 测试「标签不进入特征」与「train/test 列一致」**

```python
def test_label_not_used_in_features(competition_frames) -> None:
    train, test, _ = competition_frames
    result = transform_competition_frames(train, test)
    assert "addicted_label" not in result.feature_columns
    assert result.feature_columns == [
        c for c in result.test.columns if c != "id"
    ]
    assert list(result.train.columns) == ["id", *result.feature_columns, "addicted_label"] or (
        # 允许 id 与 label 位置固定为两端，中间为 feature_columns
        set(result.train.columns) == set(["id", "addicted_label", *result.feature_columns])
    )


def test_train_test_feature_alignment(competition_frames) -> None:
    train, test, _ = competition_frames
    result = transform_competition_frames(train, test)
    train_feats = result.train[result.feature_columns]
    test_feats = result.test[result.feature_columns]
    assert list(train_feats.columns) == list(test_feats.columns)
    assert list(train_feats.dtypes) == list(test_feats.dtypes)
```

列顺序最终约定写进 `feature_manifest.json` 后锁死；实现时选定一种（推荐：`id` + features + `label`）。

- [ ] **Step 2: 实现 → 测试通过 → Commit**

```bash
git add src/smartphone_addiction/features/base.py tests/unit/test_transform_pipeline.py
git commit -m "feat: unify train/test feature transform pipeline"
```

---

### Task 6: 写出 Parquet + feature_manifest

**Files:**
- Create: `src/smartphone_addiction/features/io.py`（新建；骨架未列但落盘需要）
- Create: `data/processed/.gitkeep`（或仅运行时创建目录）
- Create: `tests/unit/test_features_io.py`
- Modify: `.gitignore`（确保 `data/processed/*.parquet` 不进 Git，可提交 `feature_manifest.json` 可选）

**Interfaces:**
- Produces:

```python
def write_processed_dataset(
    frames: TransformedFrames,
    output_dir: Path,
    version: str = "v1",
) -> dict[str, Path]:
    """Write train_features.parquet, test_features.parquet, feature_manifest.json."""
```

`feature_manifest.json` 至少包含：

```json
{
  "version": "v1",
  "id_column": "id",
  "target_column": "addicted_label",
  "raw_features": ["..."],
  "feature_columns": ["..."],
  "categorical_columns": ["..."],
  "numeric_columns": ["..."],
  "rules": {
    "numeric_missing": "keep_nan",
    "categorical_missing": "__MISSING__",
    "safe_divide_eps": 1e-12,
    "category_interaction_sep": "_"
  },
  "row_counts": {"train": 0, "test": 0}
}
```

- [ ] **Step 1: 测试往 `tmp_path` 写出并可回读，列与行数正确，无 inf**
- [ ] **Step 2: 实现 → 测试通过**
- [ ] **Step 3: 对真实数据跑一遍并核对规模**

```bash
python -c "
from pathlib import Path
from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.features.base import transform_competition_frames
from smartphone_addiction.features.io import write_processed_dataset
raw = load_competition_frames(Path('data/raw'))
frames = transform_competition_frames(raw.train, raw.test)
paths = write_processed_dataset(frames, Path('data/processed'), version='v1')
print(paths)
"
```

Expected: 三个文件生成；train 691369 行；test 296302 行；`data/raw` 文件 mtime/内容未改。

- [ ] **Step 4: Commit（不含 parquet 大数据）**

```bash
git add src/smartphone_addiction/features/io.py tests/unit/test_features_io.py .gitignore
git commit -m "feat: write processed parquet and feature manifest"
```

---

### Task 7: CLI / 脚本入口（可选但建议）

**Files:**
- Modify: `src/smartphone_addiction/cli.py`（若尚未实现完整 CLI，可先加 `scripts/build_features.py`）
- Create: `scripts/build_features.py` 或 CLI 子命令 `features build`
- Create: `tests/integration/test_build_features.py`（用合成数据）

**Interfaces:**
- Produces: 命令

```bash
python scripts/build_features.py --raw-dir data/raw --out-dir data/processed
# 或
smartphone-addiction features build --raw-dir data/raw --out-dir data/processed
```

- [ ] **Step 1: 集成测试：合成数据 → transform → 写出 → 回读校验**
- [ ] **Step 2: 实现入口 → 测试通过 → Commit**

```bash
git add scripts/build_features.py src/smartphone_addiction/cli.py \
  tests/integration/test_build_features.py
git commit -m "feat: add processed feature build entrypoint"
```

---

## 验收清单（数据处理完成标准）

- [ ] `data/raw/train.csv`、`test.csv`、`sample_submission.csv` 仍在且未被改写
- [ ] `data/processed/train_features.parquet` / `test_features.parquet` / `feature_manifest.json` 存在
- [ ] train 行数 691369，test 行数 296302；`id` 顺序与原始一致
- [ ] train/test 的 `feature_columns` 名称、顺序、dtype 一致
- [ ] 无 `±inf`；三类原始类别无真正空值（已为 `__MISSING__`）
- [ ] `feature_columns` 不含 `id`、`addicted_label`
- [ ] 单元测试全部通过；合成数据覆盖缺失、除零、负的 unaccounted、类别组合

## 明确不做（本阶段）

- 不删行、不数值统一填补、不标准化、不 SMOTE、不目标编码
- 不按 Public LB 改特征规则
- 不在本阶段训练完整模型（下一阶段：CatBoost / LightGBM 5 折 OOF）

## 与后续模型的衔接

处理后的数据直接进 CatBoost / LightGBM：

1. 从 parquet 读入；`categorical_columns` 交给模型原生类别处理  
2. 5 折分层 CV，指标 ROC-AUC（OOF）  
3. 特征是否有效以 OOF 为准，再考虑融合与提交  

---

## Spec 覆盖自检

| 定稿章节 | 对应 Task |
| --- | --- |
| §1 四原则 | Global Constraints + Task 5 |
| §2 原始结构 | Task 1 schema |
| §3 第一步检查 | Task 1–2 |
| §3 第二步分离 | Task 5 |
| §3 第三步缺失 | Task 3–4 |
| §3 第四～七步特征 | Task 4 |
| §3 第八步一致性 | Task 5–6 |
| §4 产物 | Task 6 |
| §5 不做事项 | Global Constraints |
| §6 进模型 | 验收后下一阶段（本计划止于落盘） |
