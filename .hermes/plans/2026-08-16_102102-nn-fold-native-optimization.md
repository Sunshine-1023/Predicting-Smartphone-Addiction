# NN Fold-Native Imputation Optimization Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 固化当前 fold-native imputation 的稳定融合收益，并依次验证 MC-dropout 插补不确定性与冻结 encoder 分类头，寻找能够稳定超过当前最佳 OOF `0.966863` 的方案。

**Architecture:** 保持 LightGBM masked-v3、34 个基础特征、masking 0.20 和 seed 42 folds 不变。神经 encoder 始终在 LightGBM outer fold 内使用，同一 `encoder_f` 编码 train、masked-train、valid 和 test；跨 fold 只融合最终概率，绝不拼接或平均 latent 坐标。所有后续候选都与当前动态冠军做同 ID、同 OOF 的严格比较，并设置一次实验一个假设的停止门槛。

**Tech Stack:** Python 3.11、PyTorch/MPS、LightGBM、pandas、NumPy、scikit-learn、Pydantic/YAML、Typer、pytest、Ruff、项目现有 artifact 体系。

---

## 1. 当前事实与基准

### 1.1 锁定的源产物

| 角色 | 路径 | OOF |
|---|---|---:|
| seed42 LightGBM masked-v3 | `artifacts/runs/20260813T100627Z-lightgbm-dev-80612a1` | 0.9661480649 |
| 3-seed LightGBM masked-v3 | `artifacts/runs/20260813T101620Z-lightgbm-final-80612a1` | 0.9666295747 |
| 有效 MLP reconstruction | `artifacts/reconstruction/20260815T170953Z-mlp-mae-c7fbc69` | gate1 通过 |
| fold-native imputed seed42 | `artifacts/runs/20260815T171743Z-lightgbm-dev-c7fbc69` | 0.9664336210 |

MLP reconstruction 在统一 `input_available` 语义下五个字段全部通过：

| 字段 | R² | Spearman | RMSE 改善 |
|---|---:|---:|---:|
| daily | 0.6075 | 0.7736 | 37.42% |
| weekend | 0.5307 | 0.7367 | 31.53% |
| social | 0.3926 | 0.6309 | 22.63% |
| work | 0.3129 | 0.5516 | 17.85% |
| gaming | 0.2787 | 0.5193 | 15.89% |

### 1.2 当前已验证的融合

seed42 同折融合：

```text
0.40 * seed42 masked-v3 + 0.60 * seed42 fold-native imputed
OOF = 0.966622456
folds up = 5/5
```

面向最终提交的现有最强组合：

```text
0.60 * 3-seed masked-v3 + 0.40 * seed42 fold-native imputed
OOF = 0.966863289
```

对应切片：

| 指标 | 3-seed masked-v3 | 当前融合 | 增益 |
|---|---:|---:|---:|
| overall | 0.966630 | 0.966863 | +0.000234 |
| complete | 0.973452 | 0.973540 | +0.000088 |
| incomplete | 0.956149 | 0.956618 | +0.000468 |
| low | 0.919959 | 0.920008 | +0.000049 |
| mid | 0.962902 | 0.963423 | +0.000521 |

从本计划开始，把 `0.966863289` 视为动态冠军。后续方案不能再只和 `0.966148` 比较后宣称成功。

## 2. 强制边界

- 不修改 LightGBM masked-v3 参数。
- 不重新搜索 `max_bin`、叶子数、正则、bagging 或 feature fraction。
- 不使用全局 OOF latent 表训练树模型。
- 不平均不同 encoder 的 latent；只能平均最终概率。
- 不使用 Public LB 选择特征、模型或融合权重。
- 每个新实验只允许一个明确变化，不做 Optuna。
- uncertainty 只允许 `passes=8` 一个版本，不搜索 4/8/16。
- 冻结分类头只允许一个固定结构；未过门槛不微调 encoder。
- 只有最终 OOF 候选确定后才生成一次 submission。
- 本计划不包含任何版本控制操作。
- 不承诺 Public LB 一定超过 0.97；只接受 OOF 与切片证据。

## 3. 候选与停止规则

### 3.1 候选编号

| 编号 | 候选 | 是否需要训练 |
|---|---|---|
| C0 | 3-seed masked-v3 | 已完成 |
| C1 | `0.60*C0 + 0.40*fold-native imputed` | 不需要 |
| C2 | LightGBM + imputed core + MC uncertainty | 需要 1 次 seed42 5-fold |
| C3 | 冻结 MLP encoder + 轻量分类头 | 需要 1 次 seed42 5-fold |
| C4 | C1/C2 与 C3 的固定 OOF 融合 | 不需要重新训练 |

### 3.2 C2 不确定性晋级门槛

相对当前 fold-native imputed `0.966433621`：

- OOF `>= 0.966448`；
- 至少 3/5 folds 高于当前 imputed；
- incomplete AUC `>= 0.956083`；
- low AUC不得低于当前 imputed 的 `0.918348`；
- OOF coverage 必须为 1.0；
- 特征只有 5 个 `imputed_std_*` 新列，不能同时加入 latent 或改树参数。

若 C2 通过以上门槛，再和 C0/C1 做固定粗粒度融合。若最终融合不能超过动态冠军 C1 至少 `+0.00010`，C2 不晋级最终候选。

### 3.3 C3 冻结分类头门槛

- 独立 NN OOF `>= 0.960`；
- OOF coverage 为 1.0；
- Pearson correlation 与 C0 `< 0.99`；
- 与 C0/C1 的粗网格融合必须超过当时动态冠军至少 `+0.00015`；
- 至少 3/5 seed42 folds 相对对应融合基线提升；
- incomplete AUC 不下降。

若独立 OOF `< 0.960`，立即停止，不调隐藏层、不调 dropout、不解冻 encoder。

### 3.4 唯一允许的微调条件

只有冻结分类头满足：

```text
single OOF >= 0.960
Pearson < 0.99
blend gain >= 0.00015
```

才允许解冻 encoder 最后一层，使用固定 `learning_rate=1e-4`、最多 10 epochs、patience 3 做一次确认。否则整个 NN 分类头路线停止。

## 4. 目标数据流

```text
                                  ┌──────────────────────────────┐
raw outer-train ──masking 0.20──> │ masked outer-train raw rows │
       │                          └──────────────┬───────────────┘
       │                                        │
       ├────────────────────────────────────────┤
       │                                        ▼
       │                               encoder_f + tensorizer_f
       ▼                                        │
encoder_f + tensorizer_f                        ▼
       │                              imputed values + MC std
       ▼                                        │
imputed values + MC std                         │
       └──────────────────────┬─────────────────┘
                              ▼
                  LightGBM_f train / early stop
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
       outer-valid       test encoded       OOF prediction
       encoder_f         by encoder_f        fold f only

最终：平均 5 个 test probabilities，不平均 latent
```

## 5. 实施任务

### Task 1: 为 blend 增加固定权重模式

**Objective:** 让已经确定的 `probability + fixed weight` 可以生成正式 artifact，避免 CLI 再次在同一 OOF 上搜索方法和权重。

**Files:**

- Modify: `src/smartphone_addiction/evaluation/blend.py`
- Modify: `src/smartphone_addiction/cli.py`
- Modify: `tests/unit/test_blend.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: 写固定权重单元测试**

新增测试，要求：

```python
result = evaluate_fixed_blend(
    y,
    first,
    second,
    first_weight=0.60,
    method="probability",
)
assert result.first_weight == pytest.approx(0.60)
assert result.second_weight == pytest.approx(0.40)
assert result.method == "probability"
```

同时覆盖：

- weight 小于 0 或大于 1 时失败；
- 只提供 method、不提供 weight 时 CLI 失败；
- 只提供 weight、不提供 method 时 CLI 失败；
- 固定模式 artifact 记录 `selection_mode=fixed`；
- 原搜索模式继续记录 `selection_mode=oof_grid_search`，保持兼容。

**Step 2: 运行失败测试**

```bash
python -m pytest tests/unit/test_blend.py tests/integration/test_cli.py -q
```

Expected: 新固定权重测试失败，已有测试继续通过。

**Step 3: 增加固定模式接口**

建议接口：

```python
def evaluate_fixed_blend(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_weight: float,
    method: BlendMethod,
) -> BlendResult:
    ...


def blend_run_predictions(
    *,
    first_run_dir: Path | str,
    second_run_dir: Path | str,
    output_dir: Path | str,
    step: float = 0.05,
    fixed_method: BlendMethod | None = None,
    fixed_first_weight: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ...
```

固定模式只能在 `fixed_method` 和 `fixed_first_weight` 同时存在时启用。`blend_result.json`、`metrics.json`、`manifest.json` 都要记录：

```json
{
  "selection_mode": "fixed",
  "method": "probability",
  "first_weight": 0.6,
  "second_weight": 0.4
}
```

**Step 4: 扩展 CLI**

```text
--method probability|rank
--first-weight FLOAT
```

不传这两个参数时保持当前 OOF 搜索行为。

**Step 5: 运行测试**

```bash
python -m pytest tests/unit/test_blend.py tests/integration/test_cli.py -q
```

Expected: 全部通过。

### Task 2: 固化 C1 融合 artifact

**Objective:** 在不重新训练的情况下生成当前最强、可用于 submission builder 的融合产物。

**Files:**

- No source changes after Task 1
- Create by CLI: `artifacts/blends/<source-run-a>__<source-run-b>/`

**Step 1: 先生成 seed42 同折 gate artifact**

```bash
smartphone-addiction blend \
  --runs artifacts/runs/20260813T100627Z-lightgbm-dev-80612a1 \
  --runs artifacts/runs/20260815T171743Z-lightgbm-dev-c7fbc69 \
  --output-dir artifacts/blends/seed42-imputed-gate \
  --method probability \
  --first-weight 0.40
```

Expected:

```text
OOF approximately 0.966622456
first_weight=0.40
second_weight=0.60
method=probability
```

**Step 2: 验证 seed42 artifact**

- OOF 行数为 691,369；
- ID 唯一且与 source run 完全一致；
- target 完全一致；
- prediction 全部有限且位于 `[0,1]`；
- 5/5 folds 高于 seed42 masked-v3；
- incomplete、complete 指标不低于预期。

**Step 3: 生成当前最终候选 C1**

```bash
smartphone-addiction blend \
  --runs artifacts/runs/20260813T101620Z-lightgbm-final-80612a1 \
  --runs artifacts/runs/20260815T171743Z-lightgbm-dev-c7fbc69 \
  --output-dir artifacts/blends/final3-imputed \
  --method probability \
  --first-weight 0.60
```

Expected:

```text
OOF approximately 0.966863289
complete approximately 0.973540
incomplete approximately 0.956618
low approximately 0.920008
mid approximately 0.963423
```

**Step 4: 冻结 C1**

把该 artifact 记录为 `incumbent_candidate`，但暂不生成 submission。后续所有方案必须与它比较。

### Task 3: 增加 MC uncertainty 配置契约

**Objective:** 为 fold-native encoder 增加一个默认关闭、参数固定且可验证的 MC-dropout 配置。

**Files:**

- Modify: `src/smartphone_addiction/config.py`
- Modify: `configs/base.yaml`
- Create: `configs/experiments/lightgbm_imputed_uncertainty_v1.yaml`
- Modify: `tests/unit/test_config.py`

**Step 1: 写配置失败测试**

目标配置：

```yaml
features:
  neural_encoder:
    reconstruction_run: artifacts/reconstruction/20260815T170953Z-mlp-mae-c7fbc69
    include: [imputed_core]
    device: mps
    uncertainty:
      enabled: true
      passes: 8
      seed: 4200
```

验证：

- 默认 `enabled=false`；
- `enabled=true` 时 `passes>=2`；
- `passes=1` 明确报错；
- seed 必须是整数；
- 当前 `lightgbm_masked_v3.yaml` 解析结果不变。

**Step 2: 运行失败测试**

```bash
python -m pytest tests/unit/test_config.py -q
```

**Step 3: 实现配置模型**

建议：

```python
class MCUncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    passes: int = 8
    seed: int = 4200
```

在 `NeuralEncoderFeatureConfig` 中增加：

```python
uncertainty: MCUncertaintyConfig = Field(default_factory=MCUncertaintyConfig)
```

**Step 4: 写实验 YAML**

`configs/experiments/lightgbm_imputed_uncertainty_v1.yaml` 必须完整复制 `lightgbm_imputed_v1.yaml` 的树参数与特征组，只增加 uncertainty 配置，不改其他值。

**Step 5: 运行配置测试**

```bash
python -m pytest tests/unit/test_config.py tests/unit/test_runner_alignment.py -q
```

### Task 4: 实现可复现的 MC-dropout 标准差

**Objective:** 在保持 deterministic imputed mean 不变的情况下，为每个 core 字段计算 8 次 dropout 重建预测的标准差。

**Files:**

- Create: `src/smartphone_addiction/neural/uncertainty.py`
- Create: `tests/unit/test_neural_uncertainty.py`

**Step 1: 写失败测试**

覆盖以下行为：

1. 输出 shape 为 `[n_rows, 5]`；
2. 同 fold、同 seed、同输入得到相同结果；
3. `dropout=0` 时 std 接近 0；
4. `passes<2` 时失败；
5. 函数退出后模型恢复 `eval()`；
6. 只激活 `torch.nn.Dropout`，不把整个模型永久切到 train mode；
7. 输出无 NaN/Inf；
8. 不启用梯度。

**Step 2: 运行失败测试**

```bash
python -m pytest tests/unit/test_neural_uncertainty.py -q
```

**Step 3: 实现核心接口**

建议接口：

```python
def predict_reconstruction_std(
    model,
    frame: TensorizedFrame,
    tensorizer: FoldTensorizer,
    *,
    device: object,
    batch_size: int,
    passes: int,
    seed: int,
) -> np.ndarray:
    """Return raw-scale MC-dropout std with shape [rows, core5]."""
    ...
```

实现规则：

- 首先 `model.eval()`；
- 只把 Dropout module 临时设置为 train；
- 每次 pass 使用 `seed + pass_index`；
- 在 `torch.no_grad()` 中执行；
- 每个 pass 保存 raw-scale reconstruction；
- 使用 `np.std(stack, axis=0, ddof=0)`；
- `finally` 中恢复所有 module 的原 mode；
- deterministic `imputed_*` 继续来自现有 eval prediction，不能改成 MC mean，从而保证消融只增加 uncertainty。

**Step 4: 运行测试**

```bash
python -m pytest tests/unit/test_neural_uncertainty.py -q
```

### Task 5: 将 uncertainty 接入 FoldEncoder

**Objective:** 对缺失字段附加 `imputed_std_*`，对原本观察到的字段固定填 0，避免给完整样本增加无意义噪声。

**Files:**

- Modify: `src/smartphone_addiction/neural/fold_features.py`
- Modify: `tests/unit/test_fold_encoder_features.py`

**Step 1: 写失败测试**

期望列：

```text
imputed_std_daily_screen_time_hours
imputed_std_weekend_screen_time
imputed_std_social_media_hours
imputed_std_work_study_hours
imputed_std_gaming_hours
```

断言：

- 原值观察到时 std 特征严格为 `0.0`；
- 原值缺失时 std 非负且有限；
- 原有 `imputed_*` 和 `*_is_imputed` 不变；
- feature name 顺序稳定；
- masked training copies 允许重复 ID，但位置不乱；
- train/valid/test 使用相同 fold seed 规则。

**Step 2: 扩展 FoldEncoderBank**

建议 `FoldEncoderBank` 保存：

```python
uncertainty_enabled: bool
uncertainty_passes: int
uncertainty_seed: int
```

每折实际 seed：

```python
fold_seed = uncertainty_seed + fold * 1000
```

每个 frame 的调用再使用固定 offset，避免 train、masked-train、valid、test 意外获得完全相同 dropout 序列，同时保证重跑可复现：

```text
train offset = 0
masked-train offset = 100
valid offset = 200
test offset = 300
```

offset 必须由调用方显式传入，不能依赖调用顺序。

**Step 3: 实现 missing-only std**

```python
out[f"imputed_std_{field}"] = np.where(missing, std[:, field_i], 0.0)
```

不要增加 mean/max 聚合特征，不要增加 latent。

**Step 4: 运行测试**

```bash
python -m pytest \
  tests/unit/test_fold_encoder_features.py \
  tests/unit/test_neural_uncertainty.py \
  -q
```

### Task 6: 在 runner 中记录 uncertainty provenance

**Objective:** 保证正式 run 可以证明使用了哪套 encoder、MC passes、seed 和生成顺序。

**Files:**

- Modify: `src/smartphone_addiction/training/runner.py`
- Modify: `src/smartphone_addiction/cli.py`
- Modify: `tests/unit/test_runner_alignment.py`
- Modify: `tests/integration/test_smoke_pipeline.py`

**Step 1: 写失败测试**

要求 `resolved_config.yaml` 和 `manifest.json` 至少记录：

```yaml
features:
  neural_encoder_run: ...
  neural_encoder_features:
    - imputed_daily_screen_time_hours
    - daily_screen_time_hours_is_imputed
    - imputed_std_daily_screen_time_hours
  neural_encoder_uncertainty:
    enabled: true
    passes: 8
    seed: 4200
```

测试 masked copies 必须在 raw mask 之后编码，并且 `imputed_std_*` 与 masked raw 行对应。

**Step 2: 传递显式 context offset**

runner 调用顺序保持：

```python
x_masked = augment_training_fold(...)
x_train = attach_encoder_features(..., context="train")
x_masked = attach_encoder_features(..., context="masked_train")
x_valid = attach_encoder_features(..., context="valid")
x_test = attach_encoder_features(..., context="test")
```

不能先编码再 mask。

**Step 3: 运行定向测试**

```bash
python -m pytest \
  tests/unit/test_runner_alignment.py \
  tests/integration/test_smoke_pipeline.py \
  -q
```

### Task 7: C2 CPU/MPS smoke 验收

**Objective:** 在全量 5-fold 前验证 MPS、确定性、内存和 artifact 链路。

**Files:**

- No additional source changes unless smoke exposes a defect

**Step 1: 环境检查**

```bash
conda activate smartphone-addiction
cd "/Users/sunshine/Desktop/kaggle/Predicting Smartphone Addiction"
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
python -m pip check
```

Expected: MPS available 为 True，pip check 无 broken requirements。

**Step 2: 定向测试**

```bash
python -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_neural_uncertainty.py \
  tests/unit/test_fold_encoder_features.py \
  tests/unit/test_runner_alignment.py \
  -q
```

**Step 3: 运行 smoke**

使用项目现有 smoke profile：

```bash
smartphone-addiction train \
  --profile configs/profiles/smoke.yaml \
  --model-config configs/models/lightgbm.yaml \
  --experiment configs/experiments/lightgbm_imputed_uncertainty_v1.yaml
```

只验收：

- MPS 实际参与 encoder 推理；
- 15 个 neural features：5 imputed、5 flags、5 std；
- OOF coverage 1.0；
- 无 NaN/Inf；
- masked rows 在 mask 后编码；
- 内存不足时只允许 batch size 按现有规则下降，不改 passes。

smoke AUC 不参与晋级判断。

### Task 8: 运行唯一一次 C2 全量 5-fold

**Objective:** 判断 MC uncertainty 是否能把 fold-native imputed 单模型推过门槛。

**Step 1: 运行全量 seed42**

```bash
smartphone-addiction train \
  --profile configs/profiles/dev.yaml \
  --model-config configs/models/lightgbm.yaml \
  --experiment configs/experiments/lightgbm_imputed_uncertainty_v1.yaml \
  --allow-dirty
```

**Step 2: 生成比较报告**

对照：

```text
artifacts/runs/20260815T171743Z-lightgbm-dev-c7fbc69
```

报告必须包含：

- overall OOF；
- 5 个 fold AUC 与 delta；
- complete/incomplete；
- low/mid；
- test-pattern-weighted AUC；
- best iterations；
- prediction Pearson/Spearman；
- 15 个 neural features 的 gain/split importance。

**Step 3: 应用 C2 门槛**

不通过则：

- 保留代码和 run；
- 标记 `stopped_reason=uncertainty_gate_failed`；
- 不测试 passes 4/16；
- 不增加 aggregate uncertainty；
- 进入 Task 10 冻结分类头。

通过则：

- 与 C0 做固定粗网格融合；
- 新模型权重只允许 `{0.30, 0.40, 0.50}`；
- 选择宽平台中心，不选择单点极值；
- 只有融合超过 C1 `0.966863289 + 0.00010` 才替换动态冠军。

### Task 9: 全工程回归检查

**Objective:** 在进入新模型前确认 uncertainty 改动没有破坏当前稳定链路。

```bash
python -m pytest -m "not slow" -q
python -m ruff format --check .
python -m ruff check .
python -m build --no-isolation
python -m pip check
```

Expected:

- 全测试通过；
- `lightgbm_masked_v3.yaml` 仍解析为 34 列、masking 0.20；
- uncertainty 默认关闭；
- 不带 neural extra 的 LightGBM 路径不导入 torch；
- C0/C1 artifact 未被覆盖。

### Task 10: 增加冻结 encoder 分类头配置与模型

**Objective:** 产生一个可能与树模型相关性更低的直接 NN 概率模型，而不是继续把 latent 当作跨 fold 树特征。

**Files:**

- Create: `src/smartphone_addiction/neural/classifier.py`
- Create: `src/smartphone_addiction/neural/classification_config.py`
- Create: `configs/experiments/mlp_frozen_classifier_v1.yaml`
- Create: `tests/unit/test_neural_classifier.py`
- Create: `tests/unit/test_neural_classification_config.py`

**Step 1: 固定模型结构**

输入：

```text
32维 mean_latent
+ 5个 core unavailable flags
= 37维
```

分类头：

```python
nn.Sequential(
    nn.Linear(37, 32),
    nn.LayerNorm(32),
    nn.GELU(),
    nn.Dropout(0.10),
    nn.Linear(32, 1),
)
```

第一版 encoder 全部冻结：

```python
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
encoder.eval()
```

**Step 2: 固定训练配置**

```yaml
reconstruction_run: artifacts/reconstruction/20260815T170953Z-mlp-mae-c7fbc69
device: mps
batch_size: 4096
max_epochs: 50
patience: 5
learning_rate: 0.001
weight_decay: 0.0001
loss: bce_with_logits
seed: 42
masking_fraction: 0.20
```

不使用 class weight，不调隐藏层，不加入 raw 34 列。

**Step 3: 写单元测试**

覆盖：

- 输出 shape `[batch]`；
- 概率有限且在 `[0,1]`；
- backward 后 encoder gradient 全为 None；
- classifier head 至少一个参数更新；
- 同 seed 预测可复现；
- 输入行数或 latent width 错误时明确失败。

**Step 4: 运行测试**

```bash
python -m pytest \
  tests/unit/test_neural_classifier.py \
  tests/unit/test_neural_classification_config.py \
  -q
```

### Task 11: 实现 fold-local 冻结分类 OOF runner

**Objective:** 每折 encoder 与分类头成对使用，最终只拼接概率，不拼接 latent。

**Files:**

- Create: `src/smartphone_addiction/neural/classification.py`
- Modify: `src/smartphone_addiction/cli.py`
- Create: `tests/integration/test_neural_classification_pipeline.py`

**Step 1: 定义 CLI**

```bash
smartphone-addiction neural classify \
  --config configs/experiments/mlp_frozen_classifier_v1.yaml
```

支持 `--smoke` 和 `--device`，不支持运行时任意覆盖模型宽度。

**Step 2: 每折严格顺序**

对 fold `f`：

1. 读取 reconstruction `checkpoint/fold_f.pt`；
2. 验证 reconstruction folds 与 classification folds 完全一致；
3. outer-train 标签只用于 classifier head；
4. outer-train 内部划 10% stratified holdout，用于 early stopping；
5. outer-valid 不参与 epoch 选择；
6. 使用同一个 encoder_f 编码 train、masked-train、valid、test；
7. train masking 固定 0.20，并在 mask 后编码；
8. 输出 valid probability 和 test probability；
9. 五折 valid 拼接成 OOF；
10. 五折 test 概率取平均。

**Step 3: Artifact 结构**

```text
artifacts/neural_classification/<run_id>/
├── config_resolved.yaml
├── environment.json
├── fold_assignments.parquet
├── fold_metrics.csv
├── oof_predictions.parquet
├── test_predictions.parquet
├── metrics.json
├── correlation.json
├── manifest.json
└── models/
    ├── fold_0.pt
    └── ...
```

`manifest.json` 最后写入，只有五折完成、OOF coverage 1.0 后状态才是 `completed`。

**Step 4: 集成测试**

使用小型 synthetic competition frames 验证：

- 每个 ID 只出现在一个 outer-valid fold；
- OOF coverage 1.0；
- valid 标签未进入 encoder；
- test ID 顺序完整；
- 中断 artifact 不能被 blend/submission 使用；
- encoder checkpoint hash 写入 manifest。

**Step 5: 运行测试**

```bash
python -m pytest \
  tests/unit/test_neural_classifier.py \
  tests/integration/test_neural_classification_pipeline.py \
  -q
```

### Task 12: 冻结分类头 smoke 与全量门槛

**Objective:** 用一次固定实验决定是否保留独立 NN 分类路线。

**Step 1: MPS smoke**

```bash
smartphone-addiction neural classify \
  --config configs/experiments/mlp_frozen_classifier_v1.yaml \
  --device mps \
  --smoke
```

smoke 只检查：

- loss 有限；
- classifier 参数更新；
- encoder 参数不更新；
- checkpoint 可恢复；
- OOF/test artifact schema 正确。

**Step 2: 全量 seed42 5-fold**

```bash
smartphone-addiction neural classify \
  --config configs/experiments/mlp_frozen_classifier_v1.yaml \
  --device mps
```

**Step 3: 计算比较指标**

- NN 独立 overall/complete/incomplete/low/mid AUC；
- Pearson 与 Spearman：NN vs C0、NN vs C1；
- 固定 folds 的 fold AUC；
- 概率均值、标准差、min/max；
- 与 C0/C1 的概率和 rank blend 粗网格，step 固定 0.05。

**Step 4: 应用 C3 门槛**

失败时：

```text
single OOF < 0.960
或 Pearson >= 0.99 且 blend gain不足
或 incomplete下降
```

立即停止，不微调、不改变分类头宽度。

通过时才进入 Task 13。

### Task 13: 唯一一次最后一层微调确认（条件任务）

**Objective:** 仅在冻结头已经证明低相关融合价值后，确认轻微 label-aware 表示调整能否继续提升。

**Files:**

- Modify: `src/smartphone_addiction/neural/classifier.py`
- Modify: `src/smartphone_addiction/neural/classification.py`
- Create: `configs/experiments/mlp_last_block_finetune_v1.yaml`
- Modify: `tests/unit/test_neural_classifier.py`

固定设置：

```yaml
encoder_trainable: last_block_only
encoder_learning_rate: 0.0001
head_learning_rate: 0.001
max_epochs: 10
patience: 3
```

测试必须证明：

- 只有最后 encoder block 和 head 有 gradient；
- 其余 encoder 参数保持 bitwise 不变；
- outer-valid 不参与 early stopping；
- 每折仍使用对应 encoder_f。

晋级要求：最终融合相对当时动态冠军至少 `+0.00015`，且 incomplete 不下降。未达到则保留冻结版本结论，停止 NN 分类路线。

### Task 14: 最终候选选择与 submission 前审计

**Objective:** 从 C1、C2 blend、C3/C4 中选择一个 OOF 最强且切片稳定的唯一提交候选。

**Step 1: 建立最终比较表**

至少包含：

| candidate | OOF | delta vs C1 | complete | incomplete | low | mid | Pearson vs C0 | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---|

**Step 2: 选择规则**

1. OOF 优先；
2. OOF 差小于 `0.00005` 时选择结构更简单、切片更稳的候选；
3. incomplete 或 low 明显下降的候选不提交；
4. 不参考 Public LB 选择；
5. 一次只提交最终候选，不提交每个消融。

**Step 3: 全工程验收**

```bash
python -m pytest -m "not slow" -q
python -m ruff format --check .
python -m ruff check .
python -m build --no-isolation
python -m pip check
```

**Step 4: submission schema 验收**

生成 submission 前检查：

- 行数与 `sample_submission.csv` 一致；
- 列严格为 `id,addicted_label`；
- ID 顺序完全一致；
- 概率有限、无空值、位于 `[0,1]`；
- source run/blend manifest 状态为 completed；
- submission 元数据记录所有 source runs 和固定权重。

仅在这些检查全部通过后，才允许调用现有 submission builder。Kaggle 上传仍由用户明确确认后执行。

## 6. 推荐执行顺序

```text
Task 1-2   固化 C1 融合 artifact
    ↓
Task 3-9   唯一一次 MC uncertainty 实验
    ↓
是否超过 C1 至少 0.00010？
    ├── 是：更新动态冠军
    └── 否：停止 uncertainty
    ↓
Task 10-12 冻结 encoder 分类头
    ↓
独立 OOF、相关性、融合门槛是否全部通过？
    ├── 否：停止 NN 分类路线
    └── 是：Task 13 唯一一次 last-block 微调
    ↓
Task 14    选择唯一最终候选并审计
```

## 7. 预计资源

| 阶段 | 主要计算 | 预计风险 |
|---|---|---|
| 固定融合 | 读取两个 OOF/test parquet | 很低 |
| MC uncertainty smoke | 8 次 MPS dropout 推理 | 低 |
| MC uncertainty 全量 | 每折 train/masked/valid/test 各 8 次推理 + LightGBM | 中等，主要是时间 |
| 冻结分类头 | 5 折预计算 latent + 小型 head | 中等 |
| last-block 微调 | 5 折 MPS 反向传播 | 较高，仅条件触发 |

如果 MPS 内存不足，只允许降低 encoder batch size；不能同时减少 MC passes 或改变网络结构，否则实验含义发生变化。

## 8. 风险与应对

### 风险 1：MC uncertainty 只增加噪声

应对：只在缺失字段上输出 std，观察字段固定为 0；只跑一个 passes=8 版本，未过门槛立即停止。

### 风险 2：MC dropout 的随机性影响复现

应对：fold、context、pass 使用显式 seed；artifact 记录 seed 规则；同输入重复测试要求误差在固定容差内。

### 风险 3：高相关融合只在当前 OOF 权重上过拟合

应对：使用已验证的宽平台中心固定权重；不使用 0.01 精细搜索；同时检查切片与 seed42 五折方向。

### 风险 4：冻结分类头单模型较弱

这是预期风险。它只有在相关性足够低、融合明显提升时才有价值；不能因为单模型弱就无限调结构。

### 风险 5：OOF 提升仍不足以让 Public LB 到 0.97

本计划优化的是可靠 OOF 与模型互补性。Public LB 0.97 取决于隐藏测试分布，不能通过本地调门槛保证。

## 9. 完成定义

本计划只有在以下任一状态出现时结束：

1. **成功晋级**：产生一个 OOF 高于 C1 至少 0.00015、切片稳定、artifact 完整的最终候选；
2. **保留 C1**：uncertainty 与分类头均未过门槛，锁定 `0.60*C0 + 0.40*imputed`；
3. **工程阻塞**：只有测试证明 MPS/依赖/数据损坏且无法在当前范围修复时才算阻塞。

无论哪种结果，都必须留下可复算的 OOF、test prediction、配置、环境、切片指标和停止原因。
