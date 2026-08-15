# MPS Masked Autoencoder / TabM 编码器：核心字段重建能力验证计划

> 日期：2026-08-15
> 项目：Predicting Smartphone Addiction
> 本文性质：实施计划，不代表相关代码或实验已经完成

## 1. 目标

本路线暂时不直接训练新的分类器，而是先回答一个更基础的问题：

> 当核心行为字段缺失时，其余字段中是否包含足够的信息，可以把缺失字段较可靠地重建出来？

需要重建的 `core5` 为：

1. `daily_screen_time_hours`
2. `weekend_screen_time`
3. `social_media_hours`
4. `work_study_hours`
5. `gaming_hours`

如果这些字段无法从其余信息中恢复，神经网络编码器也很难真正改善当前 LightGBM 的缺失样本排序；此时应及时停止，避免直接投入长时间分类训练。只有重建能力通过预先设定的门槛，才允许导出编码向量并接入当前冠军模型。

本路线不能保证 Public LB 一定达到 0.97。它的价值是用低风险、可解释的实验，判断“神经表示学习”是否值得继续。

## 2. 当前基线与实验边界

当前锁定冠军为 `configs/experiments/lightgbm_masked_v3.yaml`：

| 指标 | 当前结果 |
|---|---:|
| seed 42 OOF AUC | 0.966148 |
| 3-seed OOF AUC | 0.966630 |
| Public LB | 0.96781 |
| complete AUC | 0.973094 |
| incomplete AUC | 0.955465 |
| low（观察到 0–2 个 core5）AUC | 0.919019 |
| mid（观察到 3–4 个 core5）AUC | 0.962274 |

固定规则：

- LightGBM masked-v3 的 34 列、masking 0.20 和树参数保持不变，作为唯一比较基线。
- 重建阶段不读取 `addicted_label` 作为模型输入或训练目标。
- 不用 Kaggle test 的字段值训练 autoencoder；只允许使用 test 的**缺失模式频率汇总**来定义人工遮盖分布。
- 所有标准化、类别词表和统计量只能在当前 outer-train fold 拟合。
- 不能用一个在全量 train 上训练的 encoder 生成 OOF 特征。
- 门槛未通过时，不训练神经分类器、不跑多 seed、不生成 submission。
- 整条路线只允许固定架构对比，不进行无边界超参数搜索。

## 3. 为什么先做重建实验

当前完整样本 AUC 已经超过 0.973，主要缺口在核心字段缺失的样本。继续调整 LightGBM 或换成高度相似的树模型，通常只能产生很小提升。

Masked autoencoder 的思路是：人为隐藏一个或多个原本存在的字段，让模型根据剩余字段预测它们。若模型能够稳定恢复 `daily_screen_time_hours`、`weekend_screen_time`、`social_media_hours` 等强字段，则说明编码器学到了行为变量之间的结构，而不只是记忆类别标签。

TabM 本身不是一种“缺失值插补算法”。这里使用的是 **TabM 风格的参数高效集成编码器**作为 masked autoencoder 的骨干网络：共享主干参数，同时保留少量成员特定参数，让多个子模型学习略有差异的表示。最终可以同时获得重建均值、成员间不确定性和低维 latent embedding。

## 4. 两道强制门槛

```text
原始数据
  -> fold-local 预处理
  -> 人工遮盖 core5
  -> MLP-MAE / TabM-MAE 重建评估
  -> [门槛一：重建是否有效？]
       未通过：停止整条路线
       通过：导出 fold-local latent
  -> LightGBM masked-v3 + latent
  -> [门槛二：OOF 是否真实提升？]
       未通过：不提交
       通过：才考虑多 seed、TabM 分类头和 submission
```

### 4.1 门槛一：核心字段重建能力

每个字段都与当前 fold 的训练中位数预测基线比较。一个字段只有同时满足以下条件才算通过：

- OOF `R² >= 0.10`；
- OOF `Spearman >= 0.30`；
- 相对 fold-train 中位数基线，`RMSE` 改善至少 10%；
- 5 个 outer fold 中至少 3 个 fold 的 RMSE 改善为正。

整体验收要求：

- 至少 3/5 个 core 字段通过；
- `daily_screen_time_hours`、`weekend_screen_time`、`social_media_hours` 三个强字段中至少 2 个通过；
- 指标必须来自固定人工遮盖验证集，不能来自训练 loss；
- MLP-MAE 和 TabM-MAE 至少一个达到上述门槛。

若两者都通过，按以下顺序选择下游 encoder：

1. 通过字段数量更多；
2. top3 字段平均 RMSE 改善更高；
3. fold 波动更小；
4. 前三项仍接近时，选择结构更简单的 MLP-MAE。

### 4.2 门槛二：分类 OOF 增益

先把 encoder 冻结，只给当前 LightGBM 增加 fold-local latent 特征。使用 seed 42、相同 5 folds，与当前 seed 42 冠军逐行比较。

晋级要求：

- OOF AUC `>= 0.966448`，即相对 0.966148 至少提升 0.00030；
- 至少 3/5 folds 上升；
- incomplete AUC 不下降；
- OOF coverage 为 1.0，行号和 ID 完全对齐。

只有这一门槛通过，才允许运行 3 seeds 和生成 Kaggle submission。

如果之后训练独立 TabM 分类头，则使用另一套门槛：

- 单模型 OOF AUC `>= 0.960`；
- 与 LightGBM OOF 预测 Pearson `< 0.99`；
- 交叉拟合 blend 相对 LightGBM 提升 `>= 0.00030`；
- blend 至少 3/5 folds 上升，incomplete AUC 不下降。

## 5. 数据与防泄漏设计

### 5.1 输入字段

第一阶段只使用原始特征：

- 数值字段：除 `id` 和 `addicted_label` 外的原始数值列；
- 类别字段：`gender`、`stress_level`、`academic_work_impact`；
- 每个原始字段的自然缺失标记；
- 每个 core5 字段的人工遮盖标记。

明确禁止把以下内容直接作为重建输入：

- `id`；
- `addicted_label`；
- 在遮盖前生成的 totals、deltas、ratios、logs；
- 任何使用全量 train 或 outer-valid 统计量生成的特征。

原因是：如果先用真实 core 值计算派生特征、再把 core 值遮掉，模型可以从派生列反推出真实值，得到虚假的高重建分数。第一版最稳妥的做法是完全从 raw 列构造神经网络输入。后续若要加入派生列，必须在遮盖后重新计算，并另做一次消融，不能替换第一版诊断结果。

### 5.2 fold-local 预处理

沿用当前 seed 42 的 5 折 `StratifiedKFold` 划分，以便后续和 LightGBM 逐 fold 对齐。每个 fold 中：

1. 只在 outer-train 拟合数值中位数、均值和标准差；
2. 数值列先标准化，自然缺失和人工遮盖位置填 0；
3. 每个数值列同时提供 `observed_flag`；
4. core5 额外提供 `artificial_mask_flag`，避免把自然缺失和训练遮盖混为一谈；
5. 类别词表只在 outer-train 建立，保留 `__MISSING__` 和 `__UNKNOWN__`；
6. outer-valid 只调用 transform，不允许 fit；
7. 保存每折预处理器，保证 valid/test 使用同一套变换。

### 5.3 人工遮盖生成

监督重建只对“原本存在、后来被人工遮盖”的值计算 loss。

- 每次从具备目标真值的 outer-train 行中取样；
- 每行人工遮盖 1–3 个 core 字段；
- pattern 优先按 test 的 core5 缺失模式频率采样，但只读取一次聚合频率，不输入任何 test 字段值；
- 如果某个字段在真实 pattern 中被遮次数过少，增加字段平衡采样，保证五个字段都有稳定监督量；
- 每个 epoch 重新生成 train mask；
- outer-valid 使用固定 seed 生成不可变化的 mask bank；
- 每条可评估 valid 行生成 3 个固定遮盖副本，降低一次随机遮盖带来的指标噪声；
- 自然缺失位置既不作为重建真值，也不计入 loss 或指标。

输出两个布尔张量：

```python
natural_observed_mask: Tensor  # 原始数据里是否有真值
artificial_mask: Tensor        # 本次是否被人为隐藏
```

有效监督位置必须严格等于：

```python
loss_mask = natural_observed_mask & artificial_mask
```

### 5.4 重建指标

每个 fold、每个字段记录：

- MAE；
- RMSE；
- 使用 outer-train 标准差归一化的 NRMSE；
- R²；
- Spearman；
- fold-train 中位数基线 RMSE；
- `1 - model_rmse / median_rmse`，即相对基线改善率；
- 有效评估样本数。

最终指标使用五折所有固定 valid 重建预测拼接后的 OOF 结果计算，同时保留逐 fold 结果，防止均值掩盖不稳定性。

## 6. 模型设计

### 6.1 模型 A：普通 MLP-MAE 控制组

用途是验证任务本身是否可学，并作为 TabM 结构的简单对照。

建议固定参数：

```yaml
hidden_dim: 128
latent_dim: 32
n_blocks: 3
dropout: 0.10
activation: gelu
normalization: layer_norm
```

输入由标准化数值、缺失/遮盖标记和类别 embedding 拼接；encoder 输出 32 维 latent，decoder 输出五个 core 数值的标准化预测。

### 6.2 模型 B：TabM 风格 MAE

使用 PyTorch 原生算子实现轻量 BatchEnsemble 风格骨干，第一版不依赖第三方 TabM 包，减少 MPS 不支持算子的风险。

建议固定参数：

```yaml
ensemble_size: 4
hidden_dim: 128
latent_dim: 32
n_blocks: 3
dropout: 0.10
activation: gelu
normalization: layer_norm
```

模型输出形状约定：

```python
member_predictions: Tensor  # [batch, ensemble_size, 5]
member_latents: Tensor       # [batch, ensemble_size, 32]
mean_prediction: Tensor      # [batch, 5]
mean_latent: Tensor          # [batch, 32]
```

第一版不加人为 diversity loss。各成员都对同一个 masked-only 重建目标计算损失，预测均值用于重建指标，成员标准差只作为不确定性诊断。

### 6.3 损失与训练参数

采用标准化目标上的 Huber loss，只在 `loss_mask` 为真的位置计算：

```yaml
loss: huber
huber_delta: 1.0
optimizer: adamw
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 4096
max_epochs: 50
early_stopping_patience: 5
gradient_clip_norm: 1.0
dtype: float32
seed: 42
```

early stopping 监控 outer-train 内部划出的 reconstruction holdout loss，outer-valid 只用于最终 fold 评估，不能参与 epoch 选择。

如果 batch 4096 在实际机器上触发 MPS OOM，只允许按 `4096 -> 2048 -> 1024` 顺序下降，并把实际值写入 artifact；不得同时改动模型宽度、学习率等参数。

## 7. MPS 环境方案

当前 `smartphone-addiction` Miniconda 环境是 Python 3.11，但尚未安装 PyTorch。依赖应放进独立可选组，避免普通 LightGBM 安装和离线 bundle 被强制带上 PyTorch：

```toml
[project.optional-dependencies]
neural = [
  "torch>=2.5,<3",
]
```

实施依赖改动后，由用户执行：

```bash
conda activate smartphone-addiction
cd "/Users/sunshine/Desktop/kaggle/Predicting Smartphone Addiction"
python -m pip install -e ".[analysis,dev,tools,neural]"
python -m pip check
python -c "import torch; print('torch=', torch.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available())"
```

设备选择规则：

```text
配置 device=auto：mps available -> mps，否则 cpu
配置 device=mps：不可用时直接报清楚的错误，不静默切换
配置 device=cpu：强制 CPU，用于兼容性 smoke test
```

MPS 第一版固定：

- `float32`；
- `num_workers=0`；
- `pin_memory=False`；
- 不启用 AMP；
- 不强制 `torch.use_deterministic_algorithms(True)`，因为部分 MPS 算子可能不支持；
- 固定 Python、NumPy、PyTorch 和 DataLoader generator seed，并允许极小浮点误差。

如果某个算子在 MPS 上不支持，优先改写为 PyTorch 基础算子；CPU fallback 只用于诊断和 smoke test，不能把 CPU 与 MPS 的耗时或结果混在同一个正式 run 中。

## 8. 建议工程结构

```text
src/smartphone_addiction/
├── neural/
│   ├── __init__.py
│   ├── config.py               # 独立的 NeuralReconstructionConfig
│   ├── device.py               # MPS/CPU 选择、seed、设备信息
│   ├── preprocessing.py        # fold-local 数值与类别 tensorizer
│   ├── masking.py              # 固定/动态 mask bank
│   ├── losses.py               # masked-only Huber loss
│   ├── autoencoder.py          # 普通 MLP-MAE
│   ├── tabm.py                 # TabM 风格 BatchEnsemble 编码器
│   ├── trainer.py              # epoch、early stop、checkpoint
│   ├── reconstruction.py       # 五折重建 runner 与指标
│   └── export.py               # fold-local latent OOF/test 导出
├── evaluation/
│   └── reconstruction.py       # 字段级重建指标和门槛判断
└── cli.py                      # 增加 neural 子命令，不改变现有 train 行为

configs/
├── neural/
│   └── masked_autoencoder.yaml
└── experiments/
    ├── masked_autoencoder_reconstruction_v1.yaml
    └── lightgbm_latent_v1.yaml

tests/
├── unit/
│   ├── test_neural_config.py
│   ├── test_neural_device.py
│   ├── test_neural_preprocessing.py
│   ├── test_neural_masking.py
│   ├── test_neural_losses.py
│   ├── test_autoencoder.py
│   ├── test_tabm_encoder.py
│   └── test_reconstruction_metrics.py
└── integration/
    ├── test_reconstruction_pipeline.py
    └── test_latent_export_alignment.py
```

第一阶段使用独立 `NeuralReconstructionConfig`，不立刻修改现有 `ModelConfig` 对 `catboost/lightgbm` 的限制。只有门槛一通过、确实需要把 TabM 作为分类模型接入统一 `train` 命令时，才扩展模型注册表。这样不会影响当前稳定树模型链路。

## 9. 核心接口约定

```python
@dataclass(frozen=True)
class TensorizedFrame:
    numeric: torch.Tensor
    categorical: torch.Tensor
    natural_observed: torch.Tensor
    row_ids: np.ndarray


class FoldTensorizer:
    def fit(self, frame: pd.DataFrame) -> "FoldTensorizer": ...
    def transform(self, frame: pd.DataFrame) -> TensorizedFrame: ...


@dataclass(frozen=True)
class MaskBatch:
    masked_numeric: torch.Tensor
    artificial_mask: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor


def build_train_mask_batch(
    batch: TensorizedFrame,
    *,
    generator: torch.Generator,
    pattern_distribution: PatternDistribution,
) -> MaskBatch: ...


def build_fixed_validation_mask_bank(
    frame: TensorizedFrame,
    *,
    seed: int,
    repeats: int,
) -> ValidationMaskBank: ...


class MaskedAutoencoder(nn.Module):
    def forward(self, batch: MaskBatch) -> ReconstructionOutput: ...


class TabMReconstructionModel(nn.Module):
    def forward(self, batch: MaskBatch) -> ReconstructionOutput: ...


def masked_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor: ...


def run_reconstruction_cv(config: NeuralReconstructionConfig) -> Path: ...


def evaluate_reconstruction_gate(
    field_metrics: pd.DataFrame,
    gate: ReconstructionGate,
) -> GateDecision: ...
```

接口必须保证：空 `loss_mask` 明确报错；tensor 行数不一致明确报错；所有 artifact 都保留原始 `id` 和 fold id，便于审计对齐。

## 10. Artifact 设计

每次正式重建实验写入独立目录：

```text
artifacts/reconstruction/<run_id>/
├── config_resolved.yaml
├── environment.json
├── data_fingerprints.json
├── fold_assignments.parquet
├── mask_distribution.json
├── fold_metrics.csv
├── field_metrics.csv
├── reconstruction_oof.parquet
├── training_history.csv
├── gate_decision.json
├── summary.md
└── checkpoints/
    ├── fold_0.pt
    └── ...
```

`environment.json` 至少记录 Python、PyTorch、macOS、device、MPS built/available、dtype 和实际 batch size。`reconstruction_oof.parquet` 至少包含 `id`、fold、字段名、真值、预测值、median baseline、mask pattern 和模型成员标准差。

如果门槛一通过，才生成：

```text
artifacts/reconstruction/<run_id>/latent/
├── train_oof_latent.parquet
├── test_fold_0_latent.parquet
├── ...
├── test_latent_mean.parquet
└── latent_manifest.json
```

## 11. 分阶段实施任务

### 任务 1：加入独立神经网络依赖和 MPS 设备检查

**涉及文件**

- 修改：`pyproject.toml`
- 新增：`src/smartphone_addiction/neural/device.py`
- 新增：`tests/unit/test_neural_device.py`

**实施内容**

- 添加 `neural` optional dependency；
- 实现 `auto/mps/cpu` 三种设备模式；
- 记录 MPS 环境信息；
- 配置为 `mps` 但不可用时明确失败；
- 测试不依赖真实 MPS，使用 monkeypatch 覆盖选择分支。

**验收**

- 用户完成安装后 `pip check` 通过；
- MPS smoke check 能创建张量、执行一次 Linear 前后向并更新参数；
- 原有 LightGBM 环境安装方式仍可不安装 torch。

### 任务 2：实现 fold-local tensorizer

**涉及文件**

- 新增：`src/smartphone_addiction/neural/preprocessing.py`
- 新增：`tests/unit/test_neural_preprocessing.py`

**实施内容**

- raw 数值标准化、缺失填充和 observed flag；
- 三个类别字段建 fold-local 词表与 embedding index；
- valid 未见类别映射到 `__UNKNOWN__`；
- 排除 id、label 和全部预先计算的派生列；
- 可序列化每 fold 统计量和词表。

**验收**

- 改动 valid 数值不会影响 fitted mean/std；
- outer-valid 特有类别不会进入词表；
- transform 后无 NaN/Inf；
- 行 ID、输入张量和 fold assignment 顺序一致。

### 任务 3：实现 test-like 人工遮盖器

**涉及文件**

- 新增：`src/smartphone_addiction/neural/masking.py`
- 新增：`tests/unit/test_neural_masking.py`

**实施内容**

- 从 core5 缺失 pattern 生成概率表；
- 训练 mask 每 epoch 动态生成；
- valid mask bank 固定且可复现；
- 自然缺失目标永远不进入 loss；
- 保证每个 valid 字段具有足够评估样本。

**验收**

- 同 seed 的 validation mask bank 完全一致；
- 不同 seed 至少有一个遮盖位置不同；
- `loss_mask == natural_observed & artificial_mask`；
- 人工遮盖后不能从任何未刷新派生列看到目标真值。

### 任务 4：实现 MLP-MAE、TabM-MAE 和 masked-only loss

**涉及文件**

- 新增：`src/smartphone_addiction/neural/autoencoder.py`
- 新增：`src/smartphone_addiction/neural/tabm.py`
- 新增：`src/smartphone_addiction/neural/losses.py`
- 新增对应单元测试

**实施内容**

- 固定 32 维 latent、五字段 decoder；
- TabM 输出成员维度、均值和标准差；
- Huber loss 只看人工遮盖且有真值的位置；
- 测试单次 CPU 前后向，再做真实 MPS smoke。

**验收**

- shape 与接口约定一致；
- 修改非 loss-mask 位置的 target 不改变 loss；
- 至少一个参数在 optimizer step 后发生变化；
- MPS 前后向无 unsupported-op 错误。

### 任务 5：实现 trainer、重建指标和 artifact

**涉及文件**

- 新增：`src/smartphone_addiction/neural/trainer.py`
- 新增：`src/smartphone_addiction/neural/reconstruction.py`
- 新增：`src/smartphone_addiction/evaluation/reconstruction.py`
- 修改：`src/smartphone_addiction/cli.py`
- 新增：`configs/neural/masked_autoencoder.yaml`
- 新增：`configs/experiments/masked_autoencoder_reconstruction_v1.yaml`

**建议命令**

```bash
smartphone-addiction neural reconstruct \
  --config configs/experiments/masked_autoencoder_reconstruction_v1.yaml
```

**实施内容**

- 内层 reconstruction holdout early stopping；
- best checkpoint 保存与恢复；
- 五折预测、字段级指标和 gate 自动判定；
- 中断后可按已完成 fold 恢复；
- run 不覆盖已有 artifact。

**验收**

- 五折 OOF coverage 为 1.0；
- 每个评估预测都来自未见该行的 encoder；
- 自动生成 `gate_decision.json` 和通俗版 `summary.md`；
- 中位数基线使用对应 fold 的 outer-train 统计量。

### 任务 6：先做单 fold 小样本 smoke

建议固定：20,000 行、1 fold、2 epochs，先跑 MLP，再跑 TabM。smoke 只验证链路，不用于门槛判断。

```bash
smartphone-addiction neural reconstruct \
  --config configs/experiments/masked_autoencoder_reconstruction_v1.yaml \
  --smoke --model mlp

smartphone-addiction neural reconstruct \
  --config configs/experiments/masked_autoencoder_reconstruction_v1.yaml \
  --smoke --model tabm
```

验收：

- MPS 被实际使用并写入 environment；
- loss 有限且至少总体下降；
- artifact、checkpoint、OOF 样例和指标文件齐全；
- 模型能从 checkpoint 重新加载并得到相同 shape 的预测。

### 任务 7：运行完整五折重建能力实验

只在 smoke 全部通过后运行固定参数全量 5-fold：

```bash
smartphone-addiction neural reconstruct \
  --config configs/experiments/masked_autoencoder_reconstruction_v1.yaml \
  --model mlp

smartphone-addiction neural reconstruct \
  --config configs/experiments/masked_autoencoder_reconstruction_v1.yaml \
  --model tabm
```

执行结束后只比较预先定义的字段指标和门槛，不因结果差一点而临时降低门槛。两者均不通过则路线结束，并保留失败结论。

### 任务 8：仅在门槛一通过后导出 fold-local latent

**涉及文件**

- 新增：`src/smartphone_addiction/neural/export.py`
- 新增：`tests/integration/test_latent_export_alignment.py`

每个 outer fold：

1. 使用该 fold 的 train rows 训练 encoder；
2. 用它生成本 fold valid latent，拼成 train OOF latent；
3. 用它生成整份 test latent；
4. 五个 test latent 取均值；
5. 保存 32 维均值 latent；
6. 可选保存 5 个 core 重建均值和 5 个成员标准差，但单独做特征组，不能默认全部加入。

验收：train latent 每个 ID 只出现一次；test 每个 fold 都完整覆盖；manifest 记录每列来源、encoder fold 和 checkpoint hash。

### 任务 9：LightGBM + latent 的最小消融

**涉及文件**

- 新增：`configs/experiments/lightgbm_latent_v1.yaml`
- 最小修改现有特征加载逻辑，使 latent 按 ID 严格对齐

只允许三个有明确含义的实验，不做参数搜索：

1. 当前 34 列 + 32 维 latent；
2. 当前 34 列 + 5 个 core 重建均值；
3. 若前两者至少一个有正收益，再测 latent + 重建均值 + 不确定性。

LightGBM 树参数、masking 0.20、fold 和 seed 42 全部保持原样。按门槛二决定是否晋级。

### 任务 10：仅在需要低相关新模型时接 TabM 分类头

如果 latent 对 LightGBM 无明显增益，但重建门槛通过，说明表示可能有效、树模型却没有利用好。此时才允许：

- 冻结 encoder，先训练一个轻量二分类头；
- 只有冻结版本接近单模型门槛，才允许一次全网络微调；
- 输出严格 OOF 概率；
- 与 LightGBM 做固定 folds 的交叉拟合 blend；
- 按低相关模型门槛决定停止或晋级。

不允许在此阶段重新搜索 LightGBM、XGBoost 或已停止的 GAM/专家路线。

### 任务 11：最终工程验收

实施完成后依次执行：

```bash
python -m pytest -m "not slow" -q
python -m ruff format --check .
python -m ruff check .
python -m build --no-isolation
python -m pip check
```

额外检查：

- 原 `lightgbm_masked_v3.yaml` 解析结果和 34 列特征不变；
- 原 LightGBM smoke test 仍通过；
- neural optional dependency 不进入默认 Kaggle tree-model bundle；
- 无 train/test/OOF ID 错位或重复；
- 所有正式结论可从 artifact 复算；
- 未通过晋级门槛时没有 submission 文件。

## 12. 推荐执行顺序与停止点

| 阶段 | 预计产物 | 是否允许继续的条件 |
|---|---|---|
| A. 环境与 MPS smoke | MPS 前后向结果 | MPS 可用且基础算子正常 |
| B. 数据、遮盖、模型 smoke | 小样本 artifact | 无泄漏、loss 正常、可恢复 |
| C. MLP/TabM 五折重建 | 字段级 OOF 指标 | 门槛一通过 |
| D. fold-local latent 导出 | OOF/test latent | 对齐与 coverage 验收通过 |
| E. LightGBM + latent | 同 folds OOF 对比 | 门槛二通过 |
| F. TabM 分类头或 3 seeds | 分类 OOF / 最终 run | 仅按对应门槛触发 |
| G. Kaggle submission | submission CSV | 只有正式晋级方案可生成 |

## 13. 最终决策规则

完成本计划后，只会出现以下三种结论：

1. **重建失败**：其余字段不足以恢复 core5，停止 autoencoder/TabM 路线。
2. **重建成功但分类无提升**：encoder 学到了字段关系，但没有补充 LightGBM 所需的排序信息；保留研究结果，不提交。
3. **重建成功且分类晋级**：再运行多 seed，确认稳定后才生成 submission。

这样可以把“神经网络也许有用”变成一组可验证、可停止、不会污染 OOF 的工程实验。
