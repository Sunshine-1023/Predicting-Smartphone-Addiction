# Smartphone Addiction AUC 0.97+ 优化实施计划

> **For Hermes:** 仅在用户逐项确认后按任务顺序实施；本计划不包含任何 Git 初始化、提交、推送或分支操作。

**Goal:** 在保持严格无泄漏 5 折 OOF 的前提下，优先改善核心行为字段缺失样本的排序能力，并通过特征、模型和融合实验尽可能把整体 ROC-AUC 推向 0.97+。

**Architecture:** 保留现有“确定性预处理 → 5 折训练 → OOF/测试预测 → artifact/report”的主链路；新增一个只在每个 CV 训练折上拟合的 `FoldFeatureAugmenter`，负责统计量、条件百分位和核心字段插补，避免把全量数据统计提前泄露给验证折。实验严格一次只改变一个特征组或一个训练策略，先用 LightGBM 单 seed 开发，再把胜出方案迁移到 CatBoost、多 seed 和 blend。

**Tech Stack:** Python 3.11、Miniconda、pandas、NumPy、scikit-learn、LightGBM、CatBoost、Pydantic/YAML、pytest、现有 artifact/report 系统。

---

## 1. 当前基线与判断依据

### 已确认基线

| 方案 | OOF AUC |
|---|---:|
| CatBoost dev，5-fold，seed 42 | 0.9616867 |
| LightGBM dev，5-fold，seed 42 | 0.9617918 |
| LightGBM final，5-fold × 3 seeds | 0.9623916 |
| CatBoost final，5-fold × 3 seeds | 0.9618384 |
| 0.6 LightGBM + 0.4 CatBoost | **0.9628134** |

### 数据诊断结论

- 五个核心字段全部存在的 57.27% 样本，现有 blend AUC 已达 **0.97044**。
- 至少缺一个核心字段的 42.73% 样本，现有 blend AUC 只有 **0.95110**。
- 最主要核心字段：`daily_screen_time_hours`、`weekend_screen_time`、`social_media_hours`、`work_study_hours`、`gaming_hours`。
- 无标签字段重建 R²：daily 0.799、weekend 0.634、social 0.535、work 0.436、gaming 0.410。
- 控制 daily 后仍有信息的字段依次是 social、weekend、work、gaming；opens 和 notifications 的条件增量很弱。
- 简单加性/二维/三维分箱风险面没有胜过现有树模型，因此不把“监督风险表”作为第一轮重点。

### 本轮边界

- 只使用官方 train/test 数据。
- 不使用外部数据、伪标签或全量标签生成的 target encoding。
- 不改变最终评估指标：正类概率的 5-fold `StratifiedKFold` OOF ROC-AUC。
- 不重新搜索所有参数；先证明新信息有效，再调参。
- 不进行任何 Git 操作。

---

## 2. 统一实验规则

### 固定比较条件

所有开发实验必须固定：

```text
model        = LightGBM
folds        = 已有 seed42 的相同 5 折划分
seed         = 42
threads      = 4
metric       = ROC-AUC
data         = 全量 train
early stop   = 100
```

禁止为了某一个特征组重新换折，否则细小差异不可比较。

### 每个实验必须报告

```text
overall_oof_auc
core_complete_auc
core_incomplete_auc
top3_incomplete_auc
test_pattern_weighted_auc
5 个 fold_auc
best_iteration
training_seconds
prediction_correlation_with_baseline
```

### 晋级标准

一个开发实验满足以下条件才进入下一轮：

1. 整体 OOF 相对同 seed 基线提升至少 `+0.00030`；
2. 至少 3/5 个 fold 提升；
3. 如果是缺失优化，`core_incomplete_auc` 至少提升 `+0.0010`；
4. `core_complete_auc` 不得下降超过 `0.00020`；
5. 没有 train/valid/test 列错位、标签泄漏或不可复现的全局统计量。

`+0.00015` 以下视为噪声区，不直接进入多 seed。只有通过单 seed 门槛的方案才运行 seeds `[42, 2026, 3407]`。

---

## 3. Task 1：增加分层评估，先让优化目标可见

**Objective:** 每次训练结束后自动输出完整/缺失核心字段等关键切片的 AUC，避免只看总 AUC。

**Files:**

- Create: `src/smartphone_addiction/evaluation/slices.py`
- Create: `tests/unit/test_slice_metrics.py`
- Modify: `src/smartphone_addiction/training/runner.py`
- Modify: `src/smartphone_addiction/evaluation/report.py`
- Test: `tests/integration/test_smoke_pipeline.py`

### Steps

1. 在 `tests/unit/test_slice_metrics.py` 写测试，覆盖：完整核心字段、至少缺一个核心字段、核心字段观察数量 0–5、某切片只有一个标签时返回空值而不是报错。
2. 运行：

   ```bash
   pytest tests/unit/test_slice_metrics.py -q
   ```

   预期：新模块尚不存在，测试失败。

3. 在 `evaluation/slices.py` 实现 `compute_slice_metrics(frame, y, prediction)`，只读输入，不改变行顺序。
4. 在 runner 聚合 OOF 后写入：

   ```text
   artifacts/runs/<run>/slice_metrics.json
   ```

5. 在实验汇总中增加 `core_complete_auc`、`core_incomplete_auc`、`test_pattern_weighted_auc`。
6. 运行单元与 smoke 测试，预期全部通过，并验证 OOF coverage 仍为 1.0。

**Acceptance:** 对现有 final OOF 复算时，完整核心样本 AUC 应约为 0.9704，不完整样本约为 0.9511。

---

## 4. Task 2：建立无泄漏的折内特征框架

**Objective:** 为条件统计、标准化和字段插补提供统一的 `fit(train_fold) → transform()` 能力。

**Files:**

- Create: `src/smartphone_addiction/features/fold.py`
- Create: `tests/unit/test_fold_features.py`
- Modify: `src/smartphone_addiction/config.py`
- Modify: `src/smartphone_addiction/cli.py`
- Modify: `src/smartphone_addiction/training/runner.py`
- Modify: `src/smartphone_addiction/features/io.py`
- Modify: `src/smartphone_addiction/evaluation/importance.py`
- Test: `tests/integration/test_smoke_pipeline.py`

### Proposed interface

```python
class FoldFeatureAugmenter:
    def fit(self, frame: pd.DataFrame) -> "FoldFeatureAugmenter": ...
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...
    @property
    def output_columns(self) -> list[str]: ...
```

约束：

- `fit()` 不接收 `addicted_label`。
- 只能在当前 outer-fold 的训练行上拟合。
- train、valid、test 的输出列名、顺序和 dtype 必须完全一致。
- 原始列保留，新增列追加；不得覆盖 raw 字段。
- 每折保存 `models/<fold_key>.features.joblib`，保证模型检查和 permutation importance 可以重放同一变换。

### Config contract

在 `FeatureConfig` 下增加：

```yaml
features:
  groups: [raw, missingness, behavioral_totals]
  fold:
    enabled: false
    groups: []
    conditional_bins: 30
    imputation_targets: []
```

### Steps

1. 写失败测试：验证 augmenter 未 fit 时不能 transform；验证 fit 只读取训练折；验证未知配置被 Pydantic 拒绝。
2. 实现空操作 augmenter，`enabled=false` 时必须得到与当前基线完全相同的 OOF。
3. runner 每折按以下顺序运行：

   ```text
   augmenter.fit(x_train)
   x_train_aug = augmenter.transform(x_train)
   x_valid_aug = augmenter.transform(x_valid)
   x_test_aug  = augmenter.transform(x_test)
   model.fit(...)
   ```

4. 把 fold 配置、输出特征名和 augmenter artifact 纳入 resume 一致性检查。
5. 把 `features/fold.py` 纳入 `feature_code_fingerprint()`。
6. 运行完整单元测试和 5k smoke；`fold.enabled=false` 的预测应与修改前数值一致或仅有机器浮点差异。

**Acceptance:** 关闭折内增强时零行为变化；打开后不存在验证折统计进入 `fit()` 的路径。

---

## 5. Task 3：实验 A——缺失鲁棒的核心综合特征

**Objective:** 用仍然存在的核心字段形成稳定的综合成瘾强度，不先引入模型插补。

**Files:**

- Modify: `src/smartphone_addiction/features/fold.py`
- Modify: `tests/unit/test_fold_features.py`
- Create: `configs/experiments/lightgbm_core_aggregate_v2.yaml`

### 新增特征

对五个核心字段使用训练折均值/标准差进行标准化，然后逐行计算：

```text
available_core_mean_z
available_core_max_z
available_core_min_z
available_core_std_z
available_core_count
top3_available_count
```

其中 top3 指 daily、weekend、social。统计量只能从训练折计算；逐行聚合忽略当前行缺失值，五项全缺时保持 NaN。

### Steps

1. 写测试覆盖：部分缺失、全部缺失、零标准差、valid 极端值、输入不被修改。
2. 实现 `core_aggregates` 组。
3. 运行同 seed 全量 5 折实验。
4. 对比 overall、complete、incomplete 和每折 AUC。

**Decision:** 达到统一晋级标准则保留，否则只保留 `available_core_count` 用于切片，不进入最终特征。

---

## 6. Task 4：实验 B——相对 daily 的条件百分位与残差

**Objective:** 表达“在总屏幕时间相近的人中，社交/周末/工作/游戏是否异常高”。

**Files:**

- Modify: `src/smartphone_addiction/features/fold.py`
- Modify: `tests/unit/test_fold_features.py`
- Create: `configs/experiments/lightgbm_conditional_v2.yaml`

### 新增特征

先用训练折把 daily 划为 30 个分位区间，再为每个区间保存其他字段的经验分布与中位数/IQR：

```text
social_given_daily_pct
weekend_given_daily_pct
work_given_daily_pct
gaming_given_daily_pct

social_given_daily_robust_z
weekend_given_daily_robust_z
work_given_daily_robust_z
gaming_given_daily_robust_z
```

计算方式：

```text
conditional_pct = ECDF(value | daily_bin)
robust_z = (value - median(value | daily_bin)) / max(IQR, epsilon)
```

daily 或对应字段缺失时返回 NaN，不做隐式填充。

### Experiment sequence

1. 只加 social 条件特征。
2. social 通过门槛后，再加 weekend。
3. 前两项通过后，再整体加入 work、gaming。
4. 不同时测试不同 bins；固定 30，避免形成隐性调参搜索。

**Decision:** 每一步单独记录增量。若四项整体不如 social+weekend，只保留胜出子集。

---

## 7. Task 5：实验 C——核心字段折内预测与缺失填充

**Objective:** 为核心字段缺失行增加由其他字段预测的伴随值，同时保留原始 NaN 给树模型。

**Files:**

- Create: `src/smartphone_addiction/features/imputation.py`
- Create: `tests/unit/test_fold_imputation.py`
- Modify: `src/smartphone_addiction/features/fold.py`
- Modify: `src/smartphone_addiction/config.py`
- Create: `configs/experiments/lightgbm_impute_daily_v2.yaml`
- Create: `configs/experiments/lightgbm_impute_top3_v2.yaml`
- Create: `configs/experiments/lightgbm_impute_core_v2.yaml`

### First-version imputer

- 模型：`LGBMRegressor(objective="regression_l1")`。
- 每个 outer fold 中，只使用该 fold 训练行里目标字段非缺失的样本拟合。
- 输入排除当前要预测的字段、`id` 和 `addicted_label`。
- 原始字段不被覆盖。

每个目标字段输出：

```text
<field>_filled          # 原值存在用原值，缺失时用预测值
<field>_predicted       # 仅缺失行填预测值，原值存在时为 NaN
<field>_was_imputed     # 0/1
```

第一版不为观察值计算 residual，避免训练折内预测与验证折外推形成分布差异。残差特征只有在填充方案证明有效后，再采用 inner-fold cross-fitting 单独实现。

### Experiment sequence

1. `impute_daily_v2`：只恢复 daily。
2. daily 通过门槛后，`impute_top3_v2`：daily、weekend、social。
3. top3 通过门槛后，`impute_core_v2`：再加入 work、gaming。
4. 每次同时报告各目标字段的回归 R²/MAE，并保存到 `imputation_metrics.json`。

### Leakage tests

- 修改验证折某个已观察字段的真实值，不得改变训练折 imputer。
- 修改验证折标签，不得改变任何 imputation feature。
- test 行不得参与 outer-fold imputer 拟合。
- resume 时缺失 imputer artifact 必须拒绝继续运行。

**Decision:** 重点观察 `core_incomplete_auc`。若 daily-only 提升而 top3/all-core 下降，最终只保留 daily。

---

## 8. Task 6：实验 D——训练折随机遮盖增强

**Objective:** 让分类模型学习在多个强字段共同缺失时利用替代字段，而不是只依赖完整样本的强路径。

**Files:**

- Create: `src/smartphone_addiction/training/masking.py`
- Create: `tests/unit/test_masking.py`
- Modify: `src/smartphone_addiction/training/runner.py`
- Modify: `src/smartphone_addiction/config.py`
- Create: `configs/experiments/lightgbm_masked_v2.yaml`

### Strategy

- 只复制 outer-fold 训练数据中的一部分完整/较完整样本。
- 从 test 的五核心字段缺失组合分布中抽样 mask pattern。
- 第一轮 augmentation fraction 固定为 0.20。
- 遮盖 raw 字段后，重新调用同一 fold augmenter 生成伴随特征。
- valid/test 不做人工遮盖。
- 随机数由 `seed + fold_id` 固定。

### Steps

1. 写测试验证：标签与复制行对齐；原训练数据不被原地修改；验证折不参与；同 seed 结果完全相同。
2. 实现 20% 单一版本，不同时搜索 5%、10%、30%、50%。
3. 仅基于 Task 5 的胜出插补配置运行。
4. 如果 incomplete AUC 提升但 complete AUC 下降超过门槛，降低 augmentation fraction 到 0.10，仅补做一次。

**Decision:** 最多测试两个 fraction，防止把验证集当调参集。

---

## 9. Task 7：现有 45 特征消融与精简

**Objective:** 删除高基数、重复或反直觉特征，减少树模型把容量浪费在噪声上。

**Files:**

- Create experiment YAMLs under: `configs/experiments/ablation/`
- No source change unless某个组最终确认删除

### Fixed ablations

按顺序一次移除一组：

1. `missing_pattern`，保留逐字段 missing flag 和 count；
2. `categorical_interactions`；
3. `behavioral_deltas`；
4. notifications/opens 相关比例与 log 重复项；
5. 所有旧 `behavioral_ratios`，只保留新条件特征；
6. lean set：raw + missing flags + 胜出的 core/conditional/imputation 特征。

### Decision

- 删除组若 OOF 持平或提升，并且训练更快，则进入 lean candidate。
- 不因单个 fold 的随机提升删除整组。
- `missing_pattern` 不作为专家模型分组依据；只使用少数预定义核心可用性状态。

---

## 10. Task 8：模型层优化

**Objective:** 在胜出特征固定后提高单模型能力和预测多样性。

### 8.1 LightGBM 单调约束实验

**Files:**

- Create: `configs/experiments/lightgbm_monotone_v2.yaml`
- Modify if needed: `src/smartphone_addiction/models/lightgbm.py`
- Test: `tests/unit/test_lightgbm_model.py`

只对证据极强且近似单调的字段施加正向约束：

```text
daily_screen_time_hours
weekend_screen_time
social_media_hours
```

第一轮不约束 gaming、sleep、notifications、opens。比较 constrained 与 unconstrained 的同折 OOF及相关性。

### 8.2 CatBoost 收敛实验

**Files:**

- Create: `configs/experiments/catboost_extended_v2.yaml`

当前 CatBoost 多数 fold 撞到 2000 iterations。只测试一个受控版本：

```yaml
iterations: 4000
learning_rate: 0.03
early_stopping_rounds: 200
thread_count: 4
```

其他结构参数保持不变。若仍持续撞上限但 AUC没有实质提升，不再扩大到 6000。

### 8.3 第三类模型的进入条件

只有增强后的 LightGBM/CatBoost blend 仍明显停滞，才考虑 XGBoost histogram/DART。引入前先向用户确认新依赖；不在本轮直接引入神经网络、TabPFN 或大规模 ExtraTrees。

---

## 11. Task 9：OOF 融合与缺失状态融合

**Objective:** 只用严格对齐的 OOF 结果选择最终概率组合。

**Files:**

- Modify: `src/smartphone_addiction/evaluation/blend.py`
- Modify: `tests/unit/test_blend.py`
- Create: `configs/experiments/blends/optimized_v2.yaml`（若当前 blend 不使用 YAML，则在现有 artifact 格式下记录）

### Sequence

1. 先对胜出的增强 LightGBM 与增强 CatBoost做全局概率权重网格。
2. 比较 probability blend 与 rank blend。
3. 仅在样本量足够的粗粒度状态中测试 pattern-aware weights：

   ```text
   core_complete
   top3_complete_but_other_missing
   top3_incomplete
   ```

4. 模式权重必须通过 nested OOF 或独立 meta-fold 学习，不能直接在同一 OOF 上自由搜索几十组权重。
5. 检查融合后所有概率有限并处于 `[0, 1]`，ID 顺序完全一致。

**Final gate:** 多 seed 融合相对当前 0.962813 至少提升 `+0.00020` 才替换当前 final；0.97 是追求目标，不作为伪造完成条件。

---

## 12. Task 10：最终复验与提交候选

**Objective:** 对唯一胜出方案完成多 seed、产物校验和 submission 构建。

### Steps

1. 使用 seeds `[42, 2026, 3407]` 跑增强 LightGBM final。
2. 只有 dev 证据支持时才跑增强 CatBoost final。
3. 生成并验证：

   ```text
   metrics.json
   slice_metrics.json
   fold_metrics.csv
   oof_predictions.parquet
   test_predictions.parquet
   feature_names.json
   imputation_metrics.json
   fold augmenter artifacts
   ```

4. 检查 OOF coverage=1.0、测试 ID 唯一且顺序与 sample_submission 一致。
5. 构建 submission，但不自动上传 Kaggle。
6. 在 `reports/experiment_summary.csv` 中保留当前 final 与新 candidate 的完整可比较记录。

---

## 13. 推荐实验顺序与停止规则

| 顺序 | 实验 | 目的 | 失败后处理 |
|---:|---|---|---|
| 0 | 关闭 fold augmenter 的回归基线 | 确认零行为变化 | 修复框架，不进入特征实验 |
| 1 | core aggregates | 缺失鲁棒潜变量 | 不通过则删除该组 |
| 2 | social 条件特征 | 最大条件增量 | 不通过则停止其余条件扩展 |
| 3 | weekend 条件特征 | 第二强条件增量 | 单独判断是否保留 |
| 4 | daily imputation | 最高可重建核心字段 | 不通过则停止大规模插补 |
| 5 | top3 / all-core imputation | 扩大缺失恢复 | 只保留最优子集 |
| 6 | masking augmentation | 多字段共同缺失 | 最多两个 fraction |
| 7 | 旧特征消融 | 降噪与提速 | 保留没有伤害的旧组 |
| 8 | monotone LightGBM | 平滑主风险关系 | 不通过则回普通 LightGBM |
| 9 | extended CatBoost | 收敛与多样性 | 无提升则不再加 iterations |
| 10 | multi-seed + blend | 最终稳定收益 | 不达 gate 保留当前 final |

连续三个主实验的整体提升都低于 `+0.00015` 时，暂停继续堆特征，重新评估是否需要第三类模型或比赛允许范围内的源数据研究。

---

## 14. 风险与防护

### 标签泄漏

风险最高的是插补、条件百分位和监督风险编码。防护：所有拟合统计都在 outer-fold 训练集内部产生；第一轮完全不使用标签构建新特征。

### 训练/验证特征分布不一致

观察值 residual 若在训练行内拟合、验证行外预测，会造成差异。第一版只给缺失行生成预测伴随值；residual 延后并使用 inner-fold cross-fitting。

### 计算成本与 M4 温度

- 始终保持 4 threads；不并行启动多个 fold 或多个模型。
- 插补先 daily，再 top3，再 all-core，不一次训练全部方案。
- smoke → 单 seed full → multi-seed final，禁止直接多 seed 探索。

### OOF 小幅波动

同折比较、检查 5 个 fold 的方向一致性，并设置 `+0.00030` 开发门槛。不要因为 `+0.00005` 就宣布提升。

### train/test 缺失模式差异

总体完整比例接近，但具体缺失字段组合不同。使用 `test_pattern_weighted_auc` 辅助选择，仍以未加权 OOF 为主，不把加权估计当真实排行榜分数。

---

## 15. 预计会涉及的文件汇总

### 新建

```text
src/smartphone_addiction/evaluation/slices.py
src/smartphone_addiction/features/fold.py
src/smartphone_addiction/features/imputation.py
src/smartphone_addiction/training/masking.py
tests/unit/test_slice_metrics.py
tests/unit/test_fold_features.py
tests/unit/test_fold_imputation.py
tests/unit/test_masking.py
configs/experiments/*_v2.yaml
```

### 修改

```text
src/smartphone_addiction/config.py
src/smartphone_addiction/cli.py
src/smartphone_addiction/training/runner.py
src/smartphone_addiction/features/io.py
src/smartphone_addiction/evaluation/report.py
src/smartphone_addiction/evaluation/importance.py
src/smartphone_addiction/evaluation/blend.py
tests/integration/test_smoke_pipeline.py
tests/unit/test_config.py
tests/unit/test_blend.py
tests/unit/test_lightgbm_model.py
```

---

## 16. 执行前需要逐项确认的决策

按照用户“细节问题一个一个确认”的要求，真正实施前只先确认第一个问题：

> 是否同意先实施 Task 1–2 的评估与折内特征基础设施，再开始任何新特征实验？

后续的插补目标范围、masking augmentation 和新模型依赖分别在到达对应阶段时单独询问，不提前一次性决定。
