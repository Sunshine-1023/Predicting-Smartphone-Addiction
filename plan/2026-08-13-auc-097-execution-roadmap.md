# Smartphone Addiction AUC 0.97+ 可执行优化计划

> 本计划只定义后续实施与实验顺序。本轮不修改训练代码、不运行实验、不提交 Kaggle，也不进行任何 Git 操作。

## 1. 目标与当前基线

最终目标：在严格无泄漏的 5 折 OOF 下持续提高 ROC-AUC，并尽可能使 Public LB 达到 `0.97000+`。

当前唯一正式基线：

```text
run                  = 20260813T063020Z-lightgbm-final-dcaf81b
model                = LightGBM
features             = 34
cv                   = 5 folds × seeds [42, 2026, 3407]
masking              = 0.20
OOF AUC              = 0.96425464
Public LB            = 0.96543
core complete AUC    = 0.97126097
core incomplete AUC  = 0.95356798
```

目标差距为 `+0.00575 OOF`。这是结构性差距，不能指望一次普通调参补齐。

## 2. 全程统一实验规则

所有开发实验固定以下条件：

```text
数据             = 全量官方 train.csv
模型             = LightGBM（除非该阶段明确测试新模型）
CV               = seed 42 对应的同一套 5 折
metric           = ROC-AUC
线程             = 4
基线             = 34 特征 + masking 0.20
valid/test       = 禁止人工遮盖
```

每个实验必须保存并比较：

- overall OOF AUC；
- 5 个 fold AUC；
- core complete / incomplete AUC；
- 核心字段观察数 0–5 的 AUC；
- test-pattern-weighted AUC；
- 与当前冠军 OOF 的 Pearson、Spearman 相关性；
- 训练时间和 best iteration。

开发方案只有同时满足以下条件才晋级：

1. overall OOF 至少提升 `+0.00030`；
2. 至少 3/5 folds 提升；
3. 缺失优化方案的 core incomplete AUC 至少提升 `+0.00100`；
4. core complete AUC 下降不得超过 `0.00020`；
5. OOF 覆盖率必须为 1.0，且不存在标签泄漏或行错位。

判定规则：

- 增益 `< +0.00015`：视为噪声，停止该方向；
- 增益 `+0.00015～+0.00030`：只允许补一次确认实验；
- 增益 `≥ +0.00030`：进入下一阶段或多 seed；
- 只有单 seed 晋级方案才运行 `[42, 2026, 3407]`。

---

## 3. 阶段一：修复调参与正式训练不一致

### 目的

保证 Optuna/候选复评实际优化的是当前 `34 特征 + masking 0.20`，而不是未 masking 的旧基线。

### 修改范围

- 修改 `src/smartphone_addiction/training/tuning.py`
- 修改 `src/smartphone_addiction/cli.py`
- 修改 `tests/unit/test_tuning.py`
- 必要时修改 `tests/integration/test_cli.py`

### 实施内容

1. 为 `evaluate_params_oof()` 增加 `test_features` 和 `masking` 输入。
2. 每个 trial 的每个训练折调用与正式 runner 相同的 `augment_training_fold()`。
3. `evaluate_candidates()` 把实验配置中的 masking 原样传给 `run_training()`。
4. 调参 artifact 记录 feature columns、exclude columns、masking 配置和数据 hash。
5. 增加测试，证明 tuning 与正式 runner 在相同参数、相同折、相同 seed 下得到同样的训练样本数量和 masking 设置。

### 验收

```bash
python -m pytest tests/unit/test_tuning.py tests/unit/test_masking.py -q
python -m pytest tests/integration/test_cli.py -q
python -m ruff check src tests
```

全部通过后才能进入参数搜索。此阶段不要求 AUC 提升，只要求实验链路正确。

---

## 4. 阶段二：Masking v3

### 目的

改善只剩 2–3 个核心字段的困难样本，并让训练缺失分布更接近 test。

### 修改范围

- 修改 `src/smartphone_addiction/training/masking.py`
- 修改 `src/smartphone_addiction/config.py`
- 修改 `src/smartphone_addiction/training/runner.py`
- 修改 `src/smartphone_addiction/models/lightgbm.py`
- 修改 `src/smartphone_addiction/models/catboost.py`
- 修改 `tests/unit/test_masking.py`
- 新增 `configs/experiments/lightgbm_masked_v3_core5.yaml`
- 新增 `configs/experiments/lightgbm_masked_v3_top8.yaml`

### top8 字段

```text
daily_screen_time_hours
weekend_screen_time
social_media_hours
work_study_hours
gaming_hours
notifications_per_day
sleep_hours
app_opens_per_day
```

### 按顺序只做三个实验

#### M1：兼容源行遮盖

- 仍遮盖 core5；
- 不再只从五字段全完整行取样；
- 只要源行的已观察字段是目标 pattern 的超集，就允许遮盖；
- 其他设置与当前冠军一致。

目的：单独验证“更真实的源样本”是否有效。

#### M2：扩展到 top8 pattern

- 以 M1 为基础；
- 从 test 的 top8 缺失组合分布抽样；
- 遮盖后同步重算全部 missing flags、count/ratio、totals、deltas 和 logs。

目的：覆盖 notifications、sleep、app opens 三个高重要性替代字段。

#### M3：样本权重守恒

- 只在 M2 晋级后实施；
- 同一源样本的原始视图与 masked 视图共享总权重；
- runner 和模型 adapter 显式传递 `sample_weight`；
- 防止重复复制某些完整行标签，改变总体标签权重。

目的：验证 masking 收益是否能在更稳定的训练权重下继续扩大。

### 测试要求

- 目标 pattern 不得要求恢复源行已经缺失的字段；
- 同 seed、fold 必须完全可复现；
- labels 与复制行严格对齐；
- 原始 frame 不得被原地修改；
- valid/test 不得进入人工遮盖；
- masking 后所有派生特征必须同步；
- sample weights 长度、顺序和总量正确。

### 停止条件

- M1 未达到 `+0.00015`：停止兼容源行分支，但仍允许单独测试一次 M2；
- M2 未达到 `+0.00030`：不实施 M3；
- M2 或 M3 达到晋级标准：将胜出方案命名为新的单 seed 基线。

预计实验量：2–3 次完整单 seed 5 折，约 15–25 分钟。

---

## 5. 阶段三：定向 LightGBM 搜索

### 目的

在 Masking v3 胜出基线之上，优化连续变量分箱、叶子复杂度和正则；不进行无边界随机搜索。

### 前置条件

阶段一已经确保 tuning 与 masking 一致。若阶段二没有胜出方案，则继续使用当前 masked-v2。

### 第一轮：分箱与容量，固定 6 组候选

```text
A  max_bin=255   num_leaves=63   min_child_samples=20
B  max_bin=511   num_leaves=63   min_child_samples=50
C  max_bin=1023  num_leaves=63   min_child_samples=50
D  max_bin=511   num_leaves=31   min_child_samples=100
E  max_bin=511   num_leaves=95   min_child_samples=100
F  max_bin=1023  num_leaves=95   min_child_samples=200
```

其他参数保持冠军配置不变，`n_estimators` 提高到 3000，由 early stopping 决定实际轮数。

### 第二轮：只围绕第一名测试 4 组正则

```text
reg_alpha / reg_lambda / path_smooth
0 / 1  / 0
0 / 3  / 1
0.1 / 3 / 5
1 / 10 / 5
```

### 第三轮：必要时测试 3 组采样

仅当第一、二轮累计提升达到 `+0.00030` 时执行：

```text
feature_fraction=1.0  bagging_fraction=1.0  bagging_freq=0
feature_fraction=0.9  bagging_fraction=0.9  bagging_freq=1
feature_fraction=0.8  bagging_fraction=0.9  bagging_freq=1
```

### 验收与停止

- 每轮只让第一名进入下一轮；
- 最终第一名必须用正式 5 折 seed42 重跑确认；
- 最终增益小于 `+0.00030`，参数不升级到 final；
- 不因为某一个 fold 或 Public LB 单独更高而选参数。

预计实验量：10–13 个 3 折筛选候选 + 1 个正式 5 折确认。

---

## 6. 阶段四：粗粒度缺失专家

### 目的

针对不同信息完整度学习不同决策边界，同时保持所有样本的预测分数可以全局比较。

### 专家分组

第一版只使用三个专家：

```text
Expert A：5 个核心字段全部存在
Expert B：3–4 个核心字段存在
Expert C：0–2 个核心字段存在
```

不按 32 种 pattern 分别建模。

### 修改范围

- 新增 `src/smartphone_addiction/training/experts.py`
- 新增 `src/smartphone_addiction/evaluation/expert_blend.py`
- 修改 `src/smartphone_addiction/training/runner.py`
- 修改 `src/smartphone_addiction/config.py`
- 新增 `tests/unit/test_experts.py`
- 新增 `tests/unit/test_expert_blend.py`
- 新增 `configs/experiments/lightgbm_experts_v1.yaml`

### 实施顺序

1. 每个 outer fold 只用训练折拟合三个专家。
2. valid 行按自身核心观察数路由到对应专家。
3. test 行使用同样的确定性路由。
4. 保存三个专家的 raw margin/logit，而不是只保存概率。
5. 使用训练折内 cross-fitting 学习统一的线性校准器：输入为专家 logit、核心观察数和缺失组；输出统一概率。
6. meta 校准器不得看到当前 outer valid 的标签。
7. 与单一 LightGBM 使用完全相同的 outer folds 比较。

### 晋级条件

- overall OOF 至少 `+0.00030`；
- Expert B、C 对应切片至少一个提升 `+0.001`；
- 完整样本不得明显退化；
- 跨完整/缺失组的 pairwise AUC 不得下降。

若三个专家失败，只允许再补一个“daily 是否存在”的二专家版本；仍失败则停止专家路线。

---

## 7. 阶段五：引入低相关新模型

### 目的

获得与 LightGBM 预测相关性更低的 OOF，而不是重复训练相似树模型。

### 依赖决策

进入本阶段前需单独确认是否允许在 Miniconda 环境增加依赖。当前环境没有 XGBoost、InterpretML 或 PyTorch。

### 模型顺序

#### N1：EBM/GAM

- 首选 InterpretML EBM；
- 使用 raw numeric、三个原始类别、缺失标记；
- 重点学习 daily、weekend、social 等平滑主效应和少量二阶交互；
- 先运行单 seed 5 折。

#### N2：XGBoost Hist

- 只有 EBM 未达到单模型门槛或与 LightGBM 相关性仍过高时测试；
- 使用相同 34 特征和相同 folds；
- 第一版只用 `hist`，不同时引入 DART；
- Hist 有互补价值后，再测试一个 DART 版本。

#### N3：CatBoost masked-v3

- 只在前两者没有形成有效互补时执行；
- 使用新的精简特征与 masking，而不是旧 45 特征结果；
- 先只跑一个 fold 估算时间，再决定是否完成 5 折。

### 新模型晋级标准

满足以下任意一条：

1. 单模型 OOF 比新基线高 `+0.00030`；
2. 单模型稍弱，但与 LightGBM Pearson `< 0.985`，且严格 OOF blend 提升 `≥ +0.00030`。

若相关性 `> 0.99` 且融合增益 `< +0.00015`，立即停止该模型。

---

## 8. 阶段六：核验官方原始数据来源

### 目的

确认比赛官方是否说明数据由某个公开原始数据集生成，以及比赛规则是否允许使用它。

### 执行内容

联网后只查：

1. Kaggle Overview；
2. Kaggle Data 页面；
3. Kaggle Rules；
4. 官方 Host 发布的 discussion/announcement。

输出必须记录：来源 URL、官方原文、数据许可证、字段对应关系、是否允许外部数据。

### 决策

- 官方明确给出并允许：另写独立的 original-data 融合方案；
- 来源不明确、字段无法可靠对应或规则不允许：停止，不猜测、不混入相似网络数据。

本阶段不与前面的模型实验互相阻塞，网络恢复后可单独进行。

---

## 9. 阶段七：多 seed、融合与提交

### 进入条件

只有单 seed 相对当前冠军提升至少 `+0.00030` 的方案才进入 final。

### 执行顺序

1. 用 seeds `[42, 2026, 3407]` 跑 5 折 final。
2. 生成每个 seed 的 OOF/test prediction。
3. 先比较等权 seed 平均，不搜索细碎 seed 权重。
4. 只融合通过阶段五门槛的低相关模型。
5. blend 权重使用严格 OOF，步长先用 0.05；只有最优点稳定才缩小到 0.01。
6. 检查 OOF、test-pattern-weighted AUC、折间方差和 test prediction 分布。
7. 生成 submission，但上传 Kaggle 前由用户确认。
8. Public LB 只用于最终验证，不回头反复按 Public LB 调参。

### 最终候选数量

最多保留三个提交：

1. 新最强单模型；
2. 新最强低相关 blend；
3. 风险较低的当前冠军或稳定版本。

---

## 10. 总体执行顺序与检查点

```text
阶段一：修复 tuning/masking 一致性
    ↓ 必须完成
阶段二：Masking v3
    ↓ 选出新缺失适配基线
阶段三：定向 LightGBM 搜索
    ↓ 得到最强单树基线
阶段四：缺失专家
    ↓ 判断结构化路由是否有效
阶段五：低相关新模型
    ↓ 判断融合是否有真正空间
阶段六：官方原始数据核验（可独立进行）
    ↓
阶段七：3 seed final → OOF blend → 用户确认提交
```

建议设置三个检查点：

### Checkpoint A：阶段三结束

- 若 OOF 已达到 `≥ 0.9660`：继续专家和新模型；
- 若仍低于 `0.9650`：说明 masking/调参收益不足，优先转向新模型或官方原始数据，不继续微调 LightGBM。

### Checkpoint B：阶段五结束

- 若单 seed OOF 达到 `≥ 0.9670`：进入多 seed final，存在冲击 Public LB 0.97 的现实机会；
- 若仍低于 `0.9660`：本数据内仅靠常规模型达到 0.97 的概率较低，应把重点放到 original dataset 或接受当前上限。

### Checkpoint C：最终提交前

- final OOF 至少比当前 `0.96425464` 提高 `+0.00050`；
- 5 折方向稳定；
- test prediction 无异常；
- 由用户决定是否上传。

## 11. 明确不再执行的方向

- 旧 CatBoost/旧 LightGBM 与当前冠军继续调 blend；
- 搜索三个 seed 的细碎权重；
- 恢复 `missing_pattern`；
- 恢复旧 categorical interactions 和全部 ratios；
- 重复已失败的折内插补、条件分位和单调约束；
- 加入 `id`；
- 精确重复行 target encoding；
- 仅按缺失组做简单概率校准；
- 未经官方确认使用网络上的相似外部数据。

这些方向已经被真实 OOF、相关性或数据结构诊断排除。
