# Predicting Smartphone Addiction 工程设计

## 1. 项目目标

本项目为 Kaggle Playground Series Season 6 Episode 8：Predicting Smartphone Addiction 构建一套完整、可复现、适合公开展示的表格二分类竞赛工程。

首要目标是学习和实践可信的机器学习工程流程，并争取稳定进入比赛前 20%。排名目标不是工程验收的硬条件，因为它还受到参赛人数、其他选手方案和赛程变化影响。

核心版本计划在 14 天内完成，每天投入约 2～4 小时。第一周建立可提交的完整流程，第二周进行适度特征工程、有限调参、多种子复核、模型融合和作品集整理。

## 2. 已确认的约束

- 运行方式：本地开发和小样本测试，Kaggle CPU 执行完整训练。
- 本地环境：Miniconda、Python 3.11。
- 依赖管理：`environment.yml` 创建环境，`pyproject.toml` 管理 Python 包依赖和工具配置。
- 代码组织：Python 包为核心，Notebook 只负责 EDA 和结果展示。
- 模型范围：Dummy/Logistic、CatBoost、LightGBM，以及经过 OOF 验证的简单融合。
- 验证强度：开发阶段 5 折单种子，正式阶段 5 折乘 3 个种子。
- 最终种子：`42`、`2026`、`3407`。
- 特征范围：原始特征、缺失模式和少量可解释领域特征。
- 预处理：各模型使用适合自身的折内预处理。
- 配置：YAML 配置加命令行覆盖。
- 实验追踪：本地文件化追踪，不使用 MLflow 或外部服务。
- 调参：每个主模型进行有限的 Optuna 搜索，候选参数必须在完整验证上复核。
- 数据范围：只使用官方比赛数据；第一版不使用外部数据或伪标签。
- 数据获取：Kaggle CLI，凭据位于 `~/.kaggle/kaggle.json`。
- 提交：程序生成并校验 CSV，用户手动提交。
- GitHub：立即建立公开仓库 `predicting-smartphone-addiction`。
- Python 包名：`smartphone_addiction`。
- 许可证：MIT License，只覆盖原创代码和文档。
- 文档：中文为主，README 提供英文摘要。
- 质量保障：pytest、Ruff、pre-commit、GitHub Actions。
- 协作：由 Codex 实现，用户逐阶段审核。

## 3. 总体架构

项目采用配置驱动的模块化 Python 单体。只有一套核心训练逻辑，本地命令行和 Kaggle Notebook 调用相同入口，不维护第二套 Notebook 训练实现。

```text
predicting-smartphone-addiction/
├── README.md
├── LICENSE
├── environment.yml
├── pyproject.toml
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── base.yaml
│   ├── profiles/
│   │   ├── smoke.yaml
│   │   ├── dev.yaml
│   │   └── final.yaml
│   ├── models/
│   │   ├── logistic.yaml
│   │   ├── catboost.yaml
│   │   └── lightgbm.yaml
│   └── experiments/
│       └── README.md
├── data/
│   ├── raw/
│   └── README.md
├── src/
│   └── smartphone_addiction/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── paths.py
│       ├── data/
│       │   ├── load.py
│       │   ├── schema.py
│       │   └── validate.py
│       ├── features/
│       │   ├── base.py
│       │   └── domain.py
│       ├── models/
│       │   ├── base.py
│       │   ├── logistic.py
│       │   ├── catboost.py
│       │   └── lightgbm.py
│       ├── training/
│       │   ├── cv.py
│       │   ├── runner.py
│       │   └── tuning.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── blend.py
│       │   └── importance.py
│       ├── artifacts/
│       │   ├── manifest.py
│       │   └── store.py
│       └── submission.py
├── scripts/
│   ├── package_kaggle.py
│   └── verify_environment.py
├── kaggle/
│   └── run_competition.ipynb
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_validation_analysis.ipynb
│   └── 03_final_report.ipynb
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
│       └── test_smoke_pipeline.py
├── artifacts/
├── submissions/
└── reports/
    ├── figures/
    ├── experiment_summary.csv
    └── final_report.md
```

### 3.1 模块边界

- `data/` 只负责加载、字段校验和数据完整性检查。
- `features/` 只负责确定性的特征变换，不训练模型、不写产物。
- `models/` 封装模型专属预处理、拟合和概率预测。
- `training/` 负责交叉验证、随机种子、early stopping 和 Optuna 调度。
- `evaluation/` 负责 AUC、稳定性统计、融合、消融和特征重要性。
- `artifacts/` 负责实验 Manifest、指标、模型和预测的可靠保存。
- `submission.py` 只负责生成和校验提交文件，不上传 Kaggle。
- Notebook 只调用包接口并读取已有产物，不保存唯一业务逻辑。
- `reports/` 保存可公开的轻量结果；`data/raw/`、`artifacts/` 和 `submissions/` 不进入 Git。

## 4. 配置体系

配置按以下顺序合并，后层覆盖前层：

```text
base.yaml
  -> profiles/{smoke|dev|final}.yaml
  -> models/{model}.yaml
  -> experiments/{experiment}.yaml
  -> CLI overrides
```

- `base.yaml`：路径、目标列、评价指标、默认日志和全局约定。
- `profiles/smoke.yaml`：分层小样本、单折，验证程序能否完成。
- `profiles/dev.yaml`：完整数据、5 折、种子 42。
- `profiles/final.yaml`：完整数据、5 折、3 个种子。
- `models/`：模型默认参数和专属预处理选项。
- `experiments/`：描述具体实验假设和启用的特征组。

运行开始后必须保存解析完成的完整配置。历史实验不依赖后来可能被修改的源 YAML。

## 5. 数据契约与处理流程

项目只读取官方 `train.csv`、`test.csv` 和 `sample_submission.csv`。原始文件只读，程序不得覆盖或修改。

训练前检查：

- 三个文件存在且可读；
- 列名、目标列和预期结构一致；
- `id` 唯一；
- 目标只包含整数 `0` 和 `1`；
- 训练集与测试集的特征集合一致；
- 样例提交 ID 与测试集完全对齐；
- 缺失值被视为合法数据；
- 非法无穷值和不可解析类型会触发错误。

数据流为：

```text
Kaggle CLI 下载
  -> 文件与 Schema 校验
  -> 分层交叉验证切分
  -> 折内特征工程
  -> 模型专属预处理
  -> 拟合与概率预测
  -> OOF AUC 和稳定性统计
  -> 测试预测跨折、跨种子平均
  -> 单模型或融合预测
  -> 提交文件校验
```

`id` 默认只用于数据对齐和提交，不进入模型。是否保留 `id` 只能通过独立消融实验决定。所有需要学习参数的预处理都只在当前训练折拟合，防止交叉验证泄漏。

## 6. 特征设计

特征按组开关，每组都需要消融验证。

### 6.1 原始特征

使用比赛提供的 12 个原始预测特征。

### 6.2 缺失特征

- 每行缺失字段数量；
- 每行缺失比例；
- 关键行为字段是否缺失的布尔标记。

### 6.3 行为总量特征

- 社交媒体时间加游戏时间；
- 工作学习时间与娱乐时间差；
- 总屏幕时间与已知用途时间之差。

### 6.4 行为比值与差值

- 屏幕时间除以睡眠时间；
- 周末屏幕时间除以日常屏幕时间；
- App 打开次数除以通知数；
- 通知数除以屏幕小时；
- App 打开次数除以屏幕小时；
- 周末与日常屏幕时间差。

除法使用统一安全实现。分母接近零时输出缺失值，不产生无穷值。组成字段缺失时派生字段也保持缺失，避免把不完整总量误当成真实值。

第一版不包含目标编码、伪标签、外部数据和无解释的大规模多项式组合。

## 7. 模型设计

### 7.1 DummyClassifier

根据训练折正类先验输出固定概率，用于验证指标、OOF 和提交管道。其 AUC 应接近 0.5。

### 7.2 LogisticRegression

可解释线性基线：

- 数值特征使用中位数插补和标准化；
- 类别特征使用众数插补和 One-Hot；
- 所有转换封装在折内 Pipeline。

### 7.3 CatBoost

主力模型：

- 原生使用类别特征；
- 数值缺失值保留；
- 类别缺失值转换为 `__MISSING__`；
- 使用 ROC-AUC 监控和 early stopping；
- CPU 训练并由配置限制线程数。

### 7.4 LightGBM

独立对照模型：

- 使用稳定的类别映射；
- 保留数值缺失值；
- 使用 early stopping；
- 与 CatBoost 产生差异化预测以支持融合。

## 8. 验证、调参与融合

### 8.1 验证分级

- `smoke`：分层小样本、单折，只验证代码和产物。
- `dev`：完整数据、5 折、种子 42。
- `final`：完整数据、5 折乘种子 `42`、`2026`、`3407`，共 15 次拟合。

每个模型记录：

- 每折 ROC-AUC；
- 每个种子的 pooled OOF AUC；
- 多次拟合的均值、标准差、最小值和最大值；
- 训练和预测耗时；
- 最佳迭代次数；
- OOF 覆盖率和预测分布。

候选方案通常需要相对当前基线平均提升至少约 0.0002，或者在分数基本持平时明显降低耗时和复杂度。提升不得只来自单个折，最终方案必须通过三种子复核。该阈值是实验决策参考，可根据实际折间方差调整。

### 8.2 Optuna

- 先建立手工基线；
- 每个主模型最多约 20 个 Trial；
- 调参使用 50% 分层样本、3 折、单种子；
- 每个模型选取前三组参数；
- 在完整数据、5 折上重新比较；
- 最优方案再执行正式 15 次拟合；
- Optuna 阶段分数不直接写作最终成绩。

Optuna 使用 SQLite 保存 Study，支持从中断位置继续。

### 8.3 融合

- 只融合完成完整 OOF 验证的模型；
- 分析 CatBoost 和 LightGBM 的预测相关性；
- 使用 OOF 预测按 0.05 步长搜索简单权重；
- 比较概率加权与排名平均；
- Logistic 只有在提供稳定增益时才进入融合；
- 选定权重原样应用于测试预测；
- 不根据 Public Leaderboard 反复调整权重。

## 9. 实验产物

每次运行生成不可覆盖的目录：

```text
artifacts/runs/<run-id>/
├── resolved_config.yaml
├── manifest.json
├── metrics.json
├── fold_metrics.csv
├── feature_names.json
├── oof_predictions.parquet
├── test_predictions.parquet
├── training.log
├── models/
└── importance/
```

实验 ID 由 UTC 时间、模型与实验简称、当前 Git commit 短哈希组成。

Manifest 记录：

- Git commit 和工作区是否存在未提交修改；
- 数据文件 SHA-256；
- Python、Conda 和关键依赖版本；
- 本地或 Kaggle 环境；
- 训练行数、特征数量、随机种子和折数；
- 产物文件列表及状态；
- 开始、结束时间和退出状态。

正式训练逐折保存检查点。显式 `--resume <run-id>` 恢复前必须验证配置哈希和数据哈希一致，不一致时拒绝继续。

公开的 `reports/experiment_summary.csv` 只收录主动选择的实验以及轻量指标。模型、预测、提交和完整日志不得提交到 GitHub。

## 10. Kaggle 离线运行

本地打包生成：

```text
dist/
├── smartphone_addiction-<version>.whl
├── kaggle_bundle-<git-sha>.zip
└── bundle_manifest.json
```

Kaggle Notebook 只负责：

1. 挂载比赛数据和离线代码包；
2. 解压到 `/kaggle/working/`；
3. 使用 `--no-deps` 安装项目 Wheel；
4. 检查关键预装依赖；
5. 调用与本地相同的 CLI；
6. 将产物写入 `/kaggle/working/artifacts/`；
7. 提供产物下载，不自动提交。

缺失关键依赖或版本不兼容时应在训练开始前失败，并显示明确差异。

## 11. CLI 设计

使用 Typer 提供统一命令：

```bash
smartphone-addiction data download
smartphone-addiction data validate
smartphone-addiction train --config <config>
smartphone-addiction tune --config <config>
smartphone-addiction blend --runs <run-id> <run-id>
smartphone-addiction submission build --run <run-id>
smartphone-addiction report experiments
smartphone-addiction package kaggle
```

`Makefile` 仅提供常用快捷命令，实际业务逻辑全部位于 Python 包中。

## 12. 提交文件

提交生成器验证：

- 行数与测试集一致；
- 列名严格为 `id,addicted_label`；
- ID 与样例提交完全一致且顺序相同；
- 预测不含缺失值或无穷值；
- 概率全部位于 `[0,1]`；
- 文件写入后重新读取仍通过相同校验。

每个 CSV 配套一个 JSON，记录来源 Run、OOF AUC、融合权重、生成时间和 SHA-256。项目不自动上传 Kaggle。

## 13. 异常处理与日志

定义：

- `ConfigurationError`；
- `DataValidationError`；
- `TrainingError`；
- `ArtifactError`；
- `SubmissionValidationError`。

CLI 输出简洁摘要和非零退出码，完整上下文写入实验日志。配置、指标和预测采用临时文件写入后原子重命名。Run 状态使用 `running`、`completed`、`failed` 和 `interrupted`，不得把失败实验显示为成功。

## 14. 测试策略

### 14.1 单元测试

覆盖配置合并、Schema、Fold 确定性、安全除法、特征计算、模型接口、AUC 聚合、Manifest、原子写入、融合和提交校验。

### 14.2 集成测试

测试代码动态生成几百行符合 Schema 的合成数据，完整运行 Logistic smoke pipeline，并检查 OOF 覆盖、测试预测、实验产物和提交文件。CatBoost 和 LightGBM 提供极小适配器测试，不在 CI 中运行完整交叉验证。

### 14.3 人工验收

真实数据依次运行：

```bash
make download-data
make validate-data
make smoke MODEL=logistic
make smoke MODEL=catboost
make smoke MODEL=lightgbm
```

全部通过后才能启动 `dev` 或 `final`。

## 15. CI 与安全

GitHub Actions 在 Push 和 Pull Request 时：

1. 创建 Python 3.11 Miniconda 环境；
2. 校验环境文件；
3. 运行 Ruff 格式和静态检查；
4. 运行不依赖比赛数据的 pytest；
5. 构建 Wheel；
6. 检查 Notebook 输出；
7. 检查大文件和疑似密钥。

CI 不下载 Kaggle 数据、不访问 Kaggle API、不训练完整模型、不使用用户凭据。

Kaggle 凭据只允许位于 `~/.kaggle/kaggle.json`，权限必须为 `600`。项目不得读取、复制或输出 Token 内容。

`.gitignore` 排除数据、产物、提交、构建目录、环境文件、`kaggle.json`、模型文件和预测文件。pre-commit 使用 Ruff、YAML 检查、大文件检查、Notebook 输出清理和 `detect-secrets`。

## 16. 工具范围

使用：

- pandas、NumPy、PyArrow；
- scikit-learn、CatBoost、LightGBM；
- Optuna、SQLite；
- PyYAML、Pydantic、Typer；
- Matplotlib、Seaborn、JupyterLab；
- pytest、Ruff、pre-commit、detect-secrets、nbstripout；
- `python -m build`、Kaggle CLI、Git、GitHub Actions。

不使用 MLflow、Weights & Biases、DVC、Kedro、Docker、深度学习框架、SHAP 和 XGBoost。

## 17. 14 天安排

### 第一周

1. 初始化仓库、环境、包、质量工具和 CI。
2. 配置 Kaggle Token，实现下载、Schema 和数据校验。
3. 完成 EDA、缺失模式和 Train/Test 分布分析。
4. 实现配置、Fold、Artifact Store、日志和 Logistic 基线。
5. 实现 CatBoost 适配器并完成开发验证。
6. 实现 LightGBM 适配器并完成开发验证。
7. 完成提交生成、离线包和第一次手动提交。

### 第二周

8. 实现缺失和领域特征，完成分组消融。
9. 进行 CatBoost 有限 Optuna 搜索。
10. 进行 LightGBM 有限 Optuna 搜索。
11. 在完整数据上比较候选配置。
12. 完成 CatBoost 5 折乘 3 种子训练。
13. 完成 LightGBM 正式训练、融合和 Permutation Importance。
14. 完成最终提交、报告、README、Kaggle Notebook 和全新环境复现检查。

训练可以在 Kaggle 后台运行，等待期间并行完成测试、文档和分析。

## 18. 提交节奏

计划三次有明确意义的提交：

1. 完整管道基线；
2. 最佳单模型；
3. 最终融合。

只有发现明确问题时才增加修正提交。Public Leaderboard 只用于合理性检查，不形成调参反馈循环。

## 19. 公开作品集

README 包含英文摘要、中文完整说明、比赛与数据、架构、环境安装、Token 安全配置、复现命令、验证方法、实验结果、主要结论、排行榜结果、限制和改进方向。

Notebook 清除运行输出。展示图表导出到 `reports/figures/`，结论写入 `reports/final_report.md`。

## 20. 验收标准

项目完成需要同时满足：

- 全新克隆可通过 `environment.yml` 建立环境；
- GitHub Actions 全绿；
- 仓库不包含数据、Token、模型和预测；
- Kaggle CLI 能安全下载和验证官方数据；
- 三类模型均通过 smoke pipeline；
- CatBoost 和 LightGBM 至少完成一次完整 5 折 OOF；
- 最终候选完成 5 折乘 3 种子复核；
- 正式实验具备配置、数据哈希、Git commit、指标和日志；
- OOF 覆盖、预测概率和提交 ID 全部通过程序校验；
- Kaggle 离线包调用同一套训练代码；
- 至少生成一个被 Kaggle 接受的合法提交；
- 公开报告能够解释数据泄漏、OOF、AUC、特征消融和融合方法。

进入前 20% 是目标，不是工程完成的硬性验收条件。
