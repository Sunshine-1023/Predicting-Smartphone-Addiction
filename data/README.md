# 比赛数据说明

本目录用于存放 Kaggle 官方数据，**不得重新分发**比赛原始 CSV。

## 约定

- `data/raw/`：只读官方文件
  `train.csv`、`test.csv`、`sample_submission.csv`
- `data/processed/`：由本仓库特征流水线生成的 parquet / manifest
  （大文件默认不进入 Git）

## 安全下载（推荐）

```bash
mkdir -p ~/.kaggle
mv /path/to/downloaded/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
# 或新版 CLI：把 token 写入 ~/.kaggle/access_token 并 chmod 600

kaggle competitions files -c playground-series-s6e8
smartphone-addiction data download --output-dir data/raw
smartphone-addiction data validate --data-dir data/raw
```

下载器会：检查凭据权限 → 下载到临时目录 → 只解压三份官方 CSV → 校验 → 原子写入 `data/raw` → 清理临时文件。
**切勿**把 `kaggle.json`、`access_token` 或原始 CSV 提交到公开仓库。

生成校验报告与 EDA 图：

```bash
python scripts/write_data_validation_report.py
```
