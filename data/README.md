# 比赛数据说明

本目录用于存放 Kaggle 官方数据，**不得重新分发**比赛原始 CSV。

## 约定

- `data/raw/`：只读官方文件  
  `train.csv`、`test.csv`、`sample_submission.csv`
- `data/processed/`：由本仓库特征流水线生成的 parquet / manifest  
  （大文件默认不进入 Git）

请通过 Kaggle CLI 自行下载，并确保已接受比赛规则：

```bash
kaggle competitions download -c playground-series-s6e8 -p data/raw --force
```

切勿提交 `kaggle.json`、API Token 或原始数据到公开仓库。
