# Official data validation

- Validated at (UTC): `2026-08-07T02:36:07Z`
- Source directory: `data/raw`
- Loader: `smartphone_addiction.data.load.load_competition_frames`

## File fingerprints (SHA-256)

- `train.csv`: `f4669147311c76eb03496061a852af283efcf0f12cf5c19274e775def81edd9c`
- `test.csv`: `8b462dd47fe8165cd0b082bf33b56523c5811453070af48b9f86b2eb928de49e`
- `sample_submission.csv`: `206763fe5786fb9c80d4e9289a3b812030d3dbb36450c6eb63348098154ce63e`

## Scale

- train rows: **691,369**
- test rows: **296,302**
- sample_submission rows: **296,302**
- raw feature columns: **12**

## Target distribution (train)

- positive rate: **0.709424**
- label `0`: **200,895**
- label `1`: **490,474**

## Missing value counts (top features)

### Train

- `social_media_hours`: 133,995
- `gaming_hours`: 126,821
- `weekend_screen_time`: 112,063
- `daily_screen_time_hours`: 95,854
- `app_opens_per_day`: 80,710
- `notifications_per_day`: 67,584
- `stress_level`: 55,148
- `work_study_hours`: 51,518
- `sleep_hours`: 44,480
- `academic_work_impact`: 44,224
- `gender`: 29,034
- `age`: 28,929

### Test

- `gaming_hours`: 59,420
- `weekend_screen_time`: 50,697
- `social_media_hours`: 47,397
- `notifications_per_day`: 34,221
- `daily_screen_time_hours`: 32,788
- `work_study_hours`: 27,777
- `academic_work_impact`: 25,721
- `app_opens_per_day`: 25,705
- `sleep_hours`: 22,455
- `stress_level`: 19,626
- `age`: 17,138
- `gender`: 14,212

## EDA figures

- `reports/figures/eda_target_screen.png`
- `reports/figures/eda_missing_counts.png`
- `reports/figures/eda_train_test_mean_diff.png`
- `reports/figures/eda_categorical_*.png`
- `reports/figures/eda_train_test_numeric_means.csv`

All facts above come from the official validated frames; no community notebook conclusions were copied.
