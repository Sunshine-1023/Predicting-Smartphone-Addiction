#!/usr/bin/env python3
"""Write reports/data_validation.md and EDA figures from official validated data."""

from __future__ import annotations

from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smartphone_addiction.data.download import fingerprint_files
from smartphone_addiction.data.load import load_competition_frames
from smartphone_addiction.data.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from smartphone_addiction.paths import project_root


def main() -> None:
    root = project_root()
    raw_dir = root / "data" / "raw"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = load_competition_frames(raw_dir)
    digests = fingerprint_files(raw_dir)
    train = frames.train
    test = frames.test
    validated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    target_counts = train["addicted_label"].value_counts().sort_index()
    target_rate = float(train["addicted_label"].mean())
    train_missing = train[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)
    test_missing = test[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    target_counts.plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Train target counts")
    axes[0].set_xlabel("addicted_label")
    train["daily_screen_time_hours"].hist(bins=40, ax=axes[1], color="#4C78A8")
    axes[1].set_title("daily_screen_time_hours (train)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_target_screen.png", dpi=120)
    plt.close(fig)

    miss = pd.DataFrame({"train": train_missing, "test": test_missing})
    ax = miss.plot(kind="bar", figsize=(12, 4))
    ax.set_title("Missing value counts by feature")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_missing_counts.png", dpi=120)
    plt.close()

    cmp = pd.DataFrame(
        {
            "train_mean": train[NUMERIC_COLUMNS].mean(),
            "test_mean": test[NUMERIC_COLUMNS].mean(),
        }
    )
    cmp["abs_diff"] = (cmp["train_mean"] - cmp["test_mean"]).abs()
    cmp = cmp.sort_values("abs_diff", ascending=False)
    cmp.to_csv(fig_dir / "eda_train_test_numeric_means.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    cmp["abs_diff"].plot(kind="bar", ax=ax, color="#54A24B")
    ax.set_title("Train vs test numeric mean absolute difference")
    ax.set_ylabel("|mean_train - mean_test|")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_train_test_mean_diff.png", dpi=120)
    plt.close(fig)

    for column in CATEGORICAL_COLUMNS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        train[column].fillna("__MISSING__").value_counts().plot(kind="bar", ax=axes[0])
        test[column].fillna("__MISSING__").value_counts().plot(kind="bar", ax=axes[1])
        axes[0].set_title(f"train {column}")
        axes[1].set_title(f"test {column}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"eda_categorical_{column}.png", dpi=120)
        plt.close(fig)

    # --- markdown report ---
    lines = [
        "# Official data validation",
        "",
        f"- Validated at (UTC): `{validated_at}`",
        "- Source directory: `data/raw`",
        "- Loader: `smartphone_addiction.data.load.load_competition_frames`",
        "",
        "## File fingerprints (SHA-256)",
        "",
    ]
    for name, digest in digests.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "## Scale",
            "",
            f"- train rows: **{len(train):,}**",
            f"- test rows: **{len(test):,}**",
            f"- sample_submission rows: **{len(frames.sample_submission):,}**",
            f"- raw feature columns: **{len(FEATURE_COLUMNS)}**",
            "",
            "## Target distribution (train)",
            "",
            f"- positive rate: **{target_rate:.6f}**",
        ]
    )
    for label, count in target_counts.items():
        lines.append(f"- label `{label}`: **{int(count):,}**")
    lines.extend(["", "## Missing value counts (top features)", "", "### Train", ""])
    for column, count in train_missing.head(12).items():
        lines.append(f"- `{column}`: {int(count):,}")
    lines.extend(["", "### Test", ""])
    for column, count in test_missing.head(12).items():
        lines.append(f"- `{column}`: {int(count):,}")
    lines.extend(
        [
            "",
            "## EDA figures",
            "",
            "- `reports/figures/eda_target_screen.png`",
            "- `reports/figures/eda_missing_counts.png`",
            "- `reports/figures/eda_train_test_mean_diff.png`",
            "- `reports/figures/eda_categorical_*.png`",
            "- `reports/figures/eda_train_test_numeric_means.csv`",
            "",
            "All facts above come from the official validated frames; "
            "no community notebook conclusions were copied.",
            "",
        ]
    )
    report_path = root / "reports" / "data_validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"figures under {fig_dir}")


if __name__ == "__main__":
    main()
