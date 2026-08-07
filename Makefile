.PHONY: setup setup-update test test-model lint format build features validate-data train tune submission package help

help:
	@echo "Available targets:"
	@echo "  setup         Create conda env from environment.yml"
	@echo "  setup-update  Update env deps via pip editable install"
	@echo "  test          Run unit/integration tests (exclude slow)"
	@echo "  test-model    Run CatBoost/LightGBM adapter tests"
	@echo "  lint          Ruff format check + lint"
	@echo "  format        Auto-format with Ruff"
	@echo "  build         Build wheel/sdist"
	@echo "  features      Build processed parquet via CLI"
	@echo "  validate-data Validate raw CSVs and write data_validation.md + EDA figures"
	@echo "  train         Smoke train via CLI (catboost + smoke profile)"
	@echo "  tune          Bounded Optuna tune via CLI"
	@echo "  submission    Build submission CSV from RUN_DIR (required)"
	@echo "  package       Build offline Kaggle bundle (CONFIG required)"

setup:
	conda env create -f environment.yml

setup-update:
	pip install -e ".[analysis,dev]"

test:
	python -m pytest -m "not slow" -q

test-model:
	python -m pytest -m model -q

lint:
	python -m ruff format --check .
	python -m ruff check .

format:
	python -m ruff format .
	python -m ruff check --fix .

build:
	python -m build

features:
	smartphone-addiction features build --raw-dir data/raw --out-dir data/processed

validate-data:
	smartphone-addiction data validate --data-dir data/raw
	python scripts/write_data_validation_report.py

train:
	smartphone-addiction train \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/catboost.yaml

tune:
	smartphone-addiction tune \
		--profile configs/profiles/smoke.yaml \
		--experiment configs/experiments/catboost_tune_v1.yaml \
		--n-trials 20

submission:
	@test -n "$(RUN_DIR)" || (echo "Usage: make submission RUN_DIR=artifacts/runs/<run>" && exit 1)
	smartphone-addiction submission build --run "$(RUN_DIR)"

package:
	@test -n "$(CONFIG)" || (echo "Usage: make package CONFIG=configs/experiments/catboost_domain_v1.yaml" && exit 1)
	smartphone-addiction package kaggle --config "$(CONFIG)"
