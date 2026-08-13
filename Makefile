.PHONY: setup setup-update test test-model lint format build features validate-data train train-raw train-lgbm train-lgbm-raw train-dev train-lgbm-dev train-final train-lgbm-final tune submission submit package pre-commit help

COMPETITION ?= playground-series-s6e8
# Leave unset to derive submissions/<run_dir_name>.csv via the CLI.
SUBMISSION_CSV ?=

help:
	@echo "Available targets:"
	@echo "  setup         Create conda env from environment.yml"
	@echo "  setup-update  Update env deps via pip editable install"
	@echo "  pre-commit    Install git hooks from .pre-commit-config.yaml"
	@echo "  test          Run unit/integration tests (exclude slow)"
	@echo "  test-model    Run CatBoost/LightGBM adapter tests"
	@echo "  lint          Ruff format check + lint"
	@echo "  format        Auto-format with Ruff"
	@echo "  build         Build wheel/sdist"
	@echo "  features      Build processed parquet via CLI"
	@echo "  validate-data Validate raw CSVs and write data_validation.md + EDA figures"
	@echo "  train         Smoke train CatBoost with full domain feature groups"
	@echo "  train-raw     Smoke train CatBoost with raw features only"
	@echo "  train-lgbm    Smoke train LightGBM with full domain feature groups"
	@echo "  train-lgbm-raw Smoke train LightGBM with raw features only"
	@echo "  train-dev     Full-data CatBoost 5-fold (single seed)"
	@echo "  train-lgbm-dev Full-data LightGBM 5-fold (single seed)"
	@echo "  train-final   Full-data CatBoost 5-fold x 3 seeds"
	@echo "  train-lgbm-final Full-data LightGBM 5-fold x 3 seeds"
	@echo "  tune          Bounded Optuna tune via CLI"
	@echo "  submission    Build submission CSV from RUN_DIR (default: submissions/<run>.csv; FORCE=1 to overwrite)"
	@echo "  submit        Build submission CSV then upload via Kaggle CLI (RUN_DIR required; FORCE=1 to overwrite)"
	@echo "  package       Build offline Kaggle bundle (CONFIG required)"

setup:
	conda env create -f environment.yml

setup-update:
	pip install -e ".[analysis,dev,tools]"

pre-commit:
	pre-commit install
	pre-commit run --all-files

test:
	@before=$$(mktemp); \
	bash scripts/check_reports_unchanged.sh snapshot "$$before"; \
	python -m pytest -m "not slow" -q; \
	status=$$?; \
	if [ $$status -ne 0 ]; then rm -f "$$before"; exit $$status; fi; \
	bash scripts/check_reports_unchanged.sh check "$$before"; \
	rm -f "$$before"

test-model:
	python -m pytest -m model -q

lint:
	python -m ruff format --check .
	python -m ruff check .

format:
	python -m ruff format .
	python -m ruff check --fix .

build:
	python -m build --no-isolation

features:
	smartphone-addiction features build --raw-dir data/raw --out-dir data/processed

validate-data:
	smartphone-addiction data validate --data-dir data/raw
	python scripts/write_data_validation_report.py

train:
	smartphone-addiction train \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/catboost.yaml \
		--experiment configs/experiments/catboost_domain_v1.yaml

train-raw:
	smartphone-addiction train \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/catboost.yaml

train-lgbm:
	smartphone-addiction train \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/lightgbm.yaml \
		--experiment configs/experiments/lightgbm_domain_v1.yaml

train-lgbm-raw:
	smartphone-addiction train \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/lightgbm.yaml

train-dev:
	smartphone-addiction train \
		--profile configs/profiles/dev.yaml \
		--model-config configs/models/catboost.yaml \
		--experiment configs/experiments/catboost_domain_v1.yaml

train-lgbm-dev:
	smartphone-addiction train \
		--profile configs/profiles/dev.yaml \
		--model-config configs/models/lightgbm.yaml \
		--experiment configs/experiments/lightgbm_domain_v1.yaml

train-final:
	smartphone-addiction train \
		--profile configs/profiles/final.yaml \
		--model-config configs/models/catboost.yaml \
		--experiment configs/experiments/catboost_domain_v1.yaml

train-lgbm-final:
	smartphone-addiction train \
		--profile configs/profiles/final.yaml \
		--model-config configs/models/lightgbm.yaml \
		--experiment configs/experiments/lightgbm_domain_v1.yaml

tune:
	smartphone-addiction tune \
		--profile configs/profiles/smoke.yaml \
		--model-config configs/models/lightgbm.yaml \
		--experiment configs/experiments/lightgbm_masked_v3.yaml \
		--n-trials 20

submission:
	@test -n "$(RUN_DIR)" || (echo "Usage: make submission RUN_DIR=artifacts/runs/<run> [SUBMISSION_CSV=...] [FORCE=1]" && exit 1)
	smartphone-addiction submission build --run "$(RUN_DIR)" \
		$(if $(SUBMISSION_CSV),--output "$(SUBMISSION_CSV)",) \
		$(if $(FORCE),--force,)

submit:
	@test -n "$(RUN_DIR)" || (echo "Usage: make submit RUN_DIR=artifacts/runs/<run> [MSG=...] [COMPETITION=...] [SUBMISSION_CSV=...] [FORCE=1]" && exit 1)
	@set -e; \
	csv="$(if $(SUBMISSION_CSV),$(SUBMISSION_CSV),submissions/$(notdir $(RUN_DIR)).csv)"; \
	smartphone-addiction submission build --run "$(RUN_DIR)" --output "$$csv" $(if $(FORCE),--force,) && \
	kaggle competitions submit \
		-c "$(COMPETITION)" \
		-f "$$csv" \
		-m "$(or $(MSG),submission from $(RUN_DIR))"

package:
	@test -n "$(CONFIG)" || (echo "Usage: make package CONFIG=configs/experiments/catboost_domain_v1.yaml" && exit 1)
	smartphone-addiction package kaggle --config "$(CONFIG)"
