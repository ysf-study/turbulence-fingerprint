.PHONY: help setup setup-dev setup-features verify test lint eval ablation fusion all docker clean

PY ?= python

help:
	@echo "make setup      install pinned dependencies"
	@echo "make setup-dev  install everything above plus pytest and ruff"
	@echo "make setup-features  add OpenCV so the five feature tests run too"
	@echo "make verify     integrity + consistency checks (26, ~5 s, no training)"
	@echo "make test       pytest (~25 s, 5 tests skipped without OpenCV)"
	@echo "make lint       ruff over src/, scripts/, tests/"
	@echo "make eval       five-fold and leave-one-generator-out (~60 s)"
	@echo "make ablation   feature-set ablation, Table 4 (~15 s)"
	@echo "make fusion     fusion with D3, Table 9 (~20 s)"
	@echo "make all        verify + eval + ablation + fusion"
	@echo "make docker     build the container that runs scripts/reproduce_all.sh"

setup:
	$(PY) -m pip install -r requirements.txt

setup-dev: setup
	$(PY) -m pip install -r requirements-dev.txt

setup-features: setup-dev
	$(PY) -m pip install -r requirements-features.txt

verify:
	$(PY) scripts/verify_release.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

eval:
	$(PY) src/train_eval.py --out results

ablation:
	$(PY) src/ablation.py --out results

fusion:
	$(PY) src/fusion_eval.py --out results

all: verify eval ablation fusion

docker:
	docker build -t tf-repro .

clean:
	rm -rf results/*.json results/*.csv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
