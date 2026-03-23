.PHONY: test lint run setup

setup:
	venv/bin/pip install -e ".[dev]"

test:
	venv/bin/pytest tests/ -v

lint:
	venv/bin/ruff check .

run:
	venv/bin/uvicorn src.api.main:app --reload

mlflow:
	venv/bin/mlflow ui
