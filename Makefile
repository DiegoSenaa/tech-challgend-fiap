.PHONY: test lint run setup setup-prod mlflow

setup:
	pip install -e ".[dev,train,eda]"

setup-prod:
	pip install .

test:
	pytest tests/ -v

lint:
	ruff check .

run:
	uvicorn src.api.main:app --reload

mlflow:
	mlflow ui
