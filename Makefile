.PHONY: install-dev fmt lint typecheck test precommit ci

install-dev:
pip install -e ".[dev]"
pre-commit install

fmt:
ruff format .
black .

lint:
ruff check .

typecheck:
mypy .

test:
pytest -q

precommit:
pre-commit run --all-files

ci: lint typecheck test
