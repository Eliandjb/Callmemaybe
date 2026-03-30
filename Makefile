PYTHON = uv run python
MAIN = src/__main__.py

all: install

install:
	uv sync

run:
	uv run $(MAIN)

debug:
	uv run python -m pdb $(MAIN)

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: all install run debug clean lint lint-strict
