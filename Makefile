PYTHON = python3
PIP = pip3
MAIN = src/__main__.py
LINT_FLAGS = --warn-return-any
--warn-unused-ignores--ignore-missing-imports--disallow-untyped-defs
--check-untyped-defs


.PHONY: all install run debug clean lint lint-strict m ms

all: install run

install:
	$(PIP) install pydantic python-dotenv flake8 mypy build


run:
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN) $(CONFIG)


clean:
	rm -rf __pycache__ .mypy_cache .env dist build *.egg-info

lint:
	flake8 .
	mypy $(LINT_FLAGS) .

lint-strict:
	flake8 .
	mypy --strict .

package:
	$(PYTHON) -m build
