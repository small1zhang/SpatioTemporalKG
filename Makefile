.PHONY: setup test run lint clean help

setup:
	pip install -e .
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=stk --cov-report=term-missing

run:
	python -m stk.cli --help

lint:
	black --check stk/ tests/
	isort --check-only stk/ tests/
	flake8 stk/ tests/

format:
	black stk/ tests/
	isort stk/ tests/

typecheck:
	mypy stk/

clean:
	rmdir /s /q __pycache__ .pytest_cache 2>nul || true

help:
	@echo setup     - install stk package + dev deps
	@echo test      - run all tests
	@echo run       - show CLI help
	@echo lint      - check code style
	@echo format    - auto-format code
	@echo typecheck - type checking
