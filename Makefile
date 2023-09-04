PROJECT=src

.PHONY: test
test:
	poetry run pytest --cov=$(PROJECT) --cov=tests --cov-branch --junitxml=report.xml --cov-report=term-missing --cov-report html:cov_html --cov-report xml:coverage.xml tests/

.PHONY: format
format:
	poetry run black .
	poetry run isort .

.PHONY: mypy
mypy:
	poetry run mypy $(PROJECT) tests

.PHONY: pylint
pylint:
	poetry run pylint $(PROJECT)

.PHONY: ruff
ruff:
	poetry run ruff $(PROJECT) tests

.PHONY: lint
lint: mypy ruff pylint

.PHONY: check
check: format lint
