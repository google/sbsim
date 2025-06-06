# NOTE: you have to use tabs (not spaces) to define commands in the Makefile

# these are commands to be run, not files to be created:
.PHONY: activate docs docs-build docs-quiet mdformat pyink isort pylint lint test

# variable pointing to the virtual environment executable files:
VENV_BIN := .venv/bin

#
# ENVIRONMENT
#

# https://stackoverflow.com/questions/33839018/activate-virtualenv-in-makefile
# this command uses a different shell, so you must run it manually:
# bash -c "source .venv/bin/activate"
activate:
	@echo "Run this command to activate the environment: source .venv/bin/activate"

#
# LINTING AND FORMATTING
#

mdformat:
	${VENV_BIN}/mdformat README.md docs/

pyink:
	${VENV_BIN}/pyink .

isort:
	$(VENV_BIN)/isort .

pylint:
	$(VENV_BIN)/pylint --rcfile=.pylintrc --ignore=proto smart_control

# mega-command for running all formatters:
format: mdformat pyink isort pylint

#
# TESTING
#

test:
	$(VENV_BIN)/pytest --disable-pytest-warnings

#
# DOCS
#

docs:
	$(VENV_BIN)/mkdocs serve

docs-quiet:
	${VENV_BIN}/mkdocs serve --quiet

docs-build:
	$(VENV_BIN)/mkdocs build
