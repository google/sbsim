# NOTE: you have to use tabs (not spaces) to define commands in the Makefile

# these are commands to be run, not files to be created:
.PHONY: venv-activate docs docs-build docs-quiet mdformat pyink isort pylint lint test

#
# ENVIRONMENT
#

# https://stackoverflow.com/questions/33839018/activate-virtualenv-in-makefile
# this command uses a different shell, so you must run it manually:
# 	bash -c "source .venv/bin/activate"
# or consider setting as a bash alias:
# 	alias make-env='source .venv/bin/activate'
activate:
	@echo "Run this command to activate the environment: source .venv/bin/activate"

#
# LINTING AND FORMATTING
#

mdformat:
	poetry run mdformat README.md docs/*

pyink:
	poetry run pyink .

isort:
	poetry run isort .

pylint:
	poetry run pylint --rcfile=.pylintrc --ignore=proto smart_control

# mega-command for running all formatters:
format: mdformat pyink isort pylint

#
# TESTING
#

test:
	poetry run pytest --disable-pytest-warnings

#
# DOCS
#

docs:
	poetry run mkdocs serve

docs-quiet:
	poetry run mkdocs serve --quiet

docs-build:
	poetry run mkdocs build
