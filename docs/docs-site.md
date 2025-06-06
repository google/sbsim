# Documentation Site Guide

This document outlines how to set up the documentation site locally and how to
maintain it. The site is built using [MkDocs](https://www.mkdocs.org/) with the
[mkdocstrings](https://mkdocstrings.github.io/) plugin to generate documentation
from Python docstrings.

## Setup

Follow the instructions in the [Setup Guide](./setup.md) before moving on.

Also install packages from the "docs" group:

```sh
poetry install --with docs
```

## Building and Serving Locally

First, navigate to the project root directory:

```sh
cd path/to/sbsim
```

### Previewing

Start a local server that auto-reloads when changes are detected:

```bash
poetry run mkdocs serve

# suppress warnings:
poetry run mkdocs serve --quiet
```

While the server is running you can view the site at
[http://localhost:8000](http://localhost:8000).

> NOTE: the server hot reloads after configuration changes, however if you
> change a docstring in one of the documented Python files, you may need to
> restart the server for the changes to take effect.

### Building

Build the static site including HTML files (for deployment purposes):

```bash
poetry run mkdocs build
```

The output will be placed in the "docs_site/" directory.

## GitHub Actions Deployment

A GitHub Actions workflow is set up in ".github/workflows/deploy-docs.yml". This
workflow automatically builds and deploys the documentation to GitHub Pages
whenever changes are pushed to the default branch.

You can view the hosted site at
[https://google.github.io/sbsim](https://google.github.io/sbsim/).
