# Local Development Setup Guide

This document provides instructions for getting the repository set up for local
development. By default, we use Linux OS, as some dependencies are not supported
by other operating systems, however we are also providing a "Dockerfile" to
facilitate running the code on non-Linux (Mac and Windows) systems.

This project requires the following dependencies:

- [Protocol Buffer Compiler](https://grpc.io/docs/protoc-installation/) (i.e.
  "protoc")
- [FFmpeg](https://ffmpeg.org/)
- Python (>=3.10.12 and \<3.12), as installed using
  [Pyenv](https://github.com/pyenv/pyenv)
- [Venv](https://docs.python.org/3/library/venv.html), for managing a Python
  virtual environment

## Repository Setup

Clone the repository, for example using an SSH approach:

```sh
git clone git@github.com:google/sbsim.git
cd sbsim/
```

## Linux Package Installation

Install Linux package dependencies:

```sh
sudo apt install -y protobuf-compiler
sudo apt install -y ffmpeg
```

You may need to also install `venv`:

```sh
sudo apt install python3.12-venv
```

## Python Installation

We are using `pyenv` to manage and install specific versions of Python.

First
[install and configure `pyenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

The configuration results in adding some lines like the following to your
"~/.bashrc" file:

```sh
# this is the "~/.bashrc" file...

# Load pyenv automatically:
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Load pyenv-virtualenv automatically:
eval "$(pyenv virtualenv-init -)"
```

Remember to restart your shell afterwards.

Then use `pyenv` to install a compatible Python version (e.g. Python 3.11):

```sh
pyenv install 3.11
```

Listing the installed versions:

```sh
pyenv versions
```

Use a specific version that was installed (e.g. 3.11.11):

```sh
pyenv global 3.11.11
```

## Virtual Environment Setup

Create a Python virtual environment:

```sh
python -m venv .venv
```

> NOTE: on Google machines you may need to use `python3` instead of `python`.

Activate the virtual environment:

```sh
source .venv/bin/activate
```

## Python Package Installation

We are using [Poetry](https://python-poetry.org/) to manage, install, and
configure Python package dependencies.

Install poetry:

```sh
pip install poetry
```

You may need to specify a Python version that is compatible with this project
(e.g. Python version 3.11):

```sh
poetry env use 3.11
```

Use poetry to install dependencies, including development dependencies:

```sh
poetry install --with dev
```

## Protocol Buffer Compilation

Build the ".proto" files defined in the "smart_control/proto" directory into
Python files:

```bash
cd smart_control/proto

protoc --python_out=. smart_control_building.proto \
  smart_control_normalization.proto \
  smart_control_reward.proto

cd ../..
```

## Environment Variable Setup

By default, simulation videos are stored in the "simulator/videos" directory
(which is ignored from version control). If you would like to customize this
location, use the `SIM_VIDEOS_DIRPATH` environment variable.

You can pass environment variable(s) at runtime, or create a local ".env" file
and set your desired value(s) there:

```bash
# this is the ".env" file...

# customizing the directory where simulation videos are stored:
SIM_VIDEOS_DIRPATH="/cns/oz-d/home/smart-buildings-control-team/smart-buildings/geometric_sim_videos/"
```

## Notebook Setup

If you are running the Demo notebooks in the "smart_control/notebooks"
directory, you must modify the values of `data_path`, `metrics_path`,
`output_data_path` and `root_dir` in those notebooks. Specifically, the
`data_path` should point to the directory where the "sim_config.gin" file is
located (i.e. "smart_control/configs/sim_config.gin").

> NOTE: in the future we plan on refactoring notebook code to leverage the local
> module code and simplify this notebook setup experience. See
> [issue #83](https://github.com/google/sbsim/issues/83) (contributions
> welcome)!

You can run the notebooks using Jupyter or VS Code. Either approach requires you
to first install Jupyter. We have defined a separate installation group for
notebook-related dependencies:

```sh
poetry install --with notebooks
```

Create a kernel (required for VS Code, optional for Jupyter):

```sh
poetry run python -m ipykernel install --user --name=sbsim-kernel
```

Finally you can run the notebook using Jupyter or VS Code:

A. Run the notebooks using Jupyter (then visit the resulting
[localhost:8000](localhost:8000) URL in the browser, and optionally choose the
"sbsim-kernel" from the kernel drop-down menu):

```sh
poetry run jupyter notebook
```

B. Run the notebooks using VS Code (choosing the "sbsim-kernel" kernel from the
kernel drop-down menu).

<hr>

# Docker (Alternative Setup)

To avoid OS compatibility issues, and get the repository set up on non-Linux
environments, you can use the pre-configured Docker environment (Linux/amd64)
specified by the "Dockerfile".

Build the image:

```bash
docker build -t sbsim-env .
```

Run the container, in interactive mode, with open ports:

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace sbsim-env
```

> NOTE: the container will copy the repository into "/workspace/sbsim" on the
> first run. Use -v to persist changes.

To access Jupyter notebooks, visit
[http://localhost:8888](http://localhost:8888) in the browser.

To run scripts or tests inside the actively running docker container:

```sh
# activate the virtual environment:
source /opt/venv/bin/activate

# navigate to the repository:
cd /workspace/sbsim

# running scripts:
python path/to/script.py

# running tests:
pytest
```

To stop the container:

```sh
docker stop sbsim-env
```

> NOTE: in the future we would like to further update these instructions and
> improve the Dockerfile. See
> [issue #80](https://github.com/google/sbsim/issues/80) (contributions
> welcome)!
