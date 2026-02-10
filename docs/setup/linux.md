# Linux OS Setup Guide

This guide helps you get the project setup on Linux OS.

## System Package Installation

Install Linux package dependencies:

```sh
sudo apt install -y protobuf-compiler
sudo apt install -y ffmpeg
sudo apt install -y python3.12-venv
```

## Python Installation

We are using Pyenv to manage and install specific versions of Python.

First
[install and configure Pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

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

Then use Pyenv to install a compatible Python version (e.g. Python 3.11):

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
pip install poetry==2.1.2
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

protoc --python_out=. smart_control_building.proto
protoc --python_out=. smart_control_normalization.proto
protoc --python_out=. smart_control_reward.proto

cd ../..
```

> NOTE: the generated "\*pb2.py" files have been checked in to the repository to
> facilitate publishing this package on PyPI.

> NOTE: contributors can skip this step and just use the current versions of the
> protos that have been checked in to the repository. Maintainers can run this
> step periodically to update the protos.

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

Create a kernel:

```sh
poetry run python -m ipykernel install --user --name=sbsim-kernel
```

Finally you can run the notebook using Jupyter or VS Code:

- A) Run the notebooks using Jupyter by running `poetry run jupyter notebook`
  (then visit the resulting [localhost:8000](localhost:8000) URL in the browser,
  and choose the "sbsim-kernel" from the kernel drop-down menu).

- B) Run the notebooks using VS Code (choosing the "sbsim-kernel" kernel from
  the kernel drop-down menu).
