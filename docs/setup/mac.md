# Mac OS Setup Guide

This guide helps you get the project setup on Mac OS.

## System Package Installation

First install [Homebrew](https://brew.sh/) (i.e. `brew`).

Then use Homebrew to install system dependencies:

```sh
brew install protobuf@21
brew install ffmpeg
```

Checking installations:

```sh
$(brew --prefix protobuf@21)/bin/protoc --version
#> libprotoc 3.21.12

ffmpeg -version
#> 7.1.1
```

NOTE: by installing a pinned version of Protobuf, it may not be symlinked, so
normal `protoc` commands may not work without using a prefix of
`$(brew --prefix protobuf@21)/bin/protoc`. To avoid needing the prefix, as a one
time setup step, update the ".zshrc" file to add the installed location to the
path:

```sh
# this is the "~/.zshrc" file:
export PATH="$(brew --prefix protobuf@21)/bin:$PATH"
```

Remember to restart your shell afterwards:

```sh
source ~/.zshrc
```

Then you should be able to run normal `protoc` commands without the prefix:

```sh
protoc --version
#> libprotoc 3.21.12
```

## Anaconda Installation

Install [Anaconda](https://www.anaconda.com/download), which we will use to
install Python and manage a virtual environment.

The installation results in automatically adding some content to your "~/.zshrc"
file. You may need to run a `conda init zsh` command, if prompted to do so.

Remember to restart your shell afterwards.

## Virtual Environment Setup

Create the virtual environment:

```sh
conda create -n sbsim-env python=3.11
```

Activate the virtual environment:

```sh
conda activate sbsim-env
```

## Python Package Installation

We are using [Poetry](https://python-poetry.org/) to manage, install, and
configure Python package dependencies.

Install poetry:

```sh
pip install poetry==2.1.2
```

Use poetry to install dependencies, including development dependencies:

```sh
poetry install --with dev
```

> NOTE: there may be issues with the `dm-reverb` package on Mac. See:
> [https://github.com/google/sbsim/issues/102](https://github.com/google/sbsim/issues/102).
> This issue only affects reinforcement learning functionality related to replay
> buffers, so you should still be able to run all other parts of the codebase.
> We welcome contributions to fix this issue and get all the functionality
> working on Mac!

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
