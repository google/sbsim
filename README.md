# Google Smart Buildings Control

This repository accompanies Goldfeder, J., Sipple, J., Real-World Data and
Calibrated Simulation Suite for Offline Training of Reinforcement Learning
Agents to Optimize Energy and Emission in Office Buildings, currently under
review at Neurips 2024, and builds off of Goldfeder, J., Sipple, J., (2023).
[A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings](https://dl.acm.org/doi/10.1145/3600100.3625682),
BuildSys '23, November 15–16, 2023, Istanbul, Turkey

## Real World Data

In addition to our calibrated simulator, we have released six years of data from
three buildings. This data can be used for further simulator calibration, and
for training and evaluating reinforcement learning (RL) models.

The dataset is available for download from
[Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/smart_buildings).

Alternatively, a smaller version of the dataset can be downloaded as a
[zip file](https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip)
from cloud storage.

## Documentation

View the official [Documentation Site](https://google.github.io/sbsim/) for a
complete auto-generated API reference.

There is also a legacy unofficial
[Community-run Documentation Site](https://gitwyd.github.io/sbsim_documentation/)
containing more information about the project and the codebase. We plan to merge
all this content into the official documentation site soon.

## Getting Started

A great place to start is by reviewing the
[Soft Actor Critic Demo notebook](smart_control/notebooks/SAC_Demo.ipynb). This
notebook will walk you through:

1. Creating a gym-compatible Reinforcement Learning (RL) environment.

2. Visualizing the environment.

3. Training an agent using the
   [Tensorflow Agents Library](https://www.tensorflow.org/agents).

Alternatively, RL agents can be trained by running various scripts in the
"smart_control/reinforcement_learning/scripts" directory.

Before running notebooks or scripts, make sure to complete the setup
instructions linked below.

## Setup

The [Setup Guide](docs/setup.md) provides all the information you need to run
the code locally.

## Contributing

The [Contributor's Guide](docs/contributing.md) provides more information on how
to contribute to this repository.

## [License](LICENSE)
