# Google Smart Buildings Control

This repository accompanies Goldfeder, J., Sipple, J., Real-World Data and Calibrated
Simulation Suite for Offline Training of Reinforcement Learning Agents to Optimize
Energy and Emission in Office Buildings, currently under review at Neurips 2024,
and builds off of Goldfeder, J., Sipple, J., (2023).
[A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings](https://dl.acm.org/doi/10.1145/3600100.3625682),
BuildSys ’23, November 15–16, 2023, Istanbul, Turkey

## Getting Started

The best place to jump in is the Soft Actor Critic Demo notebook,
available in notebooks/SAC_Demo.ipynb

This will walk you through:

1. Creating an RL (gym compatible) envronment

2. Visualizing the env

3. Training an agent using the [Tensorflow Agents Library](https://www.tensorflow.org/agents)

Before you run this notebook, make sure to go through the setup instructions below to ensure the notebook runs successfully.

## Setup

Follow these steps to setup locally before you run the `notebooks/SAC_Demo.ipynb` notebook. Note: this will only work on linux, as some libraries are not supported by other operating systems.

1. Clone the repository

2. Ensure you have `protoc` and `ffmpeg` installed, as well as `python >=3.10.12 and <3.12`. You can install these running `sudo apt install -y protobuf-compiler` and `sudo apt install -y ffmpeg`

3. Create a virtual environment by running `python -m venv .venv`. Activate the environment `source .venv/bin/activate`. Then, install poetry with `pip install poetry`

4. Install the dependencies by running `poetry install`

5. Build the `.proto` files at `smart_control/proto`into python files by running `cd smart_control/proto && protoc --python_out=. smart_control_building.proto smart_control_normalization.proto smart_control_reward.proto && cd ../..`  

6. Modify the value of `VIDEO_PATH_ROOT` at `smart_control/simulator/constants.py`. This is the path where simulation videos will be stored

7. Now in the `notebooks/SAC_Demo.ipynb` notebook, modify the values of `data_path`, `metrics_path`, `output_data_path` and `root_dir`. In particular, `data_path` should point to the `sim_config.gin` file at `smart_control/configs/sim_config.gin`

8. Now you are ready to run the `notebooks/SAC_Demo.ipynb` notebook 

## Real World Data

In addition to our calibrated simulator, we released 6 years of data on 3 buildings, for further calibration, and to use, in conjunction with the simulator, for training and evaluating RL models. The dataset is part of [Tensorflow Datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/smart_buildings_dataset)
