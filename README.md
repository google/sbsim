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

## Real World Data

In addition to our calibrated simulator, we released 6 years of data on 3 buildings, for further calibration, and to use, in conjunction with the simulator, for training and evaluating RL models. The dataset is part of [Tensorflow Datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/smart_buildings_dataset)
