---
layout: "default"
title: "Twin Delayed Deep Deterministic Policy Gradient"
nav_order: 4
parent: "Learning Algorithms"
grand_parent: "Smart Control Project Documentation"
---

# Twin Delayed Deep Deterministic Policy Gradient (TD3)

Implementation of TD3 can be found in the `TD3_Demo.ipynb` notebook.

---

Twin Delayed Deep Deterministic Policy Gradient (TD3) is a model-free reinforcement learning algorithm for continuous action spaces. It improves upon DDPG by addressing overestimation bias and stabilizing training. Key features include:

- **Delayed Policy Updates**: The actor network is updated less frequently than the critic networks.
- **Target Policy Smoothing**: Noise is added and clipped on target actions during critic updates.
- **Twin Critic Networks**: Two critic networks compute Q-values; the minimum is used to reduce overestimation.
- **Custom Actor Network**: A tailored actor scales outputs to valid action ranges.
- **Replay Buffer with Reverb**: A Reverb-based replay buffer stores trajectories for efficient off-policy learning.
- **TF-Agents Integration**: Leverages TensorFlow Agents for agent construction, training, and evaluation.

## Agent Architecture

### Actor Network

- Built using fully connected layers (e.g., 128, 128 units) with ReLU activations.
- The final layer uses a tanh activation to scale outputs into the action space.
- Ensures actions remain within specified bounds.

### Critic Network

- Uses separate fully connected layers for observations and actions.
- Joins the outputs via a joint network to estimate Q-values.
- Implemented with two critics to select the minimum value for target computation.

## Training Details

- **Optimizers**: Both actor and critic use the Adam optimizer (learning rate = 3e-4).
- **Hyperparameters**:
  - Discount factor (`gamma`): 0.99
  - Target update tau: 0.005
  - Target update period: 2 steps (delayed actor updates)
- **Noise Parameters**:
  - Exploration noise standard deviation: 0.1
  - Target policy noise: 0.2 (clipped at 0.5)
- **Replay Buffer**:
  - Capacity: 50,000 transitions
  - Built with Reverb to sample batches and decorrelate training data

## Training Loop

1. **Data Collection**  
   The collect policy interacts with the environment to accumulate trajectories.

2. **Gradient Updates**  
   The agent performs critic updates on every step and actor updates at a delayed frequency to improve stability.

3. **Evaluation**  
   The eval policy is used to measure performance after each training iteration.

## Noteworthy Aspects

- **Stabilized Learning**: Target policy smoothing and delayed actor updates reduce variance and mitigate overestimation.
- **Efficient Off-Policy Learning**: The use of a Reverb replay buffer ensures diverse and decorrelated training samples.
- **Custom Network Design**: A bespoke actor network enforces action bounds, while twin critics improve Q-value estimates.
- **TF-Agents Framework**: Seamless integration with TF-Agents simplifies policy definition, training, and evaluation.

[Back to Learning Algorithms](learning-algorithms.md)

[Back to Home](../index.md)