---
layout: "default"
title: "Deep Deterministic Policy Gradient"
nav_order: 3
parent: "Learning Algorithms"
grand_parent: "Smart Control Project Documentation"
---

# Deep Deterministic Policy Gradient (DDPG)
Implementation of this algorithm can be found in the `DDPG_Demo.ipynb` notebook.

---

Deep Deterministic Policy Gradient (DDPG) is a model-free reinforcement learning algorithm designed for continuous action spaces. It combines the actor-critic framework with deterministic policy gradients. Key features of DDPG include:

- **Actor-Critic Architecture**: Utilizes an actor network to select actions and a critic network to evaluate the Q-values of state-action pairs.
- **Off-Policy Learning**: Learns using a replay buffer to sample past experiences, improving stability and efficiency.
- **Target Networks**: Employs slowly updated target networks to stabilize training.
- **Exploration**: Uses noise (e.g., Ornstein-Uhlenbeck process) for exploration in continuous spaces.


[Back to Learning Algorithms](learning-algorithms.md)

[Back to Home](../index.md)