---
layout: "default"
title: "Reinforcement Learning Module"
nav_order: 10
parent: "Smart Control Project Documentation"
---
# Reinforcement Learning Module
## Structure
The reinforcement learning module is meant to have the code necessary to train and evaluate RL agents. It has the
following structure:
```
smart_control/reinforcement_learning/
├── agents/              # RL agent implementations (SAC, TD3, DDPG)
│   └── networks/           # Neural networks for agents
├── observers/           # Monitoring and data collection during training/evaluation
├── policies/            # Policy implementations (including baseline policies)
├── replay_buffer/       # Experience replay buffer management
├── scripts/             # Training and evaluation scripts
├── utils/               # Utility functions and helpers
└── visualization/       # Visualization tools for analysis
```

## Tutorials

Check out this [tutorial](https://youtu.be/RbpkKciw0IQ) to help get started with the RL module.
