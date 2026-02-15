---
layout: "default"
title: "RL Agents"
nav_order: 1
parent: "Reinforcement Learning Module"
grand_parent: "Smart Control Project Documentation"
---

# Reinforcement Learning Agents

The RL agents module provides implementations of reinforcement learning algorithms tailored for building control applications. All agents follow the TF-Agents interface for consistency and interoperability.

## Agent Interface

All agents in the Smart Control framework implement the [`tf_agents.agents.tf_agent.TFAgent`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/TFAgent) interface, providing a consistent API for interacting with the models.

## Agent Factory Functions

The Smart Control framework provides factory functions for creating agents with sensible defaults:

### SAC Agent

```python
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent

# Create an SAC agent with default parameters
agent = create_sac_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

# Create an SAC agent with custom parameters
agent = create_sac_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=(256, 128),
    critic_action_fc_layers=(256, 128),
    critic_joint_fc_layers=(256, 128),
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4
)
```

### TD3 Agent

```python
from smart_control.reinforcement_learning.agents.td3_agent import create_td3_agent

# Create a TD3 agent with default parameters
agent = create_td3_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

# Create a TD3 agent with custom parameters
agent = create_td3_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=(256, 128),
    critic_action_fc_layers=(256, 128),
    critic_joint_fc_layers=(256, 128)
)
```

### DDPG Agent

```python
from smart_control.reinforcement_learning.agents.ddpg_agent import create_ddpg_agent

# Create a DDPG agent with default parameters
agent = create_ddpg_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

# Create a DDPG agent with custom parameters
agent = create_ddpg_agent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_fc_layers=(128, 128),
    critic_obs_fc_layers=(128, 64),
    critic_action_fc_layers=(128, 64),
    critic_joint_fc_layers=(128, 64)
)
```

## Networks

The networks used by each agent are kept in the `networks/` subdirectory in this module.

## Adding a New Agent

To experiment with different agents, you can create a new agent by adding a new agent file in the `agents/` directory,
and any necessary networks in the `agents/networrks/` directory. Follow the existing agents as a template,
and ensure that your agent implementation implements the
[`tf_agents.agents.tf_agent.TFAgent`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/TFAgent) interface to work out-of
the box.

After that, any training script (e.g. `train.py`) to include this new agent, which is simply adding another item to
this part of the code:

```python
# train.py script
...
# Add your agent implementation here
# Create agent based on type
    logger.info(f"Creating {agent_type} agent")
    if agent_type.lower() == 'sac':
        logger.info("Creating SAC agent")
        agent = create_sac_agent(time_step_spec=time_step_spec, action_spec=action_spec)
    elif agent_type.lower() == 'td3':
        logger.info("Creating TD3 agent")
        agent = create_td3_agent(time_step_spec=time_step_spec, action_spec=action_spec)
    elif agent_type.lower() == 'ddpg':
        logger.info("Creating DDPG agent")
        agent = create_ddpg_agent(time_step_spec=time_step_spec, action_spec=action_spec)
    else:
        logger.exception(f"Unsupported agent type: {agent_type}")
        raise ValueError(f"Unsupported agent type: {agent_type}")
...
```

---

[Back to Reinforcement Learning](../reinforcement-learning-module.md)

[Back to Home](../../index.md)
