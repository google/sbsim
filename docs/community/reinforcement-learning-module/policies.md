---
layout: "default"
title: "Policies"
nav_order: 2
parent: "Reinforcement Learning Module"
grand_parent: "Smart Control Project Documentation"
---

# Policies

Policies determine the agent's behavior by mapping observations to actions. Policies can be deterministic (e.g. greedy policies) or stochastic (e.g. exploratory policies). A policy can also be learned by an agent,
or it can be just a simple rule-based policy (e.g. a fixed building schedule).

In this module, we provide two classes. `SchedulePolicy` is used to define rule-based policies that can be used as
baselines for RL agents. `SavedModelPolicy` is a wrapper used to load and interact with policies that have been
learned and saved.

All of the policies in this project should [`tf_agents.policies.TFPolicy`](https://www.tensorflow.org/agents/api_docs/python/tf_agents/policies/TFPolicy) to provide a consistent interface to interact with policies.

## SchedulePolicy

**Purpose**: Implements traditional rule-based control strategies based on time schedules.

**Implementation**: `SchedulePolicy` in `schedule_policy.py`

**Usage Example**: see the implementation of the `create_baseline_schedule_policy` function in `policies/schedule_policy.py`, which
instantiates an example baseline policy.

## SavedModelPolicy

**Purpose**: Loads and uses a previously trained policy that has been saved.

**Implementation**: `SavedModelPolicy` in `saved_model_policy.py`

**Usage Example**:

```python
from smart_control.reinforcement_learning.policies.saved_model_policy import SavedModelPolicy

# Load a saved policy
policy = SavedModelPolicy(
    saved_model_path="path/to/saved/policy",
    time_step_spec=time_step_spec, # obtained through tf_env.time_step_spec()
    action_spec=action_spec # obtained through tf_env.action_spec()
)
```

---

[Back to Reinforcement Learning](../reinforcement-learning-module.md)

[Back to Home](../../index.md)
