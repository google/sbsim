---
layout: "default"
title: "Environment Module"
nav_order: 2
parent: "Smart Control Project Documentation"
---

# Environment Module

The `environment` module provides the reinforcement learning environment where the agent interacts with the building simulation to control HVAC systems.

## `environment/environment.py`

**Purpose**: Implements a controllable building RL environment compatible with TF-Agents, allowing agents to control various setpoints with the goal of making the HVAC system more efficient.

### Key Classes and Functions

- **`Environment`** (inherits from `py_environment.PyEnvironment`):

  - **Attributes**:

    - **Building and Simulation Components**:

      - `_building`: Instance of `BaseBuilding`, representing the simulated building environment.
      - `_time_zone`: Time zone of the building/environment.
      - `_start_timestamp` and `_end_timestamp`: Start and end times of the episode.
      - `_step_interval`: Time interval between environment steps.
      - `_num_timesteps_in_episode`: Number of timesteps in an episode.
      - `_observation_request`: Template for requesting observations from the building.

    - **Agent Interaction Components**:

      - `_action_spec`: Specification of the actions that can be taken.
      - `_observation_spec`: Specification of the observations.
      - `_action_normalizers`: Mapping from action names to their normalizers.
      - `_action_names`: List of action field names.
      - `_field_names`: List of observation field names.
      - `_observation_normalizer`: Normalizes observations received from the building.
      - `_default_policy_values`: Default actions in the policy.

    - **Reward and Metrics**:

      - `_reward_function`: Instance of `BaseRewardFunction` used to compute rewards.
      - `_discount_factor`: Discount factor for future rewards.
      - `_metrics`: Stores metrics for analysis and visualization.
      - `_metrics_reporting_interval`: Interval for reporting metrics.
      - `_summary_writer`: Writes summary data for visualization tools like TensorBoard.

    - **Episode and Step Tracking**:

      - `_episode_ended`: Boolean flag indicating if the episode has ended.
      - `_step_count`: Number of steps taken in the current episode.
      - `_global_step_count`: Total number of steps taken across episodes.
      - `_episode_count`: Number of episodes completed.
      - `_episode_cumulative_reward`: Cumulative reward for the current episode.

  - **Methods**:

    - `__init__(...)`: Initializes the environment with the specified parameters and configurations.

    - **Environment Lifecycle Methods**:

      - `_reset()`: Resets the environment to its initial state, preparing for a new episode.
      - `_step(action)`: Executes one time step in the environment given an action from the agent.
      - `_has_episode_ended(last_timestamp)`: Checks if the episode has ended based on time or steps.

    - **Agent Interaction Methods**:

      - `action_spec()`: Returns the specification of the actions that can be taken.
      - `observation_spec()`: Returns the specification of the observations.
      - `_get_observation()`: Retrieves and processes observations from the building, including normalizing and handling missing data.
      - `_create_action_request(action_array)`: Converts normalized agent actions into native action values for the building.

    - **Reward Calculation Methods**:

      - `_get_reward()`: Computes the reward for the last action taken based on the reward function.
      - `_write_summary_reward_info_metrics(reward_info)`: Writes reward input metrics into summary logs.
      - `_write_summary_reward_response_metrics(reward_response)`: Writes reward output metrics into summary logs.
      - `_commit_reward_metrics()`: Aggregates and writes reward metrics, and resets accumulators.

    - **Utility Methods**:

      - `_get_action_spec_and_normalizers(...)`: Defines the action space and normalizers based on the building devices and action configurations.
      - `_get_observation_spec(...)`: Defines the observation space based on the building devices and observation configurations.
      - `_format_action(action, action_names)`: Reformat actions if necessary (extension point for subclasses).
      - `render(mode)`: (Not implemented) Intended for rendering the environment state.

    - **Properties**:

      - `steps_per_episode`: Number of steps in an episode.
      - `start_timestamp`: Start time of the episode.
      - `end_timestamp`: End time of the episode.
      - `default_policy_values`: Default actions in the policy.
      - `label`: Label for the environment or episode.
      - `current_simulation_timestamp`: Current simulation time.

## Environment Workflow

1. **Initialization**:

   - The environment is initialized with specified parameters, including the building simulation, reward function, observation normalizer, and action configurations.
   - Action and observation specifications are set up based on devices and configurations, involving action normalizers and mappings.
   - Auxiliary features such as time of day and day of week are prepared.

2. **Resetting the Environment**:

   - The `_reset()` method resets the building simulation and environment metrics.
   - Episode counters and timestamps are initialized.
   - The initial observation is generated by calling `_get_observation()`.

3. **Stepping through the Environment**:

   - **Action Application**:

     - The `_step(action)` method processes the action from the agent.
     - Actions are normalized and converted into native values using `_create_action_request(action_array)`.
     - The action request is sent to the building simulation via `self.building.request_action(action_request)`.
     - The environment handles action responses, including logging and handling rejections.

   - **Observation Retrieval**:

     - After the action is applied, `_get_observation()` retrieves observations from the building.
     - Observations are normalized using the observation normalizer.
     - Time features (hour of day, day of week) and occupancy features are added to the observation.
     - Missing or invalid observations are handled using past data.

   - **Reward Calculation**:

     - The reward is computed using `_get_reward()`, which invokes the reward function's `compute_reward()` method.
     - Reward metrics are logged and written to summary writers if configured.

   - **Episode Termination**:

     - The environment checks if the episode has ended using `_has_episode_ended(last_timestamp)`.
     - If the episode has ended, a terminal time step is returned, and the environment is reset for the next episode.

---

[Back to Home](../index.md)
