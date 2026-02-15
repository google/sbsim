---
layout: "default"
title: "System Interaction and Architecture"
nav_order: 7
parent: "Smart Control Project Documentation"
---

# System Interaction and Architecture

## Data Flow

The system operates in discrete time steps within an episode, following the reinforcement learning loop:

1. **Agent Action**:

   - The agent selects an action based on the current observation.
   - The action is normalized and sent to the environment.

2. **Environment Response**:

   - The environment applies the action to the building simulation.
   - Actions are converted from normalized values to native setpoint values using action normalizers.
   - The building simulation updates its state based on the action.

3. **Observation Retrieval**:

   - The environment retrieves observations from the building after the action.
   - Observations are normalized and processed, including time and occupancy features.
   - Missing or invalid observations are handled using previous data or default values.

4. **Reward Calculation**:

   - The reward function computes the reward based on productivity, energy cost, and carbon emissions.
   - The reward is provided to the agent.

5. **State Update**:

   - The environment updates internal metrics and logs information.
   - Checks if the episode has ended based on the number of steps or time.
   - If the episode has ended, the environment resets for the next episode.

## Component Interactions

- **Environment and Building Simulation**:

  - The `Environment` interacts with the `SimulatorBuilding`, which integrates the building simulation (`TFSimulator`), HVAC systems, weather controller, and occupancy model.
  - Actions are applied to the building simulation, and observations are retrieved after each step.

- **Reward Functions**:

  - The environment uses the reward function (e.g., `SetpointEnergyCarbonRegretFunction`) to compute rewards based on the `RewardInfo` from the building.
  - The reward function accesses energy consumption data, occupancy levels, and temperatures to compute productivity and costs.

- **Energy Cost Models**:

  - `ElectricityEnergyCost` and `NaturalGasEnergyCost` provide cost and carbon emission calculations based on energy usage and time.
  - The reward function uses these models to compute energy costs and carbon emissions for the reward.

- **Normalization and Configuration**:

  - `ActionConfig` defines how actions are normalized and mapped to building setpoints.
  - Observation normalizers are defined for each measurement to ensure consistent scaling.
  - Gin configuration files specify parameters and bindings for all components.

---

[Back to Home](../index.md)
