---
layout: "default"
title: "SetpointEnergyCarbonRewardFunction"
nav_order: 2
parent: "Reward Functions"
grand_parent: "Smart Control Project Documentation"
---

# SetpointEnergyCarbonRewardFunction

**Purpose**: Implements a reward function that balances productivity, energy cost, and carbon emissions.

## Key Attributes

- Inherits from `BaseSetpointEnergyCarbonRewardFunction`.
- `electricity_energy_cost`: Instance of `BaseEnergyCost` for electricity.
- `natural_gas_energy_cost`: Instance of `BaseEnergyCost` for natural gas.
- `energy_cost_weight`: Weight for energy cost in the reward.
- `carbon_cost_weight`: Weight for carbon emissions in the reward.
- `carbon_cost_factor`: Cost per kilogram of carbon emitted.
- `reward_normalizer_shift`: Shift applied to the reward for normalization.
- `reward_normalizer_scale`: Scale applied to the reward for normalization.

## Key Methods

- `__init__(...)`: Initializes the reward function with energy cost and carbon emission parameters.
- `compute_reward(energy_reward_info)`: Computes the reward value by considering productivity, energy costs, and carbon emissions.

## Reward Calculation Logic

- Calculates the productivity reward based on zone temperatures and occupancy.
- Computes the energy costs and carbon emissions for electricity and natural gas.
- Applies weights to each component (productivity, energy cost, carbon cost).
- Normalizes and combines these components to produce the final reward.

---

[Back to Reward Functions](reward-functions.md)

[Back to Home](../index.md)
