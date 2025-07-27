---
layout: "default"
title: "SetpointEnergyCarbonRegretFunction"
nav_order: 3
parent: "Reward Functions"
grand_parent: "Smart Control Project Documentation"
---

# SetpointEnergyCarbonRegretFunction

**Purpose**: Implements a reward function that calculates regret based on deviations from optimal productivity, energy cost, and carbon emissions.

## Key Attributes

- Inherits from `BaseSetpointEnergyCarbonRewardFunction`.
- `max_productivity_personhour_usd`: Maximum productivity per person-hour in USD.
- `min_productivity_personhour_usd`: Minimum productivity per person-hour in USD.
- `max_electricity_rate`: Maximum electricity energy rate for normalization.
- `max_natural_gas_rate`: Maximum natural gas energy rate for normalization.
- `productivity_weight`: Weight for productivity in the regret calculation.
- `energy_cost_weight`: Weight for energy cost in the regret calculation.
- `carbon_emission_weight`: Weight for carbon emissions in the regret calculation.

## Key Methods

- `__init__(...)`: Initializes the reward function with parameters for regret calculation.
- `compute_reward(energy_reward_info)`: Computes the normalized regret based on productivity, energy cost, and carbon emissions.

## Regret Calculation Logic

- Determines the maximum and minimum possible productivity.
- Calculates the normalized productivity regret.
- Normalizes the energy costs and carbon emissions against their maximum values.
- Combines the normalized components using specified weights.
- Produces a final reward value representing the regret.

---

[Back to Reward Functions](reward-functions.md)

[Back to Home](../index.md)
