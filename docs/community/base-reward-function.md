---
layout: "default"
title: "BaseSetpointEnergyCarbonRewardFunction"
nav_order: 1
parent: "Reward Functions"
grand_parent: "Smart Control Project Documentation"
---

# BaseSetpointEnergyCarbonRewardFunction

**Purpose**: Provides a base class for reward functions that consider productivity, energy cost, and carbon emissions.

## Key Attributes

- `max_productivity_personhour_usd`: Maximum productivity per person-hour in USD.
- `productivity_midpoint_delta`: Temperature difference from setpoint at which productivity is half of the maximum.
- `productivity_decay_stiffness`: Controls the slope of the productivity decay curve.

## Key Methods

- `__init__(...)`: Initializes the reward function with productivity parameters.
- `compute_reward(energy_reward_info)`: Abstract method to compute the reward; to be implemented by subclasses.
- `_sum_zone_productivities(energy_reward_info)`: Calculates cumulative productivity across all zones.
- `_get_zone_productivity_reward(...)`: Computes productivity reward for a single zone based on temperature.
- `_get_delta_time_sec(energy_reward_info)`: Calculates the time interval in seconds.
- `_sum_electricity_energy_rate(energy_reward_info)`: Sums up electrical energy rates from devices.
- `_sum_natural_gas_energy_rate(energy_reward_info)`: Sums up natural gas energy rates from devices.

---

[Back to Reward Functions](reward-functions.md)

[Back to Home](../index.md)
