---
layout: "default"
title: "NaturalGasEnergyCost"
nav_order: 5
parent: "Reward Functions"
grand_parent: "Smart Control Project Documentation"
---

# NaturalGasEnergyCost

**Purpose**: Models the cost and carbon emissions associated with natural gas consumption.

## Key Attributes

- `gas_price_by_month`: Gas prices by month.

## Key Methods

- `__init__(gas_price_by_month)`: Initializes the energy cost model with monthly gas prices.
- `cost(start_time, end_time, energy_rate)`: Calculates the cost of natural gas consumed over a time interval.
- `carbon(start_time, end_time, energy_rate)`: Calculates the carbon emissions from natural gas consumption over a time interval.

## Calculation Logic

- Uses monthly gas prices to determine the cost.
- Converts energy rates to costs and emissions based on standard conversion factors.
- Accounts for natural gas being used primarily for heating.

---

[Back to Reward Functions](reward-functions.md)

[Back to Home](../index.md)
