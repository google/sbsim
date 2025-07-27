---
layout: "default"
title: "ElectricityEnergyCost"
nav_order: 4
parent: "Reward Functions"
grand_parent: "Smart Control Project Documentation"
---

# ElectricityEnergyCost

**Purpose**: Models the cost and carbon emissions associated with electricity consumption.

## Key Attributes

- `weekday_energy_prices`: Energy prices for weekdays by hour.
- `weekend_energy_prices`: Energy prices for weekends by hour.
- `carbon_emission_rates`: Carbon emission rates by hour.

## Key Methods

- `__init__(...)`: Initializes the energy cost model with price and emission schedules.
- `cost(start_time, end_time, energy_rate)`: Calculates the cost of electricity consumed over a time interval.
- `carbon(start_time, end_time, energy_rate)`: Calculates the carbon emissions from electricity consumption over a time interval.

## Calculation Logic

- Determines the appropriate energy price and carbon emission rate based on the time and day.
- Converts energy rates to costs and emissions using provided schedules.
- Supports variable pricing and emission factors throughout the day.

---

[Back to Reward Functions](reward-functions.md)

[Back to Home](../index.md)
