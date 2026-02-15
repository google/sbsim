---
layout: "default"
title: "HVAC Systems"
nav_order: 2
parent: "Simulation Components"
grand_parent: "Smart Control Project Documentation"
---

# HVAC Systems

**Purpose**: Models the heating, ventilation, and air conditioning systems of the building, including air handlers, boilers, and variable air volume (VAV) boxes.

## Key Classes and Components

- **`AirHandler`**:

  - Controls the air flow and temperature in the building.

  - **Attributes**:

    - `recirculation`: Percentage of fresh air in the recirculation.
    - `heating_air_temp_setpoint`: Setpoint for heating air temperature.
    - `cooling_air_temp_setpoint`: Setpoint for cooling air temperature.
    - `fan_differential_pressure`: Pressure difference across the fan.
    - `fan_efficiency`: Efficiency of the fan.
    - `max_air_flow_rate`: Maximum air flow rate.
    - `sim_weather_controller`: Weather controller for ambient conditions.

- **`Boiler`**:

  - Provides heating by controlling water temperature.

  - **Attributes**:

    - `reheat_water_setpoint`: Setpoint for reheat water temperature.
    - `water_pump_differential_head`: Pressure difference across the water pump.
    - `water_pump_efficiency`: Efficiency of the water pump.
    - `heating_rate`: Rate at which the boiler can increase temperature.
    - `cooling_rate`: Rate at which the boiler can decrease temperature.

- **`FloorPlanBasedHvac`**:

  - Integrates the HVAC components into the building simulation.

  - **Attributes**:

    - `air_handler`: Instance of `AirHandler`.
    - `boiler`: Instance of `Boiler`.
    - `schedule`: Setpoint schedule for HVAC operation.
    - `vav_max_air_flow_rate`: Maximum air flow rate for VAV boxes.
    - `vav_reheat_max_water_flow_rate`: Maximum water flow rate for reheating.

---

[Back to Simulation Components](simulation-components.md)

[Back to Home](../index.md)
