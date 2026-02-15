---
layout: "default"
title: "Configuration"
nav_order: 6
parent: "Smart Control Project Documentation"
---

# Configuration

The project uses Gin configuration files (`*.gin`) to manage simulation settings, reward function parameters, and environment configurations.

## 1. Gin Configuration Files

Gin is a lightweight configuration framework for Python that allows parameter bindings via configuration files or command-line arguments.

### Structure of Gin Files

- **Parameter Definitions**: Define values for various parameters used throughout the simulation.
- **Function and Class Bindings**: Bind parameters to specific functions and classes.
- **References**: Use `@` to reference functions or classes, and `%` to reference parameters.

### Example from a Gin File

To illustrate, here is an example of a Gin configuration:

```
# paths
controller_reader.ProtoReader.input_dir = @get_histogram_path()
floor_plan_filepath = @get_zone_path()
zone_map_filepath = @get_zone_path()
metrics_path = @get_metrics_path()
```

## 2. Key Configuration Parameters

### Simulation Parameters

- **Weather Conditions**:
  - `convection_coefficient`: Coefficient for heat convection between the building and the environment.
  - `ambient_high_temp`, `ambient_low_temp`: High and low ambient temperatures for sinusoidal temperature variation.

- **Building Properties**:
  - `control_volume_cm`: Size of the control volume in centimeters.
  - `floor_height_cm`: Height of each floor.
  - `initial_temp`: Initial temperature inside the building.
  - `exterior_cv_conductivity`, `exterior_cv_density`, `exterior_cv_heat_capacity`: Thermal properties of the exterior building.
  - `interior_wall_cv_conductivity`, `interior_wall_cv_density`, `interior_wall_cv_heat_capacity`: Thermal properties of the interior walls.
  - `interior_cv_conductivity`, `interior_cv_density`, `interior_cv_heat_capacity`: Thermal properties of the interior air.

- **HVAC Settings**:
  - `water_pump_differential_head`, `water_pump_efficiency`: Parameters for the water pump.
  - `reheat_water_setpoint`: Setpoint temperature for reheating water.
  - `boiler_heating_rate`, `boiler_cooling_rate`: Heating and cooling rates for the boiler.
  - `fan_differential_pressure`, `fan_efficiency`: Parameters for the HVAC fan.
  - `air_handler_heating_setpoint`, `air_handler_cooling_setpoint`: Temperature setpoints for the air handler.
  - `air_handler_recirculation_ratio`: Recirculation ratio for the air handler.
  - `vav_max_air_flowrate`, `vav_reheat_water_flowrate`: Maximum flow rates for VAV boxes.

- **Occupancy Model**:
  - `morning_start_hour`, `evening_start_hour`: Hours defining the occupancy schedule.
  - `heating_setpoint_day`, `cooling_setpoint_day`: Setpoints during the day.
  - `heating_setpoint_night`, `cooling_setpoint_night`: Setpoints during the night.
  - `work_occupancy`, `nonwork_occupancy`: Occupancy levels during work and non-work hours.
  - `earliest_expected_arrival_hour`, `latest_expected_arrival_hour`: Arrival times.
  - `earliest_expected_departure_hour`, `latest_expected_departure_hour`: Departure times.

- **Time Settings**:
  - `time_step_sec`: Simulation time step in seconds.
  - `start_timestamp`: Start time of the simulation.
  - `time_zone`: Time zone for the simulation.

### Reward Function Parameters

- `max_productivity_personhour_usd`, `min_productivity_personhour_usd`: Productivity per person-hour.
- `productivity_midpoint_delta`, `productivity_decay_stiffness`: Parameters for productivity decay curve.
- `max_electricity_rate`, `max_natural_gas_rate`: Maximum energy rates for normalization.
- `productivity_weight`, `energy_cost_weight`, `carbon_emission_weight`: Weights for reward components.

### Action Normalization Parameters

- **Supply Water Setpoint**:
  - `min_normalized_value`, `max_normalized_value`: Normalized action value range.
  - `min_native_value`, `max_native_value`: Native action value range (e.g., temperature in Kelvin).

- **Supply Air Heating Temperature Setpoint**:
  - Similar normalization parameters as above.

### Observation Normalization Parameters

- **Per-Measurement Normalizers**:
  - For each measurement (e.g., `building_air_static_pressure_sensor`, `cooling_percentage_command`), define:
    - `field_id`: Identifier for the field.
    - `sample_mean`: Mean value used for normalization.
    - `sample_variance`: Variance used for normalization.

### Environment Parameters

- `discount_factor`: Discount factor for future rewards.
- `num_days_in_episode`: Number of days in an episode.
- `metrics_reporting_interval`: Interval for reporting metrics.
- `label`: Label for the simulation or environment.
- `num_hod_features`, `num_dow_features`: Number of hour-of-day and day-of-week features.

### Bindings and References

Bind classes and functions to configured parameters, for example:

```
sim_building/TFSimulator:
  building = @sim/FloorPlanBasedBuilding()
  hvac = @sim/FloorPlanBasedHvac()
  weather_controller = %weather_controller
  time_step_sec = %time_step_sec
  convergence_threshold = %convergence_threshold
  iteration_limit = %iteration_limit
  iteration_warning = %iteration_warning
  start_timestamp = @sim/to_timestamp()
```

Reference parameters using `%` and functions or classes using `@`.

---

[Back to Home](../index.md)
