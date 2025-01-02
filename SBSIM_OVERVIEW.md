Smart Control Project Documentation
===================================

* * *

Overview
--------

The **Smart Control** project is a reinforcement learning (RL) environment designed for controlling building HVAC (Heating, Ventilation, and Air Conditioning) systems to optimize energy efficiency, cost, and occupant comfort. It provides tools and frameworks for:

*   Simulating building environments and HVAC systems.
*   Implementing custom reward functions that balance productivity, energy costs, and carbon emissions.
*   Integrating with TensorFlow Agents (TF-Agents) for developing and testing RL algorithms.
*   Configuring simulations using flexible and modular settings.
*   Facilitating the development and evaluation of intelligent control strategies for building systems.

This project is ideal for researchers and developers focused on smart building technologies, energy optimization, and HVAC control using reinforcement learning.

* * *

Table of Contents
-----------------

1.  [Key Modules and Components](#key-modules-and-components)
    *   [Environment](#1-environment)
    *   [Simulation Components](#2-simulation-components)
        *   [Building Simulation](#21-building-simulation)
        *   [HVAC Systems](#22-hvac-systems)
        *   [Weather Controllers](#23-weather-controllers)
        *   [Occupancy Models](#24-occupancy-models)
    *   [Reward Functions](#3-reward-functions)
        *   [BaseSetpointEnergyCarbonRewardFunction](#31-basesetpointenergycarbonrewardfunction)
        *   [SetpointEnergyCarbonRewardFunction](#32-setpointenergycarbonrewardfunction)
        *   [SetpointEnergyCarbonRegretFunction](#33-setpointenergycarbonregretfunction)
        *   [ElectricityEnergyCost](#34-electricityenergycost)
        *   [NaturalGasEnergyCost](#35-naturalgasenergycost)
    *   [Configuration](#4-configuration)
        *   [Gin Configuration Files](#41-gin-configuration-files)
        *   [Key Configuration Parameters](#42-key-configuration-parameters)
2.  [System Interaction and Architecture](#system-interaction-and-architecture)
    *   [Data Flow](#data-flow)
    *   [Component Interactions](#component-interactions)
3.  [Contributing to the Project](#contributing-to-the-project)
    *   [Coding Standards](#coding-standards)
    *   [Development Workflow](#development-workflow)
    *   [Testing](#testing)
4.  [Best Practices](#best-practices)
5.  [Additional Resources](#additional-resources)
6.  [Appendix: Glossary](#appendix-glossary)

* * *

Key Modules and Components
--------------------------

### 1\. Environment

The `environment` module provides the reinforcement learning environment where the agent interacts with the building simulation to control HVAC systems.

#### `environment/environment.py`

**Purpose**: Implements a controllable building RL environment compatible with TF-Agents, allowing agents to control various setpoints with the goal of making the HVAC system more efficient.

**Key Classes and Functions**:

*   **`Environment`** (inherits from `py_environment.PyEnvironment`):
    
    *   **Attributes**:
        
        *   **Building and Simulation Components**:
            
            *   `_building`: Instance of `BaseBuilding`, representing the simulated building environment.
            *   `_time_zone`: Time zone of the building/environment.
            *   `_start_timestamp` and `_end_timestamp`: Start and end times of the episode.
            *   `_step_interval`: Time interval between environment steps.
            *   `_num_timesteps_in_episode`: Number of timesteps in an episode.
            *   `_observation_request`: Template for requesting observations from the building.
        *   **Agent Interaction Components**:
            
            *   `_action_spec`: Specification of the actions that can be taken.
            *   `_observation_spec`: Specification of the observations.
            *   `_action_normalizers`: Mapping from action names to their normalizers.
            *   `_action_names`: List of action field names.
            *   `_field_names`: List of observation field names.
            *   `_observation_normalizer`: Normalizes observations received from the building.
            *   `_default_policy_values`: Default actions in the policy.
        *   **Reward and Metrics**:
            
            *   `_reward_function`: Instance of `BaseRewardFunction` used to compute rewards.
            *   `_discount_factor`: Discount factor for future rewards.
            *   `_metrics`: Stores metrics for analysis and visualization.
            *   `_metrics_reporting_interval`: Interval for reporting metrics.
            *   `_summary_writer`: Writes summary data for visualization tools like TensorBoard.
        *   **Episode and Step Tracking**:
            
            *   `_episode_ended`: Boolean flag indicating if the episode has ended.
            *   `_step_count`: Number of steps taken in the current episode.
            *   `_global_step_count`: Total number of steps taken across episodes.
            *   `_episode_count`: Number of episodes completed.
            *   `_episode_cumulative_reward`: Cumulative reward for the current episode.
    *   **Methods**:
        
        *   `__init__(...)`: Initializes the environment with the specified parameters and configurations.
            
        *   **Environment Lifecycle Methods**:
            
            *   `_reset()`: Resets the environment to its initial state, preparing for a new episode.
            *   `_step(action)`: Executes one time step in the environment given an action from the agent.
            *   `_has_episode_ended(last_timestamp)`: Checks if the episode has ended based on time or steps.
        *   **Agent Interaction Methods**:
            
            *   `action_spec()`: Returns the specification of the actions that can be taken.
            *   `observation_spec()`: Returns the specification of the observations.
            *   `_get_observation()`: Retrieves and processes observations from the building, including normalizing and handling missing data.
            *   `_create_action_request(action_array)`: Converts normalized agent actions into native action values for the building.
        *   **Reward Calculation Methods**:
            
            *   `_get_reward()`: Computes the reward for the last action taken based on the reward function.
            *   `_write_summary_reward_info_metrics(reward_info)`: Writes reward input metrics into summary logs.
            *   `_write_summary_reward_response_metrics(reward_response)`: Writes reward output metrics into summary logs.
            *   `_commit_reward_metrics()`: Aggregates and writes reward metrics, and resets accumulators.
        *   **Utility Methods**:
            
            *   `_get_action_spec_and_normalizers(...)`: Defines the action space and normalizers based on the building devices and action configurations.
            *   `_get_observation_spec(...)`: Defines the observation space based on the building devices and observation configurations.
            *   `_format_action(action, action_names)`: Reformat actions if necessary (extension point for subclasses).
            *   `render(mode)`: (Not implemented) Intended for rendering the environment state.
        *   **Properties**:
            
            *   `steps_per_episode`: Number of steps in an episode.
            *   `start_timestamp`: Start time of the episode.
            *   `end_timestamp`: End time of the episode.
            *   `default_policy_values`: Default actions in the policy.
            *   `label`: Label for the environment or episode.
            *   `current_simulation_timestamp`: Current simulation time.

**Environment Workflow**:

1.  **Initialization**:
    
    *   The environment is initialized with specified parameters, including the building simulation, reward function, observation normalizer, and action configurations.
    *   Action and observation specifications are set up based on devices and configurations, involving action normalizers and mappings.
    *   Auxiliary features such as time of day and day of week are prepared.
2.  **Resetting the Environment**:
    
    *   The `_reset()` method resets the building simulation and environment metrics.
    *   Episode counters and timestamps are initialized.
    *   The initial observation is generated by calling `_get_observation()`.
3.  **Stepping through the Environment**:
    
    *   **Action Application**:
        
        *   The `_step(action)` method processes the action from the agent.
        *   Actions are normalized and converted into native values using `_create_action_request(action_array)`.
        *   The action request is sent to the building simulation via `self.building.request_action(action_request)`.
        *   The environment handles action responses, including logging and handling rejections.
    *   **Observation Retrieval**:
        
        *   After the action is applied, `_get_observation()` retrieves observations from the building.
        *   Observations are normalized using the observation normalizer.
        *   Time features (hour of day, day of week) and occupancy features are added to the observation.
        *   Missing or invalid observations are handled using past data.
    *   **Reward Calculation**:
        
        *   The reward is computed using `_get_reward()`, which invokes the reward function's `compute_reward()` method.
        *   Reward metrics are logged and written to summary writers if configured.
    *   **Episode Termination**:
        
        *   The environment checks if the episode has ended using `_has_episode_ended(last_timestamp)`.
        *   If the episode has ended, a terminal time step is returned, and the environment is reset for the next episode.

**Action and Observation Specifications**:

*   **Actions**:
    
    *   Defined by `action_spec`, which specifies the shape, data type, and bounds of the action array.
        
    *   Actions are mapped to device setpoints via `_action_names` and normalizers.
        
    *   **Action Normalizers**:
        
        *   Map agent action values (e.g., between -1 and 1) to native device setpoint values.
        *   Ensure that the agent's actions are within valid operational ranges.
        *   Example: Adjusting the supply water temperature setpoint of a boiler.
*   **Observations**:
    
    *   Defined by `observation_spec`, which specifies the shape and data type of the observation array.
        
    *   Observations include measurements from devices and auxiliary features such as time of day and occupancy.
        
    *   **Observation Normalizers**:
        
        *   Normalize raw observation values to have zero mean and unit variance.
        *   Handle different scales and units of measurements.
        *   Include handling for missing data and apply histogram reduction if configured.

### 2\. Simulation Components

The simulation components model the physical building, HVAC systems, weather conditions, and occupancy patterns.

#### 2.1 Building Simulation

**Purpose**: Represents the thermal and physical properties of the building, simulating how it responds to HVAC inputs and environmental conditions.

**Key Classes and Components**:

*   **`FloorPlanBasedBuilding`**:
    
    *   Simulates the building's structure based on a floor plan and zone mappings.
    *   **Attributes**:
        *   `cv_size_cm`: Control volume size in centimeters.
        *   `floor_height_cm`: Height of each floor.
        *   `initial_temp`: Initial uniform interior temperature.
        *   `inside_air_properties`: Thermal properties of the air inside the building.
        *   `inside_wall_properties`: Thermal properties of the interior walls.
        *   `building_exterior_properties`: Thermal properties of the exterior building.
        *   `floor_plan_filepath`: Path to the floor plan file.
        *   `zone_map_filepath`: Path to the zone mapping file.
        *   `convection_simulator`: Simulates heat convection within the building.
        *   `reset_temp_values`: Function to reset temperature values.

#### 2.2 HVAC Systems

**Purpose**: Models the heating, ventilation, and air conditioning systems of the building, including air handlers, boilers, and variable air volume (VAV) boxes.

**Key Classes and Components**:

*   **`AirHandler`**:
    
    *   Controls the air flow and temperature in the building.
    *   **Attributes**:
        *   `recirculation`: Percentage of fresh air in the recirculation.
        *   `heating_air_temp_setpoint`: Setpoint for heating air temperature.
        *   `cooling_air_temp_setpoint`: Setpoint for cooling air temperature.
        *   `fan_differential_pressure`: Pressure difference across the fan.
        *   `fan_efficiency`: Efficiency of the fan.
        *   `max_air_flow_rate`: Maximum air flow rate.
        *   `sim_weather_controller`: Weather controller for ambient conditions.
*   **`Boiler`**:
    
    *   Provides heating by controlling water temperature.
    *   **Attributes**:
        *   `reheat_water_setpoint`: Setpoint for reheat water temperature.
        *   `water_pump_differential_head`: Pressure difference across the water pump.
        *   `water_pump_efficiency`: Efficiency of the water pump.
        *   `heating_rate`: Rate at which the boiler can increase temperature.
        *   `cooling_rate`: Rate at which the boiler can decrease temperature.
*   **`FloorPlanBasedHvac`**:
    
    *   Integrates the HVAC components into the building simulation.
    *   **Attributes**:
        *   `air_handler`: Instance of `AirHandler`.
        *   `boiler`: Instance of `Boiler`.
        *   `schedule`: Setpoint schedule for HVAC operation.
        *   `vav_max_air_flow_rate`: Maximum air flow rate for VAV boxes.
        *   `vav_reheat_max_water_flow_rate`: Maximum water flow rate for reheating.

#### 2.3 Weather Controllers

**Purpose**: Simulates external weather conditions affecting the building.

**Key Classes and Components**:

*   **`ReplayWeatherController`**:
    
    *   Uses recorded weather data to simulate ambient conditions.
    *   **Attributes**:
        *   `local_weather_path`: Path to local weather data.
        *   `convection_coefficient`: Coefficient for heat convection between the building and the environment.

#### 2.4 Occupancy Models

**Purpose**: Simulates occupancy patterns within the building, affecting internal heat gains and productivity.

**Key Classes and Components**:

*   **`RandomizedArrivalDepartureOccupancy`**:
    
    *   Models occupancy with randomized arrival and departure times.
    *   **Attributes**:
        *   `zone_assignment`: Occupancy level assigned to zones.
        *   `earliest_expected_arrival_hour`: Earliest possible arrival time.
        *   `latest_expected_arrival_hour`: Latest possible arrival time.
        *   `earliest_expected_departure_hour`: Earliest possible departure time.
        *   `latest_expected_departure_hour`: Latest possible departure time.
        *   `time_step_sec`: Time step in seconds.
        *   `time_zone`: Time zone for the simulation.

### 3\. Reward Functions

The reward functions define how the agent's actions are evaluated, guiding the learning process towards desired outcomes.

#### 3.1 `BaseSetpointEnergyCarbonRewardFunction`

**Purpose**: Provides a base class for reward functions that consider productivity, energy cost, and carbon emissions.

**Key Attributes**:

*   `max_productivity_personhour_usd`: Maximum productivity per person-hour in USD.
*   `productivity_midpoint_delta`: Temperature difference from setpoint at which productivity is half of the maximum.
*   `productivity_decay_stiffness`: Controls the slope of the productivity decay curve.

**Key Methods**:

*   `__init__(...)`: Initializes the reward function with productivity parameters.
*   `compute_reward(energy_reward_info)`: Abstract method to compute the reward; to be implemented by subclasses.
*   `_sum_zone_productivities(energy_reward_info)`: Calculates cumulative productivity across all zones.
*   `_get_zone_productivity_reward(...)`: Computes productivity reward for a single zone based on temperature.
*   `_get_delta_time_sec(energy_reward_info)`: Calculates the time interval in seconds.
*   `_sum_electricity_energy_rate(energy_reward_info)`: Sums up electrical energy rates from devices.
*   `_sum_natural_gas_energy_rate(energy_reward_info)`: Sums up natural gas energy rates from devices.

#### 3.2 `SetpointEnergyCarbonRewardFunction`

**Purpose**: Implements a reward function that balances productivity, energy cost, and carbon emissions.

**Key Attributes**:

*   Inherits from `BaseSetpointEnergyCarbonRewardFunction`.
*   `electricity_energy_cost`: Instance of `BaseEnergyCost` for electricity.
*   `natural_gas_energy_cost`: Instance of `BaseEnergyCost` for natural gas.
*   `energy_cost_weight`: Weight for energy cost in the reward.
*   `carbon_cost_weight`: Weight for carbon emissions in the reward.
*   `carbon_cost_factor`: Cost per kilogram of carbon emitted.
*   `reward_normalizer_shift`: Shift applied to the reward for normalization.
*   `reward_normalizer_scale`: Scale applied to the reward for normalization.

**Key Methods**:

*   `__init__(...)`: Initializes the reward function with energy cost and carbon emission parameters.
*   `compute_reward(energy_reward_info)`: Computes the reward value by considering productivity, energy costs, and carbon emissions.

**Reward Calculation Logic**:

*   Calculates the productivity reward based on zone temperatures and occupancy.
*   Computes the energy costs and carbon emissions for electricity and natural gas.
*   Applies weights to each component (productivity, energy cost, carbon cost).
*   Normalizes and combines these components to produce the final reward.

#### 3.3 `SetpointEnergyCarbonRegretFunction`

**Purpose**: Implements a reward function that calculates regret based on deviations from optimal productivity, energy cost, and carbon emissions.

**Key Attributes**:

*   Inherits from `BaseSetpointEnergyCarbonRewardFunction`.
*   `max_productivity_personhour_usd`: Maximum productivity per person-hour in USD.
*   `min_productivity_personhour_usd`: Minimum productivity per person-hour in USD.
*   `max_electricity_rate`: Maximum electricity energy rate for normalization.
*   `max_natural_gas_rate`: Maximum natural gas energy rate for normalization.
*   `productivity_weight`: Weight for productivity in the regret calculation.
*   `energy_cost_weight`: Weight for energy cost in the regret calculation.
*   `carbon_emission_weight`: Weight for carbon emissions in the regret calculation.

**Key Methods**:

*   `__init__(...)`: Initializes the reward function with parameters for regret calculation.
*   `compute_reward(energy_reward_info)`: Computes the normalized regret based on productivity, energy cost, and carbon emissions.

**Regret Calculation Logic**:

*   Determines the maximum and minimum possible productivity.
*   Calculates the normalized productivity regret.
*   Normalizes the energy costs and carbon emissions against their maximum values.
*   Combines the normalized components using specified weights.
*   Produces a final reward value representing the regret.

#### 3.4 `ElectricityEnergyCost`

**Purpose**: Models the cost and carbon emissions associated with electricity consumption.

**Key Attributes**:

*   `weekday_energy_prices`: Energy prices for weekdays by hour.
*   `weekend_energy_prices`: Energy prices for weekends by hour.
*   `carbon_emission_rates`: Carbon emission rates by hour.

**Key Methods**:

*   `__init__(...)`: Initializes the energy cost model with price and emission schedules.
*   `cost(start_time, end_time, energy_rate)`: Calculates the cost of electricity consumed over a time interval.
*   `carbon(start_time, end_time, energy_rate)`: Calculates the carbon emissions from electricity consumption over a time interval.

**Calculation Logic**:

*   Determines the appropriate energy price and carbon emission rate based on the time and day.
*   Converts energy rates to costs and emissions using provided schedules.
*   Supports variable pricing and emission factors throughout the day.

#### 3.5 `NaturalGasEnergyCost`

**Purpose**: Models the cost and carbon emissions associated with natural gas consumption.

**Key Attributes**:

*   `gas_price_by_month`: Gas prices by month.

**Key Methods**:

*   `__init__(gas_price_by_month)`: Initializes the energy cost model with monthly gas prices.
*   `cost(start_time, end_time, energy_rate)`: Calculates the cost of natural gas consumed over a time interval.
*   `carbon(start_time, end_time, energy_rate)`: Calculates the carbon emissions from natural gas consumption over a time interval.

**Calculation Logic**:

*   Uses monthly gas prices to determine the cost.
*   Converts energy rates to costs and emissions based on standard conversion factors.
*   Accounts for natural gas being used primarily for heating.

### 4\. Configuration

The project uses Gin configuration files (`*.gin`) to manage simulation settings, reward function parameters, and environment configurations.

#### 4.1 Gin Configuration Files

Gin is a lightweight configuration framework for Python that allows parameter bindings via configuration files or command-line arguments.

**Structure of Gin Files**:

*   **Parameter Definitions**: Define values for various parameters used throughout the simulation.
*   **Function and Class Bindings**: Bind parameters to specific functions and classes.
*   **References**: Use `@` to reference functions or classes, and `%` to reference parameters.

**Example from the Provided Gin File**:

python

Copy code

`# paths controller_reader.ProtoReader.input_dir = @get_histogram_path() floor_plan_filepath = @get_zone_path() zone_map_filepath = @get_zone_path() metrics_path = @get_metrics_path()`

#### 4.2 Key Configuration Parameters

**Simulation Parameters**:

*   **Weather Conditions**:
    
    *   `convection_coefficient`: Coefficient for heat convection between the building and the environment.
    *   `ambient_high_temp`, `ambient_low_temp`: High and low ambient temperatures for sinusoidal temperature variation.
*   **Building Properties**:
    
    *   `control_volume_cm`: Size of the control volume in centimeters.
    *   `floor_height_cm`: Height of each floor.
    *   `initial_temp`: Initial temperature inside the building.
    *   `exterior_cv_conductivity`, `exterior_cv_density`, `exterior_cv_heat_capacity`: Thermal properties of the exterior building.
    *   `interior_wall_cv_conductivity`, `interior_wall_cv_density`, `interior_wall_cv_heat_capacity`: Thermal properties of the interior walls.
    *   `interior_cv_conductivity`, `interior_cv_density`, `interior_cv_heat_capacity`: Thermal properties of the interior air.
*   **HVAC Settings**:
    
    *   `water_pump_differential_head`, `water_pump_efficiency`: Parameters for the water pump.
    *   `reheat_water_setpoint`: Setpoint temperature for reheating water.
    *   `boiler_heating_rate`, `boiler_cooling_rate`: Heating and cooling rates for the boiler.
    *   `fan_differential_pressure`, `fan_efficiency`: Parameters for the HVAC fan.
    *   `air_handler_heating_setpoint`, `air_handler_cooling_setpoint`: Temperature setpoints for the air handler.
    *   `air_handler_recirculation_ratio`: Recirculation ratio for the air handler.
    *   `vav_max_air_flowrate`, `vav_reheat_water_flowrate`: Maximum flow rates for VAV boxes.
*   **Occupancy Model**:
    
    *   `morning_start_hour`, `evening_start_hour`: Hours defining the occupancy schedule.
    *   `heating_setpoint_day`, `cooling_setpoint_day`: Setpoints during the day.
    *   `heating_setpoint_night`, `cooling_setpoint_night`: Setpoints during the night.
    *   `work_occupancy`, `nonwork_occupancy`: Occupancy levels during work and non-work hours.
    *   `earliest_expected_arrival_hour`, `latest_expected_arrival_hour`: Arrival times.
    *   `earliest_expected_departure_hour`, `latest_expected_departure_hour`: Departure times.
*   **Time Settings**:
    
    *   `time_step_sec`: Simulation time step in seconds.
    *   `start_timestamp`: Start time of the simulation.
    *   `time_zone`: Time zone for the simulation.

**Reward Function Parameters**:

*   `max_productivity_personhour_usd`, `min_productivity_personhour_usd`: Productivity per person-hour.
*   `productivity_midpoint_delta`, `productivity_decay_stiffness`: Parameters for productivity decay curve.
*   `max_electricity_rate`, `max_natural_gas_rate`: Maximum energy rates for normalization.
*   `productivity_weight`, `energy_cost_weight`, `carbon_emission_weight`: Weights for reward components.

**Action Normalization Parameters**:

*   **Supply Water Setpoint**:
    
    *   `min_normalized_value`, `max_normalized_value`: Normalized action value range.
    *   `min_native_value`, `max_native_value`: Native action value range (e.g., temperature in Kelvin).
*   **Supply Air Heating Temperature Setpoint**:
    
    *   Similar normalization parameters as above.

**Observation Normalization Parameters**:

*   **Per-Measurement Normalizers**:
    
    *   For each measurement (e.g., `building_air_static_pressure_sensor`, `cooling_percentage_command`), define:
        *   `field_id`: Identifier for the field.
        *   `sample_mean`: Mean value used for normalization.
        *   `sample_variance`: Variance used for normalization.

**Environment Parameters**:

*   `discount_factor`: Discount factor for future rewards.
*   `num_days_in_episode`: Number of days in an episode.
*   `metrics_reporting_interval`: Interval for reporting metrics.
*   `label`: Label for the simulation or environment.
*   `num_hod_features`, `num_dow_features`: Number of hour-of-day and day-of-week features.

**Bindings and References**:

*   Bind classes and functions to configured parameters, e.g.:
    
    python
    
    Copy code
    
    `sim_building/TFSimulator:   building = @sim/FloorPlanBasedBuilding()   hvac  = @sim/FloorPlanBasedHvac()   weather_controller = %weather_controller   time_step_sec = %time_step_sec   convergence_threshold = %convergence_threshold   iteration_limit = %iteration_limit   iteration_warning = %iteration_warning   start_timestamp = @sim/to_timestamp()`
    
*   Reference parameters using `%` and functions or classes using `@`.
    

* * *

System Interaction and Architecture
-----------------------------------

### Data Flow

The system operates in discrete time steps within an episode, following the reinforcement learning loop:

1.  **Agent Action**:
    
    *   The agent selects an action based on the current observation.
    *   The action is normalized and sent to the environment.
2.  **Environment Response**:
    
    *   The environment applies the action to the building simulation.
    *   Actions are converted from normalized values to native setpoint values using action normalizers.
    *   The building simulation updates its state based on the action.
3.  **Observation Retrieval**:
    
    *   The environment retrieves observations from the building after the action.
    *   Observations are normalized and processed, including time and occupancy features.
    *   Missing or invalid observations are handled using previous data or default values.
4.  **Reward Calculation**:
    
    *   The reward function computes the reward based on productivity, energy cost, and carbon emissions.
    *   The reward is provided to the agent.
5.  **State Update**:
    
    *   The environment updates internal metrics and logs information.
    *   Checks if the episode has ended based on the number of steps or time.
    *   If the episode has ended, the environment resets for the next episode.

### Component Interactions

*   **Environment and Building Simulation**:
    
    *   The `Environment` interacts with the `SimulatorBuilding`, which integrates the building simulation (`TFSimulator`), HVAC systems, weather controller, and occupancy model.
    *   Actions are applied to the building simulation, and observations are retrieved after each step.
*   **Reward Functions**:
    
    *   The environment uses the reward function (e.g., `SetpointEnergyCarbonRegretFunction`) to compute rewards based on the `RewardInfo` from the building.
    *   The reward function accesses energy consumption data, occupancy levels, and temperatures to compute productivity and costs.
*   **Energy Cost Models**:
    
    *   `ElectricityEnergyCost` and `NaturalGasEnergyCost` provide cost and carbon emission calculations based on energy usage and time.
    *   The reward function uses these models to compute energy costs and carbon emissions for the reward.
*   **Normalization and Configuration**:
    
    *   `ActionConfig` defines how actions are normalized and mapped to building setpoints.
    *   Observation normalizers are defined for each measurement to ensure consistent scaling.
    *   Gin configuration files specify parameters and bindings for all components.

* * *

Contributing to the Project
---------------------------

### Coding Standards

*   **Style Guide**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
*   **Type Annotations**: Use type hints throughout the code for better readability and tooling support.
*   **Docstrings**: Include docstrings for all modules, classes, and functions using the Google Python Style Guide.
*   **Naming Conventions**:
    *   Use descriptive names for variables, functions, and classes.
    *   Class names should follow `CamelCase`.
    *   Function and variable names should be in `snake_case`.

### Development Workflow

1.  **Branching**:
    
    *   Create a new branch for each feature or bug fix.
    *   Branch names should be descriptive, e.g., `feature/add-new-reward-function`.
2.  **Writing Tests**:
    
    *   Write unit tests for new features and bug fixes.
    *   Use `unittest` or `pytest` frameworks.
    *   Place tests in the `tests` directory, mirroring the module structure.
3.  **Commits**:
    
    *   Make small, atomic commits with clear commit messages.
    *   Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification if possible.
4.  **Pull Requests**:
    
    *   Submit pull requests to the `main` branch when ready.
    *   Include a detailed description of changes and any related issues.
5.  **Code Review**:
    
    *   Participate in code reviews for peers.
    *   Be constructive and focus on improving code quality.

### Testing

*   **Unit Tests**:
    
    *   Ensure high test coverage across modules.
    *   Tests should be independent and repeatable.
*   **Integration Tests**:
    
    *   Test interactions between modules, especially environment and reward functions.
*   **Continuous Integration**:
    
    *   Use CI tools like GitHub Actions or Jenkins to automate testing on each commit or pull request.

* * *

Best Practices
--------------

*   **Modular Design**:
    
    *   Keep modules and classes focused on single responsibilities.
    *   Facilitate reuse and maintainability.
*   **Error Handling**:
    
    *   Use try-except blocks where exceptions might occur.
    *   Provide meaningful error messages and log exceptions.
*   **Logging**:
    
    *   Use the `logging` module instead of print statements.
    *   Set appropriate logging levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   **Documentation**:
    
    *   Keep documentation up to date with code changes.
    *   Use comments to explain complex logic or decisions.
*   **Performance Optimization**:
    
    *   Profile code to identify bottlenecks.
    *   Optimize algorithms and data structures where necessary.
*   **Configuration Management**:
    
    *   Use configuration files (e.g., Gin) to manage parameters.
    *   Avoid hardcoding values in the codebase.

* * *

Additional Resources
--------------------

*   **Gin Configuration Guide**:
    
    *   Learn more about Gin configurations at [Gin Config Documentation](https://github.com/google/gin-config).
*   **TF-Agents Documentation**:
    
    *   Explore TF-Agents for reinforcement learning environments and agents at TF\-Agents.
*   **Energy Cost Modeling**:
    
    *   Research energy cost and carbon emission modeling for deeper understanding.
*   **Reinforcement Learning Concepts**:
    
    *   Review reinforcement learning fundamentals to effectively develop and test agents.

* * *

Appendix: Glossary
------------------

*   **HVAC**: Heating, Ventilation, and Air Conditioning systems.
*   **RL**: Reinforcement Learning.
*   **TF-Agents**: A library for reinforcement learning in TensorFlow.
*   **Setpoint**: The desired temperature or condition that a control system aims to maintain.
*   **Regret Function**: A type of reward function that measures the difference between the actual performance and the optimal performance.
*   **Gin Configuration**: A lightweight configuration framework for Python.
*   **Productivity Decay Curve**: A function describing how productivity decreases as conditions deviate from optimal setpoints.
*   **Energy Cost Models**: Models that calculate the cost and carbon emissions associated with energy consumption.
*   **Normalization**: The process of scaling data to a standard range or distribution.
*   **VAV**: Variable Air Volume systems used in HVAC to control the amount of air flow to different areas.
*   **Control Volume**: A defined region in space through which fluid may flow in and out, used in simulations to model physical systems.

* * *

_This documentation provides an overview of the Smart Control project to assist new developers in understanding and contributing to the codebase. For detailed API documentation, refer to the docstrings within the code and any auto-generated documentation._
