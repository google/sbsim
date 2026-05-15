# Simulation Configuration for Building 'SB-1'

## Gin Configuration Files

There are some older versions of the gin config located at "sim_config.gin" and
"legacy_config.gin". These are deprecated by Judah's HVAC updates in December
2025.

The config Gabriel used for his RL experiments is located at "train_sim_configs"
directory. These are deprecated by Judah's HVAC updates in December 2025.

The newest version of the full gin config file is found at
"sim_202512/full_config.gin". This file represents the **current operational
configuration** with the current HVAC setup. We have been updating it even after
the namespaced date stamp.

## Data Files

### Floor Plan

The "double_resolution_zone_1_2.npy" file provides the floor plan for the
building.

### Zone Temperatures

The "reset_temps.npy" file provides initial zone temperatures that can be used
to seed / setup the environment. It is the default setup for simulation
experiments.

The "demo_temps.npy" file provides a slightly different set of initial zone
temperatures that have been determined to produce a more interesting temperature
gradients in live demos.

### Weather Data

The "local_weather_moffett_field_20230701_20231122.csv" file was the original
source of weather data for the specified time range in 2023.

We have since added more years of historical weather data, in the "weather_data"
directory, including the "2023.csv" file which deprecates the original file.
From time to time we can refresh the data in this directory. It's also possible
that this weather data can be shared by multiple buildings, so we could consider
moving it to a more central location, like
"configs/resources/weather_data/{station_name}", where multiple buildings like
"sb1" can share the same weather station.

## Config Utils

The "config_utils" directory provides helper methods to load data files and
ensure the necessary dependencies have been imported.

It provides a helper function to apply the configuration, and customize certain
configuration settings that we might want to parameterize in experimental
trials.

Finally, it provides tests to ensure the configuration settings are being
applied as desired.
