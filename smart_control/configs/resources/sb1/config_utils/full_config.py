"""Gin configuration utilities for Building 'SB-1'."""

import os

import gin

# pylint: disable=unused-import # these imports are needed by the gin config:
from smart_buildings.smart_control.configs.resources.sb1.config_utils import data_files
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.reward import electricity_energy_cost
from smart_buildings.smart_control.reward import natural_gas_energy_cost
from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret
from smart_buildings.smart_control.simulator import building
from smart_buildings.smart_control.simulator import hvac_floorplan_based
from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy as occupancy
from smart_buildings.smart_control.simulator import setpoint_schedule
from smart_buildings.smart_control.simulator import simulator_building
from smart_buildings.smart_control.simulator import stochastic_convection_simulator
from smart_buildings.smart_control.simulator import tf_simulator
from smart_buildings.smart_control.simulator import weather_controller
from smart_buildings.smart_control.utils import environment_utils
from smart_buildings.smart_control.utils import observation_normalizer
# pylint: enable=unused-import

FULL_CONFIG_FILEPATH = os.path.join(
    data_files.DIRPATH, "sim_202512", "full_config.gin"
)

START_TIMESTAMP = "2024-12-16 00:00:00"
N_DAYS = 7


def parse_gin_config(
    config_filepath: str = FULL_CONFIG_FILEPATH,
) -> gin.config.ParsedConfigFileIncludesAndImports:
  return gin.parse_config_file(config_filepath)


def set_gin_config(
    config_filepath: str = FULL_CONFIG_FILEPATH,
    # episode settings:
    start_timestamp: str = START_TIMESTAMP,
    n_days: int = N_DAYS,
    # reward function settings:
    productivity_weight: float = 0.6,
    energy_cost_weight: float = 0.2,
    carbon_emission_weight: float = 0.2,
    # occupancy settings (centered around building operational hours):
    earliest_expected_arrival_hour: int = 7,
    latest_expected_arrival_hour: int = 12,
    earliest_expected_departure_hour: int = 13,
    latest_expected_departure_hour: int = 19,
    # building settings:
    floor_plan_filepath: str = data_files.FLOOR_PLAN_FILEPATH,
    # weather settings:
    weather_data_filepath: str | None = None,
) -> None:
  """Overrides specified parameters in the provided gin config."""
  if weather_data_filepath is None:
    year = int(start_timestamp[0:4])
    weather_data_filepath = data_files.get_weather_data_filepath(year)

  # clear settings:
  gin.clear_config()

  gin.add_config_file_search_path(os.path.dirname(config_filepath))
  gin.parse_config_file(config_filepath)

  # override settings:
  gin.parse_config(f"start_timestamp = '{start_timestamp}'")
  gin.parse_config(f"num_days_in_episode = {n_days}")
  gin.parse_config(f"weather_data_filepath = '{weather_data_filepath}'")
  gin.parse_config(f"floor_plan_filepath = '{floor_plan_filepath}'")
  gin.parse_config(f"productivity_weight = {productivity_weight}")
  gin.parse_config(f"energy_cost_weight = {energy_cost_weight}")
  gin.parse_config(f"carbon_emission_weight = {carbon_emission_weight}")
  gin.parse_config(
      f"earliest_expected_arrival_hour = {earliest_expected_arrival_hour}"
  )
  gin.parse_config(
      f"latest_expected_arrival_hour = {latest_expected_arrival_hour}"
  )
  gin.parse_config(
      f"earliest_expected_departure_hour = {earliest_expected_departure_hour}"
  )
  gin.parse_config(
      f"latest_expected_departure_hour = {latest_expected_departure_hour}"
  )
