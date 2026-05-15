"""Reinforcement learning environment."""

import gin

from smart_buildings.smart_control.environment.environment import Environment
from smart_buildings.smart_control.reinforcement_learning.utils.constants import DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT


def load_environment(gin_config_file: str):
  """Returns an Environment from a config file."""
  # Global definition is required by Gin library to instantiate Environment.
  # global environment  # pylint: disable=global-variable-not-assigned

  with gin.unlock_config():
    gin.clear_config()
    gin.parse_config_file(gin_config_file)
    return Environment()  # pylint: disable=no-value-for-parameter


def create_and_setup_environment(
    gin_config_file: str,
    metrics_path: str = None,
    occupancy_normalization_constant: float = DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT,  # pylint: disable=line-too-long
):
  """Creates and sets up the environment."""
  env = load_environment(gin_config_file)
  env.metrics_path = metrics_path
  env.occupancy_normalization_constant = occupancy_normalization_constant

  return env
