"""Setup environment for fast testing."""

import gin

from smart_control.environment.environment import Environment
from smart_control.reinforcement_learning.utils.constants import DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT


def create_and_setup_test_environment(
    gin_config_file: str,
    metrics_path: str = None,
    occupancy_normalization_constant: float = DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT,  # pylint: disable=line-too-long
):
  """Creates and sets up the environment."""
  with gin.unlock_config():
    gin.clear_config()
    gin.parse_config_file(gin_config_file)

    # start to end is one day long ?
    # update time step interval to be longer, to decrease number of steps
    seconds_in_a_day = 60 * 60 * 24
    time_step_sec = seconds_in_a_day / 2  # produces 28 steps?
    time_step_sec = time_step_sec * 10  # produces 2 steps?
    time_step_sec = time_step_sec * 2  # produces 1 step
    gin.bind_parameter("sim_building/TFSimulator.time_step_sec", time_step_sec)

    env = Environment()  # pylint: disable=no-value-for-parameter

  # print(env._num_timesteps_in_episode)  # updated from 4032 to 1
  # breakpoint()
  env.metrics_path = metrics_path
  env.occupancy_normalization_constant = occupancy_normalization_constant

  return env
