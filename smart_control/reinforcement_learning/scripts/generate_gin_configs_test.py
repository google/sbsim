"""Tests for gin config generation script."""

from absl.testing import absltest
from absl.testing import parameterized

from smart_control.reinforcement_learning.scripts.generate_gin_configs import modify_config

GIN_CONFIG_EXCERPT = """

    # Finite difference settings.
    time_step_sec =  300
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 30
    start_timestamp = '2023-07-06 07:00:00+00:00'

    # Top-level Environment parameters
    discount_factor = 0.9
    num_days_in_episode=14
    metrics_reporting_interval=10
    label='tunable_simulator_sb1'
    num_hod_features = 1
    num_dow_features = 1

"""  # this was copied directly from the sb1 gin config file


class ConfigGenerationTest(parameterized.TestCase):

  MODIFICATION_PARAMS = [
      ("time_step_sec", 60, "time_step_sec =60"),
      ("time_step_sec", 180, "time_step_sec =180"),
      ("num_days_in_episode", 7, "num_days_in_episode=7"),
      ("num_days_in_episode", 14, "num_days_in_episode=14"),
      (
          "start_timestamp",
          "'2024-01-01 07:00:00+00:00'",  # todo: work without ''
          "start_timestamp ='2024-01-01 07:00:00+00:00",
      ),
  ]

  @parameterized.parameters(MODIFICATION_PARAMS)
  def test_modify_config(self, param_name, param_value, expected_content):

    # if param_name == "start_timestamp":
    #  breakpoint()
    modified = modify_config(GIN_CONFIG_EXCERPT, param_name, param_value)
    self.assertIn(expected_content, modified)


if __name__ == "__main__":
  absltest.main()
