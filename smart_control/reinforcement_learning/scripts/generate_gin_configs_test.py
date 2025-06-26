"""Tests for gin config generation script."""

import os
import shutil

from absl.testing import absltest
from absl.testing import parameterized

from smart_control.reinforcement_learning.scripts.generate_gin_configs import generate_configs
from smart_control.reinforcement_learning.scripts.generate_gin_configs import modify_config
from smart_control.utils.constants import SB1_TRAIN_CONFIGS_DIR

GIN_CONFIG_EXCERPT = """

    # Finite difference settings.
    time_step_sec =  300
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 30
    start_timestamp = '2023-07-06 07:00:00+00:00'

    ...

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

  def test_generate_configs(self):
    # setup, using separate temporary directory for generating test files:
    test_output_dir = os.path.join(SB1_TRAIN_CONFIGS_DIR, "generation_test")
    if os.path.isdir(test_output_dir):
      shutil.rmtree(test_output_dir)
    self.assertEqual(os.path.isdir(test_output_dir), False)

    grid = {
        "time_step_sec": ["300"],
        "num_days_in_episode": ["1", "7", "14", "30"],
        "start_timestamp": ["2023-07-06 07:00:00+00:00"],
    }
    generate_configs(output_dir=test_output_dir, params_grid=grid)

    # it creates the output directory:
    self.assertEqual(os.path.isdir(test_output_dir), True)
    # it generates a number of gin files in there:
    generated_file_names = sorted(os.listdir(test_output_dir))
    expected_file_names = [
        "step_300_days_14_start_20230706.gin",
        "step_300_days_1_start_20230706.gin",
        "step_300_days_30_start_20230706.gin",
        "step_300_days_7_start_20230706.gin",
    ]
    self.assertEqual(generated_file_names, expected_file_names)

    # cleanup:
    shutil.rmtree(test_output_dir)


if __name__ == "__main__":
  absltest.main()
