"""Tests for mppi_utils.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from unittest import mock

from smart_buildings.smart_control.agents.mppi import mppi_utils
from smart_buildings.smart_control.utils import bounded_action_normalizer
from tf_agents.trajectories import policy_step


class GetEstimatedTempFromHistogramTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.all_field_names = [
        'some_other_feature',
        'prefix_zone_air_temperature_sensor_h_20.0',
        'prefix_zone_air_temperature_sensor_h_22.5',
        'prefix_zone_air_temperature_sensor_h_25.0',
        'another_feature',
    ]
    self.sensor_base_name = 'zone_air_temperature_sensor'
    self.observation_array = np.array([10.0, 0.2, 0.6, 0.2, 20.0])

  def test_weighted_average(self):
    estimated_temp = mppi_utils.get_estimated_temp_from_histogram(
        self.observation_array,
        self.all_field_names,
        self.sensor_base_name,
        method=mppi_utils.HistogramEstimationMethod.WEIGHTED_AVERAGE,
    )
    expected_temp = (20.0 * 0.2 + 22.5 * 0.6 + 25.0 * 0.2) / 1.0
    self.assertAlmostEqual(estimated_temp, expected_temp)

  def test_max_probability(self):
    estimated_temp = mppi_utils.get_estimated_temp_from_histogram(
        self.observation_array,
        self.all_field_names,
        self.sensor_base_name,
        method=mppi_utils.HistogramEstimationMethod.MAX_PROBABILITY,
    )
    self.assertAlmostEqual(estimated_temp, 22.5)

  def test_no_active_bins(self):
    observation_array = np.zeros_like(self.observation_array)
    estimated_temp = mppi_utils.get_estimated_temp_from_histogram(
        observation_array, self.all_field_names, self.sensor_base_name
    )
    self.assertIsNone(estimated_temp)

  def test_unparseable_bin_name_is_ignored(self):
    all_field_names = [
        'prefix_zone_air_temperature_sensor_h_20.0',
        'prefix_zone_air_temperature_sensor_h_invalid',
    ]
    observation_array = np.array([0.5, 0.5])
    estimated_temp = mppi_utils.get_estimated_temp_from_histogram(
        observation_array,
        all_field_names,
        self.sensor_base_name,
        method=mppi_utils.HistogramEstimationMethod.WEIGHTED_AVERAGE,
    )
    self.assertAlmostEqual(estimated_temp, 20.0)

  def test_unknown_method_raises_value_error(self):
    with self.assertRaises(ValueError):
      mppi_utils.get_estimated_temp_from_histogram(
          self.observation_array,
          self.all_field_names,
          self.sensor_base_name,
          method='not_a_real_method',
      )


class ApplyActionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_discrete_normalizer = mock.create_autospec(
        bounded_action_normalizer.BoundedActionNormalizer, instance=True
    )
    self.mock_continuous_normalizer = mock.create_autospec(
        bounded_action_normalizer.BoundedActionNormalizer, instance=True
    )

    self.discrete_action_normalizers = {
        'dev1_discrete_action': self.mock_discrete_normalizer
    }
    self.continuous_action_normalizers = {
        'dev1_continuous_action': self.mock_continuous_normalizer
    }

    self.mock_policy_step = mock.create_autospec(
        policy_step.PolicyStep, instance=True
    )
    self.mock_policy_step.action = {
        'discrete_action': np.array([0]),
        'continuous_action': np.array([0.0]),
    }

  def test_apply_discrete_action(self):
    self.mock_discrete_normalizer.agent_value.return_value = 1

    result_step = mppi_utils.apply_action(
        self.mock_policy_step,
        self.discrete_action_normalizers,
        self.continuous_action_normalizers,
        device_id='dev1',
        action_name='discrete_action',
        native_action_value=10.0,
    )

    self.mock_discrete_normalizer.agent_value.assert_called_once_with(10.0)
    np.testing.assert_array_equal(result_step.action['discrete_action'], [1])
    np.testing.assert_array_equal(
        result_step.action['continuous_action'], [0.0]
    )

  def test_apply_continuous_action(self):
    self.mock_continuous_normalizer.agent_value.return_value = 0.5

    result_step = mppi_utils.apply_action(
        self.mock_policy_step,
        self.discrete_action_normalizers,
        self.continuous_action_normalizers,
        device_id='dev1',
        action_name='continuous_action',
        native_action_value=25.0,
    )

    self.mock_continuous_normalizer.agent_value.assert_called_once_with(25.0)
    np.testing.assert_array_almost_equal(
        result_step.action['continuous_action'], [0.5]
    )
    np.testing.assert_array_equal(result_step.action['discrete_action'], [0])

  def test_action_not_found_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'Device action name dev2_unknown_action not found in action'
        ' normalizers.',
    ):
      mppi_utils.apply_action(
          self.mock_policy_step,
          self.discrete_action_normalizers,
          self.continuous_action_normalizers,
          device_id='dev2',
          action_name='unknown_action',
          native_action_value=1.0,
      )


if __name__ == '__main__':
  absltest.main()

