"""Tests for environment.

Copyright 2023 Google LLC

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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bidict
import numpy as np
import pandas as pd
from smart_control.environment import environment
from smart_control.environment import environment_test_utils
from smart_control.models import base_building
from smart_control.models import base_reward_function
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.utils import bounded_action_normalizer
from smart_control.utils import conversion_utils
from smart_control.utils import histogram_reducer
from smart_control.utils import observation_normalizer
from smart_control.utils import test_utils
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


def _get_histogram_reducer():
  reader = mock.create_autospec(test_utils.BaseReader, instance=True)
  reader.read_action_responses.return_value = [
      test_utils.get_test_action_response(
          pd.Timestamp("2022-03-13 00:00:00"),
          [
              ("boiler_1", "measurement_1", 350.0),
              ("vav_2", "measurement_2", 68.0),
              ("boiler_3", "measurement_3", 310.0),
              ("boiler_3", "measurement_4", 20000.0),
              ("vav_4", "measurement_5", 75.0),
          ],
      ),
  ]
  return histogram_reducer.HistogramReducer(
      histogram_parameters_tuples=[
          ("measurement_2", np.arange(70.0, 78.0, 2.0)),
          ("measurement_5", np.arange(70.0, 78.0, 2.0)),
      ],
      reader=reader,
      normalize_reduce=True,
  )


class EnvironmentTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([], 0.0),
      ([np.array([1, 0])], 0.0),
      ([np.array([1, 0]), np.array([1, 0])], 0.0),
      ([np.array([0, 1]), np.array([1, 0])], 1.4142),
      ([np.array([0, 0]), np.array([1, 0])], 1.0),
      (
          [
              np.array([0, 0, 0, 0, 0]),
              np.array([1, 0, -1, 1, 0]),
              np.array([1, 0, 1, 1, 1]),
          ],
          2.236067,
      ),
  )
  def test_comput_actions_regularization_cost_valid(
      self, action_history, expected
  ):
    cost = environment.compute_action_regularization_cost(action_history)
    self.assertAlmostEqual(expected, cost, places=3)

  def test_comput_actions_regularization_cost_invalid(self):
    action_history = [np.array([1, 0]), np.array([1, 0, 1])]
    with self.assertRaises(ValueError):
      _ = environment.compute_action_regularization_cost(action_history)

  def _create_bounded_action_config(self, min_value, max_value):
    action_normalizer = bounded_action_normalizer.BoundedActionNormalizer(
        min_value, max_value
    )

    action_normalizer_inits = {
        "setpoint_1": action_normalizer,
        "setpoint_2": action_normalizer,
        "setpoint_3": action_normalizer,
        "setpoint_4": action_normalizer,
        "setpoint_5": action_normalizer,
        "setpoint_6": action_normalizer,
    }

    return environment.ActionConfig(action_normalizer_inits)

  def _assert_time_step(self, actual_time_step, expected_time_step):
    self.assertAlmostEqual(
        actual_time_step.discount, expected_time_step.discount, places=5
    )
    self.assertAlmostEqual(
        actual_time_step.reward, expected_time_step.reward, places=5
    )
    self.assertAlmostEqual(
        actual_time_step.step_type, expected_time_step.step_type, places=5
    )
    self.assertAllClose(
        actual_time_step.observation, expected_time_step.observation
    )

  def _create_observation_normalizer(self):
    normalization_constants = {
        "temperature": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="temperature", sample_mean=310.0, sample_variance=50 * 50
        ),
        "supply_water_setpoint": (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id="supply_water_setpoint",
                sample_mean=310.0,
                sample_variance=50 * 50,
            )
        ),
        "air_flowrate": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="air_flowrate", sample_mean=0.5, sample_variance=4.0
        ),
        "differential_pressure": (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id="differential_pressure",
                sample_mean=20000.0,
                sample_variance=100000.0,
            )
        ),
        "percentage": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="percentage", sample_mean=0.5, sample_variance=1.0
        ),
        "request_count": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="request_count", sample_mean=9, sample_variance=25.0
        ),
        "measurement": smart_control_normalization_pb2.ContinuousVariableInfo(
            id="measurement", sample_mean=0.0, sample_variance=1.0
        ),
    }

    return observation_normalizer.StandardScoreObservationNormalizer(
        normalization_constants
    )

  def test_generate_field_id(self):
    id_map = bidict.bidict()

    input_tuples = [
        ("a", "b_c"),
        ("a", "d"),
        ("a_b", "c"),
        ("a", "b_c"),
        ("a_b", "c"),
    ]

    expected_field_ids = [
        "a_b_c",
        "a_d",
        "a_b_c_1",
        "a_b_c",
        "a_b_c_1",
    ]

    output_field_ids = []
    for device, field in input_tuples:
      field_id = environment.generate_field_id(device, field, id_map)
      id_map[(device, field)] = field_id
      output_field_ids.append(field_id)

    self.assertCountEqual(expected_field_ids, output_field_ids)

  def test_init(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config, 0.5
    )
    env.reset()
    self.assertEqual(env.building, building)
    self.assertEqual(env.reward_function, reward_function)
    self.assertEqual(env.discount_factor, 0.5)
    self.assertEqual(env.reward_function, reward_function)

  def test_init_default(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()
    self.assertEqual(env.discount_factor, 1)

  @parameterized.named_parameters(
      (
          "No_params",
          None,
          [
              "boiler_1_setpoint_1",
              "vav_2_setpoint_2",
              "vav_2_setpoint_3",
              "vav_2_setpoint_4",
              "vav_4_setpoint_5",
              "air_handler_5_setpoint_6",
          ],
      ),
      (
          "2_params",
          [("boiler_1", "setpoint_1"), ("air_handler_5", "setpoint_6")],
          ["boiler_1_setpoint_1", "air_handler_5_setpoint_6"],
      ),
      ("1_param", [("boiler_1", "setpoint_1")], ["boiler_1_setpoint_1"]),
  )
  def test_init_device_action_tuples(self, device_action_tuples, action_names):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    # Of the 6 possible setpoints, limit only to 2 using device_action_tuples.

    env = environment.Environment(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        device_action_tuples=device_action_tuples,
    )
    env.reset()

    expected = array_spec.BoundedArraySpec(
        (len(action_names),), np.float32, minimum=-1, maximum=1, name="action"
    )

    self.assertEqual(env.action_spec(), expected)
    self.assertListEqual(action_names, env._action_names)

  def test_init_raises_value_error(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    with self.assertRaises(ValueError):
      environment.Environment(
          building, reward_function, obs_normalizer, action_config, 0
      )
    with self.assertRaises(ValueError):
      environment.Environment(
          building, reward_function, obs_normalizer, action_config, 1.01
      )

  def test_init_steps_per_episode(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()
    self.assertEqual(864.0, env.steps_per_episode)

  def test_init_action_spec(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()

    expected = array_spec.BoundedArraySpec(
        (6,), np.float32, minimum=-1, maximum=1, name="action"
    )

    self.assertEqual(env.action_spec(), expected)

  @parameterized.parameters(
      (
          None,
          array_spec.ArraySpec(
              shape=(12,), dtype=np.float32, name="observation"
          ),
      ),
      (
          _get_histogram_reducer(),
          array_spec.ArraySpec(
              shape=(18,), dtype=np.float32, name="observation"
          ),
      ),
  )
  def test_init_observation_spec(self, observation_histogram_reducer, expected):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        observation_histogram_reducer=observation_histogram_reducer,
        time_zone="UTC",
    )
    env.reset()
    self.assertEqual(env.observation_spec(), expected)

  def test_create_action_request(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()

    action = [1.0, -0.25, -0.456, 0.001, 0.12, -0.3]

    timestamp = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp("2021-06-07 12:00:01")
    )

    actual_request = env._create_action_request(action)
    expected_request = smart_control_building_pb2.ActionRequest(
        timestamp=timestamp
    )
    # for field_id in env._action_names:
    for i in range(len(env._action_names)):
      field_id = env._action_names[i]
      device, setpoint = env._id_map.inv[field_id]
      action_normalizer = action_config._action_normalizers[setpoint]
      normalized_value = action_normalizer.setpoint_value(action[i])
      expected_request.single_action_requests.append(
          smart_control_building_pb2.SingleActionRequest(
              device_id=device,
              setpoint_name=setpoint,
              continuous_value=normalized_value,
          )
      )

    self.assertEqual(actual_request, expected_request)

  def test_create_action_request_rejected_exception(self):
    class RejectionBuilding(environment_test_utils.SimpleBuilding):
      """A Building that throws exception simulating no authorization."""

      def request_action(
          self, action_request: smart_control_building_pb2.ActionRequest
      ) -> smart_control_building_pb2.ActionResponse:
        raise RuntimeError("PhysicalAssetService.WriteFieldValues")

    building = RejectionBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )

    env.reset()

    action = [1.0, -0.25, -0.456, 0.001, 0.12, -0.3]
    time_step = env._step(action)
    self.assertLessEqual(env.current_simulation_timestamp, env.end_timestamp)
    self.assertEqual(time_step.reward, -np.inf)

  def test_create_action_request_rejected_no_accepted_status(self):
    class StatusRejectionBuilding(environment_test_utils.SimpleBuilding):
      """A Building that throws exception simulating no authorization."""

      def request_action(
          self, action_request: smart_control_building_pb2.ActionRequest
      ) -> smart_control_building_pb2.ActionResponse:
        action_response = super().request_action(action_request)
        action_response.single_action_responses[0].response_type = (
            smart_control_building_pb2.SingleActionResponse.REJECTED_INVALID_DEVICE
        )
        return action_response

    building = StatusRejectionBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )

    env.reset()

    action = [1.0, -0.25, -0.456, 0.001, 0.12, -0.3]
    time_step = env._step(action)
    self.assertLessEqual(env.current_simulation_timestamp, env.end_timestamp)
    self.assertEqual(time_step.reward, -np.inf)

  def test_get_observation(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        num_hod_features=4,
        num_dow_features=2,
    )
    env.reset()

    expected_observation = tf.convert_to_tensor(
        np.array([
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            -1.0000000e00,
            7.2722054e-05,
            1.0000000e00,
            -7.2722054e-05,
            -7.2722054e-05,
            -1.0000000e00,
            7.2722054e-05,
            1.0000000e00,
            1.0000000e00,
            -1.0000000e00,
            0.0000000e00,
            1.2246469e-16,
            1.0000000e00,
            1.0000000e00,
            1.0000000e01,
        ]),
        dtype=np.float32,
    )

    actual_observation = env._get_observation()

    self.assertAllClose(actual_observation, expected_observation)

    building.values["measurement_1"] = 1
    building.values["measurement_2"] = 2
    building.values["measurement_3"] = 3
    building.values["measurement_4"] = 4
    building.values["measurement_5"] = 5

    expected_observation = tf.convert_to_tensor(
        np.array([
            1.0000000e00,
            3.0000000e00,
            4.0000000e00,
            2.0000000e00,
            5.0000000e00,
            -1.0000000e00,
            7.2722054e-05,
            1.0000000e00,
            -7.2722054e-05,
            -7.2722054e-05,
            -1.0000000e00,
            7.2722054e-05,
            1.0000000e00,
            1.0000000e00,
            -1.0000000e00,
            0.0000000e00,
            1.2246469e-16,
            1.0000000e00,
            1.0000000e00,
            1.0000000e01,
        ]),
        dtype=np.float32,
    )
    actual_observation = env._get_observation()

    self.assertAllClose(actual_observation, expected_observation)

  def test_get_observation_histogram_reducer(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()

    observation_histogram_reducer = _get_histogram_reducer()
    env = environment.Environment(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        observation_histogram_reducer=observation_histogram_reducer,
        time_zone="UTC",
        num_hod_features=4,
        num_dow_features=2,
    )
    env.reset()

    expected_observation = tf.convert_to_tensor(
        np.array([
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            7.272205402841792e-05,
            1.0,
            -7.272205402841792e-05,
            -7.272205402841792e-05,
            -1.0,
            7.272205402841792e-05,
            1.0,
            1.0,
            -1.0,
            0.0,
            1.2246468525851679e-16,
            1.0,
            1.0,
            10.0,
        ]),
        dtype=np.float32,
    )

    actual_observation = env._get_observation()

    self.assertAllClose(actual_observation, expected_observation)

    building.values["measurement_1"] = 1
    building.values["measurement_2"] = 2
    building.values["measurement_3"] = 3
    building.values["measurement_4"] = 4
    building.values["measurement_5"] = 5

    expected_observation = tf.convert_to_tensor(
        np.array([
            1.0,
            3.0,
            4.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            7.272205402841792e-05,
            1.0,
            -7.272205402841792e-05,
            -7.272205402841792e-05,
            -1.0,
            7.272205402841792e-05,
            1.0,
            1.0,
            -1.0,
            0.0,
            1.2246468525851679e-16,
            1.0,
            1.0,
            10.0,
        ]),
        dtype=np.float32,
    )
    actual_observation = env._get_observation()

    self.assertAllClose(actual_observation, expected_observation)

  def test_get_observation_invalid(self):
    class BadObservationBuilding(environment_test_utils.SimpleBuilding):

      def request_observations(
          self,
          observation_request: smart_control_building_pb2.ObservationRequest,
      ) -> smart_control_building_pb2.ObservationResponse:
        observation_response = (
            environment_test_utils.SimpleBuilding.request_observations(
                self, observation_request
            )
        )
        bad_observation_response = smart_control_building_pb2.ObservationResponse(
            timestamp=observation_response.timestamp,
            request=observation_response.request,
            single_observation_responses=observation_response.single_observation_responses[
                :3
            ],
        )
        return bad_observation_response

    building = BadObservationBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )

    with self.assertRaises(ValueError):
      env.reset()

  def test_compute_reward(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()

    rewards = []
    for _ in range(10):
      rewards.append(env._get_reward())
    self.assertSequenceAlmostEqual(
        rewards, [0, 1, 6, 43, 0.8, -1, 54, 12, -50, 0], delta=0.01
    )

  def test_reset(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()

    actual_time_step = env.reset()

    expected_time_step = ts.restart(env._get_observation())
    self._assert_time_step(actual_time_step, expected_time_step)

  def test_action_spec(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    self.assertEqual(env._action_spec, env.action_spec())

  def test_observation_spec(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    self.assertEqual(env._observation_spec, env.observation_spec())

  def test_step(self):
    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = environment.Environment(
        building, reward_function, obs_normalizer, action_config
    )
    env.reset()

    # test a few timesteps
    action = [
        1.0,
        0.25,
        -0.456,
        0.001,
        0.12,
        -0.3,
    ]
    actual = env._step(action)
    expected = ts.transition(env._get_observation(), 0, 1)

    self._assert_time_step(actual, expected)

    action = [
        0.21,
        0.225,
        -0.56,
        0.310001,
        0,
        -0.33,
    ]
    actual = env._step(action)
    expected = ts.transition(env._get_observation(), 1.0, 1)

    self._assert_time_step(actual, expected)

    action = [-0.23, -0.225, 0.156, -0.310001, 0.4, 1]
    actual = env._step(action)
    expected = ts.transition(env._get_observation(), 6.0, 1)

    self._assert_time_step(actual, expected)

  @parameterized.parameters(
      (pd.Timedelta(5, unit="minute")),
      (pd.Timedelta(1, unit="minute")),
      (pd.Timedelta(1, unit="hour")),
  )
  def test_validate_environment(self, step_interval):
    class TerminatingEnv(environment.Environment):

      def __init__(
          self,
          building: base_building.BaseBuilding,
          reward_function: base_reward_function.BaseRewardFunction,
          obs_normalizer,
          action_config,
          discount_factor: float = 1,
          step_interval: pd.Timedelta = pd.Timedelta(1, unit="minute"),
      ):
        super().__init__(
            building,
            reward_function,
            obs_normalizer,
            action_config,
            discount_factor,
            step_interval=step_interval,
        )
        self.counter = 0

      def _step(self, action) -> ts.TimeStep:
        self.counter += 1
        time_step = super()._step(action)
        if self.counter < 100:
          return time_step
        return ts.termination(env._get_observation(), reward=0.0)

    building = environment_test_utils.SimpleBuilding()
    reward_function = environment_test_utils.SimpleRewardFunction()
    action_config = self._create_bounded_action_config(200, 300)
    obs_normalizer = self._create_observation_normalizer()
    env = TerminatingEnv(
        building,
        reward_function,
        obs_normalizer,
        action_config,
        step_interval=step_interval,
    )

    utils.validate_py_environment(env, episodes=5)

  def test_replace_observations_missing_past(self):
    current_observation_request = self._get_test_observation_response(
        pd.Timestamp("2020-05-31 10:00:00"),
        pd.Timestamp("2020-05-31 10:30:00"),
        ["d0", "d0", "d1", "d1"],
        ["m0", "m1", "m0", "m1"],
        [1, 2, 3, 4],
        [True, True, False, True],
    )
    with self.assertRaises(ValueError):
      _ = environment.replace_missing_observations_past(
          current_observation_request, None
      )

  def test_no_replace_observations_missing_past(self):
    current_observation_request_in = self._get_test_observation_response(
        pd.Timestamp("2020-05-31 10:00:00"),
        pd.Timestamp("2020-05-31 10:30:00"),
        ["d0", "d0", "d1", "d1"],
        ["m0", "m1", "m0", "m1"],
        [1, 2, 3, 4],
        [True, True, True, True],
    )

    current_observation_request_out = (
        environment.replace_missing_observations_past(
            current_observation_request_in, None
        )
    )

    self.assertEqual(
        current_observation_request_in, current_observation_request_out
    )

  def test_replace_observations_with_past(self):
    past_observation_request_in = self._get_test_observation_response(
        pd.Timestamp("2020-05-31 10:00:00"),
        pd.Timestamp("2020-05-31 10:00:00"),
        ["d0", "d0", "d1", "d1"],
        ["m0", "m1", "m0", "m1"],
        [1, 2, 3, 4],
        [True, True, True, True],
    )

    current_observation_request_in = self._get_test_observation_response(
        pd.Timestamp("2020-05-31 10:00:00"),
        pd.Timestamp("2020-05-31 10:00:00"),
        ["d0", "d0", "d1", "d1"],
        ["m0", "m1", "m0", "m1"],
        [2, 0, 4, 0],
        [True, False, True, False],
    )

    current_observation_request_expected = self._get_test_observation_response(
        pd.Timestamp("2020-05-31 10:00:00"),
        pd.Timestamp("2020-05-31 10:00:00"),
        ["d0", "d0", "d1", "d1"],
        ["m0", "m1", "m0", "m1"],
        [2, 2, 4, 4],
        [True, True, True, True],
    )

    current_observation_request_out = (
        environment.replace_missing_observations_past(
            current_observation_request_in, past_observation_request_in
        )
    )

    self.assertEqual(
        current_observation_request_out,
        current_observation_request_expected,
    )

  def _get_test_observation_response(
      self,
      request_timestamp,
      response_timestamp,
      device_ids,
      measurement_names,
      values,
      observation_valids,
  ):
    request_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(request_timestamp)
    )
    response_ts = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp(response_timestamp)
    )

    single_responses = []
    single_requests = []
    for i in range(len(device_ids)):
      single_request = smart_control_building_pb2.SingleObservationRequest(
          device_id=device_ids[i], measurement_name=measurement_names[i]
      )
      single_requests.append(single_request)
      single_response = smart_control_building_pb2.SingleObservationResponse(
          timestamp=response_ts,
          single_observation_request=single_request,
          observation_valid=observation_valids[i],
          continuous_value=values[i],
      )
      single_responses.append(single_response)

    request = smart_control_building_pb2.ObservationRequest(
        timestamp=request_ts, single_observation_requests=single_requests
    )
    return smart_control_building_pb2.ObservationResponse(
        timestamp=response_ts,
        request=request,
        single_observation_responses=single_responses,
    )


if __name__ == "__main__":
  absltest.main()
