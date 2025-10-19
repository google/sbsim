import unittest
from unittest import mock

import gin
import numpy as np
import pandas as pd
from smart_buildings.smart_control.agents.mppi import world_model
from smart_buildings.smart_control.utils import bounded_action_normalizer
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class EnvWorldModelTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the test environment and the world model instance."""
    super().setUp()
    self.mock_env = mock.MagicMock()

    # Define supervisor IDs and bind them for gin configuration
    ac1_id = '4192383548323266560'
    ac2_id = '202194278473007104'
    hws_id = '3111519637754347520'
    gin.bind_parameter('%AC1', ac1_id)
    gin.bind_parameter('%AC2', ac2_id)
    gin.bind_parameter('%HWS', hws_id)

    # Create normalizers, which are required inputs for the world model
    self.mock_discrete_normalizers = {
        f'{ac2_id}_supervisor_run_command': (
            bounded_action_normalizer.BoundedActionNormalizer(
                min_native_value=-1.1, max_native_value=1.1
            )
        ),
        f'{hws_id}_supervisor_run_command': (
            bounded_action_normalizer.BoundedActionNormalizer(
                min_native_value=-1.1, max_native_value=1.1
            )
        ),
        f'{ac1_id}_supervisor_run_command': (
            bounded_action_normalizer.BoundedActionNormalizer(
                min_native_value=-1.1, max_native_value=1.1
            )
        ),
    }

    # z_min and z_max are min/max NATIVE values for continuous parameters
    z_min_map = {
        f'{ac1_id}_supply_air_static_pressure_setpoint': -12.0,
        f'{ac1_id}_supply_air_temperature_setpoint': -20.0,
        f'{ac2_id}_supply_air_static_pressure_setpoint': -10.0,
        f'{ac2_id}_supply_air_temperature_setpoint': -15.0,
        f'{hws_id}_differential_pressure_setpoint': -5.0,
        f'{hws_id}_supply_water_temperature_setpoint': -8.0,
    }
    z_max_map = {
        f'{ac1_id}_supply_air_static_pressure_setpoint': 15.0,
        f'{ac1_id}_supply_air_temperature_setpoint': 25.0,
        f'{ac2_id}_supply_air_static_pressure_setpoint': 12.0,
        f'{ac2_id}_supply_air_temperature_setpoint': 18.0,
        f'{hws_id}_differential_pressure_setpoint': 8.0,
        f'{hws_id}_supply_water_temperature_setpoint': 10.0,
    }
    # for "normalized" values, normally we use -1 to 1, but for testing,
    # we make it -2 to 2 so that we can see the difference between
    # z_normalized (-1 to 1), which mppi returns
    # versus z_native (min/max native values) which the environment uses.
    agent_continuous_min_map = {key: -2.0 for key in z_min_map}
    agent_continuous_max_map = {key: 2.0 for key in z_min_map}

    self.mock_continuous_normalizers = {
        key: bounded_action_normalizer.BoundedActionNormalizer(
            min_native_value=z_min_map[key],
            max_native_value=z_max_map[key],
            min_normalized_value=agent_continuous_min_map[key],
            max_normalized_value=agent_continuous_max_map[key],
        )
        for key in sorted(z_min_map.keys())
    }

    # Configure the mock environment's action spec
    min_bounds = [
        agent_continuous_min_map[key] for key in sorted(z_min_map.keys())
    ]
    max_bounds = [
        agent_continuous_max_map[key] for key in sorted(z_min_map.keys())
    ]
    # Set the return_value to mock a method call
    self.mock_env.action_spec.return_value = {
        'discrete_action': array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32, name='discrete', minimum=-1, maximum=1
        ),
        'continuous_action': array_spec.BoundedArraySpec(
            shape=(6,),
            dtype=np.float32,
            name='continuous',
            minimum=min_bounds,
            maximum=max_bounds,
        ),
    }
    self.observation_spec = array_spec.ArraySpec(
        shape=(10,), dtype=np.float32, name='observation'
    )

    # Define par_size as an input to the world model
    self.par_size = np.array([0, 2, 2, 2, 4, 4, 4, 6])

    # Instantiate the class under test
    self.world_model = world_model.EnvWorldModel(
        env=self.mock_env,
        discrete_normalizers=self.mock_discrete_normalizers,
        continuous_normalizers=self.mock_continuous_normalizers,
        par_size=self.par_size,
    )

  def test_initialization(self):
    """Tests that the world model initializes correctly."""
    # all_z_dim = sum of par_size = 0+2+2+2+4+4+4+6 = 24
    self.assertLen(self.world_model.continuous_mapping, 24)
    # Check that the device IDs in the mapping are correct based on params_for_k
    self.assertEqual(
        self.world_model.continuous_mapping[0]['device_id'],
        gin.query_parameter('%AC2'),
    )
    self.assertEqual(
        self.world_model.continuous_mapping[2]['device_id'],
        gin.query_parameter('%HWS'),
    )
    self.assertEqual(
        self.world_model.continuous_mapping[4]['device_id'],
        gin.query_parameter('%AC1'),
    )

    # Verify z_min and z_max against the normalizers that were passed in
    unique_param_keys = sorted(self.mock_continuous_normalizers.keys())
    expected_z_min = tf.constant(
        [
            self.mock_continuous_normalizers[key]._min_native_value
            for key in unique_param_keys
        ],
        dtype=tf.float32,
    )
    expected_z_max = tf.constant(
        [
            self.mock_continuous_normalizers[key]._max_native_value
            for key in unique_param_keys
        ],
        dtype=tf.float32,
    )
    self.assertAllClose(self.world_model.z_min, expected_z_min)
    self.assertAllClose(self.world_model.z_max, expected_z_max)

  def test_format_action_from_tensor(self):
    """Tests the internal action formatting logic."""
    # Action: k=1 ([0, 1, 0, 0, 0, 0, 0, 0]), z_normalized=[0.5, -0.5]
    k_onehot = tf.constant(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32
    )
    z_normalized = tf.constant([0.5, -0.5], dtype=tf.float32)
    # The z-vector must be padded to all_z_dim. For k=1, offset is 0.
    z_padded = tf.pad(
        z_normalized,
        [[0, self.world_model.all_z_dim - tf.shape(z_normalized)[0]]],
    )
    action_tensor = tf.expand_dims(
        tf.concat([k_onehot, z_padded], axis=0), axis=0
    )

    action_dict = self.world_model._format_action_from_tensor(action_tensor)
    print('action dict: ', action_dict)
    expected_discrete = np.array([1, -1, -1], dtype=np.float32)
    expected_continuous = np.array([1, -1, 0, 0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(
        action_dict['discrete_action'], expected_discrete
    )
    np.testing.assert_array_almost_equal(
        action_dict['continuous_action'], expected_continuous
    )

  def test_next_method_steps_env(self):
    """Tests that the `next` method correctly steps the wrapped environment."""
    # Action: k=2 ([ [-1.1, 1.1, -1.1]]), z_normalized=[-1.0, 1.0]
    # For k=2, the active parameters are for HWS (indices 2, 3).
    # z_native[0] = -1 (min)
    # z_native[1] = 2 (max)
    k_onehot = tf.constant(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32
    )
    # The z-vector must be padded to all_z_dim. For k=2, the values start at
    # offset 2.
    z_values = tf.constant([-1.0, 1.0], dtype=tf.float32)
    z_padded = tf.concat([tf.zeros(2), z_values, tf.zeros(20)], axis=0)
    action_tensor = tf.expand_dims(
        tf.concat([k_onehot, z_padded], axis=0), axis=0
    )
    expected_s_pred = np.ones(
        self.observation_spec.shape, dtype=self.observation_spec.dtype
    )

    def check_step_action(action_dict):
      """Checks that the action passed to env.step is correct."""
      expected_discrete = np.array([-1, 1, -1], dtype=np.int32)
      expected_continuous = np.array(
          [0.0, 0.0, -2.0, 2.0, 0.0, 0.0], dtype=np.float32
      )
      try:
        np.testing.assert_array_equal(
            action_dict['discrete_action'], expected_discrete
        )
        np.testing.assert_array_almost_equal(
            action_dict['continuous_action'], expected_continuous
        )
        return ts.transition(observation=expected_s_pred, reward=10.0)
      except AssertionError:
        return ts.transition(observation=expected_s_pred, reward=-1.0)

    self.mock_env.step.side_effect = check_step_action

    s_pred, reward, continue_prob, _ = self.world_model.next(action_tensor)

    self.mock_env.step.assert_called_once()

    self.assertTrue(
        tf.experimental.numpy.allclose(
            tf.squeeze(s_pred, axis=0),
            tf.convert_to_tensor(expected_s_pred, dtype=tf.float32),
        )
    )
    self.assertEqual(reward.numpy().item(), 10.0)
    self.assertTrue(
        tf.experimental.numpy.allclose(
            continue_prob, tf.constant([[1.0, 0.0]], dtype=tf.float32)
        )
    )

  def test_format_action_from_tensor_k0_k7(self):
    """Tests action formatting for k=0 and k=7 edge cases."""
    # Case 1: k=0 (no continuous parameters)
    k_onehot_0 = tf.constant(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32
    )
    # For k=0, the z-values should be ignored, so we use non-zeros to test.
    z_padded_0 = tf.ones((self.world_model.all_z_dim,), dtype=tf.float32)
    action_tensor_0 = tf.expand_dims(
        tf.concat([k_onehot_0, z_padded_0], axis=0), axis=0
    )
    action_dict_0 = self.world_model._format_action_from_tensor(action_tensor_0)
    expected_discrete_0 = np.array([-1, -1, -1], dtype=np.int32)
    expected_continuous_0 = np.zeros((6,), dtype=np.float32)
    np.testing.assert_array_equal(
        action_dict_0['discrete_action'], expected_discrete_0
    )
    np.testing.assert_array_almost_equal(
        action_dict_0['continuous_action'], expected_continuous_0
    )

    # Case 2: k=7 (all 6 continuous parameters are active)
    k_onehot_7 = tf.constant(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=tf.float32
    )
    z_normalized_7 = tf.constant(
        [0.1, 0.2, 0.3, 0.4, 0.5, -0.5], dtype=tf.float32
    )
    # For k=7, the offset is 18 and num_params is 6.
    z_padded_7 = tf.concat([tf.zeros(18), z_normalized_7], axis=0)
    action_tensor_7 = tf.expand_dims(
        tf.concat([k_onehot_7, z_padded_7], axis=0), axis=0
    )
    action_dict_7 = self.world_model._format_action_from_tensor(action_tensor_7)
    expected_discrete_7 = np.array([1, 1, 1], dtype=np.int32)

    # agent_value = 2*z_normalized
    expected_continuous_7 = np.array(
        [0.2, 0.4, 0.6, 0.8, 1.0, -1.0], dtype=np.float32
    )
    np.testing.assert_array_equal(
        action_dict_7['discrete_action'], expected_discrete_7
    )
    np.testing.assert_array_almost_equal(
        action_dict_7['continuous_action'], expected_continuous_7
    )

  def test_rollout_method(self):
    """Tests the multi-step rollout functionality."""
    start_timestamp = pd.Timestamp('2023-01-01')
    action_trajectory = [
        (0, tf.constant([0.0, -1.0], dtype=tf.float32)),
        (1, tf.constant([1.0, 0.0], dtype=tf.float32)),
    ]
    obs1 = np.full(self.observation_spec.shape, 1.0, dtype=np.float32)
    obs2 = np.full(self.observation_spec.shape, 2.0, dtype=np.float32)
    self.mock_env.step.side_effect = [
        ts.transition(obs1, reward=5.0),
        ts.transition(obs2, reward=15.0),
    ]
    initial_obs = np.zeros(self.observation_spec.shape, dtype=np.float32)
    self.mock_env.reset.return_value = ts.restart(initial_obs)

    states, rewards = self.world_model.rollout(
        start_timestamp, action_trajectory
    )

    self.mock_env.reset.assert_called_once()
    self.assertEqual(
        self.world_model.env._episode_start_timestamp, start_timestamp
    )
    self.assertEqual(self.mock_env.step.call_count, 2)

    np.testing.assert_array_equal(states[0], initial_obs)
    np.testing.assert_array_equal(states[1], obs1)
    np.testing.assert_array_equal(states[2], obs2)
    self.assertEqual(rewards, [5.0, 15.0])


if __name__ == '__main__':
  unittest.main()
