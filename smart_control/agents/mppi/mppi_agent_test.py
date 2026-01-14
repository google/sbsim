"""Tests for the MPPIAgent and MPPIPolicy.

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

from unittest import mock
import gin
import numpy as np
from smart_buildings.smart_control.agents.mppi import mppi_agent
from smart_buildings.smart_control.utils import bounded_action_normalizer
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class MPPIAgentTest(tf.test.TestCase):
  """Unit tests for the MPPIAgent and MPPIPolicy."""
  # TODO(sipple): add tests to catch bad shapes.

  def _get_dummy_observation(self):
    """Generates a dummy observation based on the observation spec."""
    return np.zeros(
        (1,) + self.observation_spec.shape, dtype=self.observation_spec.dtype
    )

  def setUp(self):
    super().setUp()
    gin.clear_config()  # Isolate the test from global gin configurations

    # --- 1. Define Device IDs, Specs, and Normalizers ---
    ac1_id = 'ac1'
    ac2_id = 'ac2'
    hws_id = 'hws'
    gin.bind_parameter('%AC1', ac1_id)
    gin.bind_parameter('%AC2', ac2_id)
    gin.bind_parameter('%HWS', hws_id)

    # Discrete Normalizers
    self.mock_discrete_normalizers = {
        f'{ac1_id}_supervisor_run_command': (
            bounded_action_normalizer.BoundedActionNormalizer(
                min_native_value=-1.1, max_native_value=1.1
            )
        ),
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
    }

    # Continuous Normalizers
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
    # For testing, we use a non-standard agent-side range to distinguish it
    # from the policy's internal [-1, 1] range.
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

    # Define Observation and Action Specs
    self.observation_spec = array_spec.ArraySpec(
        shape=(10,), dtype=np.float32, name='observation'
    )
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    min_bounds = [
        agent_continuous_min_map[key] for key in sorted(z_min_map.keys())
    ]
    max_bounds = [
        agent_continuous_max_map[key] for key in sorted(z_min_map.keys())
    ]
    self.action_spec = {
        'discrete_action': array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32, name='discrete', minimum=-1, maximum=1
        ),
        'continuous_action': array_spec.BoundedArraySpec(
            shape=(6,),
            dtype=np.float32,
            name='continuous',
            minimum=np.array(min_bounds, dtype=np.float32),
            maximum=np.array(max_bounds, dtype=np.float32),
        ),
    }

    # --- 2. Calculate Dependent Parameters ---
    self.par_size = np.array([0, 2, 2, 2, 4, 4, 4, 6])
    self.k_dim = self.par_size.shape[0]  # 8
    self.all_z_dim = self.action_spec['continuous_action'].shape[0]  # 6
    self.z_dim = np.max(self.par_size)  # 6
    self.offset = np.insert(np.cumsum(self.par_size), 0, 0)[:-1]

    # Reshape z_min and z_max for the policy
    action_z_min = np.array(
        [z_min_map[key] for key in sorted(z_min_map.keys())], dtype=np.float32
    )
    action_z_max = np.array(
        [z_max_map[key] for key in sorted(z_max_map.keys())], dtype=np.float32
    )
    z_min_shaped = np.zeros((self.k_dim, self.z_dim), dtype=np.float32)
    z_max_shaped = np.zeros((self.k_dim, self.z_dim), dtype=np.float32)
    for k in range(self.k_dim):
      num_params = self.par_size[k]
      if num_params > 0:
        # Note: This simplified mapping is for testing purposes.
        # A real implementation would have a more complex mapping.
        z_min_shaped[k, :num_params] = action_z_min[:num_params]
        z_max_shaped[k, :num_params] = action_z_max[:num_params]

    # --- 3. Mock World Model and its Attributes ---
    self.mock_world_model = mock.MagicMock()
    self.mock_world_model.k_dim = self.k_dim
    self.mock_world_model.par_size = self.par_size
    self.mock_world_model.z_dim = self.z_dim
    self.mock_world_model.all_z_dim = self.all_z_dim
    self.mock_world_model.offset = self.offset
    self.mock_world_model.z_min = z_min_shaped
    self.mock_world_model.z_max = z_max_shaped
    self.mock_world_model.all_z_min = action_z_min
    self.mock_world_model.all_z_max = action_z_max
    self.mock_world_model.discrete_normalizers = self.mock_discrete_normalizers
    self.mock_world_model.continuous_normalizers = (
        self.mock_continuous_normalizers
    )

    # Create a plausible discrete action mapping for k_dim=8
    self.mock_world_model.discrete_action_mapping = {
        i: np.random.randint(-1, 2, size=(3,)) for i in range(self.k_dim)
    }
    self.mock_world_model.discrete_action_mapping[1] = np.array([1, -1, 0])

    self.mock_world_model.get_state.return_value = 'pristine'
    self.mock_world_model.time_step_spec = self.time_step_spec
    self.mock_world_model.env = mock.MagicMock()
    self.mock_world_model.env._field_names = ['zone_air_temperature_sensor']

    # --- 4. Configure MPC and Instantiate Agent ---
    self.mpc_horizon = 2
    self.mpc_popsize = 10
    self.mpc_num_elites = 2
    gin.bind_parameter('mppi_agent.MPPIPolicy.mpc_horizon', self.mpc_horizon)
    gin.bind_parameter('mppi_agent.MPPIPolicy.mpc_popsize', self.mpc_popsize)
    gin.bind_parameter(
        'mppi_agent.MPPIPolicy.mpc_num_elites', self.mpc_num_elites
    )

    self.train_step_counter = tf.Variable(0)
    self.agent = mppi_agent.MPPIAgent(
        time_step_spec=self.time_step_spec,
        action_spec=self.action_spec,
        world_model=self.mock_world_model,
        train_step_counter=self.train_step_counter,
    )

  def test_agent_creation(self):
    """Verifies that the agent and its policy are created successfully."""
    self.assertIsInstance(self.agent, mppi_agent.MPPIAgent)
    self.assertIsInstance(self.agent.policy, mppi_agent.MPPIPolicy)

  def test_train_increments_counter(self):
    """Verifies that the train method increments the step counter."""
    initial_value = self.agent.train_step_counter.numpy()
    loss_info = self.agent.train(experience=None)
    self.assertEqual(self.agent.train_step_counter.numpy(), initial_value + 1)
    self.assertEqual(loss_info.loss, 0.0)

  def test_synchronize_copies_env_state(self):
    """Tests that the agent correctly syncs with an external environment."""
    # 1. Arrange
    # Define a side effect for the mock model's synchronize method that
    # simulates the actual copying behavior.
    def mock_sync_logic(source_env):
      py_env = source_env.pyenv if hasattr(source_env, 'pyenv') else source_env
      source_building = py_env.building
      target_building = self.mock_world_model.env.building

      target_building._native_inputs = source_building._native_inputs.copy()
      target_building._current_timestamp = source_building._current_timestamp

    self.mock_world_model.synchronize.side_effect = mock_sync_logic

    mock_env = mock.MagicMock()
    source_building = mock_env.pyenv.building
    source_building._native_inputs.copy.return_value = {'input1': 123}
    source_building._current_timestamp = 456

    # 2. Action
    self.agent.synchronize(mock_env)

    # 3. Assertion
    # First, check that the agent correctly delegated the call.
    self.mock_world_model.synchronize.assert_called_once_with(mock_env)

    # Second, check that the state was copied as expected by the side effect.
    target_building = self.agent.policy.model.env.building
    self.assertEqual(target_building._native_inputs, {'input1': 123})
    self.assertEqual(target_building._current_timestamp, 456)

  @mock.patch.object(mppi_agent.MPPIPolicy, 'plan')
  def test_action_generation(self, mock_plan):
    """Tests that a planned action is correctly decoded and formatted."""
    # Arrange: Simulate planner choosing k=1 with native values [5.0, 10.0].
    keys = sorted(self.mock_continuous_normalizers.keys())
    norm1, norm2 = (
        self.mock_continuous_normalizers[keys[0]],
        self.mock_continuous_normalizers[keys[1]],
    )
    policy_val_1 = (
        (5.0 - norm1._min_native_value)
        / (norm1._max_native_value - norm1._min_native_value)
    ) * 2.0 - 1.0
    policy_val_2 = (
        (10.0 - norm2._min_native_value)
        / (norm2._max_native_value - norm2._min_native_value)
    ) * 2.0 - 1.0
    k_part = tf.one_hot(1, self.k_dim)
    z_part = tf.constant(
        [policy_val_1, policy_val_2, 0, 0, 0, 0], dtype=tf.float32
    )
    first_action = tf.concat([k_part, z_part], axis=0)
    initial_state = self.agent.policy.get_initial_state(batch_size=1)
    unbatched_state = tf.nest.map_structure(lambda t: t[0], initial_state)
    mock_plan.return_value = (
        tf.stack([first_action] * self.mpc_horizon),
        unbatched_state,
    )
    time_step = ts.transition(
        self._get_dummy_observation(),
        reward=np.array([0.0], dtype=np.float32),
        discount=np.array([1.0], dtype=np.float32),
    )
    time_step = tf.nest.map_structure(tf.convert_to_tensor, time_step)
    # Mock the final formatting step to let the test focus on decoding.
    expected_discrete = np.array([1, -1, 0])
    expected_continuous = np.array([
        norm1.agent_value(5.0),
        norm2.agent_value(10.0),
        self.mock_continuous_normalizers[keys[2]].agent_value(0.0),
        self.mock_continuous_normalizers[keys[3]].agent_value(0.0),
        self.mock_continuous_normalizers[keys[4]].agent_value(0.0),
        self.mock_continuous_normalizers[keys[5]].agent_value(0.0),
    ])
    self.mock_world_model.create_action_dict = mock.MagicMock(
        return_value={
            'discrete_action': expected_discrete,
            'continuous_action': expected_continuous,
        }
    )

    # Act
    action_step = self.agent.policy.action(time_step, initial_state)

    # Assert
    # Check that the decoding was correct by inspecting the arguments passed
    # to the mocked formatting function.
    self.mock_world_model.create_action_dict.assert_called_once()
    (
        called_k,
        called_z_native,
    ) = self.mock_world_model.create_action_dict.call_args[0]
    self.assertEqual(called_k, 1)
    self.assertAllClose(called_z_native, [5.0, 10.0], atol=1e-5)

    # The final action should also be correct because of the mock's return
    # value.
    self.assertAllClose(action_step.action['discrete_action'][0], [1, -1, 0])
    self.assertAllClose(
        action_step.action['continuous_action'][0],
        expected_continuous,
        atol=1e-5,
    )
    # The plan method is called with positional arguments for observation
    # and policy_state, and a keyword argument for step.
    mock_plan.assert_called_once()
    called_args, called_kwargs = mock_plan.call_args
    expected_observation = tf.constant(self._get_dummy_observation()[0])
    self.assertAllClose(called_args[0], expected_observation)
    tf.nest.map_structure(self.assertAllClose, called_args[1], unbatched_state)
    self.assertIn('step', called_kwargs)
    self.assertIsInstance(called_kwargs['step'], tf.Variable)

  @mock.patch.object(mppi_agent.MPPIPolicy, '_estimate_value')
  def test_plan_returns_updated_state(self, mock_estimate_value):
    """Tests that the planner returns a correctly shaped policy state."""
    # 1. Setup
    policy = self.agent.policy
    observation = self._get_dummy_observation()[0]  # plan expects unbatched
    initial_state = policy.get_initial_state(batch_size=1)
    unbatched_initial_state = tf.nest.map_structure(
        lambda t: t[0], initial_state
    )

    # Mock `_estimate_value` to isolate the planner's state management logic
    # from the value estimation logic. This makes the test more focused and
    # efficient, as we only want to verify the shape of the returned policy
    # state here.
    # The mock value should have shape (popsize, 1)
    mock_value = tf.random.normal((self.mpc_popsize, 1))
    mock_estimate_value.return_value = (
        mock_value,
        None,
    )  # trajectories not needed

    # 2. Action: Run the planner
    step_counter = tf.Variable(1)
    _, new_policy_state = policy.plan(
        observation, unbatched_initial_state, step_counter
    )

    # 3. Assertion: Check that new_policy_state has the correct structure
    self.assertIn('prev_mean_k', new_policy_state)
    self.assertIn('prev_mean_z', new_policy_state)
    self.assertEqual(
        new_policy_state['prev_mean_k'].shape, (self.mpc_horizon, self.k_dim)
    )
    self.assertEqual(
        new_policy_state['prev_mean_z'].shape,
        (self.mpc_horizon, self.all_z_dim),
    )

  def test_estimate_value_resets_environment_for_each_trajectory(self):
    """Verifies that the model state is reset for each trajectory."""
    policy = self.agent.policy
    self.mock_world_model.next.return_value = (
        tf.zeros((1, self.observation_spec.shape[0])),
        tf.constant([[1.0]]),
        False,
        {},
    )

    # Action dimension is k_dim + all_z_dim = 8 + 6 = 14
    actions = tf.zeros((self.mpc_horizon, self.mpc_popsize, 14))

    # Act
    policy._estimate_value(actions)

    # Assertion: get_state is called once, set_state is called
    # for each trajectory
    self.mock_world_model.get_state.assert_called_once()
    self.assertEqual(
        self.mock_world_model.set_state.call_count, self.mpc_popsize
    )

  @mock.patch.object(mppi_agent.MPPIPolicy, '_estimate_value')
  def test_plan_finds_optimal_action(self, mock_estimate_value):
    """Tests that the planner converges to the action with the highest reward."""
    # 1. Setup: Define an optimal action and a reward function
    optimal_k = 7
    tf.random.set_seed(123)
    # Temporarily increase sampling to make sure the planner samples enough.
    policy = self.agent.policy
    original_popsize = policy.mpc_popsize
    original_elites = policy.mpc_num_elites
    policy.mpc_popsize = 200
    policy.mpc_num_elites = 40

    def side_effect_for_estimate_value(actions):
      # This function will return a high reward only for the optimal action
      # across the entire planning horizon.
      # actions shape: (horizon, popsize, action_dim)
      k_indices = tf.argmax(actions[:, :, : self.k_dim], axis=-1)
      # Reward is based on the sum of rewards at each step.
      rewards_per_step = tf.where(k_indices == optimal_k, 100.0, -1.0)
      total_rewards = tf.reduce_sum(rewards_per_step, axis=0)
      # returned value should have shape (popsize, 1)
      return tf.expand_dims(total_rewards, axis=-1), None

    mock_estimate_value.side_effect = side_effect_for_estimate_value
    observation = self._get_dummy_observation()[0]
    initial_state = self.agent.policy.get_initial_state(batch_size=1)
    unbatched_initial_state = tf.nest.map_structure(
        lambda t: t[0], initial_state
    )
    step_counter = tf.Variable(1)

    try:
      # 2. Action: Run the planner
      action_plan, _ = self.agent.policy.plan(
          observation, unbatched_initial_state, step_counter
      )
    finally:
      # Restore original parameters to avoid affecting other tests.
      policy.mpc_popsize = original_popsize
      policy.mpc_num_elites = original_elites

    # 3. Assertion: Check that the planner chose the optimal action
    # The action_plan has shape (horizon, action_dim)
    # The first k_dim elements are the one-hot encoded k.
    chosen_k_one_hot = action_plan[:, : self.k_dim]
    chosen_k_indices = tf.argmax(chosen_k_one_hot, axis=-1)

    # The planner should converge to the optimal k for all steps
    # in the horizon.
    expected_k_indices = tf.constant(
        [optimal_k] * self.mpc_horizon, dtype=tf.int64
    )
    self.assertAllEqual(chosen_k_indices, expected_k_indices)

  def test_update_distribution_logic(self):
    """Verifies the CEM distribution update logic."""
    # 1. Arrange
    policy = self.agent.policy
    horizon = policy.mpc_horizon
    popsize = policy.mpc_popsize
    k_dim = policy.k_dim
    all_z_dim = policy.all_z_dim
    action_dim = k_dim + all_z_dim
    alpha = policy.mpc_alpha

    # Create two distinct "elite" action sequences.
    # Elite 1: k=0, z=[0.5, 0.5, ...]
    elite_action_1_k = tf.one_hot([0] * horizon, depth=k_dim)
    elite_action_1_z = tf.fill((horizon, all_z_dim), 0.5)
    elite_action_1 = tf.concat([elite_action_1_k, elite_action_1_z], axis=-1)

    # Elite 2: k=1, z=[-0.5, -0.5, ...]
    elite_action_2_k = tf.one_hot([1] * horizon, depth=k_dim)
    elite_action_2_z = tf.fill((horizon, all_z_dim), -0.5)
    elite_action_2 = tf.concat([elite_action_2_k, elite_action_2_z], axis=-1)

    # Create a batch of actions, embedding the elites at specific indices.
    actions_np = np.zeros((horizon, popsize, action_dim), dtype=np.float32)
    # For this test, we'll make the first two samples the elites.
    actions_np[:, 0, :] = elite_action_1.numpy()
    actions_np[:, 1, :] = elite_action_2.numpy()
    actions = tf.constant(actions_np)

    # Create corresponding values. The elites get high scores.
    # To make weight calculation predictable, set a fixed temperature.
    policy.mpc_temperature = 1.0
    value_np = np.zeros((popsize, 1), dtype=np.float32)
    # Elite 1 has a higher value, so it should get a higher weight.
    value_np[0] = 10.0
    value_np[1] = 5.0
    value = tf.constant(value_np)

    # Initial distribution parameters are all zeros for this test.
    initial_kmean = tf.zeros((horizon, k_dim), dtype=tf.float32)
    initial_zmean = tf.zeros((horizon, all_z_dim), dtype=tf.float32)
    initial_std = tf.zeros((horizon, all_z_dim), dtype=tf.float32)

    # 2. Act
    new_kmean, new_zmean, new_std = policy._update_distribution(
        actions, value, initial_kmean, initial_zmean, initial_std
    )

    # 3. Assert
    # Manually calculate the expected results to verify the implementation.
    # a. Calculate elite weights (softmax of values [10.0, 5.0]).
    elite_values = tf.constant([10.0, 5.0])
    max_val = tf.reduce_max(elite_values)  # 10.0
    unnorm_weights = tf.exp(policy.mpc_temperature * (elite_values - max_val))
    weights = unnorm_weights / tf.reduce_sum(unnorm_weights)

    # b. Calculate updated means (before momentum).
    # This is the weighted average of the elite actions.
    # Since actions are constant across the horizon, the mean will be too.
    expected_updated_kmean_h0 = (
        weights[0] * elite_action_1_k[0] + weights[1] * elite_action_2_k[0]
    )
    expected_updated_kmean = tf.stack([expected_updated_kmean_h0] * horizon)

    expected_updated_zmean_val = weights[0] * 0.5 + weights[1] * -0.5
    expected_updated_zmean = tf.fill(
        (horizon, all_z_dim), expected_updated_zmean_val
    )

    # c. Calculate updated std dev (before momentum).
    z_diff_sq_1 = (0.5 - expected_updated_zmean_val) ** 2
    z_diff_sq_2 = (-0.5 - expected_updated_zmean_val) ** 2
    expected_var_val = weights[0] * z_diff_sq_1 + weights[1] * z_diff_sq_2
    expected_updated_zstd = tf.fill(
        (horizon, all_z_dim), tf.sqrt(expected_var_val + 1e-9)
    )

    # d. Apply momentum to get the final expected values.
    # Since initial params are zero, this is just a scaling.
    expected_new_kmean = (1 - alpha) * expected_updated_kmean
    expected_new_zmean = (1 - alpha) * expected_updated_zmean
    expected_new_zstd = (1 - alpha) * expected_updated_zstd

    self.assertAllClose(new_kmean, expected_new_kmean)
    self.assertAllClose(new_zmean, expected_new_zmean)
    self.assertAllClose(new_std, expected_new_zstd)


if __name__ == '__main__':
  tf.test.main()
