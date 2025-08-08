"""A pure TensorFlow implementation of an MPPI-based TF-Agents policy and agent.

In this library, we implement the algorithm described in:
Zhang, Renhao, et al. "Model-based reinforcement learning for parameterized
action spaces." ICML (2024). https://arxiv.org/abs/2404.03037

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

from typing import Optional
import gin
import numpy as np
from smart_buildings.smart_control.models import base_world_model
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import tf_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


@gin.configurable
class MPPIPolicy(tf_policy.TFPolicy):
  """A TF-Agents policy that implements the MPPI planning logic from the DLPA paper.

  This policy uses Model Predictive Path Integral (MPPI) control, a variant of
  the Cross-Entropy Method (CEM), to plan actions in a parameterized action
  space. The overall approach is described in Section 4.2 and summarized in
  Algorithm 1 of the paper.
  """

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      world_model: base_world_model.BaseWorldModel,
      # --- Gin-configurable parameters ---
      seed: int = 42,
      seed_steps: int = 0,
      mpc_horizon: int = 12,
      mpc_popsize: int = 20,
      mpc_gamma: float = 0.99,
      mpc_num_elites: int = 5,
      cem_iter: int = 3,
      mpc_alpha: float = 0.1,
      mpc_temperature: float = 0.5,
      name: Optional[str] = None,
  ):
    """Initializes the MPPI policy.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      world_model: An instance of a class that inherits from
        `BaseWorldModel`, used for simulating trajectories.
      seed: The random seed for the TF random generator.
      seed_steps: The number of initial steps to take random actions before
        planning.
      mpc_horizon: The number of steps to look ahead in the planning horizon.
      mpc_popsize: The number of action sequences to sample in the population.
      mpc_gamma: The discount factor for future rewards.
      mpc_num_elites: The number of top-performing sequences to use for updating
        the distribution.
      cem_iter: The number of Cross-Entropy Method iterations.
      mpc_alpha: The learning rate for updating the distribution mean and std.
      mpc_temperature: The temperature parameter for scaling elite weights.
      name: The name of this policy.
    """
    # --- Initialize planner attributes ---
    self.model = world_model

    self.discrete_action_normalizers = self.model.discrete_action_normalizers
    self.continuous_action_normalizers = (
        self.model.continuous_action_normalizers
    )
    # --- Store Gin-injected parameters ---
    self.k_dim = self.model.k_dim
    self.all_z_dim = self.model.all_z_dim
    self.z_min = tf.constant(self.model.z_min, dtype=tf.float32)
    self.z_max = tf.constant(self.model.z_max, dtype=tf.float32)
    self.seed_steps = seed_steps
    self.mpc_horizon = mpc_horizon
    self.mpc_popsize = mpc_popsize
    self.mpc_gamma = mpc_gamma
    self.mpc_num_elites = mpc_num_elites
    self.cem_iter = cem_iter
    self.mpc_alpha = mpc_alpha
    self.mpc_temperature = mpc_temperature
    self.offset = tf.constant(self.model.offset, dtype=tf.int32)
    self.par_size = tf.constant(self.model.par_size, dtype=tf.int32)
    self._step_counter = tf.Variable(0, dtype=tf.int64, name='step_counter')
    # Use TF random generator for reproducibility
    self.tf_random_generator = tf.random.Generator.from_seed(seed)
    policy_state_spec = {
        'prev_mean_k': tf.TensorSpec(
            shape=(mpc_horizon, self.k_dim), dtype=tf.float32
        ),
        'prev_mean_z': tf.TensorSpec(
            shape=(mpc_horizon, self.all_z_dim), dtype=tf.float32
        ),
        'prev_std_z': tf.TensorSpec(
            shape=(mpc_horizon, self.all_z_dim), dtype=tf.float32
        ),
    }
    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        name=name,
    )

  def _get_initial_state(self, batch_size: int) -> types.NestedTensor:
    """Returns the initial state of the policy.

    Args:
      batch_size: The batch size for the initial state.

    Returns:
      A nest of tensors representing the initial policy state.
    """
    kmean_init = tf.ones(
        (self.mpc_horizon, self.k_dim), dtype=tf.float32
    ) / tf.cast(self.k_dim, tf.float32)
    zmean_init = tf.zeros((self.mpc_horizon, self.all_z_dim), dtype=tf.float32)
    zstd_init = 2 * tf.ones(
        (self.mpc_horizon, self.all_z_dim), dtype=tf.float32
    )
    return {
        'prev_mean_k': tf.tile(
            tf.expand_dims(kmean_init, 0), [batch_size, 1, 1]
        ),
        'prev_mean_z': tf.tile(
            tf.expand_dims(zmean_init, 0), [batch_size, 1, 1]
        ),
        'prev_std_z': tf.tile(tf.expand_dims(zstd_init, 0), [batch_size, 1, 1]),
    }

  def synchronize(self, source_env: tf_py_environment.TFPyEnvironment) -> None:
    """Synchronizes the planning environment with the acting environment.

    This delegates the synchronization logic to the world model.

    Args:
      source_env: The main TF-Agents environment to synchronize with.
    """
    self.model.synchronize(source_env)

  def _get_action_steps(
      self, time_step: ts.TimeStep, policy_state: types.NestedTensor
  ) -> policy_step.PolicyStep:
    """Runs the planner for a batch of time steps.

    This function handles the batch processing, state resetting on episode
    boundaries, and calling the core planning logic for each item in the batch.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A nest of tensors representing the policy's state.

    Returns:
      A `PolicyStep` named tuple containing the action, state, and info.
    """

    # --- Graph-compatible state reset ---
    def _reset_state_fn():
      # The step counter must be reset inside the true_fn of tf.cond to be
      # part of the graph.
      self._step_counter.assign(0)
      return self._get_initial_state(tf.shape(time_step.observation)[0])

    # Use tf.cond for graph-compatible conditional logic.
    # We assume all items in a batch are of the same step_type.
    policy_state = tf.cond(
        time_step.is_first()[0],
        true_fn=_reset_state_fn,
        false_fn=lambda: policy_state,
    )

    observation = time_step.observation

    # --- Graph-compatible batch processing ---
    # Use tf.map_fn to apply the planning logic to each item in the batch.
    # This replaces the Python for-loop and works in graph mode.
    # The model has state and is not re-entrant, so we must disable
    # parallel execution.
    action_steps = tf.map_fn(
        fn=lambda elems: self._plan_and_get_action_trajectory(
            elems[0], tf.nest.map_structure(lambda t: t, elems[1])
        ),
        elems=(observation, policy_state),
        fn_output_signature=self.policy_step_spec,
        parallel_iterations=1,
    )

    self._step_counter.assign_add(1)
    return action_steps

  def _action(
      self,
      time_step: ts.TimeStep,
      policy_state: types.NestedTensor,
      seed: Optional[types.Seed] = None,
  ) -> policy_step.PolicyStep:
    """Generates an action by running the MPPI planner.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A nest of tensors representing the policy's state.
      seed: The seed for any stochasticity in the action generation.

    Returns:
      A `PolicyStep` named tuple containing the action, state, and info.
    """
    return self._get_action_steps(time_step, policy_state)

  def _distribution(
      self, time_step: ts.TimeStep, policy_state: types.NestedTensor
  ) -> policy_step.PolicyStep:
    """Generates a distribution over actions by running the MPPI planner.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A nest of tensors representing the policy's state.

    Returns:
      A `PolicyStep` named tuple containing a distribution over actions.
    """
    action_steps = self._get_action_steps(time_step, policy_state)

    # The policy is sampling-based, so representing the output as a
    # distribution is more accurate. The chosen action is treated as the mean
    # (or mode) of this distribution.

    # For the discrete part, we select a single best action index.
    discrete_dist = tfp.distributions.Deterministic(
        loc=action_steps.action['discrete_action']
    )

    # For the continuous part, we create a narrow Normal distribution centered
    # around the chosen action to acknowledge the planner's stochastic nature.
    continuous_dist = tfp.distributions.Normal(
        loc=action_steps.action['continuous_action'],
        scale=1e-3,  # Small scale to represent a tight distribution
    )

    dist = {
        'discrete_action': discrete_dist,
        'continuous_action': continuous_dist,
    }
    return policy_step.PolicyStep(dist, action_steps.state, action_steps.info)

  def _plan_and_get_action_trajectory(
      self, observation: types.Tensor, policy_state: types.NestedTensor
  ) -> policy_step.PolicyStep:
    """Runs the planner for a single observation and returns a formatted action.

    This function is designed to be called within a `tf.map_fn` to process a
    batch of observations in a graph-compatible manner. It takes a single
    unbatched observation and the corresponding policy state, runs the MPPI
    planner, and formats the resulting action into a `PolicyStep`.

    Args:
      observation: A single tensor representing the environment's observation.
      policy_state: The policy's state for this specific observation.

    Returns:
      A `policy_step.PolicyStep` containing the chosen action, the new policy
      state, and an empty info tuple.
    """
    # This logic assumes a single, non-batched observation.
    observation_np = observation

    # Run the planner to get the best action sequence
    action_plan, new_policy_state = self.plan(
        observation_np, policy_state, step=self._step_counter
    )

    # Extract the first action from the plan.
    # Shape: (k_dim + all_z_dim,)
    first_action_normalized = action_plan[0]

    # Decode the discrete action 'k'
    # k_part shape: (k_dim,)
    k_part = first_action_normalized[: self.k_dim]
    k_tensor = tf.argmax(k_part, output_type=tf.int32)

    # Decode the continuous action 'z'. Full z-vector has shape (all_z_dim,).
    # We must slice the correct part of this vector corresponding to the
    # chosen 'k'.
    z_vector_normalized = first_action_normalized[self.k_dim :]
    offset = tf.gather(self.offset, k_tensor)
    num_params = tf.gather(self.par_size, k_tensor)
    z_normalized = tf.slice(z_vector_normalized, [offset], [num_params])

    # De-normalize the continuous part of the action.
    # Use the flattened z_min/z_max from the world model.
    z_max_active = tf.slice(self.model.all_z_max, [offset], [num_params])
    z_min_active = tf.slice(self.model.all_z_min, [offset], [num_params])
    z_native = (
        0.5
        * (tf.clip_by_value(z_normalized, -1.0, 1.0) + 1)
        * (z_max_active - z_min_active)
        + z_min_active
    )

    # Format the action into a dictionary that the environment can understand.
    # The model.create_action_dict method uses .numpy() and must be
    # wrapped in a tf.py_function to be compatible with graph mode.
    def _py_create_action(k_tensor, z_native_tensor):
      action_dict = self.model.create_action_dict(
          k_tensor.numpy(), z_native_tensor.numpy()
      )
      return [
          action_dict['discrete_action'],
          action_dict['continuous_action'],
      ]

    [discrete_action, continuous_action] = tf.py_function(
        func=_py_create_action,
        inp=[k_tensor, z_native],
        Tout=[tf.int32, tf.float32],
    )

    # Set shapes, as py_function loses them.
    discrete_action.set_shape(self.action_spec['discrete_action'].shape)
    continuous_action.set_shape(self.action_spec['continuous_action'].shape)
    action_tf = {
        'discrete_action': discrete_action,
        'continuous_action': continuous_action,
    }

    return policy_step.PolicyStep(
        action=action_tf, state=new_policy_state, info=()
    )

  def plan(
      self,
      state: types.Tensor,  # pylint: disable=unused-argument
      policy_state: types.NestedTensor,
      step: tf.Variable,
  ) -> tuple[tf.Tensor, types.NestedTensor]:
    """The core MPPI planning loop.

    This function implements the iterative planning process described in
    Section 4.2 and Algorithm 1 of the paper. It uses the Cross-Entropy
    Method (CEM) to refine a distribution over action sequences.

    Args:
      state: The current environment state (observation). Not used in the
        current implementation but kept for API consistency.
      policy_state: The policy's internal state, containing the mean of the
        action distribution from the previous step.
      step: The current time step counter.

    Returns:
      A tuple containing:
        - chosen_action_plan: The best sequence of actions found by the planner.
        - new_policy_state: The updated policy state for the next step.
    """

    def _seed_step_fn():
      """Action selection for seed steps."""
      k_int = self.tf_random_generator.uniform(
          shape=(), maxval=self.k_dim, dtype=tf.int32
      )
      k_onehot = tf.one_hot(k_int, self.k_dim)
      z_rand = self.tf_random_generator.uniform(
          shape=[self.all_z_dim], minval=-1.0, maxval=1.0
      )
      action_plan = tf.tile(
          tf.expand_dims(tf.concat([k_onehot, z_rand], axis=0), 0),
          [self.mpc_horizon, 1],
      )
      return action_plan, policy_state

    def _plan_fn():
      """The main planning logic."""
      # Initialize sampling distribution (mean and standard deviation)
      kmean_init = tf.ones((self.mpc_horizon, self.k_dim)) / self.k_dim
      zmean_init = tf.zeros((self.mpc_horizon, self.all_z_dim))
      std_init = 2 * tf.ones((self.mpc_horizon, self.all_z_dim))

      # Warm-start the distribution from the previous step
      def _warm_start_fn():
        # policy_state is already unbatched from tf.map_fn
        prev_mean_k = policy_state['prev_mean_k']
        prev_mean_z = policy_state['prev_mean_z']
        prev_std_z = policy_state['prev_std_z']
        kmean = tf.concat([prev_mean_k[1:], kmean_init[-1:]], axis=0)
        zmean = tf.concat([prev_mean_z[1:], zmean_init[-1:]], axis=0)
        std = tf.concat([prev_std_z[1:], std_init[-1:]], axis=0)
        return kmean, zmean, std

      kmean, zmean, std = tf.cond(
          tf.equal(step, 0),
          true_fn=lambda: (kmean_init, zmean_init, std_init),
          false_fn=_warm_start_fn,
      )

      # Initial sample before the loop
      actions_init = self._sample_from_distribution(kmean, zmean, std)
      value_init, _ = self._estimate_value(actions_init)
      value_init = tf.where(tf.math.is_nan(value_init), 0.0, value_init)

      # --- CEM Refinement Loop ---
      def _cem_loop_cond(i, kmean, zmean, std, actions, value):
        del kmean, zmean, std, actions, value
        return tf.less(i, self.cem_iter)

      def _cem_loop_body(i, kmean, zmean, std, actions, value):
        kmean_new, zmean_new, std_new = self._update_distribution(
            actions, value, kmean, zmean, std
        )
        actions_new = self._sample_from_distribution(
            kmean_new, zmean_new, std_new
        )
        value_new, _ = self._estimate_value(actions_new)
        value_new = tf.where(tf.math.is_nan(value_new), 0.0, value_new)
        return i + 1, kmean_new, zmean_new, std_new, actions_new, value_new

      _, kmean, zmean, std, actions, value = tf.while_loop(
          cond=_cem_loop_cond,
          body=_cem_loop_body,
          loop_vars=[0, kmean, zmean, std, actions_init, value_init],
      )

      # --- Action Selection ---
      _, elite_idxs = tf.math.top_k(
          tf.squeeze(value, axis=1), k=self.mpc_num_elites
      )
      elite_actions = tf.gather(actions, elite_idxs, axis=1)
      chosen_action_plan = elite_actions[:, 0]

      # Update policy state for the next step (unbatched)
      new_policy_state = {
          'prev_mean_k': kmean,
          'prev_mean_z': zmean,
          'prev_std_z': std,
      }
      return chosen_action_plan, new_policy_state

    return tf.cond(
        tf.less(step, self.seed_steps),
        true_fn=_seed_step_fn,
        false_fn=_plan_fn,
    )

  def _estimate_value(self, actions: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Rolls out trajectories and calculates their cumulative discounted reward.

    This function implements the trajectory return calculation described in
    Equation (4) of the paper.
    It is designed to work within a TensorFlow graph by wrapping the stateful,
    Python-based model rollout in `tf.py_function`.

    Args:
      actions: A tensor of action sequences to evaluate. Shape: (horizon,
        popsize, action_dim).

    Returns:
      A tuple containing:
        - total_rewards: The total discounted reward for each trajectory.
        - trajectories: The sequence of states for each trajectory.
    """
    # Capture the pristine model state ONCE before starting the rollouts.
    pristine_building_state = self.model.get_state()

    # This helper function contains the stateful Python logic for rolling out
    # a single trajectory. It will be wrapped in tf.py_function.
    def _py_rollout_single_trajectory(action_sequence):
      # The world model's state is reset from a pristine copy for each rollout.
      # This state was captured before the planning step began.
      self.model.set_state(pristine_building_state)

      trajectory_reward = 0.0
      discount = self.mpc_gamma
      trajectory_states = []

      # Use a standard Python loop, as this will run inside tf.py_function.
      for t in range(self.mpc_horizon):
        # The model.next() method expects a batched action.
        action_t = np.expand_dims(action_sequence[t], axis=0)
        s_pred, reward, _, _ = self.model.next(action_t)

        trajectory_states.append(s_pred[0])
        trajectory_reward += discount * reward[0, 0]
        discount *= self.mpc_gamma

      return np.array(trajectory_reward, dtype=np.float32), np.array(
          trajectory_states, dtype=np.float32
      )

    # Wrap the Python function to make it usable in a TF graph.
    def _tf_rollout_trajectory(action_sequence):
      reward, trajectory = tf.py_function(
          func=_py_rollout_single_trajectory,
          inp=[action_sequence],
          Tout=[tf.float32, tf.float32],
      )
      # Set the shape information, which is lost by tf.py_function.
      reward.set_shape(())
      trajectory.set_shape(
          [self.mpc_horizon, self.time_step_spec.observation.shape[0]]
      )
      return reward, trajectory

    # Transpose actions so that map_fn iterates over trajectories.
    # Original shape: (horizon, popsize, action_dim)
    # Transposed shape: (popsize, horizon, action_dim)
    actions_transposed = tf.transpose(actions, perm=[1, 0, 2])

    # Apply the rollout function to each action sequence in the population.
    # parallel_iterations=1 is critical because the underlying Python function
    # is stateful and not thread-safe.
    total_rewards, trajectories = tf.map_fn(
        fn=_tf_rollout_trajectory,
        elems=actions_transposed,
        fn_output_signature=(tf.float32, tf.float32),
        parallel_iterations=1,
    )

    # Reshape the outputs to match the original function's signature.
    return tf.expand_dims(total_rewards, axis=-1), trajectories

  def _sample_from_distribution(
      self, kmean: tf.Tensor, zmean: tf.Tensor, std: tf.Tensor
  ) -> tf.Tensor:
    """Samples a batch of action sequences from the current distribution.

    This implements the sampling process described by Equations (2) and (3)
    in Section 4.2 of the paper.
    - Eq (2) Discrete: k sim Cat(theta_1, ..., theta_K)
    - Eq (3) Continuous: z_k sim N(mu_k, sigma_k^2 I)

    Args:
      kmean: The mean of the categorical distribution for discrete actions
        (theta). Shape is (horizon, k_dim).
      zmean: The mean of the Gaussian distribution for continuous actions (mu).
        Shape is (horizon, all_z_dim).
      std: The standard deviation of the Gaussian distribution (sigma). Shape is
        (horizon, all_z_dim).

    Returns:
      A tensor of sampled action sequences. Shape: (horizon, popsize,
      k_dim + all_z_dim).
    """
    # Sample discrete actions
    k_logits = tf.math.log(kmean + 1e-9)  # Add epsilon for numerical stability
    k_int = tf.random.categorical(logits=k_logits, num_samples=self.mpc_popsize)
    k_onehot = tf.one_hot(k_int, depth=self.k_dim, dtype=tf.float32)
    # This transpose was causing a shape mismatch in the concatenation below.
    # k_onehot = tf.transpose(k_onehot, perm=[0, 2, 1])

    # Sample continuous actions from a Gaussian
    z_all = tf.clip_by_value(
        tf.expand_dims(zmean, 1)
        + tf.expand_dims(std, 1)
        * self.tf_random_generator.normal(
            shape=[self.mpc_horizon, self.mpc_popsize, self.all_z_dim]
        ),
        -1.0,
        1.0,
    )

    return tf.concat([k_onehot, z_all], axis=-1)

  def _update_distribution(
      self,
      actions: tf.Tensor,
      value: tf.Tensor,
      kmean: tf.Tensor,
      zmean: tf.Tensor,
      std: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Updates the distribution mean and std using Cross-Entropy Method.

    This function implements the CEM update rules from Section 4.2 of the paper,
    specifically Equations (5), (6), and (7).

    - Eq (5): Update rule for discrete action distribution mean (theta).
    - Eq (6): Update rule for continuous parameter distribution mean (mu).
    - Eq (7): Update rule for continuous parameter distribution std (sigma).

    Args:
      actions: The population of action sequences sampled from the distribution.
        Shape is (horizon, popsize, action_dim).
      value: The estimated value (return) for each action sequence. Shape is
        (popsize, 1).
      kmean: The current mean of the categorical distribution. Shape is
        (horizon, k_dim).
      zmean: The current mean of the Gaussian distribution. Shape is (horizon,
        all_z_dim).
      std: The current standard deviation of the Gaussian distribution. Shape is
        (horizon, all_z_dim).

    Returns:
      A tuple containing the updated (kmean, zmean, std).
    """
    # --- Get Elites ---
    # Select the top-performing action sequences (elites) based on their
    # estimated values.
    elite_values, elite_idxs = tf.math.top_k(
        tf.squeeze(value, axis=1), k=self.mpc_num_elites
    )
    # elite_actions shape: (horizon, num_elites, action_dim)
    elite_actions = tf.gather(actions, elite_idxs, axis=1)

    # --- Calculate Elite Weights ---
    # Convert elite values into weights using the softmax function. This gives
    # more influence to higher-rewarding trajectories.
    max_value = tf.reduce_max(elite_values)
    # elite_weights shape: (num_elites,)
    elite_weights = tf.exp(self.mpc_temperature * (elite_values - max_value))
    elite_weights /= tf.reduce_sum(elite_weights) + 1e-9

    # --- Separate k and z parts of the elite actions ---
    # elite_k shape: (horizon, num_elites, k_dim)
    elite_k = elite_actions[:, :, : self.k_dim]
    # elite_z shape: (horizon, num_elites, all_z_dim)
    elite_z = elite_actions[:, :, self.k_dim :]

    # --- Update Mean (k and z) ---
    # Calculate the weighted average of the elite actions to get the new mean.
    # Einsum computes the dot product between weights and actions over the
    # 'elites' dimension (p), for each step in the horizon (h).
    # 'hpe,p->he' where h=horizon, p=elites, e=features.
    # elite_weights is broadcast across the horizon dimension.
    updated_kmean = tf.einsum('hpe,p->he', elite_k, elite_weights)
    updated_zmean = tf.einsum('hpe,p->he', elite_z, elite_weights)

    # --- Update Standard Deviation (z only) ---
    # Calculate the weighted standard deviation for the continuous part.
    z_diff_sq = (elite_z - tf.expand_dims(updated_zmean, 1)) ** 2
    weighted_z_var = tf.einsum('hpe,p->he', z_diff_sq, elite_weights)
    updated_zstd = tf.sqrt(weighted_z_var + 1e-9)  # Add epsilon for stability

    # --- Apply Momentum (CEM smoothing) ---
    # Blend the new distribution parameters with the old ones to stabilize
    # the learning process.
    new_kmean = self.mpc_alpha * kmean + (1 - self.mpc_alpha) * updated_kmean
    new_zmean = self.mpc_alpha * zmean + (1 - self.mpc_alpha) * updated_zmean
    new_std = self.mpc_alpha * std + (1 - self.mpc_alpha) * updated_zstd

    return new_kmean, new_zmean, new_std


@gin.configurable
class MPPIAgent(tf_agent.TFAgent):
  """An agent that uses the pure TensorFlow MPPIPolicy to select actions."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      world_model,
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[str] = None,
      **kwargs,  # Pass-through for MPPIPolicy gin params
  ):
    """Initializes the MPPI agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      world_model: An instance of a class that inherits from
        `BaseWorldModel`, used by the policy for planning.
      train_step_counter: An optional `tf.Variable` to increment for each train
        step.
      name: The name of this agent.
      **kwargs: Additional keyword arguments to be passed to the MPPIPolicy.
    """
    tf.Module.__init__(self, name=name)

    policy = MPPIPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        world_model=world_model,
        **kwargs,  # Pass all other gin-configured params
    )

    super().__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=None,
        train_step_counter=train_step_counter,
    )

  def synchronize(self, env: tf_py_environment.TFPyEnvironment) -> None:
    """Synchronizes the planner's internal state with the environment.

    This is a proxy for the policy's `synchronize` method, which copies the
    state from the acting environment's building to the planning environment's
    building.

    Args:
      env: The environment to synchronize with.
    """
    self.policy.synchronize(env)

  def _initialize(self):  # pylint: disable=useless-super-delegation
    """Initializes the agent by calling the parent's `_initialize` method.

    This ensures that the policy's initial state is properly set up.

    Returns:
      An operation that initializes the agent.
    """
    return super(MPPIAgent, self)._initialize()

  def _train(
      self,
      experience: types.NestedTensor,
      weights: Optional[types.Tensor] = None,
  ) -> tf_agent.LossInfo:
    """MPPI is a planning agent and does not train from a replay buffer.

    This method is required by the `TFAgent` interface but is a no-op for MPPI,
    as the policy is optimized online through planning rather than offline
    through training on past experience.

    Args:
      experience: A batch of experience trajectories.
      weights: Optional weights for the experience.

    Returns:
      A `LossInfo` object with a constant loss of 0.0.
    """
    if self.train_step_counter is not None:
      self.train_step_counter.assign_add(1)
    return tf_agent.LossInfo(loss=tf.constant(0.0), extra=())
