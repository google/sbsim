"""Defines the EnvWorldModel class to be used by a model-based RL algorithm (MPPI).

This class is a wrapper around the environment used for our BaseBuilding Class.
Thus, it can work with any environment that inherits from BaseBuilding class.

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

import collections
import itertools
from typing import Any, List, Mapping, cast

import gin
import numpy as np
import pandas as pd
from smart_buildings.smart_control.agents.mppi import mppi_utils
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.models import base_world_model
from smart_buildings.smart_control.utils import bounded_action_normalizer
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import policy_step as tf_policy_step


@gin.configurable
class EnvWorldModel(base_world_model.BaseWorldModel):
  """A wrapper around the eval_env that provides a `next` method to serve as the world model for model-based RL algorithms."""

  def __init__(
      self,
      env: environment.Environment,
      discrete_normalizers: dict[
          str, bounded_action_normalizer.BoundedActionNormalizer
      ],
      continuous_normalizers: dict[
          str, bounded_action_normalizer.BoundedActionNormalizer
      ],
      par_size: np.ndarray | None = None,
  ):
    """Initializes the EnvWorldModel.

    Args:
      env: The evaluation environment.
      discrete_normalizers: Normalizers for discrete actions.
      continuous_normalizers: Normalizers for continuous actions.
      par_size: An array specifying the size of the continuous action space for
        each discrete action.
    """
    self._env = env
    self.par_size = tf.constant(par_size, dtype=tf.int32)
    self.discrete_action_normalizers = discrete_normalizers
    self.continuous_action_normalizers = continuous_normalizers
    self.action_spec = self.env.action_spec()
    # offset[k] is the index of the first continuous parameter in the k-th
    # discrete action. This is used to map the continuous action to the correct
    # part of the flattened continuous action space (which has size all_z_dim).
    offset = np.insert(np.cumsum(self.par_size), 0, 0)[:-1]
    self.offset = tf.constant(offset, dtype=tf.int32)
    # This is the total number of continuous parameters in the action space.
    self.all_z_dim = np.sum(par_size)

    (
        unique_param_keys,
        unique_param_key_to_idx,
    ) = self._initialize_unique_continuous_params()
    (
        supervisor_ids,
        supervisor_id_to_key,
        supervisor_to_params,
    ) = self._initialize_supervisor_info(unique_param_keys)
    params_for_k = self._generate_parameter_combinations(
        supervisor_ids, supervisor_to_params
    )
    self.k_dim = len(params_for_k)
    self.supervisor_order = supervisor_ids
    self._compute_discrete_action_mapping(
        params_for_k,
        supervisor_id_to_key,
        supervisor_to_params,
    )
    self._create_flattened_mappings(params_for_k, unique_param_key_to_idx)

  def synchronize(self, source_env: tf_py_environment.TFPyEnvironment):
    """Synchronizes the planning environment with the acting environment.

    This copies the state from the acting environment's building to the
    planning environment's building, ensuring the planner starts its
    simulation from the correct state.

    Args:
      source_env: The main TF-Agents environment to synchronize with.
    """
    # The source_env could be a TFPyEnvironment wrapper or a raw Python env.
    py_env = source_env.pyenv if hasattr(source_env, 'pyenv') else source_env

    # Access the underlying Python environment's building instance.
    if not hasattr(py_env, 'building') or not hasattr(self.env, 'building'):
      return

    source_building = py_env.building
    target_building = self.env.building

    # Copy all relevant state attributes
    target_building._native_inputs = source_building._native_inputs.copy()  # pylint: disable=protected-access
    target_building._current_observation_mapping = (  # pylint: disable=protected-access
        source_building._current_observation_mapping.copy()  # pylint: disable=protected-access
    )
    target_building._current_timestamp = source_building._current_timestamp  # pylint: disable=protected-access
    target_building._observation_index = source_building._observation_index  # pylint: disable=protected-access

    if (
        hasattr(source_building, '_current_action_mapping')
        and source_building._current_action_mapping is not None  # pylint: disable=protected-access
    ):
      target_building._current_action_mapping = (  # pylint: disable=protected-access
          source_building._current_action_mapping.copy()  # pylint: disable=protected-access
      )
    else:
      target_building._current_action_mapping = None  # pylint: disable=protected-access

  def _initialize_unique_continuous_params(
      self,
  ) -> tuple[list[str], dict[str, int]]:
    """Initializes and returns unique continuous parameter information."""
    # 1. Define the canonical list of UNIQUE continuous parameters from the
    # normalizers.
    # TODO(sipple): Remove protected access.
    unique_param_keys = sorted(self.continuous_action_normalizers.keys())
    z_min = [
        self.continuous_action_normalizers[key]._min_native_value  # pylint: disable=protected-access
        for key in unique_param_keys
    ]
    z_max = [
        self.continuous_action_normalizers[key]._max_native_value  # pylint: disable=protected-access
        for key in unique_param_keys
    ]
    self.z_min = tf.constant(z_min, dtype=tf.float32)
    self.z_max = tf.constant(z_max, dtype=tf.float32)
    unique_param_key_to_idx = {
        key: i for i, key in enumerate(unique_param_keys)
    }
    return unique_param_keys, unique_param_key_to_idx

  def _initialize_supervisor_info(
      self, unique_param_keys: list[str]
  ) -> tuple[
      list[str], dict[str, str], collections.defaultdict[str, List[str]]
  ]:
    """Initializes and returns supervisor-related information."""
    # 2. Dynamically build a map from supervisor ID to its full key
    supervisor_id_to_key = {}
    for key in self.discrete_action_normalizers.keys():
      # Assumes the ID is the part before the first underscore
      supervisor_id = key.split('_', 1)[0]
      supervisor_id_to_key[supervisor_id] = key

    # Derive the sorted list of supervisor IDs from the map
    supervisor_ids = sorted(supervisor_id_to_key.keys())

    # Group continuous parameters by supervisor ID
    supervisor_to_params = collections.defaultdict(list)
    for param_name in unique_param_keys:
      supervisor_id = param_name.split('_', 1)[0]
      supervisor_to_params[supervisor_id].append(param_name)
    return supervisor_ids, supervisor_id_to_key, supervisor_to_params

  def _generate_parameter_combinations(
      self,
      supervisor_ids: list[str],
      supervisor_to_params: collections.defaultdict[str, List[str]],
  ) -> list[list[str]]:
    """Generates all combinations of supervisor parameters."""
    supervisor_params_list = [
        supervisor_to_params.get(sid, []) for sid in supervisor_ids
    ]

    params_for_k = []
    for r in range(len(supervisor_params_list) + 1):
      for combo in itertools.combinations(supervisor_params_list, r):
        flattened_list = []
        for sublist in combo:
          flattened_list.extend(sublist)
        params_for_k.append(flattened_list)
    return params_for_k

  def _compute_discrete_action_mapping(
      self,
      params_for_k: list[list[str]],
      supervisor_id_to_key: dict[str, str],
      supervisor_to_params: collections.defaultdict[str, List[str]],
  ):
    """Computes the discrete action mapping."""
    # TODO(sipple): Remove protected access.
    # 3.Compute the discrete action mapping using the dynamic lookup
    self._discrete_action_mapping = {}
    for k, k_params in enumerate(params_for_k):
      command = []
      for supervisor_id in self.supervisor_order:
        #  Look up the full key from the map instead of reconstructing it
        normalizer_key = supervisor_id_to_key[supervisor_id]
        normalizer = self.discrete_action_normalizers[normalizer_key]
        supervisor_params = supervisor_to_params.get(supervisor_id, [])
        if supervisor_params and any(p in k_params for p in supervisor_params):
          command.append(normalizer._max_native_value)  # pylint: disable=protected-access
        else:
          command.append(normalizer._min_native_value)  # pylint: disable=protected-access
      self._discrete_action_mapping[k] = command

  def _create_flattened_mappings(
      self,
      params_for_k: list[list[str]],
      unique_param_key_to_idx: dict[str, int],
  ):
    """Creates mappings for the flattened action space."""
    # 4. Construct the final FLATTENED list of parameter names. This list has
    # `all_z_dim` elements and defines the exact layout of the model's output.
    flattened_param_names = []
    for k_idx in range(self.k_dim):
      flattened_param_names.extend(params_for_k[k_idx])

    # 5. Build the mapping from the flattened space to the unique
    # parameter space.
    self.all_dim_to_unique_dim_idx = tf.constant(
        [unique_param_key_to_idx[name] for name in flattened_param_names],
        dtype=tf.int32,
    )

    # 6. Use the mapping to create z_min/z_max vectors that align with the
    # `all_z_dim` (flattened) action space for de-normalization.
    self.all_z_min = tf.gather(self.z_min, self.all_dim_to_unique_dim_idx)
    self.all_z_max = tf.gather(self.z_max, self.all_dim_to_unique_dim_idx)
    self.all_z_min.set_shape([self.all_z_dim])
    self.all_z_max.set_shape([self.all_z_dim])

    # 7. Create the continuous_mapping list, aligned with the flattened space.
    self.continuous_mapping = [
        {
            'device_id': name.split('_', 1)[0],
            'setpoint_name': name.split('_', 1)[1],
        }
        for name in flattened_param_names
    ]

  @property
  def env(self):
    return self._env

  @property
  def discrete_action_mapping(self) -> dict[int, List[float]]:
    return self._discrete_action_mapping

  @discrete_action_mapping.setter
  def discrete_action_mapping(self, value: dict[int, List[float]]):
    self._discrete_action_mapping = value

  def get_state(self) -> dict[str, Any]:
    """Saves a snapshot of the internal state of the building environment."""
    if not hasattr(self.env, 'building'):
      # TODO(sipple): Remove protected access.
      return {}

    native_inputs = self.env.building._native_inputs.copy()  # pylint: disable=protected-access
    obs_mapping = self.env.building._current_observation_mapping.copy()  # pylint: disable=protected-access
    timestamp = self.env.building._current_timestamp  # pylint: disable=protected-access
    obs_index = self.env.building._observation_index  # pylint: disable=protected-access
    if (
        self.env.building._current_action_mapping is not None  # pylint: disable=protected-access
    ):  # pylint: disable=protected-access
      action_mapping = self.env.building._current_action_mapping.copy()  # pylint: disable=protected-access
    else:
      action_mapping = None

    pristine_state = {
        '_native_inputs': native_inputs,
        '_current_observation_mapping': obs_mapping,
        '_current_timestamp': timestamp,
        '_observation_index': obs_index,
        '_current_action_mapping': action_mapping,
    }
    return pristine_state

  def set_state(self, state_dict: Mapping[str, Any]):
    """Restores the internal state of the building environment."""
    # TODO(sipple): Remove protected access.
    if not hasattr(self.env, 'building') or not state_dict:
      return

    for key, value in state_dict.items():
      setattr(
          self.env.building,
          key,
          value.copy() if hasattr(value, 'copy') else value,
      )

  @property
  def building(self):
    return self.env.building

  def rollout(
      self,
      start_timestamp: pd.Timestamp,
      action_trajectory: list[tuple[int, tf.Tensor]],
  ):
    """Resets the environment and executes a sequence of actions."""
    predicted_states, predicted_rewards = [], []
    # TODO(sipple): Remove protected access.
    self.env._episode_start_timestamp = start_timestamp  # pylint: disable=protected-access
    time_step = self.env.reset()
    predicted_states.append(time_step.observation)  # Add initial state

    # Execute the trajectory of actions for each step of the horizon.
    for _, (k_action, z_action) in enumerate(action_trajectory):
      # Call the new, centralized helper method
      action_dict = self.create_action_dict(k_action, z_action.numpy())
      time_step = self.env.step(action_dict)
      predicted_states.append(time_step.observation)
      predicted_rewards.append(time_step.reward)

    return predicted_states, predicted_rewards

  def create_action_dict(
      self, k: int, z_native: np.ndarray
  ) -> dict[str, np.ndarray]:
    """Builds the final action dictionary required by the environment.

    This is the single source of truth for converting a discrete action 'k' and
    a de-normalized continuous vector 'z_native' into the correct format.

    Args:
      k: The chosen discrete action index.
      z_native: A numpy array of de-normalized continuous action values.

    Returns:
      The formatted action dictionary.
    """
    actions_to_apply = []
    supervisor_vals = self._discrete_action_mapping[k]

    # Get the number of active parameters and the offset for the chosen 'k'
    num_active = self.par_size[k].numpy()
    offset_val = self.offset[k].numpy()

    # Map active continuous parameters
    for i in range(num_active):
      param_idx = offset_val + i
      mapping = self.continuous_mapping[param_idx]
      actions_to_apply.append({
          'device_id': mapping['device_id'],
          'action_name': mapping['setpoint_name'],
          'native_value': z_native[i],
      })

    # Add discrete supervisor actions
    for i, device_id in enumerate(self.supervisor_order):
      actions_to_apply.append({
          'device_id': device_id,
          'action_name': 'supervisor_run_command',
          'native_value': supervisor_vals[i],
      })

    # Initialize the action dictionary structure
    action_dict = {
        'discrete_action': np.zeros(
            self.action_spec['discrete_action'].shape, dtype=np.int32
        ),
        'continuous_action': np.zeros(
            self.action_spec['continuous_action'].shape, dtype=np.float32
        ),
    }
    policy_step = collections.namedtuple(
        'PolicyStepMimic', ['action', 'state', 'info']
    )(action=action_dict, state=(), info=())

    # Populate the action dictionary using the utility function
    for action_item in actions_to_apply:
      policy_step = mppi_utils.apply_action(
          cast(tf_policy_step.PolicyStep, policy_step),
          self.discrete_action_normalizers,
          self.continuous_action_normalizers,
          action_item['device_id'],
          action_item['action_name'],
          action_item['native_value'],
      )
    return policy_step.action

  def _format_action_from_tensor(self, a: tf.Tensor) -> dict[str, np.ndarray]:
    """Decodes the model's raw action tensor and formats it for the environment."""
    if a.shape[0] != 1:
      raise ValueError(
          'EnvWorldModel._format_action_from_tensor expects a batch size of 1,'
          f' but got shape {a.shape}'
      )
    action_tensor = tf.squeeze(a, axis=0)

    # 1. Decode the discrete action 'k'
    k_tensor = tf.argmax(action_tensor[: self.k_dim], output_type=tf.int32)

    # 2. Decode and de-normalize the continuous action 'z'
    z_vector_normalized = action_tensor[self.k_dim :]
    offset = self.offset[k_tensor]
    num_params = self.par_size[k_tensor]
    z_normalized = z_vector_normalized[offset : offset + num_params]

    z_max_active = self.all_z_max[offset : offset + num_params]
    z_min_active = self.all_z_min[offset : offset + num_params]
    z_native = (
        0.5
        * (tf.clip_by_value(z_normalized, -1.0, 1.0) + 1)
        * (z_max_active - z_min_active)
        + z_min_active
    )

    # 3. Delegate final formatting to the helper function
    return self.create_action_dict(k_tensor.numpy(), z_native.numpy())

  def next(self, a: tf.Tensor):
    """Mimics the learned world_model's `next` method."""
    # 1. Decode the action tensor into the dictionary format for the env
    action_dict = self._format_action_from_tensor(a)

    time_step = self.env.step(action_dict)

    # 2. Unpack the results from the TimeStep object
    s_pred = time_step.observation
    reward = time_step.reward
    is_terminal = time_step.is_last()

    # This handles cases where the environment returns a float, int, None,
    # or a 0-dimensional numpy.ndarray.(#Q: would this be a problem
    # when reward in -inf?)
    try:
      reward_float = float(reward)
    except (TypeError, ValueError):
      reward_float = 0.0

    # 3. Format the results into tensors
    s_pred_tensor = tf.expand_dims(
        tf.constant(s_pred, dtype=tf.float32), axis=0
    )
    reward_tensor = tf.constant([[reward_float]], dtype=tf.float32)

    continue_prob = (
        tf.constant([[0.0, 1.0]]) if is_terminal else tf.constant([[1.0, 0.0]])
    )
    r1 = None

    return s_pred_tensor, reward_tensor, continue_prob, r1
