"""Defines the BaseWorldModel abstract base class to be used by a model-based RL algorithm (MPPI).

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

import abc
from typing import Any, List, Mapping, TypeAlias
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment

StateKey: TypeAlias = str
StateValue: TypeAlias = Any
StateType: TypeAlias = Mapping[StateKey, StateValue]
DiscreteActionMapping: TypeAlias = dict[int, list[float]]


class BaseWorldModel(abc.ABC):
  """Abstract base class for a world model used by the MPPI planner."""

  def __init__(self):
    """Initializes the world model's parameters."""
    self.all_z_dim: int = 0
    self.all_z_max: List[float] = []
    self.all_z_min: List[float] = []
    self.continuous_action_normalizers: Mapping[str, Any] = {}
    self.discrete_action_normalizers: Mapping[str, Any] = {}
    self.k_dim: int = 0
    self.offset: List[int] = []
    self.par_size: List[int] = []
    self.z_max: List[float] = []
    self.z_min: List[float] = []

  @abc.abstractmethod
  def next(
      self, action: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Predicts the next state, reward, and continuation probability.

    Args:
        action: A tensor representing the action to take.

    Returns:
        A tuple of:
          s_pred_tensor: Predicted next state.
          reward_tensor: Predicted reward.
          continue_prob: Probability of continuing.
          r2: An optional secondary reward tensor, which may be used for
          specific environments or for intrinsic rewards. Current implementation
          of MPPI does not use it.
    """

  @abc.abstractmethod
  def get_state(self) -> StateType:
    """Saves a snapshot of the internal state of the model's environment.

    This is used by the planner to save the state of the simulation environment
    so that it can be restored later to explore different trajectories.
    Basically, it is required to be able to reset the environment to the
    pristine state. While reset might work for some environments, it is not
    always possible to restore the environment to a previous state exactly.
    Hence, we use this method to save a snapshot of the environment's state,
    which might not be possible to achieve using the environment's reset()
    method.

    Returns:
      A dictionary representing the state of the environment.
    """

  @abc.abstractmethod
  def set_state(self, state_dict: StateType):
    """Restores the internal state of the model's environment.

    This is used by the planner to restore the state of the simulation
    environment to a previously saved state.

    Args:
      state_dict: A dictionary representing the state of the environment, as
        returned by `get_state`
    """
    pass

  @property
  @abc.abstractmethod
  def env(self) -> py_environment.PyEnvironment:
    """Returns the underlying environment."""

  @property
  @abc.abstractmethod
  def discrete_action_mapping(self) -> DiscreteActionMapping:
    """Returns the mapping from discrete action index to supervisor commands."""

  @discrete_action_mapping.setter
  @abc.abstractmethod
  def discrete_action_mapping(self, value: DiscreteActionMapping):
    pass

  @abc.abstractmethod
  def create_action_dict(self, k: int, z: np.ndarray) -> Mapping[str, Any]:
    """Creates a dictionary of actions for the environment."""

  @abc.abstractmethod
  def synchronize(self, source_env: tf_py_environment.TFPyEnvironment):
    """Synchronizes the planning environment with the acting environment."""
