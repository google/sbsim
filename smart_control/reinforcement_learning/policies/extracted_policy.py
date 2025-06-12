"""Module for a TF Policy that aggregates historical actions based on a
timedeltaand then replays the aggregated actions sequentially."""

import datetime
import logging
import math
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.specs import BoundedTensorSpec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]",
)
logger = logging.getLogger(__name__)


class ExtractedPolicy(tf_policy.TFPolicy):
  """
  A TF Policy that aggregates historical actions based on a timedelta
  and then replays the aggregated actions sequentially.

  Each aggregated action (average over a bin) is REPEATED in the replay
  sequence a number of times equal to the count of original actions
  that fell into its corresponding aggregation bin.

  The aggregation timedelta can be changed, triggering re-aggregation.
  Ignores TimeStep observation content during action selection.
  """

  def __init__(
      self,
      original_actions: np.ndarray,
      original_parsed_times: List[datetime.datetime],
      initial_aggregation_timedelta: datetime.timedelta,
      time_step_spec: ts.TimeStep,
      action_spec: BoundedTensorSpec,
      name: str = "ExtractedPolicy",
  ):
    """
    Initializes the ExtractedPolicy.

    Args:
        original_actions: Numpy array of actions recorded (N, D). Assumed
                          ordered by time.
        original_parsed_times: List of N datetime objects corresponding to
                               actions. Must be sorted chronologically and
                               reasonably regular.
        initial_aggregation_timedelta: The initial time duration for aggregation
                                       bins. Must be a multiple of the original
                                       data interval.
        time_step_spec: A `TimeStep` spec (required by base class).
        action_spec: A BoundedTensorSpec representing the actions.
        name: The name of this policy.
    """
    # number of actions should be equal to number of timestamps
    if len(original_actions) != len(original_parsed_times):
      raise ValueError(
          "original_actions and original_parsed_times must have the same"
          " length."
      )

    # there must be at least two actions to determine the interval
    if len(original_parsed_times) < 2:
      raise ValueError(
          "Need at least two original timestamps to determine interval."
      )

    self._original_actions_np = np.array(
        original_actions, dtype=action_spec.dtype.as_numpy_dtype
    )
    self._original_times = list(original_parsed_times)
    self._original_step_delta = (
        self._original_times[1] - self._original_times[0]
    )
    if self._original_step_delta <= datetime.timedelta(0):
      raise ValueError("Original timestamps must be increasing.")
    logger.info(
        "Detected original time step interval: %s", self._original_step_delta
    )

    # variable used to store the index of the next action to be returned
    policy_state_spec = tensor_spec.TensorSpec(
        shape=(), dtype=tf.int32, name="replay_index"
    )

    # store specs
    self._action_spec_dtype = action_spec.dtype
    self._action_spec_shape = action_spec.shape
    self._action_dim = self._original_actions_np.shape[1]

    super(ExtractedPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        name=name,
    )

    # this tensor will hold the actions to be returned by the policy
    fixed_shape = tf.TensorShape([len(original_actions), self._action_dim])
    self._repeated_aggregated_actions_tensor = tf.Variable(
        tf.zeros(shape=fixed_shape, dtype=self._action_spec_dtype),
        trainable=False,
        shape=fixed_shape,
        name="repeated_actions",
    )

    # call the setter to perform the initial aggregation
    self.aggregation_timedelta = initial_aggregation_timedelta

  def _validate_timedelta(self, value: datetime.timedelta):
    if not isinstance(value, datetime.timedelta):
      raise TypeError(
          "aggregation_timedelta must be a datetime.timedelta object."
      )

    if value <= datetime.timedelta(0):
      raise ValueError("aggregation_timedelta must be positive.")

    # timedelta to aggregate must be a multiple of the original step interval
    ratio = value.total_seconds() / self._original_step_delta.total_seconds()
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
      raise ValueError(
          f"aggregation_timedelta ({value}) must be a multiple of the original"
          f" step interval ({self._original_step_delta}). Ratio is {ratio}."
      )

  def _update_aggregation(self):
    logger.info(
        "Re-aggregating actions with timedelta: %s", self._aggregation_timedelta
    )
    repeated_aggregated_actions_list = []
    num_original_actions = len(self._original_actions_np)
    ratio = int(
        round(
            self._aggregation_timedelta.total_seconds()
            / self._original_step_delta.total_seconds()
        )
    )

    for i in range(0, num_original_actions, ratio):
      start_idx = i
      end_idx = min(start_idx + ratio, num_original_actions)
      actions_in_bin = self._original_actions_np[start_idx:end_idx]

      average_action = np.mean(actions_in_bin, axis=0)
      repetitions = actions_in_bin.shape[0]
      repeated_aggregated_actions_list.extend([average_action] * repetitions)

    logger.info(
        "Aggregated %d actions from %d original actions.",
        len(repeated_aggregated_actions_list),
        num_original_actions,
    )

    if not repeated_aggregated_actions_list:
      logger.error(
          "No actions were aggregated. The replay sequence will be empty."
      )
      raise ValueError(
          "No actions were aggregated. Please check the input data and"
          " timedelta."
      )

    np_actions = np.array(repeated_aggregated_actions_list)
    logger.info(
        "Aggregation resulted in %d repeated replay actions.",
        len(repeated_aggregated_actions_list),
    )
    self._repeated_aggregated_actions_tensor.assign(np_actions)

  @property
  def aggregation_timedelta(self) -> datetime.timedelta:
    return self._aggregation_timedelta

  @aggregation_timedelta.setter
  def aggregation_timedelta(self, value: datetime.timedelta):
    self._validate_timedelta(value)
    self._aggregation_timedelta = value
    self._update_aggregation()

  def _get_initial_state(
      self, batch_size: Optional[int] = None
  ) -> types.NestedTensor:
    state_shape = []
    if batch_size is not None:
      state_shape = [batch_size]
    return tf.zeros(shape=state_shape, dtype=tf.int32)

  @tf.function
  def _action(
      self,
      time_step: ts.TimeStep,
      policy_state: types.NestedTensor,
      seed: Any = None,
  ):
    current_index = policy_state
    safe_index = current_index
    action = tf.gather(
        self._repeated_aggregated_actions_tensor, safe_index, axis=0
    )
    next_index = (
        current_index + 1
    ) % self._repeated_aggregated_actions_tensor.shape[0]

    return policy_step.PolicyStep(action=action, state=next_index, info=())

  @tf.function
  def _distribution(
      self, time_step: ts.TimeStep, policy_state: types.NestedTensor
  ):
    action_step = self._action(time_step, policy_state)
    action_distribution = tf.nest.map_structure(
        lambda act: tfp.distributions.Deterministic(loc=act), action_step.action
    )
    return policy_step.PolicyStep(
        action=action_distribution,
        state=action_step.state,
        info=action_step.info,
    )
