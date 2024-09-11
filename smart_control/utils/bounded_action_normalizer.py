"""Helper classes for mapping normalized agent actions to native setpoint values.

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

import numpy as np
from smart_control.models import base_normalizer
from tf_agents import specs
# Due to floating point precision errors, it's possible that values will be
# above/under the max/min thresholds by a small amount. ACTION_TOLERANCE
# allows the action values to range within a narrow range.
ACTION_TOLERANCE = 0.00001


class BoundedActionNormalizer(base_normalizer.BaseActionNormalizer):
  """Translates normalized agent action values into native setpoint values.

  Actions involve setting real valued setpoints. The normalized agent action
  range is mapped directly to the native setpoint value range.
  """

  def __init__(
      self,
      min_native_value: float,
      max_native_value: float,
      min_normalized_value: float = -1.0,
      max_normalized_value: float = 1.0,
  ):
    """Creates BoundedActionNormalizer.

    Args:
      min_native_value: Min value to output for setpoint.
      max_native_value: Max value to output for setpoint.
      min_normalized_value: Min value as input from agent.
      max_normalized_value: Max value as input from agent.
    """
    self._min_native_value = min_native_value
    self._max_native_value = max_native_value
    self._min_normalized_value = min_normalized_value
    self._max_normalized_value = max_normalized_value
    self._tolerance = ACTION_TOLERANCE

  def get_array_spec(self, name=None) -> specs.ArraySpec:
    """Returns array_spec for the action.

    This informs the agent how many normalized values to output along with their
    range, which will get transformed into a single value for the setpoint.

    Args:
      name: Name to pass to the ArraySpec.
    """
    return specs.BoundedArraySpec(
        (),
        np.float32,
        minimum=self._min_normalized_value,
        maximum=self._max_normalized_value,
        name=name,
    )

  def setpoint_value(self, agent_action: np.ndarray) -> float:
    """Returns value to apply to building given agent action values.

    Args:
      agent_action: normalized values returned directly from agent, compatible
        with array_spec.
    """
    if np.ndim(agent_action) > 0:
      raise ValueError(
          f'agent_action expected to be scalar but received: {agent_action}'
      )
    if agent_action < (
        self._min_normalized_value - self._tolerance
    ) or agent_action > (self._max_normalized_value + self._tolerance):
      raise ValueError(
          f'agent_action: {agent_action} not within bounds'
          f' [{self._min_normalized_value}, {self._max_normalized_value}]'
      )

    # Map agent value to range (0,1).
    input_range = self._max_normalized_value - self._min_normalized_value
    agent_ratio = (agent_action - self._min_normalized_value) / input_range

    # Map value to normalized range.
    output_range = self._max_native_value - self._min_native_value
    return agent_ratio * output_range + self._min_native_value  # pytype: disable=bad-return-type  # typed-numpy

  def agent_value(self, setpoint_value: float) -> float:
    """Returns the normalized setpoint_value as an agent action.

    Args:
      setpoint_value: Value in native units.
    """
    if (
        setpoint_value > self._max_native_value
        or setpoint_value < self._min_native_value
    ):
      raise ValueError(
          f'setpoint_value {setpoint_value} not within bounds'
          f' [{self._min_native_value}, {self._max_native_value}]'
      )
    return (self._max_normalized_value - self._min_normalized_value) / (
        self._max_native_value - self._min_native_value
    ) * (setpoint_value - self._min_native_value) + self._min_normalized_value

  @property
  def setpoint_min(self) -> float:
    """Returns the minimum setpoint value."""
    return self._min_native_value

  @property
  def setpoint_max(self) -> float:
    """Returns the maximum setpoint value."""
    return self._max_native_value
