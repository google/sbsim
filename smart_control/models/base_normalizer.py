"""Defines observation and action normalizer base classes."""

import abc

import numpy as np
from tf_agents import specs

from smart_control.proto import smart_control_building_pb2


class BaseObservationNormalizer(metaclass=abc.ABCMeta):
  """Normalizer base class for Observations."""

  @abc.abstractmethod
  def normalize(
      self, native: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Normalizes an Observation response."""

  @abc.abstractmethod
  def denormalize(
      self, normalized: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """De-normalizes an Observation response."""


class BaseActionNormalizer(metaclass=abc.ABCMeta):
  """Translates native agent action values into normalized setpoint values."""

  @abc.abstractmethod
  def get_array_spec(self, name=None) -> specs.ArraySpec:
    """Returns array_spec for the action.

    This informs the agent how many values to output, which will get
    transformed into a single value for the setpoint.

    Args:
      name: Name to pass to the ArraySpec
    """

  @abc.abstractmethod
  def setpoint_value(self, agent_action: np.ndarray) -> float:
    """Returns value to apply to building given agent action values.

    Args:
      agent_action: Values returned directly from agent, compatible with
        array_spec.
    """

  @abc.abstractmethod
  def agent_value(self, setpoint_value: float) -> float:
    """Returns the normalized setpoint_value as an agent action.

    Args:
      setpoint_value: Value in native units.
    """

  @property
  @abc.abstractmethod
  def setpoint_min(self) -> float:
    """Returns the minimum setpoint value."""

  @property
  @abc.abstractmethod
  def setpoint_max(self) -> float:
    """Returns the maximum setpoint value."""
