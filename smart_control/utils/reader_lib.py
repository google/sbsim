"""Utilities to read smart control protos from endpoint."""

import abc
from typing import Final, Mapping, NewType, Sequence, TypeVar

from absl import logging
import gin
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2

VariableId = NewType('VariableId', str)

T = TypeVar('T')


class BaseReader(metaclass=abc.ABCMeta):
  """Abstract base class for writing the building and reward protos."""

  @abc.abstractmethod
  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads observation_responses from endpoint bounded by start & end time."""

  @abc.abstractmethod
  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads action_responses from endpoint bounded by start and end time."""

  @abc.abstractmethod
  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward infos from endpoint bounded by start and end time."""

  @abc.abstractmethod
  def read_reward_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward responses from endpoint bounded by start and end time."""

  @abc.abstractmethod
  def read_normalization_info(
      self,
  ) -> Mapping[
      VariableId, smart_control_normalization_pb2.ContinuousVariableInfo
  ]:
    """Reads variable normalization info from RecordIO."""

  @abc.abstractmethod
  def read_zone_infos(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Reads the zone infos for the Building."""

  @abc.abstractmethod
  def read_device_infos(
      self,
  ) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Reads the device infos for the Building."""


@gin.configurable
class Readers:

  def __init__(self, readers: Sequence[BaseReader]):
    self._readers: Final[Sequence[BaseReader]] = readers
    logging.info('There are %d readers available.', len(readers))

  @property
  def readers(self) -> Sequence[BaseReader]:
    return self._readers
