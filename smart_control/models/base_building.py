"""Base class that extends functionality outside of the building.

The base class should be extended by the simulation and actual buildings.
"""

import abc
from collections.abc import Sequence
import itertools
from typing import Any

import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils.proto_parsers import device_info_parser
from smart_buildings.smart_control.utils.proto_parsers import zone_info_parser

SerializableData = dict[str, Any]


class BaseBuilding(abc.ABC):
  """Base class for a controllable building for reinforcement learning."""

  def __init__(self, zones: Sequence[building_pb2.ZoneInfo] | None = None):
    """Initializes the instance.

    Args:
      zones: A list of thermal zones in the building.
    """
    self._zones = list(zones) if zones else []

  @property
  @abc.abstractmethod
  def reward_info(self) -> reward_pb2.RewardInfo:
    """A message containing data to compute the instantaneous reward."""

  @abc.abstractmethod
  def request_observations(
      self, observation_request: building_pb2.ObservationRequest
  ) -> building_pb2.ObservationResponse:
    """Queries the building for its current state."""

  @abc.abstractmethod
  def request_observations_within_time_interval(
      self,
      observation_request: building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_timestamp: pd.Timestamp,
  ) -> Sequence[building_pb2.ObservationResponse]:
    """Queries the building for observations between start and end times."""

  @abc.abstractmethod
  def request_action(
      self, action_request: building_pb2.ActionRequest
  ) -> building_pb2.ActionResponse:
    """Issues a command to the building to change one or more setpoints."""

  @abc.abstractmethod
  def wait_time(self) -> None:
    """Returns after a certain amount of time."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building, throwing an RuntimeError if this is impossible."""

  @property
  @abc.abstractmethod
  def devices(self) -> Sequence[building_pb2.DeviceInfo]:
    """The devices that can be queried and/or controlled in the building."""

  @property
  def devices_df(self) -> pd.DataFrame:
    """A DataFrame listing the building's devices."""
    return pd.DataFrame(
        device_info_parser.DeviceInfoParser(d).as_dict for d in self.devices
    )

  @property
  def action_fields_df(self) -> pd.DataFrame:
    """DataFrame containing the combined action fields of all building devices."""
    return pd.DataFrame(
        r.as_dict
        for r in itertools.chain.from_iterable(
            device_info_parser.DeviceInfoParser(device).action_fields
            for device in self.devices
        )
    )

  @property
  def observable_fields_df(self) -> pd.DataFrame:
    """DataFrame containing the combined observable fields of all building devices."""
    return pd.DataFrame(
        r.as_dict
        for r in itertools.chain.from_iterable(
            device_info_parser.DeviceInfoParser(device).observable_fields
            for device in self.devices
        )
    )

  @property
  def fields_df(self) -> pd.DataFrame:
    """DataFrame containing the combined fields of all building devices."""
    return pd.DataFrame(
        r.as_dict
        for r in itertools.chain.from_iterable(
            device_info_parser.DeviceInfoParser(device).fields
            for device in self.devices
        )
    )

  @property
  def zones(self) -> Sequence[building_pb2.ZoneInfo]:
    """Sequence of thermal zones in the building managed by the RL agent."""
    return self._zones

  @property
  def zones_df(self) -> pd.DataFrame:
    """A DataFrame listing the building's thermal zones."""
    return pd.DataFrame(
        zone_info_parser.ZoneInfoParser(z).as_dict for z in self.zones
    )

  @property
  @abc.abstractmethod
  def current_timestamp(self) -> pd.Timestamp:
    """The current local timestamp of the building."""

  @abc.abstractmethod
  def render(self, path: str) -> None:
    """Renders the current state of the building."""

  @abc.abstractmethod
  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Whether or not the building is in comfort mode at the given timestamp."""

  @property
  @abc.abstractmethod
  def num_occupants(self) -> int:
    """The number of occupants currently in the building."""

  @property
  @abc.abstractmethod
  def time_step_sec(self) -> float:
    """The length of the time step, in seconds."""

  @property
  def json_metadata(self) -> SerializableData:
    """A JSON-serializable dictionary of metadata about the building."""
    return {
        'n_devices': len(self.devices),
        'n_zones': len(self.zones),
        'device_ids': self.devices_df['device_id'].tolist(),
        'zone_ids': self.zones_df['zone_id'].tolist(),
    }
