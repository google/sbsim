"""Base class that extends functionality outside of the building.

The base class should be extended by the simulation and actual buildings.
"""

import abc
from collections.abc import Sequence
from typing import Any

import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2

SerializableData = dict[str, Any]


class BaseBuilding(metaclass=abc.ABCMeta):
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
    """Returns a message with data to compute the instantaneous reward."""

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
    """Lists the devices that can be queried and/or controlled."""

  @property
  def devices_df(self) -> pd.DataFrame:
    """Lists the building's devices in dataframe format."""
    device_records = []
    for device in self.devices:
      device_records.append({
          'device_id': device.device_id,
          'namespace': device.namespace,
          'code': device.code,
          'zone_id': device.zone_id,
          'device_type': building_pb2.DeviceInfo.DeviceType.Name(device.device_type),  # pylint: disable=line-too-long
          'observable_fields': sorted(list(device.observable_fields.keys())),
          'action_fields': sorted(list(device.action_fields.keys())),
          'observable_field_types': {
              k: building_pb2.DeviceInfo.ValueType.Name(v)
              for k, v in device.observable_fields.items()
          },
          'action_field_types': {
              k: building_pb2.DeviceInfo.ValueType.Name(v)
              for k, v in device.action_fields.items()
          },
      })
    return pd.DataFrame(device_records)

  @property
  def zones(self) -> Sequence[building_pb2.ZoneInfo]:
    """Sequence of thermal zones in the building managed by the RL agent."""
    return self._zones

  @property
  def zones_df(self) -> pd.DataFrame:
    """Lists the building's zones in dataframe format."""
    zone_records = []
    for zone in self.zones:
      zone_records.append({
          'zone_id': zone.zone_id,
          'building_id': zone.building_id,
          'zone_description': zone.zone_description,
          'area': zone.area,
          'devices': list(zone.devices),
          'zone_type': building_pb2.ZoneInfo.ZoneType.Name(zone.zone_type),
          'floor': zone.floor,
      })
    return pd.DataFrame(zone_records)

  @property
  @abc.abstractmethod
  def current_timestamp(self) -> pd.Timestamp:
    """Lists the current local time of the building."""

  @abc.abstractmethod
  def render(self, path: str) -> None:
    """Renders the current state of the building."""

  @abc.abstractmethod
  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""

  @property
  @abc.abstractmethod
  def num_occupants(self) -> int:
    """Returns the number of occupants in building."""

  @property
  @abc.abstractmethod
  def time_step_sec(self) -> float:
    """Returns the amount of time between time steps."""

  @property
  def json_metadata(self) -> SerializableData:
    """Returns a JSON-serializable dictionary of metadata about the building."""
    return {
        'n_devices': len(self.devices),
        'n_zones': len(self.zones),
        'device_ids': self.devices_df['device_id'].tolist(),
        'zone_ids': self.zones_df['zone_id'].tolist(),
    }
