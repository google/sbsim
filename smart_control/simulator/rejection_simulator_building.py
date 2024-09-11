"""A simulator building that initially throws RPC exceptions before start.

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

from typing import Sequence

import gin
import pandas as pd
from smart_control.models.base_building import BaseBuilding
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2


_ValueType = smart_control_building_pb2.DeviceInfo.ValueType
_ActionResponseType = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)


@gin.configurable
class RejectionSimulatorBuilding(BaseBuilding):
  """A Building that throws exception while agent is awaiting authorization."""

  def __init__(
      self, base_building: BaseBuilding, initial_rejection_count: int = 0
  ):
    """Initializes a RejectionSimulatorBuilding.

    Takes on a BaseBuilding rather than extending it so other types of
    effects (like sensor errors, etc.) can be chained together.

    Args:
      base_building: A BaseBuilding implementation.
      initial_rejection_count: Number of action rejections before start.
    """
    self._base_building = base_building
    self._action_attempt_count = 0
    self._initial_rejection_count = initial_rejection_count
    self.reset()

  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    if self._action_attempt_count < self._initial_rejection_count:
      self._action_attempt_count += 1
      raise RuntimeError('PhysicalAssetService.WriteFieldValues')
    else:
      return self._base_building.request_action(action_request)

  def reset(self) -> None:
    """Resets the building, throwing a RuntimeError if this is impossible."""
    self._base_building.reset()
    self._action_attempt_count = 0

  @property
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    return self._base_building.reward_info

  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state."""
    return self._base_building.request_observations(observation_request)

  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_time: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries the building for observations between start and end times."""
    return self._base_building.request_observations_within_time_interval(
        observation_request, start_timestamp, end_time
    )

  def wait_time(self) -> None:
    """Returns after a certain amount of time."""
    self._base_building.wait_time()

  @property
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    return self._base_building.devices

  @property
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Lists the zones in the building managed by the RL agent."""
    return self._base_building.zones

  @property
  def current_timestamp(self) -> pd.Timestamp:
    """Lists the current local time of the building."""
    return self._base_building.current_timestamp

  def render(self, path: str) -> None:
    """Renders the current state of the building."""
    self._base_building.render(path)

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""
    return self._base_building.is_comfort_mode(current_time)

  @property
  def num_occupants(self) -> int:
    """Returns the number of occupants in building."""
    return self._base_building.num_occupants

  @property
  def time_step_sec(self) -> float:
    """Returns the amount of time between time steps."""
    return self._base_building.time_step_sec
