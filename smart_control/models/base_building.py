"""Base class that extends functionality outside of the building.

    The base class should be extended by the simulation and actual buildings.

    Copyright 2022 Google LLC

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
from typing import Sequence
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2


class BaseBuilding(metaclass=abc.ABCMeta):
  """Base class for a controllable building for reinforcement learning."""

  @property
  @abc.abstractmethod
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""

  @abc.abstractmethod
  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state."""

  @abc.abstractmethod
  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_time: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries the building for observations between start and end times."""

  @abc.abstractmethod
  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    """Issues a command to the building to change one or more setpoints."""

  @abc.abstractmethod
  def wait_time(self) -> None:
    """Returns after a certain amount of time."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building, throwing an RuntimeError if this is impossible."""

  @property
  @abc.abstractmethod
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Lists the devices that can be queried and/or controlled."""

  @property
  @abc.abstractmethod
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Lists the zones in the building managed by the RL agent."""

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
