"""Utilities to write smart control protos to endpoint.

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
import os
import sys
from typing import Mapping, NewType, Sequence, TypeAlias

import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2

if sys.version_info >= (3, 11):
  from importlib.resources.abc import Traversable  # pylint: disable=g-import-not-at-top
else:
  from importlib_resources.abc import Traversable  # pylint: disable=g-import-not-at-top

PathLocation: TypeAlias = Traversable | os.PathLike[str] | str

VariableId = NewType('VariableId', str)


class BaseWriter(metaclass=abc.ABCMeta):
  """Abstract base class for writing the building and reward protos."""

  @abc.abstractmethod
  def write_observation_response(
      self,
      observation_response: smart_control_building_pb2.ObservationResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the observation response obtained from the environment."""

  @abc.abstractmethod
  def write_building_image(
      self, base64_img: bytes, timestamp: pd.Timestamp
  ) -> None:
    """Writes the rendered building image obtained from the environment."""

  @abc.abstractmethod
  def write_action_response(
      self,
      action_response: smart_control_building_pb2.ActionResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the action response obtained from the environment."""

  @abc.abstractmethod
  def write_reward_info(
      self,
      reward_info: smart_control_reward_pb2.RewardInfo,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the reward info obtained from the environment."""

  @abc.abstractmethod
  def write_reward_response(
      self,
      reward_response: smart_control_reward_pb2.RewardResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the reward response from the reward function."""

  @abc.abstractmethod
  def write_normalization_info(
      self,
      normalization_info: Mapping[
          VariableId, smart_control_normalization_pb2.ContinuousVariableInfo
      ],
  ) -> None:
    """Writes variable normalization info to RecordIO."""

  @abc.abstractmethod
  def write_device_infos(
      self, device_infos: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> None:
    """Writes the device infos to endpoint."""

  @abc.abstractmethod
  def write_zone_infos(
      self, zone_infos: Sequence[smart_control_building_pb2.ZoneInfo]
  ) -> None:
    """Writes the zone infos to endpoint."""


class BaseWriterFactory(metaclass=abc.ABCMeta):
  """Abstract base class for creating a writer."""

  @abc.abstractmethod
  def create(self, output_dir: PathLocation) -> BaseWriter:
    """Creates a writer with a output directory."""
