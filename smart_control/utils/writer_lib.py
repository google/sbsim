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
import json
import os
import sys
from typing import Any, IO, Mapping, NewType, Sequence, TypeAlias

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_normalization_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2

if sys.version_info >= (3, 11):
  from importlib.resources.abc import Traversable  # pylint: disable=g-import-not-at-top, g-importing-member
else:
  from importlib_resources.abc import Traversable  # pylint: disable=g-import-not-at-top, g-importing-member

PathLocation: TypeAlias = Traversable | os.PathLike[str] | str

SerializableData: TypeAlias = dict[str, Any]

VariableId = NewType('VariableId', str)


class BaseWriter(metaclass=abc.ABCMeta):
  """Abstract base class for writing the building and reward protos."""

  @property
  @abc.abstractmethod
  def output_dir(self) -> PathLocation:
    """The output directory for the writer."""

  # PROTO WRITING METHODS

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

  # GENERIC FILE WRITING METHODS

  def _open(self, filepath: PathLocation, mode: str) -> IO[Any]:
    """Opens a file for reading. Can be overridden in corp codebase."""
    if 'b' in mode:
      return open(filepath, mode)
    return open(filepath, mode, encoding='utf-8')

  def _write_content(self, content: Any, filepath: PathLocation) -> None:
    """Writes content to a file in the output directory."""
    with self._open(filepath, 'w') as f:
      f.write(content)

  def write_txt(self, text: str, filename: str) -> None:
    """Writes a string to a text file in the output directory."""
    filepath = os.path.join(self.output_dir, filename)
    self._write_content(text, filepath)

  def write_json(self, data: SerializableData, filename: str) -> None:
    """Writes a dictionary as a JSON file in the output directory."""
    filepath = os.path.join(self.output_dir, filename)
    content_json = json.dumps(data, indent=2)
    self._write_content(content_json, filepath)

  def write_csv(
      self, df: pd.DataFrame, filename: str, index: bool = False
  ) -> None:
    """Writes a Pandas DataFrame as a CSV file in the output directory."""
    filepath = os.path.join(self.output_dir, filename)
    content_csv = df.to_csv(index=index)
    self._write_content(content_csv, filepath)

  def write_plot_html(self, fig: go.Figure, filename: str) -> None:
    """Writes a Plotly figure as an HTML file in the output directory."""
    filepath = os.path.join(self.output_dir, filename)
    content_html = pio.to_html(fig, full_html=True)
    self._write_content(content_html, filepath)


class BaseWriterFactory(metaclass=abc.ABCMeta):
  """Abstract base class for creating a writer."""

  @abc.abstractmethod
  def create(self, output_dir: PathLocation) -> BaseWriter:
    """Creates a writer with a output directory."""
