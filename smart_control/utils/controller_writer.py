"""Utilities to write smart control protos to endpoint.

Copyright 2024 Google LLC

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

import csv
import os
from typing import Mapping, Sequence

from absl import logging
import gin
from google.protobuf import message
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import constants
from smart_control.utils import writer_lib


@gin.configurable
class ProtoWriter(writer_lib.BaseWriter):
  """Implementation for writing building and reward protos to disk.

  Writes Smart Control protos as hourly shards as a serialized file.
  Each type of message uses a different file prefix (e.g., action_response)
  to identify the type of proto. Each shard is identified with a serial
  based on the timestamp. For example, a file of ActionResponses written
  on the 4th hour (UTC) of 5/25, would be action_response_2021.05.25.04.

  Attributes:
    output_dir: destination directory
  """

  def __init__(self, output_dir: str):
    self._output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writer lib output directory %s', self._output_dir)

  def write_observation_response(
      self,
      observation_response: smart_control_building_pb2.ObservationResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the observation response obtained from the environment."""
    serial = self._get_serial(timestamp)
    filepath = self._get_file_path(
        self._output_dir, constants.OBSERVATION_RESPONSE_FILE_PREFIX, serial
    )
    self._write_msg_to_disk(observation_response, filepath)

  def write_building_image(
      self, base64_img: bytes, timestamp: pd.Timestamp
  ) -> None:
    """Writes the rendered building image obtained from the environment."""
    filepath = os.path.join(self._output_dir, constants.BUILDING_IMAGE_CSV_FILE)
    with open(filepath, 'a') as csv_file:
      csv.writer(csv_file).writerow([timestamp.timestamp(), base64_img])

  def write_action_response(
      self,
      action_response: smart_control_building_pb2.ActionResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the action response obtained from the environment."""
    serial = self._get_serial(timestamp)
    filepath = self._get_file_path(
        self._output_dir, constants.ACTION_RESPONSE_FILE_PREFIX, serial
    )
    self._write_msg_to_disk(action_response, filepath)

  def write_reward_info(
      self,
      reward_info: smart_control_reward_pb2.RewardInfo,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the reward info obtained from the environment."""
    serial = self._get_serial(timestamp)
    filepath = self._get_file_path(
        self._output_dir, constants.REWARD_INFO_PREFIX, serial
    )
    self._write_msg_to_disk(reward_info, filepath)

  def write_reward_response(
      self,
      reward_response: smart_control_reward_pb2.RewardResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes the reward response from the reward function."""
    serial = self._get_serial(timestamp)
    filepath = self._get_file_path(
        self._output_dir, constants.REWARD_RESPONSE_PREFIX, serial
    )
    self._write_msg_to_disk(reward_response, filepath)

  def _get_serial(self, timestamp: pd.Timestamp):
    return timestamp.strftime('%Y.%m.%d.%H')

  def _get_file_path(self, output_dir: str, file_prefix: str, serial: str):
    return os.path.join(output_dir, '%s_%s' % (file_prefix, serial))

  def _write_msg_to_disk(self, proto: message.Message, filepath: str):
    """Creates or appends a binary file with the proto."""

    if os.path.exists(filepath):
      mode = 'ab'
    else:
      mode = 'wb'

    try:
      with open(filepath, mode) as output_file:
        size = proto.ByteSize()
        output_file.write(size.to_bytes(4, 'little'))
        output_file.write(proto.SerializeToString())

    except IOError:
      logging.exception(
          'IOException encountered. Failed to write proto to %s', filepath
      )

  def write_normalization_info(
      self,
      normalization_info: Mapping[
          writer_lib.VariableId,
          smart_control_normalization_pb2.ContinuousVariableInfo,
      ],
  ) -> None:
    """Writes variable normalization info to disk."""
    filepath = os.path.join(self._output_dir, constants.NORMALIZATION_FILENAME)
    with open(filepath, 'wb') as output_file:
      for variable in normalization_info.values():
        size = variable.ByteSize()
        output_file.write(size.to_bytes(4, 'little'))
        output_file.write(variable.SerializeToString())

  def write_device_infos(
      self, device_infos: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> None:
    """Writes the device infos to disk."""
    filepath = os.path.join(self._output_dir, constants.DEVICE_INFO_PREFIX)
    if os.path.exists(filepath):
      logging.info('Deleting an exiting DeviceInfo file.')
      os.remove(filepath)
    for device_info in device_infos:
      self._write_msg_to_disk(device_info, filepath)

  def write_zone_infos(
      self, zone_infos: Sequence[smart_control_building_pb2.ZoneInfo]
  ) -> None:
    """Writes the zone infos to disk."""
    filepath = os.path.join(self._output_dir, constants.ZONE_INFO_PREFIX)
    if os.path.exists(filepath):
      logging.info('Deleting an exiting ZoneInfo file.')
      os.remove(filepath)
    for zone_info in zone_infos:
      self._write_msg_to_disk(zone_info, filepath)


@gin.configurable
class ProtoWriterFactory(writer_lib.BaseWriterFactory):
  """Factory for proto writers."""

  def create(self, output_dir: writer_lib.PathLocation) -> ProtoWriter:
    """Creates a writer with an output directory."""
    return ProtoWriter(output_dir)
