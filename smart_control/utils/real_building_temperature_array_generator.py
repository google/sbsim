"""Utility to go from list of VAV temperatues, to a teperature array.

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

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils as utils

Room = Sequence[tuple[int, int]]


class RealBuildingTemperatureArrayGenerator:
  """Generate temperature array from VAV temperatures.

  Attributes:
    building_layout: array of where walls are
    device_layout_map: a mapping of device names to rooms
    device_map: a mapping of device ID to device code
  """

  def __init__(
      self,
      building_layout: np.ndarray,
      device_layout_map: Mapping[str, Room],
      device_map: Mapping[str, str],
  ):
    """Constructs temperature array generator based on specifics of the building.

    Args:
      building_layout: 2d array of where walls are
      device_layout_map: a mapping of device names to rooms
      device_map: a mapping of device ID to device code
    """
    self._building_layout = building_layout
    self._device_layout_map = device_layout_map
    self._device_map = device_map

  def get_temperature_array(
      self, response: smart_control_building_pb2.ObservationResponse
  ) -> tuple[np.ndarray, pd.Timestamp]:
    """Returns a tuple of temperature array, in Kelvin, and a corresponding timestamp.

    Args:
      response: an observation response
    """
    timestamp = utils.proto_to_pandas_timestamp(response.timestamp)
    array = np.zeros(self._building_layout.shape)
    for single_response in response.single_observation_responses:
      device_id = single_response.single_observation_request.device_id
      if (
          single_response.single_observation_request.measurement_name
          != "zone_air_temperature_sensor"
      ):
        continue
      if device_id not in self._device_map:
        continue
      device_name = self._device_map[device_id]
      if device_name not in self._device_layout_map:
        continue

      temp_kelvin = utils.fahrenheit_to_kelvin(single_response.continuous_value)
      for cv in self._device_layout_map[device_name]:
        array[cv[0]][cv[1]] = temp_kelvin

    return array, timestamp
