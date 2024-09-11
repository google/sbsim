"""Utilities to visualize temperatures in a building.

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

import base64
from collections.abc import Sequence
import io
import json
import os
import pathlib
import sys
from typing import TypeAlias

from absl import logging
import gin
import numpy as np
from PIL import Image
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import building_renderer
from smart_control.utils import real_building_temperature_array_generator as temp_array_gen

if sys.version_info >= (3, 11):
  from importlib.resources.abc import Traversable  # pylint: disable=g-import-not-at-top
else:
  from importlib_resources.abc import Traversable  # pylint: disable=g-import-not-at-top

PathLocation: TypeAlias = Traversable | os.PathLike[str] | str


def _make_traversable(path_location: str | os.PathLike[str]) -> Traversable:
  if isinstance(path_location, Traversable):
    return path_location
  else:
    return pathlib.Path(path_location)


@gin.configurable
class BuildingImageGenerator:
  """Generates a base64 encoding of a building image from an observation.

  Attributes:
    device_layout_path: path of a JSON file containing layout information that
      can be used to create a map from device names to rooms
    floor_plan_path: path of a numpy array file that represents where the walls
      are in the building where 0 is interior space, 1 is wall, and 2 is
      exterior space
    device_infos: sequence of DeviceInfos used to create a map from device id to
      device code
    cv_size: how large a control volume should be rendered, in pixels
  """

  def __init__(
      self,
      device_layout_path: PathLocation,
      floor_plan_path: PathLocation,
      device_infos: Sequence[smart_control_building_pb2.DeviceInfo],
      cv_size: int,
  ):
    self._device_layout_path = _make_traversable(device_layout_path)
    self._floor_plan_path = _make_traversable(floor_plan_path)
    self._device_infos = device_infos
    self._cv_size = cv_size

  def generate_building_image(
      self, observation_response: smart_control_building_pb2.ObservationResponse
  ) -> bytes:
    """Returns a base64 encoded building image given an observation response."""
    device_map = {}
    for device_info in self._device_infos:
      device_map[device_info.device_id] = device_info.code

    with self._device_layout_path.open("rt") as f:  # pytype: disable=wrong-arg-types
      room_dict_real = json.load(f)

    with self._floor_plan_path.open("rb") as fp:
      floor_plan = np.load(fp)

    keys_not_found = set()
    device_layout_map = {}
    for key, room in room_dict_real.items():
      if not key:
        continue
      found = False
      for device_code in device_map.values():
        # adding the space to the comparison ensures we do not confuse things
        # like room 1-2-1 and 1-2-10
        if str(key) + " " in str(device_code) + " ":
          device_layout_map[device_code] = room
          found = True
      if not found:
        keys_not_found.add(key)
    if keys_not_found:
      logging.warning(
          "The following keys in the room mapping were missing from the device"
          " map (device id -> code): %s",
          keys_not_found,
      )

    renderer = building_renderer.BuildingRenderer(floor_plan, self._cv_size)
    array_gen = temp_array_gen.RealBuildingTemperatureArrayGenerator(
        floor_plan, device_layout_map, device_map
    )
    img = renderer.render(
        array_gen.get_temperature_array(observation_response)[0]
    )
    return self.image_to_png_base64(img)

  def image_to_png_base64(self, image: Image.Image) -> bytes:
    """Converts a PIL Image to a PNG base64 encoding."""
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue())
