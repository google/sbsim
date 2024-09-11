"""Utilities to log and then visualize building.

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

from typing import Optional

import numpy as np
import pandas as pd
from smart_control.utils import building_renderer


class VisualLogger:
  """Logs building temperatures and renders the building.

  Attributes:
    _temperature_arrays: stores list of temperature arrays, to be used in
      get_video().
    _timestamps: a list of timestamps to print onto images
  """

  def __init__(self, renderer: building_renderer.BuildingRenderer):
    """Initializes the forground image, ie the walls of the building.

    Args:
      renderer: a building renderer
    """

    self._renderer = renderer
    self._temperature_arrays = []

    self._timestamps = []

  def log(
      self,
      temperature_array: np.ndarray,
      timestamp: Optional[pd.Timestamp] = None,
  ) -> None:
    """Aggregates an array of temperature values, to be rendered by get_video.

    Args:
      temperature_array: an array representing CV's temperatures
      timestamp: a timestamp associated with the temperatures
    """
    if temperature_array.shape != self._renderer.get_building_dimensions():
      raise ValueError(
          'building layout and building temperatures do not have the same'
          ' dimensions.'
      )
    self._temperature_arrays.append(temperature_array.copy())
    self._timestamps.append(timestamp)

  def get_video(
      self,
      file_path: str,
      fps: int,
      vmin: int = 280,
      vmax: int = 300,
      cmap: str = 'rainbow',
      alpha: float = 1.0,
      wall_color: str = 'black',
      grid: bool = True,
  ) -> None:
    """Creates a video of all the logged temperatures.

    Args:
      file_path: path of video to be saved. Must end in .mp4
      fps: frames per second
      vmin: minimum value to be used when creating the heatmap
      vmax: maximum value to be used when creating the heatmap
      cmap: color map to use
      alpha: opacity of walls. 1 means full dark, 0 means fully invisible
      wall_color: color of the walls
      grid: boolean flag. If False, walls are solid, if True, a grid pattern
    """
    self._renderer.get_video(
        file_path=file_path,
        temperature_arrays=self._temperature_arrays,
        fps=fps,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        alpha=alpha,
        wall_color=wall_color,
        grid=grid,
        timestamps=self._timestamps,
    )
