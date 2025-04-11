"""Utilities to visualize temperature changes in a building.

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

import copy
import functools
import io
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
import PIL
from PIL import ImageDraw
import seaborn as sn
from smart_control.simulator import building_utils
from smart_control.simulator import constants


class BuildingRenderer:
  """Renders the building.

  Attributes:
    cv_size: size of how large a control volume should be rendered, in pixels.
  """

  def __init__(
      self, building_layout: building_utils.FileInputFloorPlan, cv_size: int = 6
  ):
    """Initializes the foreground image, ie the walls of the building.

    Args:
      building_layout: an array representing where the walls are in the building
        where 0 is interior space, 1 is wall and 2 is exterior space
      cv_size: how many pixels to make each CV
    """

    building_layout = building_layout.copy()
    self._building_height, self._building_width = building_layout.shape
    self.cv_size = cv_size
    # set exterior space air from 2 to 0
    building_layout[
        building_layout == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
    ] = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    self._mask = PIL.Image.fromarray(
        building_layout == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
    )
    self._mask = self._mask.resize(
        (
            self._building_width * self.cv_size,
            self._building_height * self.cv_size,
        ),
        PIL.Image.Resampling.LANCZOS,
    )

  @functools.cached_property
  def _grid_mask(self) -> PIL.Image.Image:
    """Returns an image of a grid to use as mask to project walls."""
    height = self._building_height * self.cv_size
    width = self._building_width * self.cv_size

    white = 255
    grey = 128

    image = PIL.Image.new(mode='L', size=(width, height), color=white)
    draw = ImageDraw.Draw(image)
    y_end = image.height
    x_end = image.width

    for x in range(0, image.width + self.cv_size, self.cv_size):
      line = ((x, 0), (x, y_end))
      draw.line(line, fill=grey)
      line = ((x - 1, 0), (x - 1, y_end))
      draw.line(line, fill=grey)

    for y in range(0, image.height + self.cv_size, self.cv_size):
      line = ((0, y), (x_end, y))
      draw.line(line, fill=grey)
      line = ((0, y - 1), (x_end, y - 1))
      draw.line(line, fill=grey)
    del draw

    # make binary
    image = PIL.ImageOps.grayscale(image)
    image = np.array(image)
    image[image == grey] = 1
    image[image == white] = 0
    image = np.logical_and(
        image.astype(bool), np.array(self._mask).astype(bool)
    )
    image = PIL.Image.fromarray(image)
    return image

  def render(
      self,
      temperature_array: np.ndarray,
      vmin: int = 280,
      vmax: int = 300,
      cmap: str = 'rainbow',
      alpha: float = 1.0,
      wall_color: str = 'black',
      grid: bool = True,
      ts: Optional[pd.Timestamp] = None,
      input_q: Optional[np.ndarray] = None,
      diff_range: float = 0.5,
      diff_size: int = 1,
      colorbar: bool = False,
      clip_range: int = 6,
      center: int = 21,
  ) -> PIL.Image.Image:
    """Returns a rendering of the provided temperature values.

    Args:
      temperature_array: an array representing CV temperatures
      vmin: vmin value to be used when creating the heatmap
      vmax: vmax value to be used when creating the heatmap
      cmap: color map to use for heatmap
      alpha: opacity of walls. 1 means full dark, 0 means fully invisible
      wall_color: color of the walls
      grid: boolean flag. If False, walls are solid, if True, a grid pattern
      ts: optional  timestamp to render
      input_q: optional array of diffuser values in J/s
      diff_range: optional int range of diffuser values to display, in J/s
      diff_size: optional int size to render diffusers
      colorbar: optional boolean flag to render colorbar
      clip_range: optional int range of colorbars
      center: optional int center of colorbar
    """
    if temperature_array.shape != (self._building_height, self._building_width):
      raise ValueError(
          'building layout and building temperatures do not have the same'
          ' dimensions.'
      )
    buffer = io.BytesIO()
    plt.imsave(
        buffer, temperature_array, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax
    )
    buffer.seek(0)
    background = PIL.Image.open(buffer)
    background = background.resize(
        (
            self._building_width * self.cv_size,
            self._building_height * self.cv_size,
        ),
        PIL.Image.Resampling.LANCZOS,
    )

    foreground = PIL.Image.new(
        'RGB',
        (
            self._building_width * self.cv_size,
            self._building_height * self.cv_size,
        ),
        color=wall_color,
    )

    mask = self._grid_mask if grid else self._mask
    original_background = background.copy()
    background.paste(foreground, (0, 0), mask)
    background = PIL.Image.blend(original_background, background, alpha)

    # add diffusers to image
    if input_q is not None:

      def enlarge(arr, size):
        arr2 = copy.deepcopy(arr)

        for _ in range(size - 1):
          for i in range(1, arr.shape[0] - 1):
            for j in range(1, arr.shape[1] - 1):
              for off in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if (
                    arr[i + off[0]][j + off[1]] < -diff_range
                    or arr[i + off[0]][j + off[1]] > diff_range
                ):
                  arr2[i][j] = arr[i + off[0]][j + off[1]]

          arr = arr2
          arr2 = copy.deepcopy(arr)

        return arr2

      input_q = enlarge(input_q, diff_size)

      buffer_q = io.BytesIO()
      plt.imsave(
          buffer_q,
          input_q,
          cmap=plt.get_cmap(cmap),
          vmin=-diff_range,
          vmax=diff_range,
      )
      buffer_q.seek(0)
      background_q = PIL.Image.open(buffer_q)
      background_q = background_q.resize(
          (
              self._building_width * self.cv_size,
              self._building_height * self.cv_size,
          ),
          PIL.Image.Resampling.LANCZOS,
      )
      q_mask = PIL.Image.fromarray(
          ~((input_q < -diff_range) | (input_q > diff_range))
      )
      background_q.paste(background, (0, 0), q_mask)
      background = background_q
    if ts is not None:
      draw = ImageDraw.Draw(background)
      draw.text((2, 2), ts.strftime('%Y-%m-%d %X'), fill=(0, 0, 0))

    # add colorbar to image
    if colorbar:

      def add_colorbar(im, clip_range, center, cmap):
        max_bar = center + clip_range
        min_bar = center - clip_range
        diff = np.zeros((744, 1004))
        diff[0][0] = min_bar
        diff[0][1] = max_bar
        diff = np.clip(diff, min_bar, max_bar)
        plt.figure(figsize=(16, 12))
        sn.heatmap(
            data=diff, cmap=cmap, xticklabels=False, yticklabels=False
        )
        plt.savefig('colorbar.png')
        plt.close()
        bar = PIL.Image.open('colorbar.png')
        shape = np.array(bar).shape
        bar = bar.crop((shape[1] - 270, 60, shape[1] - 180, shape[0] - 60))
        bar = bar.resize((bar.size[0], im.size[1]))
        right_img = np.array(bar.convert('RGB'))
        left_img = np.array(im.convert('RGB'))
        c = np.concatenate([left_img, right_img], axis=1)
        combined_im = PIL.Image.fromarray(c)

        return combined_im

      background = add_colorbar(
          background, clip_range=clip_range, center=center, cmap=cmap
      )

    return background

  def get_video(
      self,
      file_path: str,
      temperature_arrays: List[np.ndarray],
      fps: int,
      vmin: int = 280,
      vmax: int = 300,
      cmap: str = 'rainbow',
      alpha: float = 1.0,
      wall_color: str = 'black',
      grid: bool = True,
      timestamps: Optional[List[pd.Timestamp]] = None,
  ) -> None:
    """Creates a video of all the logged temperatures.

    Args:
      file_path: path of video to be saved. Must end in .mp4
      temperature_arrays: a list of np arrays representing CV temperatures
      fps: frames per second
      vmin: minimum value to be used when creating the heatmap
      vmax: maximum value to be used when creating the heatmap
      cmap: color map to use
      alpha: opacity of walls. 1 means full dark, 0 means fully invisible
      wall_color: color of the walls
      grid: boolean flag. If False, walls are solid, if True, a grid pattern
      timestamps: optional list of timestamps to render
    """
    with media.VideoWriter(
        file_path, shape=(self._mask.size[1], self._mask.size[0]), fps=fps
    ) as w:
      for idx, temperature_array in enumerate(temperature_arrays):
        ts = timestamps[idx] if timestamps else None
        frame = self.render(
            temperature_array, vmin, vmax, cmap, alpha, wall_color, grid, ts
        )
        w.add_image(np.array(frame)[:, :, :3])

  def get_building_dimensions(self) -> Tuple[int, int]:
    """Returns dimensions of the building."""
    return self._building_height, self._building_width
