"""Code for generating thermal diffusers in a building.

These helper functions are separated these out into their own file for
extensibility: we can easily put in another function loading these from data and
process this using similar function format.

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

import math
from typing import Collection, Dict, Optional, Tuple, Union
import warnings

from absl import logging
import cv2
import numpy as np
from smart_control.simulator import building_utils


Coordinates2D = Union[Tuple[int, int], np.ndarray]
RoomIndicesDict = Dict[str, Collection[Coordinates2D]]


def _evenly_spaced_inds_from_domain(
    start: int, end: int, spacing: int = 10
) -> Collection[int]:
  """Returns evenly-spaced indices spanning the domain covered by ind_list.

  Given a spacing, return inds that are less than or equal to that spacing
  apart from both themselves and their nearest walls.

  Args:
    start: the min value of an int valued domain
    end: the max value of an int valued domain
    spacing: how much distance there should be before another ind is added

  Returns:
    a list with the indices approximately spaced
  """

  ind_len = end - start

  if ind_len == 0:
    return [start]

  n_diffusers = np.max((1, np.round(ind_len / spacing)))

  # Our goal is to determine even spreading of spacers without one starting
  # exactly where the wall starts. np.arange() returns an evenly spaced array
  # with the starting index being the "start" entry. Since that is our wall, we
  # prefer to ask for 1 more than the number of spaces we actually want, and
  # then filter out the first index that arange returns.
  temp_placement = np.arange(start, end, ind_len / (n_diffusers + 1))
  temp_placement = temp_placement[1:]
  placement_inds = [int(math.ceil(i)) for i in temp_placement]

  return placement_inds


def _rectangularity_test(
    room_cv_indices: Collection[Coordinates2D], threshold: float = 0.5
) -> bool:
  """Tests how rectangular a room is.

  Tests to see how rectangular the room is by comparing the
  number of CVs in the room to the number of CVs that would be
  in the rectangular supershape made up of the min and max inds.

  Args:
    room_cv_indices: a list of indices making up a room.
    threshold: the proportion of rectangularity the room should make up. Must be
      between 0 and 1. Numbers closer to 1 require more strict rectangles,
      whereas numbers closer to 0 do not.

  Returns:
    a boolean value where True implies the room is
      rectangular.
  """
  room_cv_indices = np.array(room_cv_indices)
  num_cvs = len(room_cv_indices)

  xs = [x for (x, y) in room_cv_indices]
  ys = [y for (x, y) in room_cv_indices]

  start_x, end_x = min(xs), max(xs)
  start_y, end_y = min(ys), max(ys)

  if not end_x - start_x or not end_y - start_y:
    warnings.warn("""You have asked computations to be performed on a
                       nonsensical room; i.e. one that has either 0|1 width or
                       0|1 height.""")

  rectangular_vol = np.max(((end_x - start_x), 1)) * np.max(
      ((end_y - start_y), 1)
  )

  return num_cvs / rectangular_vol > threshold


def _determine_random_inds_for_thermal_diffusers(
    room_cv_indices: Collection[Coordinates2D],
    spacing: int = 10,
    random_seed: int = 23,
) -> Collection[Coordinates2D]:
  """Randomly picks n diffusers among room_cv_indices given.

    'n' is the ceiling of the number of room_cv_indices divided by
    the spacing squared to match the earlier simulation; i.e.,
    if there are 110 CVs and a spacing of 10, we should expect 2
    diffusers.

  Args:
    room_cv_indices: a list of indices making up a room.
    spacing: how many control volumes to put between each diffuser
    random_seed: what seed to input into the function.

  Returns:
    a list of inds to place diffusers.
  """
  rng = np.random.default_rng(random_seed)
  num_cvs = len(room_cv_indices)
  spacing_sq = spacing * spacing
  num_diffusers = int(np.max((1, np.round(num_cvs / spacing_sq))))

  inds = rng.choice(room_cv_indices, num_diffusers, replace=False)

  return np.array(inds)


def _determine_equal_spacing_for_thermal_diffusers(
    room_cv_indices: Collection[Coordinates2D],
    spacing: int = 10,
    buffer_from_walls: int = 3,
) -> Collection[Coordinates2D]:
  """Determines inds of even spacing that are inside the ind list.

  If there are no inds inside the ind list (i.e. if the room is very small),
  then return random inds.

  Args:
    room_cv_indices: a list of indices making up a room.
    spacing: how many control volumes to put between each diffuser.
    buffer_from_walls: how far to place a thermal diffuser away from a wall.

  Returns:
    a list of inds to place diffusers.
  """

  if not room_cv_indices:
    raise ValueError("room_cv_indices missing!")

  room_cv_indices = np.array(room_cv_indices)
  xs = [x for (x, y) in room_cv_indices]
  ys = [y for (x, y) in room_cv_indices]

  start_x, end_x = min(xs), max(xs)
  start_y, end_y = min(ys), max(ys)

  if end_x - start_x > 2 * buffer_from_walls:
    start_x += buffer_from_walls
    end_x -= buffer_from_walls
  else:
    logging.warning("WARNING: thermal_diffusers may be very close to walls.")

  placement_inds_x = set(
      _evenly_spaced_inds_from_domain(start_x, end_x, spacing)
  )
  placement_inds_y = set(
      _evenly_spaced_inds_from_domain(start_y, end_y, spacing)
  )

  inds = [
      ind
      for ind in room_cv_indices
      if ind[0] in placement_inds_x and ind[1] in placement_inds_y
  ]

  return np.array(inds)


def diffuser_allocation_switch(
    room_cv_indices: Collection[Coordinates2D],
    spacing: int = 10,
    interior_walls: Optional[building_utils.InteriorWalls] = None,
    buffer_from_walls: int = 2,
) -> Collection[Coordinates2D]:
  """Switches between random and even assignment of thermal diffusers.

  A more in-depth explanation: here we provide a method for allocating thermal
  diffusers that is general for many types of rooms. At a high level, if a room
  is rectangular enough, we allocate diffusers evenly according to a grid
  determined by the input "spacing", and then only select the points that are
  inside the original set of room indices to ensure we don't place a diffuser
  outside. If a room is too insanely shaped (i.e. imagine a long snaking hallway
  or an entryway to a few private alcoves) we instead try to ensure that the
  appropriate number of diffusers are still placed, but for simplicity we place
  them randomly. This option is almost never taken, as most rooms are deemed
  rectangular enough according to our measure:

    Here, from the number of CVs that define a room, extract the total number of
  CVs (num_CVs), the maximum and minimum x coordinates (x_max, x_min), and the
  maximum and minimum y coordinates (y_max, y_min), and set a threshold.
  If num_CVs / ((x_max - x_min) * (y_max - y_min)) > threshold,
  it is rectangular enough.

  Args:
    room_cv_indices: a list of indices making up a room.
    spacing: how many control volumes to put between each diffuser
    interior_walls: an InteriorWalls for determining whether thermal diffusers
      are allocated in walls. This measure is intended for cases in which one
      needs to provide a interior_walls and zone_map to the Building class, and
      they may not line up correctly on account of being from different photo
      sources.
    buffer_from_walls: how far to place a thermal diffuser away from a wall.

  Returns:
    a list of inds to place diffusers.
  """

  if _rectangularity_test(room_cv_indices, threshold=0.1):
    inds = _determine_equal_spacing_for_thermal_diffusers(
        room_cv_indices, spacing=spacing, buffer_from_walls=buffer_from_walls
    )
  else:
    inds = _determine_random_inds_for_thermal_diffusers(
        room_cv_indices, spacing
    )

  if inds is None:
    inds = _determine_random_inds_for_thermal_diffusers(
        room_cv_indices, spacing
    )

  if interior_walls is not None:
    kernel = np.ones((2, 2), np.uint8)

    ## TODO(spangher): This dilate function throws errors when
    # iterations is greater than 0, which takes away an effort I had
    # started to make sure that thermal diffusers are not placed closed to
    # walls. If data anchoring proceeds with trouble, please consider
    # restarting the effort to place a buffer in between diffusers and walls.

    if buffer_from_walls:
      dilated_interior_walls = cv2.dilate(interior_walls, kernel, iterations=0)
    else:
      dilated_interior_walls = interior_walls

    inds = [ind for ind in inds if dilated_interior_walls[ind[0]][ind[1]] == 0]

  return inds
