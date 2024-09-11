"""Utils for computing the physical and thermal characteristics of buildings.

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

import collections
import datetime
import pathlib
from typing import Any, NewType, Tuple, Union
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from smart_control.simulator import constants


Coordinates2D = Tuple[int, int]
RoomIndicesDict = collections.defaultdict[str, Any]
"""Note: The following four types all describe various aspects of the same

  floorplan; i.e. all four types have the same dimensionality and encode similar
  variations of the same information.
  Here I will describe the differences between the types:
  (1) FileInputFloorPlan: the floorplan format as entered into the
    program. This specifically notes exterior space as 2's, walls as 1's, and
    inside space as 0's. This file type may have various "glitches" that the
    program will correct, such as walls that abut directly the frame of the
    image.
  (2) ConnectionReadyFloorPlan: the floorplan format as processed and ready for
    the opencv function connectedComponents . This format requires a very
    specific input type, namely 1's where there is open space and 0's everywhere
    else. Therefore, prior to this step, various pieces of information must be
    removed from the floor plan, such as the location of exterior space, and
    stored in other datatypes.
  (3) Connections: the floorplan format after opencv's connectedComponents is
    run on it. The Connections floorplan has each connected component, defined
    as the contiguous space within each component of interest (room, zone, etc.)
    labelled with an increasing integer indexing the component. It is an
    np.ndarray where 0's are walls and connected contiguous space are increasing
    integers.
  (4) ExteriorSpace: the floorplan format serves an auxiliary store of
    information so that the original floorplan can be stripped of a specific
    designation for exterior space in order to be prepared for ingestion into
    connectedComponents. It is an np.ndarray where 0's are walls or interior
    space and exterior space is marked with -1.
  (5) ExteriorWalls: a floorplan format that notes where exterior walls are with
    1's and everywhere else with 0's.
  (6) InteriorWalls: a floorplan format that notes where interior walls are
    with -3 and everywhere else with 0's.
"""

FileInputFloorPlan = NewType("FileInputFloorPlan", np.ndarray)
ConnectionReadyFloorPlan = NewType("ConnectionReadyFloorPlan", np.ndarray)
Connections = NewType("Connections", np.ndarray)
ExteriorSpace = NewType("ExteriorSpace", np.ndarray)
ExteriorWalls = NewType("ExteriorWalls", np.ndarray)
InteriorWalls = NewType("InteriorWalls", np.ndarray)


def read_floor_plan_from_filepath(
    filepath: str,
    save_debugging_image: bool = False,
) -> FileInputFloorPlan:
  """Reads a file from a disk (including CNS) and returns it.

  Args:
    filepath: name of the location on CNS. CSV and NPY files are supported, and
      they are both determined by the semantic naming of the path, i.e. the file
      type suffix.
    save_debugging_image: boolean for whether we should save the debugging image
      to cns.

  Returns:
    a FileInputFloorPlan
  """

  # function to return the file extension
  t = pathlib.Path(filepath).suffix

  filepath = pathlib.Path(filepath)
  with filepath.open(mode="rb") as fp:
    if t == ".csv":
      floor_plan = np.loadtxt(fp, delimiter=",")
    elif t == ".npy":
      floor_plan = np.load(fp, allow_pickle=True)
    else:
      raise ValueError("please provide the data in csv or npy.")

  floor_plan = np.asarray(floor_plan)

  if save_debugging_image:
    save_images_to_cns_for_debugging(
        FileInputFloorPlan(floor_plan), "file_from_input"
    )

  return FileInputFloorPlan(floor_plan)


def save_images_to_cns_for_debugging(
    floor_plan: Union[
        FileInputFloorPlan,
        Connections,
        ExteriorWalls,
        InteriorWalls,
        np.ndarray,
        ConnectionReadyFloorPlan,
    ],
    path_ending: str,
    path_to_simulator_cns: str = "/cns/oi-d/home/smart_buildings/control/configs/simulation/",
) -> None:
  """Saves a .png of a floorplan array to CNS for visual debugging.

  Args:
    floor_plan: one of the floor_plan types
    path_ending: a path suffix to end saved files
    path_to_simulator_cns: base path to save the files on CNS.
  """
  full_path = (
      path_to_simulator_cns
      + "floorplan_construction_debugging_images/"
      + path_ending
      + str(datetime.datetime.now().strftime("%Y%m%d"))
  )
  plt.imshow(floor_plan)
  full_path = pathlib.Path(full_path)
  with full_path.open(mode="wb") as fp:
    plt.savefig(fp)


def guarantee_air_padding_in_frame(
    floor_plan: FileInputFloorPlan,
) -> FileInputFloorPlan:
  """Adds a row or column of air if a building is abuts its frame edge.

  Future computation relies on buildings being surrounded by at least one
    layer of air CVs between them and the edge of the floor plan frame.
    However, due to human variation in preparing floor plans for transformation
    into arrays, we may have the case where a floor plan is passed in that has
    wall CVs at the edge of the frame. Thus, this function should check that
    that is the case and will add a layer of exterior space CVs if building lies
    against the array's edge.

  A brief note on the helper functions below: If walls are touching more than
    one edge of the floor plan, the function should be able to compute this --
    but if it adds rows of CVs and then does not update the shape
    of the floor plan, it will fail. Thus, we package the dimension
    tracking and rows to concatenate in a convenient helper function so
    it can be called multiple times without repetitive code.

  Args:
    floor_plan: a FileInputFloorPlan

  Returns:
    an FileInputFloorPlan that has 2's padded along whichever array edge was
      missing them.
  """

  # handle the case of a floor_plan that is trivial in its dimensions (i.e. has
  # one dimension of 1 or 0). This will ensure that each is at least a few
  # CVs wide:
  if 1 in floor_plan.shape or 0 in floor_plan.shape:
    raise ValueError("floor plan is a 1 dimensional array")

  # to handle more sensible floor plans, we need these helpers in case multiple
  # walls are touching the edges.

  def determine_row_size_of_exterior_space_to_add() -> np.ndarray:
    """A helper function to recompute a space row to add from floor plan dims.

    Returns:
      a row of constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT's to concat if
        the y index has walls abutting it.
    """
    return np.full(
        (floor_plan.shape[0], 1), constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
    )

  def determine_column_size_of_exterior_space_to_add() -> np.ndarray:
    """A helper function to recompute a columns row to add from floor plan dim.

    Returns:
      a column of constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT's to concat if
        the x index has walls abutting it.
    """
    return np.full(
        (1, floor_plan.shape[1]), constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
    )

  if np.any(floor_plan[0, :] == 1):
    xs_to_concat = determine_column_size_of_exterior_space_to_add()
    floor_plan = np.concatenate((xs_to_concat, floor_plan), axis=0)

  if np.any(floor_plan[:, 0] == 1):
    ys_to_concat = determine_row_size_of_exterior_space_to_add()
    floor_plan = np.concatenate((ys_to_concat, floor_plan), axis=1)

  if np.any(floor_plan[-1, :] == 1):
    xs_to_concat = determine_column_size_of_exterior_space_to_add()
    floor_plan = np.concatenate((floor_plan, xs_to_concat), axis=0)

  if np.any(floor_plan[:, -1] == 1):
    ys_to_concat = determine_row_size_of_exterior_space_to_add()
    floor_plan = np.concatenate((floor_plan, ys_to_concat), axis=1)

  return floor_plan


def _determine_exterior_space(
    floor_plan: FileInputFloorPlan,
) -> Tuple[ConnectionReadyFloorPlan, ExteriorSpace]:
  """Marks which CVs are exterior space and which are not.

  Creates an ancillary array denoting the exterior space, and then make original
  a binary value. OpenCV's connectedComponents function works by considering
  1's as components and 0's as filler space.

  Args:
    floor_plan: a FileInputFloorPlan

  Returns:
    a ConnectionReadyFloorPlan
    an ExteriorSpace array
  """

  exterior_space = np.where(
      floor_plan == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT,
  )

  floor_plan = np.where(
      floor_plan == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      constants.INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT,
      constants.GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT,
  )

  return ConnectionReadyFloorPlan(floor_plan), ExteriorSpace(exterior_space)


def _run_connected_components(
    floor_plan: ConnectionReadyFloorPlan,
    connectivity: int = 4,
    save_debugging_image: bool = False,
) -> Connections:
  """Executes the openCV command connectedComponentsWithStats.

  For more info, please see:
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html

  Args:
    floor_plan: a ConnectionReadyFloorPlan
    connectivity: int, defines whether to consider corners as part of connected
      components (connectivity = 8) or just lateral sides (connectivity = 4). We
      need more testing, but our recommendation is currently 4.
    save_debugging_image: boolean for whether we should save the debugging image
      to cns.

  Returns:
    connections: a Connections array
  """

  fp = np.uint8(floor_plan.copy())

  _, connections, _, _ = cv2.connectedComponentsWithStats(
      fp, connectivity=connectivity
  )

  if np.max(connections < 5):
    warnings.warn("""Connected components is showing that there are 4 or fewer
     rooms in your building. You may have your 0's and 1's inverted in the
     floor_plan. Remember that for the connectedComponents function,
     0's must code for exterior space and exterior or interior walls,
     and 1's must code for interior space.""")

  if save_debugging_image:
    save_images_to_cns_for_debugging(connections, "connections")

  return Connections(connections)


def _set_exterior_space_neg(
    connections: Connections, exterior_space: ExteriorSpace
) -> Connections:
  """Modifies the connections array so that exterior space is negative.

  Encoding the exterior space as negative is important in the connections array
  as it will encode an aribtrarily large number of rooms as positive integers.
  Thus, setting the exterior space as negative ensures that we will always be
  able to deal with it as its own category of space.

  Args:
    connections: a Connections array containing the exact output of
      connectedComponents function.
    exterior_space: an ExteriorSpace array

  Returns:
    a Connections array in which exterior air set negative.
  """

  connections = np.where(
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      connections,
  )
  return Connections(connections)


def _label_exterior_wall_shell(
    exterior_space: ExteriorSpace,
) -> ExteriorWalls:  # TODO(spangher): replace ExteriorSpace with
  # FileInputFloorPlan or a canonical form.
  """Returns a matrix with exterior walls noted.

  Note: this function just labels the shell of external walls, not the whole
  wall. The point of this is to provide a starting point for later functions to
  then expand the number of walls into interior wall scope.

  Args:
    exterior_space: an np.ndarray where values are set to
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION if exterior space, and 0
      otherwise.

  Returns:
    an np.ndarray where exterior walls are set to
      constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  """

  is_exterior_space = (
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
  )

  struct = ndimage.generate_binary_structure(2, 1)

  is_near_exterior_space = ndimage.binary_dilation(
      is_exterior_space, structure=struct
  )

  is_exterior_wall = is_near_exterior_space & ~is_exterior_space

  return ExteriorWalls(
      np.where(is_exterior_wall, constants.EXTERIOR_WALL_VALUE_IN_FUNCTION, 0)
  )


def _label_interior_walls(
    exterior_walls: ExteriorWalls,
    original_floor_plan: FileInputFloorPlan,
) -> InteriorWalls:
  """Returns a matrix with interior walls noted."""
  interior_walls = np.full(
      exterior_walls.shape, constants.INTERIOR_SPACE_VALUE_IN_FUNCTION
  )
  interior_walls[
      original_floor_plan == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
  ] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  interior_walls[
      exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  ] = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION
  return InteriorWalls(interior_walls)


def _construct_room_dict(connections: Connections) -> RoomIndicesDict:
  """Return a dictionary with room index lists per room label.

  Note that only empty space is considered to be part of rooms.

  Args:
    connections: a Connections array with exterior_space set negative

  Returns:
    a RoomIndicesDict, i.e. a dictionary with keys;
      (room_1, room_2, ..., room_n, exterior_space) which each map onto a
      list of indices.
  """

  def _component_to_room_index(component: int) -> str:
    """Helper to set component names correctly.

    Args:
      component: the int index of a component.

    Returns:
      The name of the component to go into room_dict.
    """
    if component == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION:
      return constants.EXTERIOR_SPACE_NAME_IN_ROOM_DICT
    elif component == constants.INTERIOR_WALL_VALUE_IN_COMPONENT:
      return constants.INTERIOR_WALL_NAME_IN_ROOM_DICT
    else:
      return f"room_{component}"

  room_dict = collections.defaultdict(list)

  for i in range(connections.shape[0]):
    for j in range(connections.shape[1]):
      component = connections[i, j]
      room_index = _component_to_room_index(component)
      room_dict[room_index].append((i, j))

  return room_dict


def process_and_run_connected_components(
    floor_plan: FileInputFloorPlan,
) -> Connections:
  """Public function that takes in floor plan and outputs a Components.

  Args:
    floor_plan: FileInputFloorPlan

  Returns:
    connections output with exterior space set negative.
  """
  connection_ready_floor_plan, exterior_space = _determine_exterior_space(
      floor_plan
  )
  connections = _run_connected_components(
      connection_ready_floor_plan, connectivity=4
  )
  return _set_exterior_space_neg(connections, exterior_space)


def construct_building_data_types(
    floor_plan: FileInputFloorPlan,
    zone_map: FileInputFloorPlan,
    save_debugging_image: bool = False,
) -> Tuple[RoomIndicesDict, ExteriorWalls, InteriorWalls, ExteriorSpace]:
  """Sequentially calls all preprocessing functions in building_utils.py.

  This function links together the necessary helper functions in
  building_utils.py so that it is clear and straightforward what they do when
  put in sequence. Given a FileInputFloorPlan, this function breaks out the
  necessary pieces of information for further processing.

  Args:
    floor_plan: an FileInputFloorPlan with outside air marked as
      constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT, inside walls marked as
      constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT, and inside space marked as
      constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT.
    zone_map: a FileInputFloorPlan that has, using the same int markings as the
      floor_plan, the VAV zones noted instead of the physical rooms.
    save_debugging_image: bool for whether we should save some debugging images
      to CNS.

  Returns:
    connections output with exterior space set negative.
  """

  padded_floor_plan = guarantee_air_padding_in_frame(floor_plan)
  padded_zone_map = guarantee_air_padding_in_frame(zone_map)

  merged_floor_zone = padded_floor_plan.copy()
  merged_floor_zone = np.where(padded_zone_map == 1, 1, merged_floor_zone)

  _, exterior_space = _determine_exterior_space(padded_floor_plan)
  exterior_walls = _label_exterior_wall_shell(exterior_space)
  interior_walls = _label_interior_walls(exterior_walls, padded_floor_plan)

  if save_debugging_image:
    save_images_to_cns_for_debugging(exterior_walls, "exterior_walls")
    save_images_to_cns_for_debugging(interior_walls, "interior_walls")

  connected_components_neg = process_and_run_connected_components(
      padded_zone_map
  )
  room_dict = _construct_room_dict(connected_components_neg)

  return room_dict, exterior_walls, interior_walls, exterior_space


def enlarge_component(
    array_with_component_nonzero: np.ndarray, distance_to_augment: float
) -> np.ndarray:
  """Enlarges the component in question by CVs within a certain distance.

  Note: this function is a general purpose function intended to enlarge any
  component by measure of "distance_to_augment".

  Args:
    array_with_component_nonzero: array where the object to enlarge is nonzero
    distance_to_augment: return a new object with CVs within this distance
      selected.

  Returns:
    An array with 1 being a CV to include and 0 being a CV to exclude.
  """

  array_with_component_nonzero = np.uint8(array_with_component_nonzero)
  array_with_component_zero = 1 - array_with_component_nonzero
  distances = np.round(
      cv2.distanceTransform(array_with_component_zero, cv2.DIST_L2, 3),
      decimals=2,
  )

  return np.uint8(distances <= distance_to_augment)
