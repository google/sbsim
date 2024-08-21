"""Code for representing the control volumes within a building.

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

import abc
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gin
import numpy as np
from smart_buildings.smart_control.simulator import base_convection_simulator
from smart_buildings.smart_control.simulator import building_utils
from smart_buildings.smart_control.simulator import constants
from smart_buildings.smart_control.simulator import thermal_diffuser_utils

Coordinates2D = Tuple[int, int]
Shape2D = Tuple[int, int]

RoomIndicesDict = Dict[str, Sequence[Coordinates2D]]


@gin.configurable
@dataclasses.dataclass
class MaterialProperties:
  """Holds the physical constants for a material."""

  conductivity: float
  heat_capacity: float
  density: float


def _check_room_sizes(matrix_shape: Shape2D, room_shape: Shape2D):
  """Raises a ValueError if room_shape is not compatible with matrix_shape.

  The matrix for the building includes 2 outer wall layers, then rooms divided
  by walls.

  Args:
    matrix_shape: 2-Tuple representing shape of a matrix.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  if (matrix_shape[0] - 3) % (room_shape[0] + 1) != 0:
    raise ValueError("Room_shape[0] is not compatible with matrix_shape[0]")

  if (matrix_shape[1] - 3) % (room_shape[1] + 1) != 0:
    raise ValueError("Room_shape[1] is not compatible with matrix_shape[1]")


def assign_building_exterior_values(array: np.ndarray, value: float):
  """Assigns value to the building's exterior locations.

  The outer 2 layers of the matrix are special CVs which represent the thicker
  exterior walls as well as the ambient air.

  Args:
    array: Numpy array to assign values to.
    value: Value to assign.
  """
  array[:, [0, 1, -2, -1]] = value
  array[[0, 1, -2, -1], :] = value


def assign_interior_wall_values(
    array: np.ndarray, value: float, room_shape: Shape2D
):
  """Assigns value to interior wall locations.

  These are the walls dividing the rooms. None of these walls are on the
  outer 2 layers of the matrix which are reserved for the thicker outer walls.

  Args:
    array: Numpy array to assign values to.
    value: Value to assign.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  _check_room_sizes(array.shape, room_shape)
  nrows, ncols = array.shape

  for x in range(room_shape[0] + 2, nrows - 2, room_shape[0] + 1):
    for y in range(2, ncols - 2):
      array[x, y] = value
  for x in range(2, nrows - 2):
    for y in range(room_shape[1] + 2, ncols - 2, room_shape[1] + 1):
      array[x, y] = value


def generate_thermal_diffusers(
    matrix_shape: Shape2D, room_shape: Shape2D
) -> np.ndarray:
  """Returns a matrix with four thermal air diffusers for a VAV in each zone.

  This function places 4 diffusers in each room. The function aims to distribute
  them evenly in the room regardless of room size.

  Args:
    matrix_shape: 2-Tuple representing shape of a matrix.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  _check_room_sizes(matrix_shape, room_shape)

  n_diffusers_per_dim = 2

  # The sum of the diffuser's values in each room sum to 1.
  diffuser_value = 1 / n_diffusers_per_dim**2

  diffusers = np.zeros(shape=matrix_shape, dtype=np.float32)
  nrows, ncols = matrix_shape

  # First, number of non-diffuser spaces across each dimension is calculated
  empty_spaces_x = room_shape[0] - n_diffusers_per_dim

  # The empty spaces are distributed evenly between the diffusers and the walls.
  # This leads to 3 zones: wall to diff_1, diff_1 to diff_2, diff_2 to wall.
  diff_1_step_x = empty_spaces_x // 3

  # Put the second diffuser the same distance from the far wall.
  diff_2_step_x = room_shape[0] - diff_1_step_x - 1

  # Same steps for y dimension
  empty_spaces_y = room_shape[1] - n_diffusers_per_dim
  diff_1_step_y = empty_spaces_y // 3
  diff_2_step_y = room_shape[1] - diff_1_step_y - 1

  # room_start is the first empty space in each room, stop at the end of the
  # building.
  for room_start_x in range(2, nrows - 3, room_shape[0] + 1):
    for room_start_y in range(2, ncols - 3, room_shape[1] + 1):
      diffusers[room_start_x + diff_1_step_x, room_start_y + diff_1_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_2_step_x, room_start_y + diff_1_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_1_step_x, room_start_y + diff_2_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_2_step_x, room_start_y + diff_2_step_y] = (
          diffuser_value
      )
  return diffusers


def get_zone_bounds(
    zone_coordinates: Coordinates2D, room_shape: Shape2D
) -> Tuple[int, int, int, int]:
  """Returns (min_x, max_x, min_y, max_y) index bounds for specified zone.

  Helper function to get the minimum and maximum indices excluding walls in
  each dimension for specified zone.

  Args:
    zone_coordinates: Tuple containing x and y coordinates for zone.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  zone_x, zone_y = zone_coordinates
  x_min = zone_x * (room_shape[0] + 1) + 2
  x_max = x_min + room_shape[0] - 1
  y_min = zone_y * (room_shape[1] + 1) + 2
  y_max = y_min + room_shape[1] - 1
  return (x_min, x_max, y_min, y_max)


#### Helper code below here marks the updated helper functions that Lucas wrote:


def enlarge_exterior_walls(
    exterior_walls: building_utils.ExteriorWalls,
    interior_walls: building_utils.InteriorWalls,
) -> Tuple[building_utils.ExteriorWalls, building_utils.InteriorWalls]:
  """Sequentially calls helper functions for expanding exterior walls.

  This function links together some necessary helper functions in
  building_utils.py so that it is clear and straightforward what they do when
  put in sequence. Given a FileInputFloorPlan, this function breaks out the
  necessary pieces of information for further processing.

  Args:
    exterior_walls: an ExteriorWalls noting where exterior walls are.
    interior_walls: an InteriorWalls noting where interior walls are.

  Returns:
    an ExteriorWalls with exterior walls expanded by
    constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT.
    an InteriorWalls with interior walls shrunk by
    constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT.
  """

  exterior_walls_binary = exterior_walls.copy()
  interior_walls_binary = interior_walls.copy()
  exterior_walls_binary = np.uint8(
      exterior_walls_binary == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  )
  interior_walls_binary = np.uint8(
      interior_walls_binary == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  )
  exterior_walls_augmented_temp = building_utils.enlarge_component(
      exterior_walls_binary, constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT
  )
  walls_or_expanded = (
      exterior_walls_augmented_temp
      + interior_walls_binary
      + exterior_walls_binary
  )
  exterior_walls_augmented = np.uint8(
      walls_or_expanded >= constants.WALLS_AND_EXPANDED_BOOLS
  ) * (constants.EXTERIOR_WALL_VALUE_IN_FUNCTION)
  interior_walls_shrunk = np.uint8(
      interior_walls + exterior_walls_augmented
      == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  ) * (constants.INTERIOR_WALL_VALUE_IN_FUNCTION)

  return exterior_walls_augmented, interior_walls_shrunk


def _assign_interior_and_exterior_values(
    exterior_walls: np.ndarray,
    interior_walls: np.ndarray,
    interior_wall_value: float,
    exterior_wall_value: float,
    interior_and_exterior_space_value: float,
) -> np.ndarray:
  """Assigns properties to interior and exterior walls.

  This differs from the original implementation in that it reads from
  pre-processed arrays noting where the exterior and interior inds are,
  whereas the original implementation simply counted, assuming rectangular
  rooms.

  Args:
    exterior_walls: an nd.array with constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
      as exterior walls and 0 otherwise.
    interior_walls: an nd.array with constants.INTERIOR_WALL_VALUE_IN_FUNCTION
      as interior walls and 0 otherwise.
    interior_wall_value: the value to assign to interior walls.
    exterior_wall_value: the value to assign to exterior spaces.
    interior_and_exterior_space_value: the value to assign to interior and
      exterior space.

  Returns:
    an np.ndarray with the appropriate values set.
  """
  array_to_return = np.where(
      interior_walls == constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
      interior_wall_value,
      np.where(
          exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION,
          exterior_wall_value,
          interior_and_exterior_space_value,
      ),
  )
  return array_to_return


def _construct_cv_type_array(
    exterior_walls: np.ndarray, exterior_space: np.ndarray
) -> np.ndarray:
  """Fills once the CV type matrix and save it.

  In the original imlementation,
      the sweep() function would call the get_cv_type() function every time,
      repeating logic that only needed to be computed once and saved.

  Args:
    exterior_walls: np.ndarray noting where exterior walls are
    exterior_space: np.ndarray noting where outside air is

  Returns:
    an np.array filled with strings of the cv type.
  """

  return np.where(
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.LABEL_FOR_EXTERIOR_SPACE,
      np.where(
          exterior_walls == constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
          constants.LABEL_FOR_INTERIOR_SPACE,
          constants.LABEL_FOR_WALLS,
      ),
  )


def _assign_thermal_diffusers(
    array_to_fill: np.ndarray,
    room_dict: RoomIndicesDict,
    interior_walls: building_utils.InteriorWalls,
    diffuser_spacing: int = 10,
    buffer_from_walls: int = 5,
) -> np.ndarray:
  """Places as many thermal diffusers in a zone as "diffuser_spacing" allows.

  The method by which assign_thermal_diffusers works has been updated to deal
  with rooms with differing geometries. It works as follows:

    First, test if the room is rectangular enough.
      If so, then allocate diffusers evenly in a 2D grid whose distance is
        formed by diffuser_spacing, and included only if the index is within the
        ind list for the entry to room_dict
      If not (and this can be fairly rare), then allocate the diffusers randomly

  It is different from the original method, which simply dispersed thermal
    diffusers in a grid determined by a value, "n_diffusers_per_zone", and
    did not consider any non-rectangular room. It would not work if, say,
    we considered the "room" made up of a windy hallway, or an "L" shaped room.

  assign_thermal_diffusers() is a placeholder until we have data on
    exactly where the diffusers are.

  Args:
    array_to_fill: an array prefilled with interior space values that this
      function will fill appropriately
    room_dict: a dict mapping room names to indices
    interior_walls: additional check to see if the allocated diffusers were
      placed in walls.
    diffuser_spacing: how many diffusers to have per control volume spacing.
    buffer_from_walls: how many CVs to leave in between each wall and each
      thermal diffuser

  Returns:
    an np.ndarray with the appropriate values set.
  """

  for key, value in room_dict.items():
    if not key.startswith(constants.ROOM_STRING_DESIGNATOR):
      continue

    inds = thermal_diffuser_utils.diffuser_allocation_switch(
        room_cv_indices=value,
        spacing=diffuser_spacing,
        interior_walls=interior_walls,
        buffer_from_walls=buffer_from_walls,
    )
    num_inds = len(inds)
    for ind in inds:
      array_to_fill[tuple(ind)] = 1.0 / float(num_inds)

  return array_to_fill


class BaseSimulatorBuilding(abc.ABC):
  """Base class for building simulators."""

  @abc.abstractmethod
  def reset(self):
    """Resets the building to its initial parameters."""

  @abc.abstractmethod
  def get_zone_average_temps(
      self,
  ) -> Union[
      Dict[Tuple[int, int], Any],
      Dict[str, Any],
  ]:
    """Returns the average temperature of each zone."""

  @property
  @abc.abstractmethod
  def density(self) -> np.ndarray:
    """Returns the density array of the building."""

  @property
  @abc.abstractmethod
  def heat_capacity(self) -> np.ndarray:
    """Returns the heat capacity array of the building."""

  @property
  @abc.abstractmethod
  def conductivity(self) -> np.ndarray:
    """Returns the conductivity array of the building."""

  @property
  @abc.abstractmethod
  def cv_type(self) -> np.ndarray:
    """Returns the CV type array of the building."""


@gin.configurable
class Building(BaseSimulatorBuilding):
  """Represents a matrix of volumes of material in a building.

  Attributes:
    cv_size_cm: Scalar in cm representing width, length and height of control
      volume.
    floor_height_cm: Height in cm floor to ceiling of each room.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
    building_shape: 2-Tuple representing the number of rooms in the width and
      length of the building.
    temp: The current temp in K of each control volume.
    conductivity: Thermal conductivity in of each control volume W/m/K.
    heat_capacity: Thermal heat cpacity of each control volume in J/kg/K.
    density: Material density in kg/m3 of each control volume.
    input_q: Heat energy applied (sign indicates heating/cooling) at the CV in W
      (J/s).
    diffusers: Proportion of the heat applied per VAV; sums to 1 for each zone.
    neighbors: Matrix containing list of neighbor coordinates for each control
      volume.
    cv_type: a matrix noting whether each CV is outside air, interior space, or
      a wall. cv_type will be used in the sweep() function.
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      room_shape: Shape2D,
      building_shape: Shape2D,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      deprecation: bool = False,
  ):
    """Initializes the ControlVolumes.

    Creates a matrix of control volumes representing the air and walls of a
    building. The size of each room (in terms of control volumes of air) is
    controlled by room_shape. The number of rooms in each building is controlled
    by building_shape. The outer 2 layers of the matrix represent special cells
    where the exterior walls and ambient air interact.

    Args:
      cv_size_cm: Width, length and height of control volume.
      floor_height_cm: Height in cm floor to ceiling of each room.
      room_shape: 2-Tuple representing the number of air control volumes in the
        width and length of each room.
      building_shape: 2-Tuple representing the number of rooms in the width and
        length of the building.
      initial_temp: Initial temperature for each control volume.
      inside_air_properties: MaterialProperties for interior air.
      inside_wall_properties: MaterialProperties for interior walls.
      building_exterior_properties: MaterialProperties for building's exterior.
      deprecation: if true, the old code has been deprecated and transitioned to
        the new, geometrically flexible code. TODO(spangher): change to True
        when the former code is deprecated.
    """

    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self.room_shape = room_shape
    self.building_shape = building_shape
    self._initial_temp = initial_temp

    if not deprecation:
      # TODO(sipple): delete the class when deprecation is finished.

      nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
      ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3

      self._conductivity = np.full(
          (nrows, ncols), inside_air_properties.conductivity
      )
      assign_interior_wall_values(
          self._conductivity,
          inside_wall_properties.conductivity,
          self.room_shape,
      )
      assign_building_exterior_values(
          self._conductivity, building_exterior_properties.conductivity
      )

      self._heat_capacity = np.full(
          (nrows, ncols), inside_air_properties.heat_capacity
      )
      assign_interior_wall_values(
          self._heat_capacity,
          inside_wall_properties.heat_capacity,
          self.room_shape,
      )
      assign_building_exterior_values(
          self._heat_capacity, building_exterior_properties.heat_capacity
      )

      self._density = np.full((nrows, ncols), inside_air_properties.density)
      assign_interior_wall_values(
          self._density, inside_wall_properties.density, self.room_shape
      )
      assign_building_exterior_values(
          self._density, building_exterior_properties.density
      )

      self.diffusers = generate_thermal_diffusers(
          (nrows, ncols), self.room_shape
      )

      self.neighbors = self._calculate_neighbors((nrows, ncols))

      self.reset()

  @property
  def density(self) -> np.ndarray:
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    raise NotImplementedError()

  def reset(self):
    """Resets the building to its initial parameters."""
    nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
    ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3
    self.temp = np.full((nrows, ncols), self._initial_temp)
    self.input_q = np.full((nrows, ncols), 0.0)

  def _calculate_neighbors(
      self, shape: Shape2D
  ) -> List[List[List[Coordinates2D]]]:
    """Returns matrix of list of neighbor indices for each location in a matrix.

    Args:
      shape: 2-Tuple representing the shape of a matrix.
    """
    neighbors = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    for x in range(shape[0]):
      for y in range(shape[1]):
        possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in possible_neighbors:
          if nx >= 0 and nx < shape[0] and ny >= 0 and ny < shape[1]:
            neighbors[x][y].append((nx, ny))
    return neighbors

  def get_zone_thermal_energy_rate(
      self, zone_coordinates: Coordinates2D
  ) -> float:
    """Returns energy rate in W being input to specified zone, summing its CVs contributions.

    Calculates and returns sum of input_q of all air CVs in a given zone.

    Args:
      zone_coordinates: Tuple containing x and y coordinates for zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    submat = self.input_q[x_min : x_max + 1, y_min : y_max + 1]
    return np.sum(submat)

  def get_zone_temp_stats(
      self, zone_coordinates: Coordinates2D
  ) -> Tuple[float, float, float]:
    """Returns the min, max, and mean temp of all air CVs in zone.

    Args:
      zone_coordinates: Tuple containing x and y coordinates for zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    submat = self.temp[x_min : x_max + 1, y_min : y_max + 1]
    return np.min(submat), np.max(submat), np.mean(submat)

  def get_zone_average_temps(self) -> Dict[Tuple[int, int], Any]:
    """Returns a dict of zone average temps, with key (zone_coordinates) and val: temp."""
    avg_temps = {}
    for zone_x in range(self.building_shape[0]):
      for zone_y in range(self.building_shape[1]):
        zone_coordinates = (zone_x, zone_y)
        _, _, avg_temp = self.get_zone_temp_stats(zone_coordinates)
        avg_temps[zone_coordinates] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(
      self, zone_coordinates: Coordinates2D, power: float
  ):
    """Applies thermal power [W] to zone zone_x, zone_y spread evenly to all diffusers.

    Args:
      zone_coordinates: Tuple containing x and y coordinates for zone.
      power: Watts to apply to zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    for x in range(x_min, x_max + 1):
      for y in range(y_min, y_max + 1):
        if self.diffusers[x, y] > 0.0:
          self.input_q[x, y] = power * self.diffusers[x, y]


@gin.configurable
class FloorPlanBasedBuilding(BaseSimulatorBuilding):
  """Creates a Building that is floor plan based to avoid a messy deprecation.

  Attributes:
    cv_size_cm: Scalar in cm representing width, length and height of control
      volume.
    floor_height_cm: Height in cm floor to ceiling of each room.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
    building_shape: 2-Tuple representing the number of rooms in the width and
      length of the building.
    temp: The current temp in K of each control volume.
    conductivity: Thermal conductivity in of each control volume W/m/K.
    heat_capacity: Thermal heat cpacity of each control volume in J/kg/K.
    density: Material density in kg/m3 of each control volume.
    input_q: Heat energy applied (sign indicates heating/cooling) at the CV in W
      (J/s).
    diffusers: Proportion of the heat applied per VAV; sums to 1 for each zone.
    cv_type: a matrix noting whether each CV is outside air, interior space, or
      a wall. cv_type will be used in the sweep() function.
    neighbors: Matrix containing list of neighbor coordinates for each control
      volume.
    len_neighbors: matrix containing the length of neighbors
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      zone_map: Optional[np.ndarray] = None,
      zone_map_filepath: Optional[str] = None,
      floor_plan: Optional[np.ndarray] = None,
      floor_plan_filepath: Optional[str] = None,
      buffer_from_walls: int = 3,
      convection_simulator: Optional[
          base_convection_simulator.BaseConvectionSimulator
      ] = None,
      reset_temp_values: np.ndarray | None = None,
  ):
    """Initializes the New Building.

    Args:
      cv_size_cm: Width, length and height of control volume.
      floor_height_cm: Height in cm floor to ceiling of each room.
      initial_temp: Initial temperature for each control volume.
      inside_air_properties: MaterialProperties for interior air.
      inside_wall_properties: MaterialProperties for interior walls.
      building_exterior_properties: MaterialProperties for building's exterior.
      zone_map: an np.ndarray noting where the VAV zones are.
      zone_map_filepath: a string of where to find the zone_map in CNS. Note
        that the user requires only to provide one of either zone_map_filepath
        or zone_map.
      floor_plan: an np.ndarray to pass into the function if one has this. If
        this is None, then the user must pass in a filepath.
      floor_plan_filepath: a string of where to find the floor_plan in CNS. Both
        floor_plan and floor_plan_filepath may not be None in the new code.
        debugging purposes.
      buffer_from_walls: int to note the space to put between thermal diffusers
        and walls
      convection_simulator: object to simulate air convection
      reset_temp_values: Temp values to use when resetting the building
    """

    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self._initial_temp = initial_temp
    self._convection_simulator = convection_simulator
    self._reset_temp_values = reset_temp_values

    # below is new code, to derive necessary artifacts from the floor plan.
    # TODO(spangher): neaten code by turning the next twenty lines into a
    #   private method.

    if floor_plan is None and floor_plan_filepath is None:
      raise ValueError(
          "Both floor_plan and floor_plan_filepath cannot be None."
      )

    elif floor_plan is None and floor_plan_filepath:
      self._floor_plan = building_utils.read_floor_plan_from_filepath(
          floor_plan_filepath
      )

    elif floor_plan is not None and floor_plan_filepath is None:
      self._floor_plan = floor_plan

    else:
      raise ValueError("floor_plan and floor_plan_filepath ")

    if zone_map_filepath is None and zone_map is None:
      raise ValueError("please provide a zone_map_filepath or a zone_map")

    if zone_map_filepath is not None and zone_map is not None:
      raise ValueError(
          "You have provided both zone_map_filepath and a zone_map"
      )

    if zone_map is not None and zone_map_filepath is None:
      self._zone_map = zone_map

    if zone_map is None and zone_map_filepath is not None:
      zone_map = building_utils.read_floor_plan_from_filepath(zone_map_filepath)
      self._zone_map = zone_map

    (self._room_dict, exterior_walls, interior_walls, self._exterior_space) = (
        building_utils.construct_building_data_types(
            floor_plan=self._floor_plan, zone_map=zone_map
        )
    )

    self._exterior_walls, self._interior_walls = enlarge_exterior_walls(
        exterior_walls=exterior_walls, interior_walls=interior_walls
    )

    self._conductivity = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.conductivity,
        exterior_wall_value=building_exterior_properties.conductivity,
        interior_and_exterior_space_value=inside_air_properties.conductivity,
    )

    self._heat_capacity = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.heat_capacity,
        exterior_wall_value=building_exterior_properties.heat_capacity,
        interior_and_exterior_space_value=inside_air_properties.heat_capacity,
    )

    self._density = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.density,
        exterior_wall_value=building_exterior_properties.density,
        interior_and_exterior_space_value=inside_air_properties.density,
    )

    self.diffusers = np.zeros(self._exterior_walls.shape)
    self.diffusers = _assign_thermal_diffusers(
        self.diffusers,
        room_dict=self._room_dict,
        interior_walls=interior_walls,
        buffer_from_walls=buffer_from_walls,
    )

    self._cv_type = _construct_cv_type_array(
        self._exterior_walls, self._exterior_space
    )

    self.neighbors = self._calculate_neighbors()
    self.len_neighbors = self._calculate_length_of_neighbors()

    self.reset()

  @property
  def density(self) -> np.ndarray:
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    return self._cv_type

  def reset(self):
    self.temp = np.full(
        shape=self._exterior_walls.shape, fill_value=self._initial_temp
    )

    if self._reset_temp_values is not None:
      self.temp = np.copy(self._reset_temp_values)

    self.input_q = np.zeros(self._exterior_walls.shape)

  def _calculate_neighbors(self) -> List[List[List[Coordinates2D]]]:
    """Returns matrix of list of neighbor indices for each location in a matrix.

    Returns:
      A list of CVs that are neighbors with respect to the building.
    """
    shape = self._exterior_walls.shape
    neighbors = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    for x in range(shape[0]):
      for y in range(shape[1]):
        if self.cv_type[x][y] == constants.LABEL_FOR_EXTERIOR_SPACE:
          continue

        possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in possible_neighbors:
          if nx >= 0 and nx < shape[0] and ny >= 0 and ny < shape[1]:
            if self.cv_type[nx][ny] != constants.LABEL_FOR_EXTERIOR_SPACE:
              neighbors[x][y].append((nx, ny))
    return neighbors

  def _calculate_length_of_neighbors(self) -> np.ndarray:
    """Calculates an array with the length of neighbors noted."""

    len_neighbors = np.full(shape=self._exterior_walls.shape, fill_value=0)
    for x in range(len_neighbors.shape[0]):
      for y in range(len_neighbors.shape[1]):
        len_neighbors[x][y] = len(self.neighbors[x][y])

    return len_neighbors

  def get_zone_thermal_energy_rate(self, zone_name: str) -> float:  # pylint: disable=arguments-renamed
    """Returns energy rate in W being input to specified zone, summing its CVs contributions.

    Calculates and returns sum of input_q of all air CVs in a given zone.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.

    Returns:
      the thermal energy rate of the zone.
    """

    if zone_name not in self._room_dict.keys():
      raise ValueError("Zone name is not present in room_dict.")

    zone_coordinates = self._room_dict[zone_name]
    qs = [self.input_q[coord] for coord in zone_coordinates]
    return np.sum(qs)

  def get_zone_temp_stats(self, zone_name: str) -> Tuple[float, float, float]:  # pylint: disable=arguments-renamed
    """Returns the min, max, and mean temp of all air CVs in zone.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.

    Returns:
      the thermal energy rate of the zone.
    """

    if zone_name not in self._room_dict.keys():
      raise ValueError("Zone name is not present in room_dict.")

    zone_coordinates = self._room_dict[zone_name]
    temps = [self.temp[coord] for coord in zone_coordinates]
    return np.min(temps), np.max(temps), np.mean(temps)

  def get_zone_average_temps(self) -> Dict[str, Any]:
    """Returns a dict of zone average temps, with key (zone_coordinates) and val: temp."""
    avg_temps = {}

    for zone in self._room_dict.keys():
      if zone.startswith(constants.ROOM_STRING_DESIGNATOR):
        _, _, avg_temp = self.get_zone_temp_stats(zone)
        avg_temps[zone] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(self, zone_name: str, power: float):  # pylint: disable=arguments-renamed
    """Applies thermal power [W] to zone zone_x, zone_y spread evenly to all diffusers.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.
      power: Watts to apply to zone.
    """

    if zone_name not in self._room_dict.keys():
      raise ValueError("Zone name is not present in room_dict.")

    zone_coordinates = self._room_dict[zone_name]

    for coord in zone_coordinates:
      if self.diffusers[coord] > 0.0:
        self.input_q[coord] = power * self.diffusers[coord]

  def apply_convection(self) -> None:
    if self._convection_simulator is not None:
      self._convection_simulator.apply_convection(self._room_dict, self.temp)
