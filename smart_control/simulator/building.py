"""Code for representing the control volumes within a building."""

import abc
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gin
import numpy as np

from smart_control.simulator import base_convection_simulator
from smart_control.simulator import building_radiation_utils
from smart_control.simulator import building_utils
from smart_control.simulator import constants
from smart_control.simulator import thermal_diffuser_utils

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


@dataclasses.dataclass
class DefaultInsideAirMaterialProperties(MaterialProperties):
  """The default material properties for inside air."""

  def __init__(self):
    super().__init__(conductivity=50.0, heat_capacity=700.0, density=1.2)


@dataclasses.dataclass
class DefaultInsideWallMaterialProperties(MaterialProperties):
  """The default material properties for inside walls."""

  def __init__(self):
    super().__init__(conductivity=2.0, heat_capacity=1000.0, density=1800.0)


@dataclasses.dataclass
class DefaultExteriorWallMaterialProperties(MaterialProperties):
  """The default material properties for building exterior."""

  def __init__(self):
    super().__init__(conductivity=0.05, heat_capacity=1000.0, density=3000.0)


@gin.configurable
@dataclasses.dataclass
class RadiationProperties:
  # pylint: disable=line-too-long
  r"""Holds the radiative properties for a material.

  Args:
    alpha (float): absorptivity. Absorptivity is the fraction of incident
      radiative heat that is absorbed by a surface. When radiation strikes a
      surface, a portion of its energy is converted into internal thermal
      energy, causing the temperature of the surface to rise.
      A value of 1 means the surface is a "black body" and absorbs all incident
      radiation, while a value of 0 means it absorbs none.
    epsilon (float): emissivity. Emissivity is a measure of a surface's ability
      to emit thermal radiation. It is the ratio of the radiation emitted by a
      surface to the radiation emitted by a perfect black body at the same
      temperature. A black body has an emissivity of 1, as it is a perfect
      emitter. A surface with an emissivity of 0 is a theoretical "white body"
      that cannot emit radiation. High emissivity surfaces (like matte black
      paint) are excellent radiators of heat, while low emissivity surfaces
      (like polished metal) are poor radiators.
    tau (float): transmittance. Transmittance is the fraction of incident
      radiative heat that passes through a medium without being absorbed or
      reflected. This property is particularly relevant for modeling radiation
      through transparent or semi-transparent materials, such as glass, air, or
      other gases. For an opaque surface, the transmittance is always 0 because
      no radiation passes through it. For a perfectly transparent medium, the
      transmittance is always 1.
    rho (float): reflectivity. Reflectivity is the fraction of incident
      radiative heat that is reflected away from a surface. When radiation hits
      a surface, some of it bounces off. A highly polished, shiny surface will
      have a high reflectivity (approaching 1), while a dull, dark surface will
      have low reflectivity (approaching 0).

  Relationship between the properties:

    + For any surface, the sum of absorptivity, reflectivity, and transmittance
      must equal 1, as all incident radiation is either absorbed, reflected, or
      transmitted.
    + For an opaque (non-transparent) surface, where transmittance is 0, the sum
      of absorptivity and reflectivity must equal 1, as all incident radiation
      is either absorbed or reflected.

  Each of the property values should be between 0 and 1 (inclusive). Example
  values for various common materials are displayed in the tables below.

  Long-wave and solar emissivity for building surfaces:

  | Material            | Long-wave emissivity (epsilon) | Solar absorptivity (alpha) |
  |---------------------|--------------------------------|------------------------------|
  | Building materials  | 0.90 - 0.96                    | 0.6 - 0.7                    |
  | Wood                | 0.9                            | 0.9 - 0.96                   |
  | Dark-colored paints | 0.91 - 0.95                    | 0.98                         |
  | Light-colored paints| 0.8                            | 0.2                          |
  | Galvanized metal    | 0.28                           | 0.8                          |
  | Aluminum, polished  | 0.03                           | 0.09                         |
  | Window glass        | 0.9 - 0.95                     | 0.02 - 0.04                  |
  | Water               | 0.96                           | 0.1 - 1*                     |
  | Ice                 | 0.95                           | 0.3 - 0.4                    |

  \* Depends strongly on zenith angle; is close to unity for small angles and
    close to zero for large angles.

  Source:
    Table 4.5, Mitchell, John W., and James E. Braun. Principles of
    heating, ventilation, and air conditioning in buildings. John Wiley & Sons,
    2012.
  """
  # pylint: enable=line-too-long

  alpha: float  # absorptivity
  epsilon: float  # emissivity
  tau: float  # transmittance
  rho: float | None = None  # reflectivity

  def __post_init__(self):
    if self.rho is None:
      self.rho = 1 - self.alpha - self.tau

    if self.alpha < 0 or self.alpha > 1:
      raise ValueError("The value for alpha should be between 0 and 1.")

    if self.epsilon < 0 or self.epsilon > 1:
      raise ValueError("The value for epsilon should be between 0 and 1.")

    if self.tau < 0 or self.tau > 1:
      raise ValueError("The value for tau should be between 0 and 1.")

    if self.rho < 0 or self.rho > 1:
      raise ValueError("The value for rho should be between 0 and 1.")

    # Check that the sum of certain radiative properties is equal to 1:
    total = self.alpha + self.rho + self.tau
    if abs(total - 1.0) > 1e-10:
      raise ValueError(
          f"The sum of alpha ({self.alpha}), rho ({self.rho}), "
          f"and tau ({self.tau}) must equal 1, but got {total}."
      )


@dataclasses.dataclass
class DefaultInsideAirRadiationProperties(RadiationProperties):
  """The default radiation properties for inside air."""

  def __init__(self):
    super().__init__(alpha=0.0, epsilon=0.0, tau=1.0, rho=0.0)


@dataclasses.dataclass
class DefaultInsideWallRadiationProperties(RadiationProperties):
  """The default radiation properties for light colored paints."""

  def __init__(self):
    super().__init__(alpha=0.2, epsilon=0.8, tau=0.0, rho=0.8)


@dataclasses.dataclass
class DefaultExteriorWallRadiationProperties(RadiationProperties):
  """The default radiation properties for building materials."""

  def __init__(self):
    super().__init__(alpha=0.65, epsilon=0.93, tau=0.0, rho=0.35)


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
  exterior_walls_augmented = np.int16(
      walls_or_expanded >= constants.WALLS_AND_EXPANDED_BOOLS
  ) * (constants.EXTERIOR_WALL_VALUE_IN_FUNCTION)
  interior_walls_shrunk = np.int16(
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
    """Returns energy rate in W being input to specified zone.

    Sums its CVs contributions.

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
    """Returns a dict of zone average temps.

    The dict is formatted as {`zone_coordinates`: `temp`}.
    """
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
    """Applies thermal power to zones, spread evenly across diffusers.

    The thermal power [W] is applied to zones `zone_x` and `zone_y`.

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
    floor_plan: an np.ndarray representing the building's floor plan.
    temp: The current temp in K of each control volume.
    conductivity: Thermal conductivity in of each control volume W/m/K.
    heat_capacity: Thermal heat capacity of each control volume in J/kg/K.
    density: Material density in kg/m3 of each control volume.
    input_q: Heat energy applied (sign indicates heating/cooling) at the CV in W
      (J/s).
    diffusers: Proportion of the heat applied per VAV; sums to 1 for each zone.
    cv_type: a matrix noting whether each CV is outside air, interior space, or
      a wall. cv_type will be used in the sweep() function.
    neighbors: Matrix containing list of neighbor coordinates for each control
      volume.
    len_neighbors: matrix containing the length of neighbors
    include_radiative_heat_transfer: bool to note whether to include radiative
      heat transfer.
    indexed_floor_plan: matrix representing the building's floor plan. Used only
      for calculating interior radiative heat transfer.
    interior_wall_mask: matrix representing the interior walls of the
      building. Used only for calculating interior radiative heat transfer.
    interior_wall_index: matrix representing the index of the interior
      walls of the building. Used only for calculating interior radiative
      heat transfer.
    interior_wall_VF: matrix representing the view factors of the
      interior walls of the building, which is denoted as F in the equation.
      Used only for calculating interior radiative heat transfer.
    epsilon: matrix representing the emissivity of the nodes of
      the building. Used only for calculating radiative heat transfer.
    alpha: matrix representing the absorptivity of the nodes of
      the building. Used only for calculating radiative heat transfer.
    tau: matrix representing the transmittance of the nodes of
      the building. Used only for calculating radiative heat transfer.
    ifa_inv: matrix representing the inverse of the IFA matrix of the nodes of
      the building. Used only for calculating radiative heat transfer.
    include_interior_mass: bool to note whether to include interior mass nodes.
    interior_mass_mask: matrix indicating which CVs have interior mass nodes.
    interior_mass_temp: matrix representing temperature of interior mass nodes.

      The longwave radiation ($q_{lwx}$) is calculated as:

      $$q_{lwx} = \\sigma(I-F)\\tilde{A}_{inv}T^4$$

      Where the term $(I-F)\\tilde{A}_{inv}$ can be pre-calculated as:

      $$IFA_{inv} = (I-F)\\tilde{A}_{inv}$$
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      initial_temp: float,
      inside_air_properties: MaterialProperties | None = None,
      inside_wall_properties: MaterialProperties | None = None,
      building_exterior_properties: MaterialProperties | None = None,
      interior_mass_properties: MaterialProperties | None = None,
      zone_map: Optional[np.ndarray] = None,
      zone_map_filepath: Optional[str] = None,
      floor_plan: Optional[np.ndarray] = None,
      floor_plan_filepath: Optional[str] = None,
      buffer_from_walls: int = 3,
      convection_simulator: Optional[
          base_convection_simulator.BaseConvectionSimulator
      ] = None,
      reset_temp_values: np.ndarray | None = None,
      inside_air_radiative_properties: RadiationProperties | None = None,
      inside_wall_radiative_properties: RadiationProperties | None = None,
      building_exterior_radiative_properties: RadiationProperties | None = None,
      interior_mass_radiative_properties: RadiationProperties | None = None,
      include_radiative_heat_transfer: bool = False,
      view_factor_method: str = "ScriptF",
      include_interior_mass: bool = False,
  ):
    """Initializes the New Building.

    Args:
      cv_size_cm: Width, length and height of control volume in cm.
      floor_height_cm: Height in cm floor to ceiling of each room.
      initial_temp: Initial temperature for each control volume in K.
      inside_air_properties: MaterialProperties for interior air. If None,
        defaults to DefaultInsideAirMaterialProperties.
      inside_wall_properties: MaterialProperties for interior walls. If None,
        defaults to DefaultInsideWallMaterialProperties.
      building_exterior_properties: MaterialProperties for building's exterior.
        If None, defaults to DefaultExteriorWallMaterialProperties.
      zone_map: an np.ndarray noting where the VAV zones are.
      zone_map_filepath: a string of where to find the zone_map in CNS. Note
        that the user requires only to provide one of either zone_map_filepath
        or zone_map.
      floor_plan: an np.ndarray to pass into the function if one has this. If
        this is None, then the user must pass in a filepath.
      floor_plan_filepath: a string of where to find the floor_plan in CNS. Both
        floor_plan and floor_plan_filepath may not be None.
      buffer_from_walls: int to note the space to put between thermal diffusers
        and walls.
      convection_simulator: object to simulate air convection.
      reset_temp_values: Temp values to use when resetting the building.
      inside_air_radiative_properties: RadiationProperties for interior air.
      inside_wall_radiative_properties: RadiationProperties for interior walls.
      building_exterior_radiative_properties: RadiationProperties for building's
        exterior.
      include_radiative_heat_transfer: bool to note whether to include radiative
        heat transfer.
      view_factor_method: str to note the method to use for view factors.
        Either "ScriptF" or "CarrollMRT". See
        [LW Radiation Exchange Among Zone Surfaces](https://bigladdersoftware.com/epx/docs/9-6/engineering-reference/inside-heat-balance.html#lw-radiation-exchange-among-zone-surfaces)
        for more details.
      interior_mass_properties: MaterialProperties for interior mass nodes
        attached to air CVs.
      interior_mass_radiative_properties: RadiationProperties for interior mass
        nodes attached to air CVs.
      include_interior_mass: bool to note whether to include interior mass nodes
        for air CVs.
    """

    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self._initial_temp = initial_temp
    self._convection_simulator = convection_simulator
    self._reset_temp_values = reset_temp_values
    self.include_radiative_heat_transfer = include_radiative_heat_transfer
    self.include_interior_mass = include_interior_mass

    # Apply default material properties if not provided
    inside_air_properties = inside_air_properties or (
        DefaultInsideAirMaterialProperties()
    )
    inside_wall_properties = inside_wall_properties or (
        DefaultInsideWallMaterialProperties()
    )
    building_exterior_properties = building_exterior_properties or (
        DefaultExteriorWallMaterialProperties()
    )

    # below is new code, to derive necessary artifacts from the floor plan.
    # TODO(spangher): neaten code by turning the next twenty lines into a
    #   private method.

    if floor_plan is None and floor_plan_filepath is None:
      raise ValueError(
          "Both floor_plan and floor_plan_filepath cannot be None."
      )

    elif floor_plan is None and floor_plan_filepath:
      self.floor_plan = building_utils.read_floor_plan_from_filepath(
          floor_plan_filepath
      )

    elif floor_plan is not None and floor_plan_filepath is None:
      self.floor_plan = floor_plan

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
    if include_interior_mass and interior_mass_properties is None:
      raise ValueError(
          "interior_mass_properties must be provided if include_interior_mass"
          " is True"
      )

    (self._room_dict, exterior_walls, interior_walls, self._exterior_space) = (
        building_utils.construct_building_data_types(
            floor_plan=self.floor_plan, zone_map=zone_map
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

    self._assign_interior_mass_properties(
        interior_mass_properties=interior_mass_properties,
        interior_mass_radiative_properties=interior_mass_radiative_properties,
    )

    self._assign_radiative_heat_transfer_properties(
        view_factor_method,
        exterior_walls,
        interior_walls,
        inside_wall_radiative_properties,
        building_exterior_radiative_properties,
        inside_air_radiative_properties,
    )

    self.reset()

  def _assign_radiative_heat_transfer_properties(
      self,
      view_factor_method,
      exterior_walls,
      interior_walls,
      inside_wall_radiative_properties,
      building_exterior_radiative_properties,
      inside_air_radiative_properties,
  ):
    if self.include_radiative_heat_transfer:
      self.view_factor_method = view_factor_method

      self.indexed_floor_plan = self.floor_plan.copy()
      # convert values from 2 to -1:
      self.indexed_floor_plan[
          self.indexed_floor_plan
          == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
      ] = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
      # convert values from 1 to -3:
      self.indexed_floor_plan[
          self.indexed_floor_plan == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
      ] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION

      self.interior_wall_mask = (
          building_radiation_utils.mark_interior_wall_adjacent_to_air(
              self.indexed_floor_plan,
              constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
              constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
          )
      )
      self.lwx_index = np.full(self.indexed_floor_plan.shape, -1)
      # convert mask index => range for view factor matrix order.
      if self.include_interior_mass:
        interior_wall_mask_all = (
            self.interior_wall_mask | self.interior_mass_mask
        )
      else:
        interior_wall_mask_all = self.interior_wall_mask
      self.lwx_index[interior_wall_mask_all] = np.arange(
          np.sum(interior_wall_mask_all)
      )
      self.interior_wall_vf = building_radiation_utils.get_vf(
          indexed_floor_plan=self.indexed_floor_plan,
          interior_wall_mask=self.interior_wall_mask,
          view_factor_method=view_factor_method,
          interior_mass_mask=self.interior_mass_mask,
      )

      # radiative properties
      inside_wall_radiative_properties = (
          inside_wall_radiative_properties
          or DefaultInsideWallRadiationProperties()
      )
      building_exterior_radiative_properties = (
          building_exterior_radiative_properties
          or DefaultExteriorWallRadiationProperties()
      )
      inside_air_radiative_properties = (
          inside_air_radiative_properties
          or DefaultInsideAirRadiationProperties()
      )

      # emissivity
      self._epsilon = _assign_interior_and_exterior_values(
          exterior_walls=exterior_walls,
          interior_walls=interior_walls,
          interior_wall_value=inside_wall_radiative_properties.epsilon,
          exterior_wall_value=building_exterior_radiative_properties.epsilon,
          interior_and_exterior_space_value=inside_air_radiative_properties.epsilon,  # pylint: disable=line-too-long
      )
      # absorptivity
      self._alpha = _assign_interior_and_exterior_values(
          exterior_walls=exterior_walls,
          interior_walls=interior_walls,
          interior_wall_value=inside_wall_radiative_properties.alpha,
          exterior_wall_value=building_exterior_radiative_properties.alpha,
          interior_and_exterior_space_value=inside_air_radiative_properties.alpha,  # pylint: disable=line-too-long
      )
      # transmittance
      self._tau = _assign_interior_and_exterior_values(
          exterior_walls=exterior_walls,
          interior_walls=interior_walls,
          interior_wall_value=inside_wall_radiative_properties.tau,
          exterior_wall_value=building_exterior_radiative_properties.tau,
          interior_and_exterior_space_value=inside_air_radiative_properties.tau,
      )
      if self.include_interior_mass:
        epsilon_temp = np.zeros_like(self._epsilon)
        epsilon_temp[self.interior_mass_mask] = self._epsilon_interior_mass[
            self.interior_mass_mask
        ]
        epsilon_temp[self.interior_wall_mask] = self._epsilon[
            self.interior_wall_mask
        ]
        interior_mask_all = self.interior_mass_mask | self.interior_wall_mask
        epsilon_vector = epsilon_temp[interior_mask_all]
      else:
        epsilon_vector = self._epsilon[self.interior_wall_mask]
      a_tilde_inv = building_radiation_utils.calculate_a_tilde_inv(
          epsilon_vector, self.interior_wall_vf
      )
      self.ifa_inv = building_radiation_utils.calculate_ifa_inv(
          self.interior_wall_vf, a_tilde_inv
      )

    else:
      self.view_factor_method = None
      self.indexed_floor_plan = None
      self.interior_wall_mask = None
      self.interior_wall_index = None
      self.interior_wall_vf = None
      self._alpha = None
      self._epsilon = None
      self._tau = None
      self.ifa_inv = None

  def _assign_interior_mass_properties(
      self,
      interior_mass_properties,
      interior_mass_radiative_properties,
  ):
    """Assigns properties for interior mass nodes."""
    if self.include_interior_mass:
      # Use provided properties or default to air properties

      # Create mask for air nodes (interior space)
      self.interior_mass_mask = (
          self.floor_plan == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
      )

      # Initialize interior mass temperature array
      self.interior_mass_temp = np.full(
          self._exterior_walls.shape, self._initial_temp
      )

      # Assign material properties for interior mass
      self._interior_mass_conductivity = np.where(
          self.interior_mass_mask,
          interior_mass_properties.conductivity,
          0.0,
      )
      self._interior_mass_heat_capacity = np.where(
          self.interior_mass_mask,
          interior_mass_properties.heat_capacity,
          0.0,
      )
      self._interior_mass_density = np.where(
          self.interior_mass_mask,
          interior_mass_properties.density,
          0.0,
      )

      if self.include_radiative_heat_transfer:
        interior_mass_radiative_properties = (
            interior_mass_radiative_properties
            or DefaultInsideWallRadiationProperties()
        )
        self._epsilon_interior_mass = np.where(
            self.interior_mass_mask,
            interior_mass_radiative_properties.epsilon,
            0.0,
        )
        self._alpha_interior_mass = np.where(
            self.interior_mass_mask,
            interior_mass_radiative_properties.alpha,
            0.0,
        )
        self._tau_interior_mass = np.where(
            self.interior_mass_mask,
            interior_mass_radiative_properties.tau,
            0.0,
        )
    else:
      self.interior_mass_mask = None
      self.interior_mass_temp = None
      self._interior_mass_conductivity = None
      self._interior_mass_heat_capacity = None
      self._interior_mass_density = None
      self._epsilon_interior_mass = None
      self._alpha_interior_mass = None
      self._tau_interior_mass = None

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

  @property
  def interior_mass_conductivity(self) -> np.ndarray:
    return self._interior_mass_conductivity

  @property
  def interior_mass_heat_capacity(self) -> np.ndarray:
    return self._interior_mass_heat_capacity

  @property
  def interior_mass_density(self) -> np.ndarray:
    return self._interior_mass_density

  def reset(self):
    self.temp = np.full(
        shape=self._exterior_walls.shape, fill_value=self._initial_temp
    )

    if self._reset_temp_values is not None:
      self.temp = np.copy(self._reset_temp_values)

    self.input_q = np.zeros(self._exterior_walls.shape)

    # Reset interior mass temperatures if enabled
    if self.include_interior_mass:
      self.interior_mass_temp = np.full(
          self._exterior_walls.shape, self._initial_temp
      )

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
    """Returns energy rate in W being input to specified zone.

    Sums its CVs contributions.

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
    """Returns a dict of zone average temps.

    The dict is formatted as: {`zone_coordinates`: `temp`}.
    """
    avg_temps = {}

    for zone in self._room_dict.keys():
      if zone.startswith(constants.ROOM_STRING_DESIGNATOR):
        _, _, avg_temp = self.get_zone_temp_stats(zone)
        avg_temps[zone] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(self, zone_name: str, power: float):  # pylint: disable=arguments-renamed
    """Applies thermal power to zones, spread evenly across diffusers.

    The thermal power [W] is applied to zones `zone_x` and `zone_y`.

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

  def apply_longwave_interior_radiative_heat_transfer(
      self, temperature_estimates: np.ndarray
  ) -> np.ndarray:
    """
    Applies long-wave interior radiative heat transfer.

    This function calculates the net radiative heat flux and radiosity for each
    interior wall.
    """
    if self.include_interior_mass:
      interior_mask_all = self.interior_mass_mask | self.interior_wall_mask
      temperature_estimates_temp = np.zeros_like(temperature_estimates)
      temperature_estimates_temp[self.interior_mass_mask] = (
          self.interior_mass_temp[self.interior_mass_mask]
      )
      temperature_estimates_temp[self.interior_wall_mask] = (
          temperature_estimates[self.interior_wall_mask]
      )
      q_lwx = building_radiation_utils.net_radiative_heatflux_function_of_t(
          temperature_estimates_temp[interior_mask_all], self.ifa_inv
      )
    else:
      q_lwx = building_radiation_utils.net_radiative_heatflux_function_of_t(
          temperature_estimates[self.interior_wall_mask], self.ifa_inv
      )
    return q_lwx
