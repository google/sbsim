"""Tensorflow-enabled Finite Difference calculator.

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

Iterative methods that loop through each control volume sequentially
are very slow with geometries that have many control volumes (CV). The
TFSimulator, instead, processe the finite differences as a set of
tensor operations.
"""

import enum
import functools
from typing import Mapping, Optional, Sequence
from absl import logging
import gin
import numpy as np
import pandas as pd
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as hvac_py
from smart_control.simulator import simulator_flexible_floor_plan as simulator
from smart_control.simulator import weather_controller as weather_controller_py
import tensorflow as tf

# List of neighbors for a specific CV.
NeighborCoordinates = Sequence[simulator.CVCoordinates]
# Neighors, indexable by a specific i,j coord.
Neighbors = Sequence[Sequence[NeighborCoordinates]]


# All CVs must be interior, exterior, or boundary positions.
class CVPositionType(enum.Enum):
  INTERIOR = 1
  EXTERIOR = 2
  BOUNDARY = 3


# All Boundary CVs must be either edge (three non-exterior neighbors), or
# corner (two non-exterior neighbors).
class CVBoundaryType(enum.Enum):
  EDGE = 1
  CORNER = 2


# All edge CVs must be top, bottom, left or right edges.
class CVEdgeOrientationType(enum.Enum):
  """Classifies edge CVs by orientation."""

  TOP = (1, 0.5, 1.0)
  BOTTOM = (2, 0.5, 1.0)
  LEFT = (3, 1.0, 0.5)
  RIGHT = (4, 1.0, 0.5)

  def __init__(
      self,
      value: int,
      vertical_scale_factor: float,
      horizontal_scale_factor: float,
  ):
    self._value_ = value
    self.vertical_scale_factor = vertical_scale_factor
    self.horizontal_scale_factor = horizontal_scale_factor


# All corner CVs must be top left, bottom right, bottom left, or top right.
class CVCornerOrientationType(enum.Enum):
  TOP_LEFT = (1, 0.5)
  BOTTOM_LEFT = (2, 0.5)
  BOTTOM_RIGHT = (3, 0.5)
  TOP_RIGHT = (4, 0.5)

  def __init__(self, value: int, scale_factor: float):
    self._value_ = value
    self.scale_factor = scale_factor


class CVType:
  """Classifies a CV based on its neighborhood and orientation.

  A CV can be interior with four neighbors, exterior with zero or one
  neighbor, or boundary with two or three neighbors. A boundary CV
  can be an edge with three neighbors, or a corner with two neighbors.

  Using a regular class instead of a data class to ensure consistent
  assignment of CV proprties.
  """

  def __init__(
      self,
      position: CVPositionType,
      boundary: Optional[CVBoundaryType] = None,
      edge: Optional[CVEdgeOrientationType] = None,
      corner: Optional[CVCornerOrientationType] = None,
  ):
    self.position = position
    self.boundary = boundary
    self.edge = edge
    self.corner = corner

    if self.position in [CVPositionType.EXTERIOR, CVPositionType.INTERIOR]:
      if (
          self.boundary is not None
          or self.edge is not None
          or self.corner is not None
      ):
        raise ValueError(
            'Cannot assign boundary properties to a non-boundary CV.'
        )
    elif self.position == CVPositionType.BOUNDARY:
      if self.edge is None and self.corner is None:
        raise ValueError('Boundary CVs must have edge or corner properties')

      if self.edge is not None and self.corner is not None:
        raise ValueError('Boundary CVs cannot be edge AND corner.')

      if self.boundary == CVBoundaryType.EDGE and self.edge is None:
        raise ValueError('Edge CVs must be assigned an edge orientation.')

      if self.boundary == CVBoundaryType.CORNER and self.corner is None:
        raise ValueError('Corner CVs must be assigned a corner orientation.')

  def __eq__(self, other) -> bool:

    if isinstance(other, CVType):
      return (
          self.position == other.position
          and self.boundary == other.boundary
          and self.edge == other.edge
          and self.corner == other.corner
      )
    return False

  @property
  def horizontal_scale_factor(self) -> float:
    """Returns the horizontal scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.horizontal_scale_factor
    raise ValueError('horizontal_scale_factor is not defined for this CV type.')

  @property
  def vertical_scale_factor(self) -> float:
    """Returns the vertical scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.vertical_scale_factor
    raise ValueError('vertical_scale_factor is not defined for this CV type.')


BoundaryCVMapping = Mapping[simulator.CVCoordinates, CVType]


def classify_cv(
    coords: simulator.CVCoordinates, neighbors: Neighbors
) -> CVType:
  """Classifies a CV based on its neighborhood.

  0-1 neighbors: EXTERIOR
  2 neighbors: CORNER
  3 neighbors: EDGE
  4 neighbors: INTERIOR

  Args:
    coords: i, j coords of the CV
    neighbors:  a lookup table of all the neighbors.

  Returns:
    a string that represents the CV
  """

  def _cv_type_corner_factory(
      corner: CVCornerOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.CORNER,
        corner=corner,
    )

  def _cv_corner_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    if set([(i + 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_LEFT)
    if set([(i - 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.BOTTOM_LEFT)
    if set([(i + 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_RIGHT)
    if set([(i - 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(
          corner=CVCornerOrientationType.BOTTOM_RIGHT
      )
    raise ValueError(
        f"wasn't able to determine which corner the CV {(i,j)} is."
    )

  def _cv_type_edge_factory(
      edge: CVEdgeOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.EDGE,
        edge=edge,
    )

  def _cv_edge_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    edge = functools.partial(_cv_type_edge_factory)
    if set([(i, j + 1), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.TOP)
    if set([(i, j - 1), (i, j + 1), (i - 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.BOTTOM)
    if set([(i - 1, j), (i, j + 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.LEFT)
    if set([(i - 1, j), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.RIGHT)
    raise ValueError(f"wasn't able to determine which edge the CV {(i,j)} is.")

  i, j = coords

  cv_neighbors = neighbors[i][j]
  match len(cv_neighbors):
    case 0:
      return CVType(position=CVPositionType.EXTERIOR)
    case 1:
      return CVType(position=CVPositionType.EXTERIOR)
    case 2:
      return _cv_corner_type(cv_neighbors)
    case 3:
      return _cv_edge_type(cv_neighbors)
    case 4:
      return CVType(position=CVPositionType.INTERIOR)
    case _:
      raise ValueError(
          f"wasn't able to determine which CV type the CV {(i,j)} is."
      )


def get_cv_mapping(
    neighbors: Neighbors,
    position_criterion: CVPositionType = CVPositionType.BOUNDARY,
) -> BoundaryCVMapping:
  """Gets a map of all the non-interior cv_types."""

  boundary_cv_mapping = {}
  ni = len(neighbors)
  nj = len(neighbors[0])
  for i in range(ni):
    for j in range(nj):
      cv_type = classify_cv((i, j), neighbors)
      if cv_type.position == position_criterion:
        boundary_cv_mapping[(i, j)] = cv_type

  return boundary_cv_mapping


def get_cv_dimension_tensors(
    control_volume_cm: float,
    boundary_cv_mapping: BoundaryCVMapping,
    shape=tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
  """Returns horizontal and vertical CV dimension tensors.

  Boundary control volumes have varying heights and widths, since they share
  a part with the outside. For example, edge boundary CVs have one side
  that is half width, depending on orientation, and corner CVs have both edges
  half width.

  Args:
    control_volume_cm: centimeters width of each interior control volume.
    boundary_cv_mapping: dict of all boundary elements.
    shape: shape of the target matrix.

  Returns:
    horizontal (u), vertical (v) dimension tensors.
  """

  def _compute_cv_dimension(cv_type: CVType) -> tuple[float, float]:
    """Sets the horizontal and vertical CV dimension for one CV."""

    return (
        control_volume_cm * cv_type.horizontal_scale_factor,
        control_volume_cm * cv_type.vertical_scale_factor,
    )

  horizontal_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float32)
  vertical_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():

    cv_horizontal_cv_dimension, cv_vertical_cv_dimension = (
        _compute_cv_dimension(cv_type)
    )
    vertical_cv_dimension[i][j] = cv_vertical_cv_dimension
    horizontal_cv_dimension[i][j] = cv_horizontal_cv_dimension

  t_horizontal_cv_dimension = tf.convert_to_tensor(
      horizontal_cv_dimension, dtype=tf.float32
  )
  t_vertical_cv_dimension = tf.convert_to_tensor(
      vertical_cv_dimension, dtype=tf.float32
  )
  return t_horizontal_cv_dimension, t_vertical_cv_dimension


def get_oriented_convection_coefficient_tensors(
    convection_coefficient_air: float,
    shape: tuple[int, int],
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns oriented convection coefficient tensors.

  Forced convection (wind) is the primary means of exchanging heat
  across the boundary CVs.

  To accommodate tensor operations, we have to create edge-oriented tensors
  for convection.

  Args:
    convection_coefficient_air: Outside air convection coefficient.
    shape: Tensor shape - should be identical to the other tensors, i.e., temp
    boundary_cv_mapping: dict of boundary CVs.

  Returns:
    tensors for convection coeffs: left, right, top, bottom, all of shape
  """

  def _set_edge_convection_coefficient(cv_type: CVType) -> None:
    """Sets the convection coefficient for one CV."""
    match cv_type.edge:
      case CVEdgeOrientationType.TOP:
        h_top_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.LEFT:
        h_left_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.RIGHT:
        h_right_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.BOTTOM:
        h_bottom_edge[i][j] = convection_coefficient_air

  def _set_corner_convection_coefficient(cv_type: CVType) -> None:
    match cv_type.corner:
      case CVCornerOrientationType.TOP_LEFT:
        h_top_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.TOP_RIGHT:
        h_top_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_LEFT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_RIGHT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air

  h_left_edge = np.full(shape, 0, dtype=np.float32)
  h_right_edge = np.full(shape, 0, dtype=np.float32)
  h_top_edge = np.full(shape, 0, dtype=np.float32)
  h_bottom_edge = np.full(shape, 0, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        _set_edge_convection_coefficient(cv_type)

      if cv_type.boundary == CVBoundaryType.CORNER:
        _set_corner_convection_coefficient(cv_type)

  t_h_left_edge = tf.convert_to_tensor(h_left_edge, dtype=tf.float32)
  t_h_right_edge = tf.convert_to_tensor(h_right_edge, dtype=tf.float32)
  t_h_top_edge = tf.convert_to_tensor(h_top_edge, dtype=tf.float32)
  t_h_bottom_edge = tf.convert_to_tensor(h_bottom_edge, dtype=tf.float32)
  return t_h_left_edge, t_h_right_edge, t_h_top_edge, t_h_bottom_edge


def get_oriented_conductivity_tensors(
    conductivity: np.ndarray,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns oriented conductivity tensors.

  Conduction is a function of two adjacent CVs. On boundary CVs, there will be
  no conduction transfer, k = 0. On internal CVs there will be heat transfer,
  k > 0.

  The conductivity is split by sides of the CV, top, bottom, left and right.

  Args:
    conductivity: conductivity of the CV itself.
    boundary_cv_mapping: mapping that contains the coordinates of boundary CVs

  Returns:
    tensors for left, right, top, bottom conductivities.
  """

  k_left_edge = conductivity.copy()
  k_right_edge = conductivity.copy()
  k_top_edge = conductivity.copy()
  k_bottom_edge = conductivity.copy()

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        if cv_type.edge == CVEdgeOrientationType.TOP:
          k_top_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.LEFT:
          k_left_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.RIGHT:
          k_right_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.BOTTOM:
          k_bottom_edge[i][j] = 0.0

      if cv_type.boundary == CVBoundaryType.CORNER:
        if cv_type.corner == CVCornerOrientationType.TOP_LEFT:
          k_top_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.TOP_RIGHT:
          k_top_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_LEFT:
          k_bottom_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_RIGHT:
          k_bottom_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0

  t_k_left_edge = tf.convert_to_tensor(k_left_edge, dtype=tf.float32)
  t_k_right_edge = tf.convert_to_tensor(k_right_edge, dtype=tf.float32)
  t_k_top_edge = tf.convert_to_tensor(k_top_edge, dtype=tf.float32)
  t_k_bottom_edge = tf.convert_to_tensor(k_bottom_edge, dtype=tf.float32)
  return t_k_left_edge, t_k_right_edge, t_k_top_edge, t_k_bottom_edge


def shift_tensor_right(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_right = tf.pad(x, [[0, 0], [1, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_right, [True] * x.shape[1] + [False], axis=1)


def shift_tensor_left(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_left = tf.pad(x, [[0, 0], [0, 1]], constant_values=padding_value)
  return tf.boolean_mask(t_left, [False] + [True] * x.shape[1], axis=1)


def shift_tensor_up(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_up = tf.pad(x, [[0, 1], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_up, [False] + [True] * x.shape[0], axis=0)


def shift_tensor_down(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_down = tf.pad(x, [[1, 0], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_down, [True] * x.shape[0] + [False], axis=0)


def apply_exterior_temps(
    t_temps: tf.Tensor, temp_inf: float, t_exterior_temps_mask: tf.Tensor
) -> tf.Tensor:
  """Applies the temp_inf to all CVs that are masked as exterior."""
  t_exterior_temp = tf.fill(
      t_temps.shape, tf.constant(temp_inf, dtype=tf.float32)
  )
  return tf.where(t_exterior_temps_mask, t_exterior_temp, t_temps)


@gin.configurable
class TFSimulator(simulator.SimulatorFlexibleGeometries):
  """Tensor-based simulator that used matrix ops to update temps."""

  def __init__(
      self,
      building: building_py.Building,
      hvac: hvac_py.FloorPlanBasedHvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
  ):

    super().__init__(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Get a mapping of all the boundary CVs that interface between interior
    # and extrior CVs. This mapping will be used to process boundary elements
    # iteratively.
    self._boundary_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.BOUNDARY
    )
    n_boundary_elements = len(self._boundary_cv_mapping)
    logging.info('Number of boundary CVs: %d', n_boundary_elements)
    # Get a binary mask that mark exterior CVs so that they will always be
    # assigned ambinent air temps.
    self._t_exerior_temps_mask = self._get_tensor_exterior_mask(building)
    n_exterior_elements = tf.math.count_nonzero(self._t_exerior_temps_mask)
    logging.info('Number of exterior CVs: %d', n_exterior_elements)

    n_elements = self._building.temp.shape[0] * self._building.temp.shape[1]
    n_interior_elements = n_elements - n_boundary_elements - n_exterior_elements
    logging.info('Number of interior CVs: %d', n_interior_elements)

    self._t_u, self._t_v = get_cv_dimension_tensors(
        self._building.cv_size_cm / 100.0,
        self._boundary_cv_mapping,
        self._building.temp.shape,
    )

    (
        self._t_conductivity_left_edge,
        self._t_conductivity_right_edge,
        self._t_conductivity_top_edge,
        self._t_conductivity_bottom_edge,
    ) = get_oriented_conductivity_tensors(
        self._building.conductivity, self._boundary_cv_mapping
    )

  def _get_tensor_exterior_mask(
      self, building: building_py.Building
  ) -> tf.Tensor:
    """Returns a binary tensor mask of all CVs that are exterior = True."""
    exterior_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.EXTERIOR
    )
    exterior_mask = np.full(building.temp.shape, False)
    for i, j in exterior_cv_mapping:
      exterior_mask[i][j] = True
    return tf.convert_to_tensor(exterior_mask)

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    """Iterates across all CVs and updates the temperature estimate.

    Corner and edge CVs are exposed to thermal exchange with the ambient air
    through convection.

    This method implements Equation 22, derived in
    go/smart-buildings-simulator-design.

    Args:
      temperature_estimates: Current temperature estimate for each CV, will be
        updated with new values.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      Maximum difference in temperture_estimates across all CVs before and after
      operation.
    """

    def _get_input_tensors(
        building,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
      """Returns the input matrices as tensors."""
      # Convert a bunch of numpy arrays into TF tensors.
      t_temp = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
      t_temp_old = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
      t_temp_minus = tf.convert_to_tensor(building.temp, dtype=tf.float32)
      t_input_q = tf.convert_to_tensor(building.input_q, dtype=tf.float32)
      t_density = tf.convert_to_tensor(building.density, dtype=tf.float32)
      t_heat_capacity = tf.convert_to_tensor(
          building.heat_capacity, dtype=tf.float32
      )
      t_z = tf.constant(building.floor_height_cm / 100.0, dtype=tf.float32)
      return (
          t_temp,
          t_temp_old,
          t_temp_minus,
          t_input_q,
          t_density,
          t_heat_capacity,
          t_z,
      )

    def _get_neighbor_temps(
        t_temp: tf.Tensor, ambient_temperature: float
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      """Creates left, right, up, down neighbor temp matrices."""

      # Create shifted tensor to be able to evaluate neighbors in the equation.
      t_temp_left = shift_tensor_left(t_temp, padding_value=ambient_temperature)

      t_temp_right = shift_tensor_right(
          t_temp, padding_value=ambient_temperature
      )

      t_temp_above = shift_tensor_down(
          t_temp, padding_value=ambient_temperature
      )

      t_temp_below = shift_tensor_up(t_temp, padding_value=ambient_temperature)

      return (t_temp_left, t_temp_right, t_temp_above, t_temp_below)

    def _get_denominator(
        t_k1_div_u: tf.Tensor,
        t_k3_div_u: tf.Tensor,
        t_convection_left_edge: tf.Tensor,
        t_convection_right_edge: tf.Tensor,
        t_convection_top_edge: tf.Tensor,
        t_convection_bottom_edge: tf.Tensor,
        t_vz: tf.Tensor,
        t_k2_div_v: tf.Tensor,
        t_k4_div_v: tf.Tensor,
        t_uz: tf.Tensor,
        t_density: tf.Tensor,
        t_heat_capacity: tf.Tensor,
        t_z: tf.Tensor,
        t_delta_t: tf.Tensor,
    ) -> tf.Tensor:
      """Returns the denominator matrix from Eqn 22 as a tensor."""

      # Compute conductivity/conduction transfer terms on the v-z surface.
      dt1 = tf.math.add(t_k1_div_u, t_k3_div_u)
      dt1 = tf.math.add(dt1, t_convection_left_edge)
      dt1 = tf.math.add(dt1, t_convection_right_edge)
      dt1 = tf.math.multiply(t_vz, dt1)

      # Compute conductivity/conduction transfer terms on the u-z surface.
      dt2 = tf.math.add(t_k2_div_v, t_k4_div_v)
      dt2 = tf.math.add(dt2, t_convection_bottom_edge)
      dt2 = tf.math.add(dt2, t_convection_top_edge)
      dt2 = tf.math.multiply(t_uz, dt2)

      # Create the thermal absorption term.
      dt3 = tf.math.multiply(t_density, self._t_u)
      dt3 = tf.math.multiply(dt3, self._t_v)
      dt3 = tf.math.multiply(dt3, t_heat_capacity)
      dt3 = tf.scalar_mul(t_z, dt3)
      dt3 = tf.math.multiply(dt3, t_heat_capacity)
      dt3 = tf.math.divide(dt3, t_delta_t)

      # Sum up u-z, u-v surface transfer and absorption terms.
      t_denom = tf.math.add(dt1, dt2)
      t_denom = tf.math.add(t_denom, dt3)
      return t_denom

    def _get_numerator(
        t_k1_div_u: tf.Tensor,
        t_k3_div_u: tf.Tensor,
        t_convection_left_edge: tf.Tensor,
        t_convection_right_edge: tf.Tensor,
        t_convection_top_edge: tf.Tensor,
        t_convection_bottom_edge: tf.Tensor,
        t_vz: tf.Tensor,
        t_k2_div_v: tf.Tensor,
        t_k4_div_v: tf.Tensor,
        t_uz: tf.Tensor,
        t_density: tf.Tensor,
        t_heat_capacity: tf.Tensor,
        t_z: tf.Tensor,
        t_delta_t: tf.Tensor,
        t_temp_left: tf.Tensor,
        t_temp_right: tf.Tensor,
        t_temp_above: tf.Tensor,
        t_temp_below: tf.Tensor,
        t_temp_inf: tf.Tensor,
        t_input_q: tf.Tensor,
        t_temp_minus: tf.Tensor,
    ) -> tf.Tensor:
      """Returns the numerator matrix from Eqn 22 as a tensor."""

      # Compute numerator's conductivity transfer terms.
      t_k1_div_u_temp_left = tf.math.multiply(t_k1_div_u, t_temp_left)
      t_k3_div_u_temp_right = tf.math.multiply(t_k3_div_u, t_temp_right)
      t_k2_div_v_temp_below = tf.math.multiply(t_k2_div_v, t_temp_below)
      t_k4_div_v_temp_above = tf.math.multiply(t_k4_div_v, t_temp_above)

      # Compute numerator's convection transfer terms.
      t_h_left_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_left_edge)
      t_h_right_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_right_edge)
      t_h_above_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_top_edge)
      t_h_below_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_bottom_edge)

      # Merge the conduction/convection transfer terms across the v-z surfaces.
      nt1 = tf.math.add(t_k1_div_u_temp_left, t_k3_div_u_temp_right)
      nt1 = tf.math.add(nt1, t_h_left_tinf)
      nt1 = tf.math.add(nt1, t_h_right_tinf)
      nt1 = tf.math.multiply(t_vz, nt1)

      # Merge the conduction/convection transfer terms across the u-z surfaces.
      nt2 = tf.math.add(t_k2_div_v_temp_below, t_k4_div_v_temp_above)
      nt2 = tf.math.add(nt2, t_h_below_tinf)
      nt2 = tf.math.add(nt2, t_h_above_tinf)
      nt2 = tf.math.multiply(t_uz, nt2)

      # Create the thermal absorption term.
      nt3 = tf.math.multiply(t_density, self._t_u)
      nt3 = tf.math.multiply(nt3, self._t_v)
      nt3 = tf.math.multiply(nt3, t_heat_capacity)
      nt3 = tf.scalar_mul(t_z, nt3)
      nt3 = tf.math.multiply(nt3, t_heat_capacity)
      nt3 = tf.math.multiply(nt3, t_temp_minus)
      nt3 = tf.math.divide(nt3, t_delta_t)

      # Add the u-z, u-v surface transfer, absorption and external source terms.
      t_numer = tf.math.add(nt1, nt2)
      t_numer = tf.math.add(t_numer, nt3)
      t_numer = tf.math.add(t_numer, t_input_q)
      return t_numer

    # Get the inputs to the equation as Tensors from the building.
    (
        t_temp,
        t_temp_old,
        t_temp_minus,
        t_input_q,
        t_density,
        t_heat_capacity,
        t_z,
    ) = _get_input_tensors(self._building)

    (
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
    ) = get_oriented_convection_coefficient_tensors(
        convection_coefficient,
        self._building.temp.shape,
        self._boundary_cv_mapping,
    )

    # Create shifted tensor to be able to evaluate neighbors in the equation.
    t_temp_left, t_temp_right, t_temp_above, t_temp_below = _get_neighbor_temps(
        t_temp, ambient_temperature
    )

    # Get the ambinet temperature as a tensor.
    t_temp_inf = tf.constant(ambient_temperature, dtype=tf.float32)

    # Convert the timestep input to tensor.
    t_delta_t = tf.constant(self._time_step_sec, dtype=tf.float32)

    # Create surface area tensors: horizontal u or v x vertical z dim.
    t_uz = tf.scalar_mul(t_z, self._t_u)
    t_vz = tf.scalar_mul(t_z, self._t_v)

    # Calculate the denominator terms.
    t_k1_div_u = tf.math.divide(self._t_conductivity_left_edge, self._t_u)
    t_k3_div_u = tf.math.divide(self._t_conductivity_right_edge, self._t_u)
    t_k2_div_v = tf.math.divide(self._t_conductivity_bottom_edge, self._t_v)
    t_k4_div_v = tf.math.divide(self._t_conductivity_top_edge, self._t_v)

    t_denom = _get_denominator(
        t_k1_div_u,
        t_k3_div_u,
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
        t_vz,
        t_k2_div_v,
        t_k4_div_v,
        t_uz,
        t_density,
        t_heat_capacity,
        t_z,
        t_delta_t,
    )

    # Calculate the numerator terms
    t_numer = _get_numerator(
        t_k1_div_u,
        t_k3_div_u,
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
        t_vz,
        t_k2_div_v,
        t_k4_div_v,
        t_uz,
        t_density,
        t_heat_capacity,
        t_z,
        t_delta_t,
        t_temp_left,
        t_temp_right,
        t_temp_above,
        t_temp_below,
        t_temp_inf,
        t_input_q,
        t_temp_minus,
    )

    # Finally, perform an elementwise division - not a matrix inversion.
    t_temperature_estimates = tf.math.divide(t_numer, t_denom)

    # The tensor operation potentially altered the exterior air conditions,
    # so we need to reset exterior CVs to the exterior air conditioners.
    t_temperature_estimates = apply_exterior_temps(
        t_temperature_estimates, t_temp_inf, self._t_exerior_temps_mask
    )

    t_delta = tf.math.subtract(t_temperature_estimates, t_temp_old)

    return t_temperature_estimates.numpy(), np.max(tf.math.abs(t_delta))
