"""Building Radiation Utility Functions

For computing the physical and thermal characteristics of buildings.
"""

from collections import deque
import math
from typing import Any, Mapping, Sequence

import numpy as np

from smart_control.simulator import constants
from smart_control.simulator import solar_radiation

TEMPORARY_MARKED_VALUE = -33
TEMPORARY_BLOCKED_VALUE = -34
AIR_IN_LINE_OF_SIGHT = 9  # Air nodes along line of sight between wall nodes
FOUR_CONNECTED_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# pylint: disable=invalid-name


def calculate_a_tilde_inv(epsilon: np.ndarray, F: np.ndarray) -> np.ndarray:
  r"""Calculates the inverse A-tilde matrix for radiosity calculations.

  Starting from the gray, diffuse-surface radiosity relation

  $$J_i = \epsilon_i E_{b,i} + (1 - \epsilon_i) G_i,$$

  and using $G_i = \sum_j F_{ij} J_j$, the equations become

  $$J_i - (1 - \epsilon_i) \sum_j F_{ij} J_j = \epsilon_i E_{b,i}.$$

  Dividing each row by $\epsilon_i$ gives

  $$\tilde{\mathbf{A}}\,\mathbf{J} = \mathbf{E}_b,$$

  where

  $$\tilde{A}_{ij} =
  \frac{\delta_{ij} - (1 - \epsilon_i) F_{ij}}{\epsilon_i}.$$

  This function returns $\tilde{\mathbf{A}}^{-1}$, which maps blackbody
  emissive power to radiosity: $\mathbf{J} =
  \tilde{\mathbf{A}}^{-1}\mathbf{E}_b$.

  Args:
    epsilon: Surface emissivities.
    F: View-factor matrix in the convention used by this calculation, where
      ``F[i, j]`` contributes radiosity from surface ``j`` to irradiation on
      surface ``i``.

  Returns:
    The inverse A-tilde matrix, $\tilde{\mathbf{A}}^{-1}$.

  Raises:
    ValueError: If inputs have incompatible dimensions, non-finite values, or
      emissivities outside the interval $(0, 1]$.
  """
  epsilon = np.asarray(epsilon, dtype=float)
  F = np.asarray(F, dtype=float)
  n = epsilon.shape[0]

  if epsilon.ndim != 1 or F.shape != (n, n):
    raise ValueError('epsilon and F must have compatible square dimensions.')
  if not np.all(np.isfinite(epsilon)) or np.any((epsilon <= 0) | (epsilon > 1)):
    raise ValueError(
        'epsilon values must be finite and in the interval (0, 1].'
    )
  if not np.all(np.isfinite(F)):
    raise ValueError('F must contain only finite values.')

  a_tilde = (np.eye(n) - (1.0 - epsilon)[:, np.newaxis] * F) / epsilon[
      :, np.newaxis
  ]
  return np.linalg.solve(a_tilde, np.eye(n))


def calculate_ifa_inv(F: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
  r"""
  Calculates the $IFA_{inv}$ matrix.

  $$IFA_{inv} = (I - F) \tilde{A}^{-1}$$

  See [`net_radiative_heatflux_function_of_T`](./#smart_control.simulator.building_radiation_utils.net_radiative_heatflux_function_of_T) for more details.

  Args:
      F (np.ndarray): The view factor matrix.
      A_inv (np.ndarray): The A inverse matrix.

  Returns:
      IFA_inv : The IFA inverse matrix.
  """

  n = F.shape[0]

  I = np.eye(n)
  ifa_inv = (I - F) @ A_inv
  return ifa_inv


def net_radiative_heatflux_function_of_t(
    T: np.ndarray, ifa_inv: np.ndarray
) -> np.array:
  r"""
  Calculates the net radiative heat flux and radiosity for all surfaces given
    surface temperatures.

  Equations:
  --------------------
  The net radiative heat flux leaving surface $i$ is:

  $$q_i = J_i - G_i$$

  where:
  - $J_i$ is the radiosity (total outgoing radiative flux) from surface $i$,
  - $G_i$ is the irradiation (total incoming radiative flux) onto surface $i$.

  The radiosity is given by:

  $$J_i = \epsilon_i E_{b,i} + \rho_i G_i$$

  where $\epsilon_i$ is the emissivity, $\rho_i = 1 - \epsilon_i$ is the
    reflectivity (for opaque surfaces), and $E_{b,i}$ is the blackbody
    emission from $i$ surface.

  The irradiation for the $i$ surface is:

  $$G_i A_i = \sum_{j=1,\, j\neq i}^n J_j A_j F_{ji}$$

  where $F_{ji}$ is the view factor from surface $j$ to $i$.

  Combining these, the radiosity equation for all surfaces can be written in
    vector-matrix form as:

  $$\tilde{\mathbf{A}}\, \mathbf{J} = \mathbf{E}_b$$

  where $\tilde{A}_{ij} =
    \frac{\delta_{ij} - (1-\epsilon_i) F_{ij}}{\epsilon_i}$.

  Solving for $\mathbf{J}$:

  $$\mathbf{J} = \tilde{\mathbf{A}}^{-1} \mathbf{E}_b$$

  The net heat flux vector for all surfaces is:

  $$\mathbf{q}=
  (\mathbf{I}-\tilde{\mathbf{F}})\tilde{\mathbf{A}}^{-1}\mathbf{E}_b$$

  where $\tilde{\mathbf{F}}$ is the matrix of view factors,
    $F_{ij}$ and$\mathbf{E}_b$ is $\sigma \mathbf{T}^4$.

  Nomenclature and Units:
  -----------------------
  - $q_i$        : Net radiative heat flux from surface $i$ [$\mathrm{W/m^2}$]
  - $\mathbf{q}$ : Vector of $q_i$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $J_i$        : Radiosity of surface $i$ [$\mathrm{W/m^2}$]
  - $\mathbf{J}$ : Vector of $J_i$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $G_i$        : Irradiation on surface $i$ [$\mathrm{W/m^2}$]
  - $E_{b,i}$    : Blackbody emissive power of surface $i$ [$\mathrm{W/m^2}$]$
  - $\mathbf{E}_b$: Vector of $E_{b,i}$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $\epsilon_i$ : Emissivity of surface $i$ [dimensionless]
  - $\rho_i$     : Reflectivity of surface $i$ [dimensionless],
                    ($\rho_i=1-\epsilon_i$)
  - $A_i$        : Area of surface $i$ [$\mathrm{m^2}$]
  - $F_{ij}$     : View factor from surface $i$ to $j$ [dimensionless]
  - $\tilde{\mathbf{A}}$: Matrix with elements
    ($\tilde{A}_{ij} =
    \frac{\delta_{ij} - (1-\epsilon_i) F_{ij}}{\epsilon_i}$)
  - $\mathbf{I}$ : $n \times n$ identity matrix
  - $\tilde{\mathbf{F}}$: Matrix of $F_{ij}$ (view factors)
  - $\delta_{ij}$: Kronecker delta ($=1$ if $i=j$, $=0$ otherwise)
  - $\sigma$: Stefan-Boltzmann constant [$\mathrm{W/m^2K^4}$]
  - $\mathbf{T}$: Vector of surface temperatures [K]

  References:
  -----------
  - Incropera, F.P., DeWitt, D.P., "Fundamentals of Heat and Mass Transfer"

  Args:
    T (np.ndarray): Surface temperatures in Kelvin.
    ifa_inv (np.ndarray): (I - F) @ A_inv.

  Returns:
      q : Net radiative heat flux [W/m^2]

  """
  sigma = (
      constants.STEFAN_BOLTZMANN_CONSTANT
  )  # [W/m^2K^4] Stefan-Boltzmann constant

  q = sigma * ifa_inv @ np.power(T, 4)
  return q


def mark_air_connected_interior_walls(
    indexed_floor_plan: np.ndarray,
    start_pos: tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """
  Mark all interior wall nodes that are connected to the same air space as the
      starting position (interior wall or air cell).
  Uses 4-directional connectivity to check wall-air adjacency.
  All connected walls are marked.

  Args:
    indexed_floor_plan (np.ndarray): 2D numpy array representing the floor plan
        where different values represent different types of cells (walls, air,
        etc.).
    start_pos (Tuple[int, int]): Starting position (row, col). Can be either an
        interior wall or an air cell. If it's an interior wall, finds all walls
        connected to the same air space. If it's an air cell, finds all walls
        connected to that air space.
    interior_wall_value (int, optional): Value used to represent interior walls
        in the floor plan. Defaults to -3 (from "constants.py").
    marked_value (int, optional): Value used to mark connected interior walls.
        Only used internally. Defaults to -33.
    air_value (int, optional): Value used to represent air spaces in the floor
        plan. Defaults to 0 (from "constants.py").

  Returns:
    A tuple containing:

      - `modified_floor_plan`: Copy of input floor plan with connected walls
          marked with marked_value. `None` if `start_pos` is invalid.

      - `interior_space_array`: Extracted interior space containing only air and
          marked walls, cropped to the bounding box of the connected region.
          `None` if `start_pos` is invalid or no interior space is found.

  Raises:
      ValueError: If the starting position is out of bounds of the floor plan.

  Note:
      This function is used as the first step in radiative heat transfer
      calculations to identify all interior wall nodes that are connected to the
      same air space. The marked_value (-33) indicates walls that can
      potentially participate in radiative heat transfer with each other.
  """
  # Make a copy to avoid modifying the original
  floor_plan = indexed_floor_plan.copy()
  if (
      start_pos[0] < 0
      or start_pos[0] >= floor_plan.shape[0]
      or start_pos[1] < 0
      or start_pos[1] >= floor_plan.shape[1]
  ):
    raise ValueError('Starting position is out of bounds')

  start_row, start_col = start_pos
  start_cell_value = floor_plan[start_row, start_col]

  # Return None if start_pos is neither interior_wall_value nor air_value
  if start_cell_value != interior_wall_value and start_cell_value != air_value:
    return None, None

  # 4-connectivity for all steps
  directions = FOUR_CONNECTED_DIRECTIONS

  # Find all air cells that are connected to the starting position
  connected_air_cells = set()
  air_queue = deque()

  if start_cell_value == air_value:
    # If starting from an air cell, start BFS from that cell
    air_queue.append((start_row, start_col))
    connected_air_cells.add((start_row, start_col))
  else:
    # If starting from an interior wall, find air cells adjacent to it
    for dr, dc in directions:
      new_row, new_col = start_row + dr, start_col + dc
      if (
          0 <= new_row < floor_plan.shape[0]
          and 0 <= new_col < floor_plan.shape[1]
          and floor_plan[new_row, new_col] == air_value
      ):
        air_queue.append((new_row, new_col))
        connected_air_cells.add((new_row, new_col))

  # BFS to find all connected air cells (4-connectivity)
  while air_queue:
    current_row, current_col = air_queue.popleft()
    for dr, dc in directions:
      new_row, new_col = current_row + dr, current_col + dc
      if (
          0 <= new_row < floor_plan.shape[0]
          and 0 <= new_col < floor_plan.shape[1]
          and floor_plan[new_row, new_col] == air_value
          and (new_row, new_col) not in connected_air_cells
      ):
        air_queue.append((new_row, new_col))
        connected_air_cells.add((new_row, new_col))

  # Now find all interior walls that are adjacent to
  #  any of the connected air cells (4-connectivity)
  walls_to_mark = set()
  for air_row, air_col in connected_air_cells:
    for dr, dc in directions:
      wall_row, wall_col = air_row + dr, air_col + dc
      if (
          0 <= wall_row < floor_plan.shape[0]
          and 0 <= wall_col < floor_plan.shape[1]
          and floor_plan[wall_row, wall_col] == interior_wall_value
      ):
        walls_to_mark.add((wall_row, wall_col))

  # Mark all the connected interior walls
  # If starting from an interior wall, exclude it from marking
  # (it will be marked separately)
  # If starting from an air cell, mark all walls found
  for wall_row, wall_col in walls_to_mark:
    if start_cell_value == interior_wall_value and (wall_row, wall_col) == (
        start_row,
        start_col,
    ):
      # Skip marking the starting wall here; will mark it below if any walls
      #  were found
      continue
    floor_plan[wall_row, wall_col] = marked_value

  # If starting from an interior wall and any walls were found, mark the
  # starting position
  if start_cell_value == interior_wall_value and walls_to_mark:
    floor_plan[start_row, start_col] = marked_value

  # Create interior space array containing only air and marked walls
  all_interior_positions = connected_air_cells.union(walls_to_mark)
  if not all_interior_positions:
    return floor_plan, None

  min_row = min(pos[0] for pos in all_interior_positions)
  max_row = max(pos[0] for pos in all_interior_positions)
  min_col = min(pos[1] for pos in all_interior_positions)
  max_col = max(pos[1] for pos in all_interior_positions)

  interior_height = max_row - min_row + 1
  interior_width = max_col - min_col + 1
  interior_space = np.full(
      (interior_height, interior_width),
      interior_wall_value,
      dtype=floor_plan.dtype,
  )

  for air_row, air_col in connected_air_cells:
    interior_space[air_row - min_row, air_col - min_col] = air_value

  # Mark all walls in interior space
  # If starting from interior wall, it will be included in walls_to_mark
  # and marked
  for wall_row, wall_col in walls_to_mark:
    interior_space[wall_row - min_row, wall_col - min_col] = marked_value

  return floor_plan, interior_space


def fix_view_factors(
    view_factors: np.ndarray,
    surface_areas: np.ndarray | None = None,
) -> np.ndarray:
  r"""Return view factors corrected for reciprocity and enclosure closure.

  The returned matrix uses the conventional orientation where
  ``view_factors[i, j]`` is the fraction of radiation leaving surface ``i``
  that reaches surface ``j``. It satisfies, within numerical tolerance:

  $$\sum_j F_{ij} = 1$$

  and

  $$A_i F_{ij} = A_j F_{ji}.$$

  The input arrays are never modified. The correction iteratively balances the
  symmetric area-weighted exchange matrix $M_{ij} = A_i F_{ij}$ so that its row
  sums equal the surface-area vector.

  Args:
    view_factors: Approximate square view-factor matrix.
    surface_areas: Positive surface-area vector. Equal areas are assumed when
      omitted.

  Returns:
    A corrected view-factor matrix.

  Raises:
    ValueError: If inputs have invalid shapes, non-finite values, negative view
      factors, or non-positive surface areas.
    RuntimeError: If the matrix cannot be balanced within the iteration limit.
  """
  convergence_tolerance = 1e-8
  max_iterations = 400

  view_factors = np.asarray(view_factors, dtype=float)
  if view_factors.ndim != 2 or view_factors.shape[0] != view_factors.shape[1]:
    raise ValueError('view_factors must be a square matrix.')
  if not np.all(np.isfinite(view_factors)):
    raise ValueError('view_factors must contain only finite values.')
  if np.any(view_factors < 0):
    raise ValueError('view_factors cannot contain negative values.')

  num_surfaces = view_factors.shape[0]
  if surface_areas is None:
    surface_areas = np.ones(num_surfaces)
  else:
    surface_areas = np.asarray(surface_areas, dtype=float)
    if surface_areas.shape != (num_surfaces,):
      raise ValueError('surface_areas must have one value per surface.')
  if not np.all(np.isfinite(surface_areas)) or np.any(surface_areas <= 0):
    raise ValueError('surface_areas must be finite and strictly positive.')
  if num_surfaces == 0:
    return view_factors.copy()

  # M is symmetric when reciprocity holds: M[i, j] = A_i * F[i, j].
  exchange_matrix = surface_areas[:, np.newaxis] * view_factors
  exchange_matrix = 0.5 * (exchange_matrix + exchange_matrix.T)

  # A zero-exchange surface has no feasible closure. Assign self-view before
  # balancing so every row can converge to its corresponding surface area.
  zero_exchange_rows = np.isclose(exchange_matrix.sum(axis=1), 0.0)
  exchange_matrix[zero_exchange_rows, zero_exchange_rows] = surface_areas[
      zero_exchange_rows
  ]

  for _ in range(max_iterations):
    row_sums = exchange_matrix.sum(axis=1)
    if np.any(row_sums <= 0):
      raise RuntimeError('Unable to balance view factors with zero row sums.')

    scale = np.sqrt(surface_areas / row_sums)
    exchange_matrix *= scale[:, np.newaxis] * scale[np.newaxis, :]

    if np.allclose(
        exchange_matrix.sum(axis=1),
        surface_areas,
        rtol=convergence_tolerance,
        atol=convergence_tolerance,
    ):
      corrected_view_factors = exchange_matrix / surface_areas[:, np.newaxis]
      break
  else:
    raise RuntimeError('View-factor correction did not converge.')

  if (
      np.any(corrected_view_factors < -convergence_tolerance)
      or not np.allclose(
          corrected_view_factors.sum(axis=1),
          1.0,
          rtol=convergence_tolerance,
          atol=convergence_tolerance,
      )
      or not np.allclose(
          surface_areas[:, np.newaxis] * corrected_view_factors,
          (surface_areas[:, np.newaxis] * corrected_view_factors).T,
          rtol=convergence_tolerance,
          atol=convergence_tolerance,
      )
  ):
    raise RuntimeError('Corrected view factors failed physical validation.')

  return np.maximum(corrected_view_factors, 0.0)


def get_vf(
    indexed_floor_plan: np.ndarray,
    interior_wall_mask: np.ndarray,
    view_factor_method: str = 'ScriptF',
    marked_value: int = TEMPORARY_MARKED_VALUE,
    interior_mass_mask: np.ndarray | None = None,
    interior_mass_value: int = AIR_IN_LINE_OF_SIGHT,
) -> np.ndarray:
  """
  Calculate view factors between interior walls in the floor plan.

  Args:
      indexed_floor_plan (np.ndarray): 2D array representing the floor plan with
          indexed values.
      view_factor_method (str, optional): Method to use for view factors.
          Defaults to 'ScriptF'. Either "ScriptF" or "CarrollMRT".
      marked_value (int, optional): Value used to mark connected interior walls.
          Only used internally. Defaults to -33.
      interior_mass_mask (Optional[np.ndarray], optional): Mask for interior
          mass nodes. Defaults to None.
      interior_mass_value (int, optional): Value used to represent interior
          mass nodes. Defaults to 9 (`AIR_IN_LINE_OF_SIGHT`).
  Returns:
      View factor matrix where `VF[i,j]` represents the view factor from wall
          `i` to wall `j`.

  """
  if view_factor_method == 'ScriptF':
    if interior_mass_mask is not None:
      interior_wall_mask_all = interior_wall_mask | interior_mass_mask
    else:
      interior_wall_mask_all = interior_wall_mask

    n_interior_wall = np.sum(interior_wall_mask_all)
    interior_wall_tuples = [
        (r, c)
        for r in range(indexed_floor_plan.shape[0])
        for c in range(indexed_floor_plan.shape[1])
        if interior_wall_mask_all[r, c]
    ]

    vf = np.zeros((n_interior_wall, n_interior_wall))

    for i in range(n_interior_wall):
      result_floor_plan, _ = mark_air_connected_interior_walls(
          indexed_floor_plan, interior_wall_tuples[i]
      )
      result_floor_plan = mark_directly_seeing_nodes(
          floor_plan=result_floor_plan, base_node=interior_wall_tuples[i]
      )
      if interior_mass_mask is not None:
        vf_ = 1 / np.sum(
            (result_floor_plan == marked_value)
            | (result_floor_plan == interior_mass_value)
        )
      else:
        vf_ = 1 / np.sum((result_floor_plan == marked_value))

      result_floor_plan_ = np.zeros_like(result_floor_plan).astype('float')

      if interior_mass_mask is not None:
        result_floor_plan_[
            (result_floor_plan == marked_value)
            | (result_floor_plan == interior_mass_value)
        ] = vf_
      else:
        result_floor_plan_[(result_floor_plan == marked_value)] = vf_
      vf[i, :] = result_floor_plan_[interior_wall_mask_all]

  elif view_factor_method == 'CarrollMRT':
    raise NotImplementedError('CarrollMRT view factor method not implemented')
  else:
    raise ValueError(
        f'Invalid view factor method: {view_factor_method}. Either "ScriptF" or'
        ' "CarrollMRT"'
    )

  vf = fix_view_factors(vf)
  return vf


def mark_interior_wall_adjacent_to_air(
    arr: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Marks interior walls that are adjacent to air spaces.

  Creates a boolean mask identifying interior walls that share an edge with an
  air space (value of 0) in the floor plan. Checks for adjacency in four
   directions:   up, down, left, and right.

  Args:
    arr: 2D array representing the floor plan with interior walls marked as
      interior_wall_value and air spaces as 0.
    interior_wall_value: Value used to represent interior walls in the floor
      plan. Defaults to -3 (constants.INTERIOR_WALL_VALUE_IN_FUNCTION).

  Returns:
    Boolean mask array where True indicates an interior wall that is adjacent to
    at least one air space.
  """
  mask_minus_interior_wall = arr == interior_wall_value
  mask_zero = arr == air_value
  # Find -3s that have a 0 neighbor (up/down/left/right)
  contact = np.zeros_like(arr, dtype=bool)
  # up
  contact[1:, :] |= mask_zero[:-1, :] & mask_minus_interior_wall[1:, :]
  # down
  contact[:-1, :] |= mask_zero[1:, :] & mask_minus_interior_wall[:-1, :]
  # left
  contact[:, 1:] |= mask_zero[:, :-1] & mask_minus_interior_wall[:, 1:]
  # right
  contact[:, :-1] |= mask_zero[:, 1:] & mask_minus_interior_wall[:, :-1]
  # Only mark the -3 cells that are adjacent to a 0
  marked = mask_minus_interior_wall & contact
  return marked


def get_line_points(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
  """Generate points where the line crosses integer grid lines.

  This function calculates all intersection points between a line segment and
  the integer grid lines. It handles vertical, horizontal, and diagonal lines
  by finding intersections with both vertical (x = integer) and horizontal
  (y = integer) grid lines.

  Args:
      start: Starting point of the line segment as (x, y) coordinates.
      end: Ending point of the line segment as (x, y) coordinates.

  Returns:
      List of intersection points sorted by distance from the start point.
          Each point is a tuple of (x, y) coordinates as floats.


  """
  x1, y1 = start
  x2, y2 = end

  points = []

  # Handle vertical line case
  if abs(x2 - x1) < 1e-10:  # Vertical line
    min_y, max_y = min(y1, y2), max(y2, y1)
    for y in range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1):
      if min_y <= y <= max_y:
        points.append((x1, float(y)))
  # Handle horizontal line case
  elif abs(y2 - y1) < 1e-10:  # Horizontal line
    min_x, max_x = min(x1, x2), max(x1, x2)
    for x in range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1):
      if min_x <= x <= max_x:
        points.append((float(x), y1))
  else:
    # General case: line has slope
    # Find intersections with vertical grid lines (x = integer)
    min_x, max_x = min(x1, x2), max(x1, x2)
    for x in range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1):
      if min_x <= x <= max_x:
        # Calculate y for this x using line equation
        t = (x - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((float(x), y))

    # Find intersections with horizontal grid lines (y = integer)
    min_y, max_y = min(y1, y2), max(y1, y2)
    for y in range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1):
      if min_y <= y <= max_y:
        # Calculate x for this y using line equation
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        points.append((x, float(y)))

  # Remove duplicates and sort by distance from start
  unique_points = []
  for point in points:
    # Check if this point is already in the list (within tolerance)
    is_duplicate = False
    for existing_point in unique_points:
      if (
          abs(point[0] - existing_point[0]) < 1e-10
          and abs(point[1] - existing_point[1]) < 1e-10
      ):
        is_duplicate = True
        break
    if not is_duplicate:
      unique_points.append(point)

  # Sort by distance from start point
  def distance_from_start(point):
    return (point[0] - x1) ** 2 + (point[1] - y1) ** 2

  unique_points.sort(key=distance_from_start)

  return unique_points


def is_line_blocked(
    floor_plan: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    blocked_value: int = TEMPORARY_BLOCKED_VALUE,
) -> bool:
  """Check if the line between start and end is blocked by walls.

  This function determines if a line of sight between two points is blocked
  by walls in the floor plan. It checks all grid intersections along the line
  and determines if the line is blocked by examining the 4 surrounding grid
  cells at each intersection point.

  Args:
      floor_plan: 2D numpy array representing the floor plan where different
          values represent different types of cells (walls, air, etc.).
      start: Starting point of the line as (x, y) coordinates.
      end: Ending point of the line as (x, y) coordinates.
      interior_wall_value: Value used to represent interior walls in the floor
          plan. Defaults to -3 (from "constants.py").
      marked_value: Value used to represent marked wall nodes. Only used
          internally. Defaults: -33. Only used internally.
      blocked_value: Value used to represent blocked wall nodes. Only used
          internally. Default: -34. Only used internally.


  Returns:
      True if the line is blocked by walls, False if the line of sight is clear.

  Note:
      The function considers a line blocked if all 4 grid cells surrounding
      an intersection point are walls (values -3, -33, or -34).
  """
  line_points = get_line_points(start, end)

  # Skip start and end points for blocking check
  for _, point in enumerate(line_points[1:-1], 1):
    x, y = point

    # Get 4 integer coordinates by rounding up/down
    coords = [
        (math.floor(x), math.floor(y)),
        (math.floor(x), math.ceil(y)),
        (math.ceil(x), math.floor(y)),
        (math.ceil(x), math.ceil(y)),
    ]

    # Check if all 4 coordinates are within bounds and get their values
    coord_values = []
    all_walls = True

    for cx, cy in coords:
      if 0 <= cx < floor_plan.shape[0] and 0 <= cy < floor_plan.shape[1]:
        value = floor_plan[cx, cy]
        coord_values.append(value)
        if (
            value != interior_wall_value
            and value != marked_value
            and value != blocked_value
        ):
          all_walls = False
      else:
        coord_values.append('OUT_OF_BOUNDS')
        all_walls = False

    # If all 4 coordinates are walls, the line is blocked
    if all_walls:
      return True

  return False


def are_neighbors(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
  """Check if two positions are physically neighboring (adjacent).

  This function determines if two grid positions are adjacent to each other
  using 4-connectivity. Two positions are considered neighbors if they are
  within 1 unit distance in both x and y directions, but not the same position.

  Args:
      pos1: First position as (row, col) coordinates.
      pos2: Second position as (row, col) coordinates.

  Returns:
      True if the positions are neighbors, False otherwise.


  """
  dx = abs(pos1[0] - pos2[0])
  dy = abs(pos1[1] - pos2[1])
  return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def mark_directly_seeing_nodes(
    floor_plan: np.ndarray,
    base_node: tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    blocked_value: int = TEMPORARY_BLOCKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Mark nodes that are directly seeing the base node as blocked_value.

  This function identifies and marks wall nodes that have a direct line of sight
  to the base node. It processes all connected wall nodes (marked with
  marked_value) and determines which ones can directly see the base node without
  being blocked by other walls. Additionally, it marks air nodes along unblocked
  lines of sight between wall nodes for interior mass radiative heat transfer.

  When the base node is an air cell, it finds directly seeing nodes among the
  interior walls, but does NOT mark air nodes as `AIR_IN_LINE_OF_SIGHT`.

  Args:
      floor_plan: 2D numpy array representing the floor plan where different
          values represent different types of cells (walls, air, etc.).
      base_node: Position of the base node as (row, col) coordinates. Can be
          either an interior wall node or an air cell.
      interior_wall_value: Value used to represent interior walls in the floor
          plan. Defaults to -3 (from "constants.py").
      marked_value: Value used to represent connected wall nodes that should
          be checked for line of sight. Only used internally. Defaults to -33.
      blocked_value: Value used to mark nodes that cannot directly see the
          base node. Only used internally. Defaults to -34.
      air_value: Value used to represent air spaces in the floor plan.
          Defaults to 0 (from "constants.py").

  Returns:
      Copy of the floor plan with nodes marked according to their visibility
          to the base node. Nodes that cannot see the base node are marked
          with blocked_value, and the base node itself is marked with
          blocked_value + marked_value. When starting from a wall node, air
          nodes along unblocked lines of sight are marked with
          `AIR_IN_LINE_OF_SIGHT` (9).
          When starting from an air node, air nodes are NOT marked.
          Air nodes along blocked lines remain as air_value (0).

  Note:
      - When starting from a wall node: Neighboring wall nodes are automatically
        marked as blocked (no line of sight calculation needed, and no air nodes
        between directly adjacent walls).
      - When starting from an air node: Neighboring wall nodes are NOT marked as
        blocked; they remain as marked_value and can participate in radiative
        transfer.
      - For non-neighboring nodes, the function first checks if the line of
        sight is blocked by walls using is_line_blocked().
      - When starting from a wall node: Air nodes are ONLY marked as 9 along
        lines that are NOT blocked. If a line is blocked, air nodes along that
        line remain as 0 (air_value).
      - When starting from an air node: Air nodes are NOT marked, even along
        unblocked lines.
      - The base node itself is marked with a special value to distinguish it.
      - Value meanings for radiative heat transfer:
        * marked_value (-33): Interior wall nodes connected to the same air
          space (can participate in radiative transfer)
        * blocked_value (-34): Interior wall nodes that cannot see the base node
          (blocked from radiative transfer)
        * blocked_value + marked_value (-67): The starting node itself
        * `AIR_IN_LINE_OF_SIGHT` (9): Air nodes along unblocked line of sight
          between wall nodes (for interior mass radiative transfer)
  """
  floor_plan_copy = floor_plan.copy()
  base_row, base_col = base_node
  base_cell_value = floor_plan_copy[base_row, base_col]
  is_base_air = base_cell_value == air_value

  # Find all marked_value nodes (connected wall nodes)
  connected_nodes = np.where(floor_plan_copy == marked_value)
  connected_positions = list(zip(connected_nodes[0], connected_nodes[1]))

  directly_seeing_count = 0

  for pos in connected_positions:
    row, col = pos

    # Skip if it's the base node itself
    if (row, col) == (base_row, base_col):
      continue
    # Check if not physically neighboring
    is_neighbor = are_neighbors((base_row, base_col), (row, col))

    if is_neighbor:
      # Neighbors are directly adjacent
      # Only mark as blocked if starting from an interior wall node
      # (not when starting from an air node)
      if not is_base_air:
        # When starting from wall, mark neighboring walls as blocked
        # (no air nodes between directly adjacent wall nodes)
        floor_plan_copy[row, col] = blocked_value
      # When starting from air node, leave neighboring walls as marked_value
      # (they can participate in radiative transfer)
    else:
      # Check if line of sight is blocked first
      blocked = is_line_blocked(
          floor_plan_copy,
          (base_row, base_col),
          (row, col),
          interior_wall_value,
          marked_value,
          blocked_value,
      )

      if blocked:
        # Line is blocked, so mark the wall node as blocked
        # and DON'T mark air nodes along this line
        floor_plan_copy[row, col] = blocked_value
        directly_seeing_count += 1
      else:
        # Line is NOT blocked
        # Only mark air nodes along the line if starting from a wall node
        # (not when starting from an air node)
        if not is_base_air:
          line_points = get_line_points(
              (float(base_row), float(base_col)), (float(row), float(col))
          )

          # Mark air nodes along the line (excluding start and end points)
          for point in line_points[1:-1]:
            px, py = point
            # Check all 4 integer coordinates around the floating point
            for cx, cy in [
                (math.floor(px), math.floor(py)),
                (math.floor(px), math.ceil(py)),
                (math.ceil(px), math.floor(py)),
                (math.ceil(px), math.ceil(py)),
            ]:
              if (
                  0 <= cx < floor_plan_copy.shape[0]
                  and 0 <= cy < floor_plan_copy.shape[1]
                  and floor_plan_copy[cx, cy] == air_value
              ):
                floor_plan_copy[cx, cy] = AIR_IN_LINE_OF_SIGHT
        # Wall node is visible (not blocked), so leave it as marked_value (-33)

  # Mark the base node with a special value
  floor_plan_copy[base_row, base_col] = blocked_value + marked_value
  return floor_plan_copy


def _ensure_irradiance_components(
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float | None = None,
    solar_azimuth: float | None = None,
) -> solar_radiation.IrradianceComponents:
  """Normalizes irradiance input to an IrradianceComponents instance.

  Args:
      irradiance_components: Either an IrradianceComponents dataclass or a
          mapping containing irradiance fields.
      solar_zenith: Optional fallback solar zenith angle in degrees. Used
          when the mapping does not contain 'solar_zenith'.
      solar_azimuth: Optional fallback solar azimuth angle in degrees. Used
          when the mapping does not contain 'solar_azimuth'.

  Returns:
      solar_radiation.IrradianceComponents: Irradiance data with guaranteed
      fields and types.

  Raises:
      KeyError: If 'ghi', 'dni', or 'dhi' are absent, or if 'solar_zenith'/
          'solar_azimuth' are absent and no fallback is provided.
  """
  if isinstance(irradiance_components, solar_radiation.IrradianceComponents):
    return irradiance_components

  required_keys = ('ghi', 'dni', 'dhi')
  missing_keys = [
      key for key in required_keys if key not in irradiance_components
  ]
  if missing_keys:
    missing_key_names = ', '.join(sorted(missing_keys))
    raise KeyError(f'Missing irradiance component keys: {missing_key_names}')

  # Use mapping values if available, else fall back to arguments
  sz = irradiance_components.get('solar_zenith', solar_zenith)
  sa = irradiance_components.get('solar_azimuth', solar_azimuth)

  if sz is None or sa is None:
    missing = []
    if sz is None:
      missing.append('solar_zenith')
    if sa is None:
      missing.append('solar_azimuth')
    missing_key_names = ', '.join(sorted(missing))
    raise KeyError(f'Missing irradiance component keys: {missing_key_names}')

  timestamp = irradiance_components.get('timestamp')
  return solar_radiation.IrradianceComponents(
      ghi=float(irradiance_components['ghi']),
      dni=float(irradiance_components['dni']),
      dhi=float(irradiance_components['dhi']),
      solar_zenith=float(sz),
      solar_azimuth=float(sa),
      timestamp=timestamp,
  )


def validate_fenestration_connectivity(
    floor_plan: np.ndarray,
    fenestration_value: int = constants.FENESTRATION_VALUE_IN_FILE_INPUT,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
) -> bool:
  """Validate that fenestration nodes bridge exterior space to interior air."""
  if not np.any(floor_plan == fenestration_value):
    return True

  fenestration_groups = _find_connected_groups(floor_plan, fenestration_value)

  directions = FOUR_CONNECTED_DIRECTIONS
  rows, cols = floor_plan.shape

  for group_name, group_info in fenestration_groups.items():
    group_indices: Sequence[tuple[int, int]] = group_info['indices']

    nodes_adjacent_to_air: set[tuple[int, int]] = set()
    nodes_adjacent_to_exterior: set[tuple[int, int]] = set()
    nodes_surrounded_by_air: list[tuple[int, int]] = []

    for row, col in group_indices:
      adjacent_to_air = False
      adjacent_to_exterior = False
      air_neighbor_count = 0

      for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          adjacent_to_exterior = True
          continue

        neighbor_val = floor_plan[nr, nc]

        if neighbor_val == air_value:
          adjacent_to_air = True
          air_neighbor_count += 1
        elif neighbor_val == exterior_space_value:
          adjacent_to_exterior = True
        elif neighbor_val == interior_wall_value:
          # Interior wall does not affect adjacent_to_air/exterior flags
          continue

      if adjacent_to_air:
        nodes_adjacent_to_air.add((row, col))
      if adjacent_to_exterior:
        nodes_adjacent_to_exterior.add((row, col))

      if air_neighbor_count == 4 and not adjacent_to_exterior:
        nodes_surrounded_by_air.append((row, col))

    if not nodes_adjacent_to_air:
      raise ValueError(
          f'Fenestration group {group_name} is not connected to indoor air. '
          f'Fenestration must border value {air_value} nodes.'
      )

    if not nodes_adjacent_to_exterior:
      raise ValueError(
          f'Fenestration group {group_name} is not exposed to exterior '
          f'(value {exterior_space_value}).'
      )

    if nodes_surrounded_by_air:
      raise ValueError(
          f'Fenestration group {group_name} has fenestration nodes fully '
          f'surrounded by air at positions {nodes_surrounded_by_air}. '
          'Fenestration must bridge exterior and interior.'
      )

    _validate_fenestration_chain_connectivity(
        floor_plan=floor_plan,
        group_name=group_name,
        group_indices=group_indices,
        nodes_adjacent_to_air=nodes_adjacent_to_air,
        nodes_adjacent_to_exterior=nodes_adjacent_to_exterior,
        fenestration_value=fenestration_value,
        air_value=air_value,
    )

  return True


def _validate_fenestration_chain_connectivity(
    floor_plan: np.ndarray,
    group_name: str,
    group_indices: Sequence[tuple[int, int]],
    nodes_adjacent_to_air: set[tuple[int, int]],
    nodes_adjacent_to_exterior: set[tuple[int, int]],
    fenestration_value: int,
    air_value: int,
) -> None:
  """Ensure each fenestration node can reach both exterior and interior."""
  directions = FOUR_CONNECTED_DIRECTIONS
  direction_names = {
      (-1, 0): 'north',
      (1, 0): 'south',
      (0, -1): 'west',
      (0, 1): 'east',
  }
  rows, cols = floor_plan.shape
  group_set = set(group_indices)

  reachable_from_exterior: set[tuple[int, int]] = set()
  queue = deque(nodes_adjacent_to_exterior)
  reachable_from_exterior.update(nodes_adjacent_to_exterior)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_exterior:
        reachable_from_exterior.add((nr, nc))
        queue.append((nr, nc))

  reachable_from_air: set[tuple[int, int]] = set()
  queue = deque(nodes_adjacent_to_air)
  reachable_from_air.update(nodes_adjacent_to_air)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_air:
        reachable_from_air.add((nr, nc))
        queue.append((nr, nc))

  for row, col in group_indices:
    if (row, col) not in reachable_from_exterior:
      raise ValueError(
          f'Fenestration group {group_name} has node {(row, col)} blocked '
          'from exterior exposure.'
      )
    if (row, col) not in reachable_from_air:
      raise ValueError(
          f'Fenestration group {group_name} has node {(row, col)} blocked '
          'from interior air.'
      )

  interior_direction = _determine_interior_direction(
      nodes_adjacent_to_exterior, rows, cols, floor_plan
  )

  if interior_direction is None:
    return

  blocked_by_interior_wall: list[tuple[int, int]] = []
  dr, dc = interior_direction
  for row, col in group_indices:
    nr, nc = row + dr, col + dc
    if (nr, nc) in group_set:
      continue
    if 0 <= nr < rows and 0 <= nc < cols:
      neighbor_val = floor_plan[nr, nc]
      if neighbor_val not in (air_value, fenestration_value):
        blocked_by_interior_wall.append((row, col))
    else:
      blocked_by_interior_wall.append((row, col))

  if blocked_by_interior_wall:
    direction_name = direction_names.get(
        interior_direction, str(interior_direction)
    )
    raise ValueError(
        f'Fenestration group {group_name} has interior-facing nodes blocked '
        f'by non-air cells in the {direction_name} direction: '
        f'{blocked_by_interior_wall}.'
    )


def _determine_interior_direction(
    nodes_adjacent_to_exterior: set[tuple[int, int]],
    rows: int,
    cols: int,
    floor_plan: np.ndarray,
) -> tuple[int, int] | None:
  """Infer the direction pointing from exterior toward interior air."""
  if not nodes_adjacent_to_exterior:
    return None

  for row, col in nodes_adjacent_to_exterior:
    if row == 0:
      return (1, 0)
    if row == rows - 1:
      return (-1, 0)
    if col == 0:
      return (0, 1)
    if col == cols - 1:
      return (0, -1)

  directions = FOUR_CONNECTED_DIRECTIONS
  opposite = {
      (-1, 0): (1, 0),
      (1, 0): (-1, 0),
      (0, -1): (0, 1),
      (0, 1): (0, -1),
  }
  exterior_space_value = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT

  for row, col in nodes_adjacent_to_exterior:
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (
          0 <= nr < rows
          and 0 <= nc < cols
          and floor_plan[nr, nc] == exterior_space_value
      ):
        return opposite[(dr, dc)]
  return None


def _find_connected_groups(
    floor_plan: np.ndarray,
    target_value: int,
) -> dict[str, dict[str, Any]]:
  """Find 4-connected groups of `target_value`cells and return their indices."""
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == target_value and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < floor_plan.shape[0]
                and 0 <= nc < floor_plan.shape[1]
                and floor_plan[nr, nc] == target_value
                and not visited[nr, nc]
            ):
              visited[nr, nc] = True
              queue.append((nr, nc))

        groups[f'group_{group_count}'] = {
            'count': len(group_indices),
            'indices': group_indices,
        }

  return groups


def mark_fenestration_positions(
    floor_plan: np.ndarray,
    fenestration_value: int = constants.FENESTRATION_VALUE_IN_FUNCTION,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
) -> np.ndarray:
  """Classify fenestration nodes into exterior, interior, or in-between."""
  result = floor_plan.copy()
  rows, cols = floor_plan.shape
  directions = FOUR_CONNECTED_DIRECTIONS

  fen_rows, fen_cols = np.where(floor_plan == fenestration_value)
  for row, col in zip(fen_rows, fen_cols):
    adjacent_to_exterior = False
    adjacent_to_air = False
    at_boundary = row == 0 or row == rows - 1 or col == 0 or col == cols - 1

    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if 0 <= nr < rows and 0 <= nc < cols:
        neighbor_val = floor_plan[nr, nc]
        if neighbor_val == exterior_space_value:
          adjacent_to_exterior = True
        elif neighbor_val == air_value:
          adjacent_to_air = True

    if at_boundary or adjacent_to_exterior and not adjacent_to_air:
      result[row, col] = exterior_fenestration
    elif adjacent_to_air and not adjacent_to_exterior:
      result[row, col] = interior_fenestration
    elif adjacent_to_air and adjacent_to_exterior:
      result[row, col] = interior_fenestration
    else:
      result[row, col] = inbetween_fenestration

  return result


def group_fenestrations(
    floor_plan: np.ndarray,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    floor_plan_orientation: float = 0.0,
) -> dict[str, dict[str, Any]]:
  """Group adjacent fenestration nodes and compute surface properties."""
  fenestration_mask = (
      (floor_plan == exterior_fenestration)
      | (floor_plan == interior_fenestration)
      | (floor_plan == inbetween_fenestration)
  )

  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if fenestration_mask[row, col] and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        exterior_count = 0
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          if floor_plan[r, c] == exterior_fenestration:
            exterior_count += 1

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < floor_plan.shape[0]
                and 0 <= nc < floor_plan.shape[1]
                and fenestration_mask[nr, nc]
                and not visited[nr, nc]
            ):
              visited[nr, nc] = True
              queue.append((nr, nc))

        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        exterior_indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          if floor_plan[r, c] == exterior_fenestration:
            exterior_indices_array[r, c] = True

        azimuth = _determine_fenestration_azimuth(
            floor_plan,
            group_indices,
            exterior_space_value,
            exterior_fenestration,
            floor_plan_orientation,
        )

        phi = float(constants.FENESTRATION_TILT_ANGLE)
        phi_rad = math.radians(phi)
        cos_phi = math.cos(phi_rad)

        factor = 0.5 * (1 + cos_phi)
        f_gnd = 0.5 * (1 - cos_phi)
        f_sky = factor * math.sqrt(factor)
        f_air = factor * (1 - math.sqrt(factor))
        beta = math.sqrt(factor)

        groups[f'fenestration_{group_count}'] = {
            'count': len(group_indices),
            'exterior_count': exterior_count,
            'indices': group_indices,
            'indices_array': indices_array,
            'exterior_indices_array': exterior_indices_array,
            'phi': phi,
            'azimuth': float(azimuth),
            'F_gnd': f_gnd,
            'F_sky': f_sky,
            'F_air': f_air,
            'beta': beta,
        }

  return groups


def _determine_fenestration_azimuth(
    floor_plan: np.ndarray,
    group_indices: Sequence[tuple[int, int]],
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration_value: int = constants.EXTERIOR_FENESTRATION_VALUE,
    floor_plan_orientation: float = 0.0,
) -> float:
  """Determine fenestration azimuth from exterior adjacency using atan2.

  The azimuth is inferred from the floor_plan grid using the Cartesian
  coordinate system. The direction vector is computed from exterior
  fenestration nodes toward adjacent exterior space nodes, then the azimuth
  is calculated using atan2.

  The floor_plan grid uses (row, col) coordinates where:
  - row increases downward (South in default orientation)
  - col increases rightward (East in default orientation)

  In the Cartesian mapping:
  - x = col direction (East)
  - y = -row direction (North, since row increases downward)

  Azimuth is measured clockwise from North:
  - 0° = North (up/decreasing row)
  - 90° = East (right/increasing col)
  - 180° = South (down/increasing row)
  - 270° = West (left/decreasing col)

  The computed azimuth is then offset by floor_plan_orientation.

  Args:
    floor_plan: 2D array of the indexed floor plan.
    group_indices: List of (row, col) indices for the fenestration group.
    exterior_space_value: Value representing exterior space in the floor plan.
    exterior_fenestration_value: Value representing exterior fenestration.
    floor_plan_orientation: Compass angle (degrees) of the floor-plan's
      "up" direction. 0/360 = North, 90 = East, 180 = South, 270 = West.

  Returns:
    Azimuth angle in degrees [0, 360).
  """
  rows, cols = floor_plan.shape
  directions = FOUR_CONNECTED_DIRECTIONS

  # Accumulate direction vectors from exterior fenestration toward exterior
  # space in Cartesian coordinates (dx = col direction, dy = -row direction)
  total_dx = 0.0
  total_dy = 0.0
  count = 0

  for row, col in group_indices:
    if floor_plan[row, col] != exterior_fenestration_value:
      continue
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      # Check adjacency to exterior space or array boundary
      if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        # Array boundary counts as exterior
        total_dx += float(dc)
        total_dy += float(-dr)
        count += 1
      elif floor_plan[nr, nc] == exterior_space_value:
        total_dx += float(dc)
        total_dy += float(-dr)
        count += 1

  if count == 0:
    # Fallback: check all nodes (not just exterior fenestration)
    for row, col in group_indices:
      for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          total_dx += float(dc)
          total_dy += float(-dr)
          count += 1
        elif floor_plan[nr, nc] == exterior_space_value:
          total_dx += float(dc)
          total_dy += float(-dr)
          count += 1

  if count == 0:
    # Default to North if no direction can be determined
    return float(floor_plan_orientation) % 360.0

  # Compute azimuth using atan2
  # atan2(x, y) gives angle clockwise from North (y-axis)
  azimuth_rad = math.atan2(total_dx, total_dy)
  azimuth_deg = math.degrees(azimuth_rad)

  # Normalize to [0, 360)
  azimuth_deg = azimuth_deg % 360.0

  # Apply floor_plan_orientation offset
  return (azimuth_deg + float(floor_plan_orientation)) % 360.0


def group_air_nodes(
    floor_plan: np.ndarray,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    fenestration_groups: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
  """Group connected interior air nodes and annotate adjacent fenestrations.

  Finds all connected components of air nodes (using 4-connectivity) and
  records which fenestration groups are adjacent to each air group.

  Args:
    floor_plan: 2D indexed floor plan array.
    air_value: Value representing interior air spaces.
    exterior_fenestration: Value for exterior fenestration nodes.
    interior_fenestration: Value for interior fenestration nodes.
    inbetween_fenestration: Value for in-between fenestration nodes.
    fenestration_groups: Pre-computed fenestration groups (from
      :func:`group_fenestrations`). If provided, each air group will list
      which fenestration groups are adjacent.

  Returns:
    Dictionary mapping group names to group info dicts with keys:
      'count', 'indices', 'indices_array', 'fenestration_groups'.
  """
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  fenestration_index_to_group: dict[tuple[int, int], str] = {}
  if fenestration_groups:
    for group_name, group_info in fenestration_groups.items():
      for idx in group_info['indices']:
        fenestration_index_to_group[idx] = group_name

  fenestration_values = {
      exterior_fenestration,
      interior_fenestration,
      inbetween_fenestration,
  }

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == air_value and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        adjacent_fenestrations: set[str] = set()
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < floor_plan.shape[0] and 0 <= nc < floor_plan.shape[1]:
              neighbor_val = floor_plan[nr, nc]

              if neighbor_val in fenestration_values:
                group_name = fenestration_index_to_group.get((nr, nc))
                if group_name:
                  adjacent_fenestrations.add(group_name)

              if neighbor_val == air_value and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        groups[f'air_{group_count}'] = {
            'count': len(group_indices),
            'indices': group_indices,
            'indices_array': indices_array,
            'fenestration_groups': sorted(adjacent_fenestrations),
        }

  return groups


def calculate_solar_absorbed_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> float:
  """Compute absorbed solar flux per node for a fenestration group."""
  exterior_count = int(fenestration_group.get('exterior_count', 0))
  if exterior_count == 0:
    return 0.0

  irr = _ensure_irradiance_components(
      irradiance_components, solar_zenith, solar_azimuth
  )

  surface_tilt = float(fenestration_group['phi'])
  surface_azimuth = float(fenestration_group['azimuth'])

  g_ts = solar_radiation.calculate_poa_irradiance(
      irr,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  total_absorbed = g_ts * alpha * exterior_count
  total_count = int(fenestration_group.get('count', 0))
  return total_absorbed / total_count if total_count > 0 else 0.0


def net_solar_absorbed_heatflux_fenestration(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> np.ndarray:
  """Return array of absorbed solar heat flux at each fenestration node."""
  q_sol_alpha_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups:
    return q_sol_alpha_array

  for group_info in fenestration_groups.values():
    q_sol_alpha_per_node = calculate_solar_absorbed_for_fenestration_group(
        group_info,
        irradiance_components,
        solar_zenith,
        solar_azimuth,
        alpha,
    )
    indices_array = group_info['indices_array']
    q_sol_alpha_array[indices_array] += q_sol_alpha_per_node

  return q_sol_alpha_array


def calculate_solar_transmitted_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> float:
  """Compute total transmitted solar radiation for a fenestration group."""
  exterior_count = int(fenestration_group.get('exterior_count', 0))
  if exterior_count == 0:
    return 0.0

  irr = _ensure_irradiance_components(
      irradiance_components, solar_zenith, solar_azimuth
  )

  surface_tilt = float(fenestration_group['phi'])
  surface_azimuth = float(fenestration_group['azimuth'])

  g_ts = solar_radiation.calculate_poa_irradiance(
      irr,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  return g_ts * tau * exterior_count


def net_solar_transmitted_heatflux_fenestration(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    air_groups: Mapping[str, Mapping[str, Any]] | None,
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> np.ndarray:
  """Distribute transmitted solar heat flux to air nodes connected to windows"""
  q_sol_tau_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups or not air_groups:
    return q_sol_tau_array

  fenestration_q_sol_tau: dict[str, float] = {}
  for group_name, group_info in fenestration_groups.items():
    fenestration_q_sol_tau[group_name] = (
        calculate_solar_transmitted_for_fenestration_group(
            group_info,
            irradiance_components,
            solar_zenith,
            solar_azimuth,
            tau,
        )
    )

  for air_group_info in air_groups.values():
    connected_fenestrations = air_group_info.get('fenestration_groups', [])
    total_q_sol_tau = sum(
        fenestration_q_sol_tau.get(group, 0.0)
        for group in connected_fenestrations
    )

    if total_q_sol_tau == 0.0:
      continue

    air_count = int(air_group_info['count'])
    q_sol_tau_per_node = total_q_sol_tau / air_count if air_count > 0 else 0.0

    indices_array = air_group_info['indices_array']
    q_sol_tau_array[indices_array] += q_sol_tau_per_node

  return q_sol_tau_array


# ---------------------------------------------------------------------------
# Interior surface adjacency (walls + fenestration)
# ---------------------------------------------------------------------------


def mark_interior_surface_adjacent_to_air(
    floor_plan: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    interior_fenestration_value: int = constants.INTERIOR_FENESTRATION_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Mark interior surfaces (walls and fenestration) adjacent to air.

  Creates a boolean mask identifying interior walls and interior fenestration
  nodes that share an edge with an air space (value 0) in the floor plan.
  Checks adjacency in 4 directions (up, down, left, right).

  Args:
    floor_plan: 2D array representing the indexed floor plan.
    interior_wall_value: Value used to represent interior walls.
    interior_fenestration_value: Value for interior fenestration nodes.
    air_value: Value for interior air spaces.

  Returns:
    Boolean mask where True indicates an interior surface (wall or
    fenestration) that is adjacent to at least one air space.
  """
  mask_surface = (floor_plan == interior_wall_value) | (
      floor_plan == interior_fenestration_value
  )
  mask_air = floor_plan == air_value

  contact = np.zeros_like(floor_plan, dtype=bool)
  # up
  contact[1:, :] |= mask_air[:-1, :] & mask_surface[1:, :]
  # down
  contact[:-1, :] |= mask_air[1:, :] & mask_surface[:-1, :]
  # left
  contact[:, 1:] |= mask_air[:, :-1] & mask_surface[:, 1:]
  # right
  contact[:, :-1] |= mask_air[:, 1:] & mask_surface[:, :-1]

  return mask_surface & contact


# ---------------------------------------------------------------------------
# Exterior longwave radiation (LWR) for fenestration
# ---------------------------------------------------------------------------


def calculate_exterior_lwr_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> float:
  """Compute net exterior longwave radiative heat flux for a fenestration group.

  Calculates the net longwave radiation exchange between the fenestration
  surface and the sky, ground, and surrounding air using:

    q_lwr = epsilon * sigma * (F_sky*(T_sky^4 - T_s^4) +
                               F_gnd*(T_air^4 - T_s^4) +
                               F_air*(T_air^4 - T_s^4))

  The result is the net heat flux INTO the surface (positive means surface
  gains heat).

  Args:
    fenestration_group: Dictionary containing fenestration group properties
      including 'F_sky', 'F_gnd', 'F_air', 'exterior_indices_array'.
    surface_temperatures: 2D array of surface temperatures in K.
    emissivity_array: 2D array of surface emissivities.
    ambient_temperature: Outdoor dry-bulb temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    Net LWR heat flux in W/m² (positive = surface gains heat).
  """
  sigma = constants.STEFAN_BOLTZMANN_CONSTANT
  f_sky = float(fenestration_group['F_sky'])
  f_gnd = float(fenestration_group['F_gnd'])
  f_air = float(fenestration_group['F_air'])

  exterior_mask = fenestration_group['exterior_indices_array']

  # Average surface temperature and emissivity over exterior fenestration nodes
  if np.any(exterior_mask):
    t_surf = np.mean(surface_temperatures[exterior_mask])
    epsilon = np.mean(emissivity_array[exterior_mask])
  else:
    return 0.0

  t_sky4 = sky_temperature**4
  t_air4 = ambient_temperature**4
  t_surf4 = t_surf**4

  q_lwr = (
      epsilon
      * sigma
      * (
          f_sky * (t_sky4 - t_surf4)
          + f_gnd * (t_air4 - t_surf4)
          + f_air * (t_air4 - t_surf4)
      )
  )

  return float(q_lwr)


def net_exterior_radiative_heatflux(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> np.ndarray:
  """Compute net exterior LWR heat flux array for all fenestration nodes.

  For each fenestration group, calculates the net longwave radiation exchange
  and assigns the result uniformly to all nodes in the group.

  Args:
    floor_plan: 2D indexed floor plan array.
    fenestration_groups: Dictionary of fenestration group properties.
    surface_temperatures: 2D array of surface temperatures in K.
    emissivity_array: 2D array of surface emissivities.
    ambient_temperature: Outdoor dry-bulb temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    2D array of net LWR heat flux at each grid position.
  """
  q_lwr_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups:
    return q_lwr_array

  for group_info in fenestration_groups.values():
    q_lwr = calculate_exterior_lwr_for_fenestration_group(
        group_info,
        surface_temperatures,
        emissivity_array,
        ambient_temperature,
        sky_temperature,
    )
    indices_array = group_info['indices_array']
    q_lwr_array[indices_array] = q_lwr

  return q_lwr_array


# ---------------------------------------------------------------------------
# Exterior wall boundary mask
# ---------------------------------------------------------------------------


def get_exterior_wall_boundary_mask(
    floor_plan: np.ndarray,
    wall_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
) -> np.ndarray:
  """Identify exterior wall nodes that form the building boundary.

  A wall node is considered an exterior wall boundary if:
  1. It has the specified wall_value.
  2. It is NOT adjacent to an enclosed interior air space.

  Walls adjacent to enclosed interior air (courtyard, rooms) are excluded
  because they face inward. Walls at the array boundary or adjacent to
  exterior_space_value are included.

  Interior walls (value 1) are treated as solid material and do NOT cause
  exclusion of adjacent exterior walls.

  Args:
    floor_plan: 2D array of the floor plan (can use FILE_INPUT or FUNCTION
      values).
    wall_value: The value representing exterior walls.
    exterior_space_value: The value representing exterior space.
    air_value: The value representing interior air spaces.

  Returns:
    Boolean mask where True indicates an exterior wall boundary node.
  """
  rows, cols = floor_plan.shape
  wall_mask = floor_plan == wall_value

  if not np.any(wall_mask):
    return np.zeros_like(floor_plan, dtype=bool)

  # Find all enclosed interior air regions using flood-fill from edges
  # Air connected to edges or exterior space is NOT enclosed interior air
  air_mask = floor_plan == air_value
  if not np.any(air_mask):
    # No interior air, all walls are exterior boundaries
    return wall_mask

  # Find enclosed air: air NOT connected to exterior
  # Use flood fill from exterior space or from array boundary
  visited_air = np.zeros_like(floor_plan, dtype=bool)
  exterior_mask = floor_plan == exterior_space_value

  # BFS from all exterior space nodes and boundary nodes
  queue = deque()

  # Add all exterior space nodes to start BFS
  for r in range(rows):
    for c in range(cols):
      if exterior_mask[r, c]:
        queue.append((r, c))
        visited_air[r, c] = True

  # Also check air at array boundaries (connected to outside)
  for r in range(rows):
    for c in range(cols):
      if (
          air_mask[r, c]
          and not visited_air[r, c]
          and (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
      ):
        queue.append((r, c))
        visited_air[r, c] = True

  directions = FOUR_CONNECTED_DIRECTIONS
  while queue:
    r, c = queue.popleft()
    for dr, dc in directions:
      nr, nc = r + dr, c + dc
      if 0 <= nr < rows and 0 <= nc < cols and not visited_air[nr, nc]:
        if air_mask[nr, nc] or exterior_mask[nr, nc]:
          visited_air[nr, nc] = True
          queue.append((nr, nc))

  # Enclosed interior air = air nodes NOT visited (not connected to exterior)
  enclosed_air = air_mask & ~visited_air

  # Walls adjacent to enclosed air should be excluded UNLESS they are also
  # at the array boundary or adjacent to exterior space
  adjacent_to_enclosed = np.zeros_like(floor_plan, dtype=bool)
  for dr, dc in directions:
    shifted = np.zeros_like(floor_plan, dtype=bool)
    if dr == -1:
      shifted[:-1, :] = enclosed_air[1:, :]
    elif dr == 1:
      shifted[1:, :] = enclosed_air[:-1, :]
    elif dc == -1:
      shifted[:, :-1] = enclosed_air[:, 1:]
    elif dc == 1:
      shifted[:, 1:] = enclosed_air[:, :-1]
    adjacent_to_enclosed |= shifted

  # Determine which walls are at the array boundary
  at_boundary = np.zeros_like(floor_plan, dtype=bool)
  at_boundary[0, :] = True
  at_boundary[-1, :] = True
  at_boundary[:, 0] = True
  at_boundary[:, -1] = True

  # Determine which walls are adjacent to exterior space
  adjacent_to_exterior = np.zeros_like(floor_plan, dtype=bool)
  for dr, dc in directions:
    shifted = np.zeros_like(floor_plan, dtype=bool)
    if dr == -1:
      shifted[:-1, :] = exterior_mask[1:, :]
    elif dr == 1:
      shifted[1:, :] = exterior_mask[:-1, :]
    elif dc == -1:
      shifted[:, :-1] = exterior_mask[:, 1:]
    elif dc == 1:
      shifted[:, 1:] = exterior_mask[:, :-1]
    adjacent_to_exterior |= shifted

  # A wall faces outward if it's at boundary or adjacent to exterior space
  faces_outward = at_boundary | adjacent_to_exterior

  # Exterior wall boundary = wall AND(faces outward OR NOT adjacent to enclosed)
  # i.e., exclude walls ONLY adjacent to enclosed air that don't face outward
  return wall_mask & (faces_outward | ~adjacent_to_enclosed)


# ---------------------------------------------------------------------------
# Exterior wall azimuth determination
# ---------------------------------------------------------------------------


def determine_exterior_wall_azimuth_array(
    exterior_wall_boundary_mask: np.ndarray,
    floor_plan: np.ndarray,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    floor_plan_orientation: float = 0.0,
) -> np.ndarray:
  """Determine azimuth for each exterior wall boundary node.

  For each exterior wall boundary node, computes the outward-facing direction
  (toward exterior space) and converts it to an azimuth angle using atan2
  in the Cartesian coordinate system.

  The grid coordinate system:
  - row increases downward (South in default orientation)
  - col increases rightward (East in default orientation)

  Cartesian mapping:
  - x = col direction (East)
  - y = -row direction (North)

  Azimuth (clockwise from North):
  - 0° = North (decreasing row)
  - 90° = East (increasing col)
  - 180° = South (increasing row)
  - 270° = West (decreasing col)

  For corner nodes with multiple adjacent exterior space directions, the
  resulting azimuth is the average direction (e.g. top-right = 45°).

  Args:
    exterior_wall_boundary_mask: Boolean mask of exterior wall nodes.
    floor_plan: 2D indexed floor plan array.
    exterior_space_value: Value representing exterior space.
    floor_plan_orientation: Compass angle (degrees) of the floor-plan's
      "up" direction. 0/360 = North.

  Returns:
    2D array of azimuth angles in degrees. Non-wall positions are 0.0.
  """
  rows, cols = floor_plan.shape
  azimuth_array = np.zeros((rows, cols), dtype=float)
  directions = FOUR_CONNECTED_DIRECTIONS

  for r in range(rows):
    for c in range(cols):
      if not exterior_wall_boundary_mask[r, c]:
        continue

      total_dx = 0.0
      total_dy = 0.0

      for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          # Array boundary counts as exterior
          total_dx += float(dc)
          total_dy += float(-dr)
        elif floor_plan[nr, nc] == exterior_space_value:
          total_dx += float(dc)
          total_dy += float(-dr)

      if total_dx == 0.0 and total_dy == 0.0:
        continue

      # atan2(x, y) gives clockwise angle from North (y-axis)
      azimuth_rad = math.atan2(total_dx, total_dy)
      azimuth_deg = math.degrees(azimuth_rad) % 360.0

      # Apply floor_plan_orientation offset
      azimuth_array[r, c] = (azimuth_deg + floor_plan_orientation) % 360.0

  return azimuth_array
