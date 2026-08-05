"""Building Radiation Utility Functions

For computing the physical and thermal characteristics of buildings.
"""

from collections import deque
import math
from typing import Optional, Tuple

import numpy as np

from smart_control.simulator import constants
from smart_control.simulator.solar_radiation import calculate_poa_irradiance  # re-export  # pylint: disable=unused-import
from smart_control.simulator.solar_radiation import IrradianceComponents  # re-export  # pylint: disable=unused-import

TEMPORARY_MARKED_VALUE = -33
TEMPORARY_BLOCKED_VALUE = -34
AIR_IN_LINE_OF_SIGHT = 9  # Air nodes along line of sight between wall nodes

# pylint: disable=invalid-name


def calculate_a_tilde_inv(epsilon: np.ndarray, F: np.ndarray) -> np.ndarray:
  """Calculates the A-tilde matrix used in radiative heat transfer calculations.

  The A-tilde matrix relates the radiosity to the blackbody emissive power in a
  radiative heat transfer system. It accounts for both emission and reflection.

  Args:
      epsilon: Array of surface emissivity values (between 0 and 1)
      F: View factor matrix

  Returns:
      The A-tilde matrix relating radiosity to blackbody emissive power

  Raises:
      AssertionError: If emissivity vector size doesn't match view factor matrix
          or if emissivity values are outside [0,1]
  """
  n = epsilon.shape[0]
  epsilon[epsilon == 0] = 1e-10

  A = np.eye(n)
  I = np.eye(n)
  for i in range(n):
    for j in range(n):
      A[i, j] = (I[i, j] - (1 - epsilon[i]) * F[i, j]) / epsilon[i]
  return np.linalg.inv(A)


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
    \delta_{ij} - \frac{(1-\epsilon_i) F_{ij}}{\epsilon_i}$.

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
    ($\tilde{A}_{ij} = \delta_{ij} - \frac{(1-\epsilon_i) F_{ij}}{\epsilon_i}$)
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
    start_pos: Tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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


def fix_view_factors(F: np.ndarray, A: np.ndarray = None) -> np.ndarray:
  """
  Fix approximate view factors and enforce reciprocity and completeness.

  Args:
      F (np.ndarray): Approximate direct view factor matrix (N x N)
      A (np.ndarray, optional): Area vector (N elements). Defaults to None.

  Returns:
      Fixed view factor matrix

  References:
      See `FixViewFactors` function in [EnergyPlus](https://github.com/NREL/EnergyPlus/blob/develop/src/EnergyPlus/HeatBalanceIntRadExchange.cc) # pylint: disable=line-too-long
  """

  # Parameter definitions
  PRIMARY_CONVERGENCE = 0.001
  DIFFERENCE_CONVERGENCE = 0.00001
  MAX_ITERATIONS = 400

  # Convert inputs to numpy arrays
  if A is None:
    A = np.ones(F.shape[0])

  # F = np.array(F, dtype=np.float64)
  F = F.T  # since EP calculation is based on F[j,i]
  N = F.shape[0]

  # Initialize return values
  results = {
      'original_check_value': 0.0,
      'fixed_check_value': 0.0,
      'final_check_value': 0.0,
      'num_iterations': 0,
      'row_sum': 0.0,
      'enforced_reciprocity': False,
  }

  # OriginalCheckValue is the first pass at a completeness check
  results['original_check_value'] = abs(np.sum(F) - N)

  # Allocate and initialize arrays
  FixedAF = F.copy()  # store for largest area check

  ConvrgOld = 10.0
  LargestArea = np.max(A)
  severe_error_present = False
  largest_surf = -1

  # Check for Strange Geometry
  # When one surface has an area that exceeds the sum of all other surface areas
  if LargestArea > 0.99 * (np.sum(A) - LargestArea) and N > 3:
    for i in range(N):
      if LargestArea == A[i]:
        largest_surf = i
        break

    if largest_surf >= 0:
      # Give self view to big surface
      FixedAF[largest_surf, largest_surf] = min(
          0.9, 1.2 * LargestArea / np.sum(A)
      )

  # Set up AF matrix (AREA * DIRECT VIEW FACTOR) MATRIX
  AF = np.zeros((N, N))
  for i in range(N):
    for j in range(N):
      AF[j, i] = FixedAF[j, i] * A[i]

  # Enforce reciprocity by averaging AiFij and AjFji
  FixedAF = 0.5 * (AF + AF.T)

  FixedF = np.zeros((N, N))
  results['num_iterations'] = 0
  results['row_sum'] = 0.0

  # Check for physically unreasonable enclosures (N <= 3)
  if N <= 3:
    for i in range(N):
      for j in range(N):
        if A[i] != 0:
          FixedF[j, i] = FixedAF[j, i] / A[i]

    results['row_sum'] = np.sum(FixedF)

    if results['row_sum'] > (N + 0.01):
      # Find the largest row summation and normalize
      sum_FixedF = np.sum(FixedF, axis=1)  # Sum along rows
      MaxFixedFRowSum = np.max(sum_FixedF)

      if MaxFixedFRowSum < 1.0:
        raise RuntimeError(
            'FixViewFactors: Three surface or less zone failing ViewFactorFix'
            ' correction which should never happen.'
        )
      else:
        FixedF *= 1.0 / MaxFixedFRowSum

      results['row_sum'] = np.sum(FixedF)  # Recalculate

    results['final_check_value'] = results['fixed_check_value'] = abs(
        results['row_sum'] - N
    )
    F[:] = FixedF  # Update F in place
    results['enforced_reciprocity'] = True
    return results

  # Regular fix cases (N > 3)
  RowCoefficient = np.zeros(N)
  Converged = False

  while not Converged:
    results['num_iterations'] += 1

    for i in range(N):
      # Determine row coefficients which will enforce closure
      sum_FixedAF_i = np.sum(FixedAF[:, i])
      if abs(sum_FixedAF_i) > 1.0e-10:
        RowCoefficient[i] = A[i] / sum_FixedAF_i
      else:
        RowCoefficient[i] = 1.0

      FixedAF[:, i] *= RowCoefficient[i]

    # Enforce reciprocity by averaging AiFij and AjFji
    FixedAF = 0.5 * (FixedAF + FixedAF.T)

    # Form FixedF matrix
    for i in range(N):
      for j in range(N):
        if A[i] != 0:
          FixedF[j, i] = FixedAF[j, i] / A[i]
          if abs(FixedF[j, i]) < 1.0e-10:
            FixedF[j, i] = 0.0
            FixedAF[j, i] = 0.0

    ConvrgNew = abs(np.sum(FixedF) - N)

    # Check convergence
    if (
        abs(ConvrgOld - ConvrgNew) < DIFFERENCE_CONVERGENCE
        or ConvrgNew <= PRIMARY_CONVERGENCE
    ):
      Converged = True

    ConvrgOld = ConvrgNew

    # Emergency exit after too many iterations
    if results['num_iterations'] > MAX_ITERATIONS:
      # Enforce reciprocity by averaging AiFij and AjFji
      FixedAF = 0.5 * (FixedAF + FixedAF.T)

      # Form FixedF matrix
      for i in range(N):
        for j in range(N):
          if A[i] != 0:
            FixedF[j, i] = FixedAF[j, i] / A[i]

      sum_FixedF = np.sum(FixedF)
      results['final_check_value'] = results['fixed_check_value'] = (
          CheckConvergeTolerance
      ) = abs(sum_FixedF - N)
      results['row_sum'] = sum_FixedF

      # pylint:disable=line-too-long
      if CheckConvergeTolerance > 0.005:
        if CheckConvergeTolerance > 0.1:
          pass
        pass
      # pylint:enable=line-too-long

      if abs(results['fixed_check_value']) < abs(
          results['original_check_value']
      ):
        F[:] = FixedF
        results['final_check_value'] = results['fixed_check_value']

      return results

  # Normal completion
  results['fixed_check_value'] = ConvrgNew

  if results['fixed_check_value'] < results['original_check_value']:
    F[:] = FixedF
    results['final_check_value'] = results['fixed_check_value']
  else:
    results['final_check_value'] = results['original_check_value']
    results['row_sum'] = np.sum(FixedF)

    if abs(results['row_sum'] - N) < PRIMARY_CONVERGENCE:
      F[:] = FixedF
      results['final_check_value'] = results['fixed_check_value']
    else:
      pass

  if severe_error_present:
    raise RuntimeError(
        'FixViewFactors: View factor calculations significantly out of'
        ' tolerance. See above messages for more information.'
    )

  F = F.T
  return F


def get_vf(
    indexed_floor_plan: np.ndarray,
    interior_wall_mask: np.ndarray,
    view_factor_method: str = 'ScriptF',
    marked_value: int = TEMPORARY_MARKED_VALUE,
    interior_mass_mask: Optional[np.ndarray] = None,
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
    start: Tuple[float, float], end: Tuple[float, float]
) -> list[Tuple[float, float]]:
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
    start: Tuple[float, float],
    end: Tuple[float, float],
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


def are_neighbors(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
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
    base_node: Tuple[int, int],
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
