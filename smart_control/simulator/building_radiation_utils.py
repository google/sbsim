"""Building Radiation Utility Functions

For computing the physical and thermal characteristics of buildings.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

from smart_control.simulator import constants

# we are choosing to keep the mathematical notation names for the functions and
# variables in this file
# pylint: disable=invalid-name


def calculate_A_tilde_inv(epsilon: np.ndarray, F: np.ndarray) -> np.ndarray:
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


def calculate_IFAinv(F: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
  """
  Calculates the $IFA_{inv}$ matrix.

  Main equation:

  $$IFA_{inv} = (I - F) @ A_{inv}$$

  Where:

    + $q=(I-F)@J$
    + $J=A_{inv}@E_b$
    + $E_b=sigma*T^4$

  So:

  $$q=sigma*(I-F)@A_{inv}@T^4$$

  Args:
      F (np.ndarray): The F matrix.
      A_inv (np.ndarray): The A inverse matrix.

  Returns:
      IFA_inv : The IFA inverse matrix.
  """

  n = F.shape[0]

  I = np.eye(n)
  IFAinv = (I - F) @ A_inv
  return IFAinv


def net_radiative_heatflux_function_of_T(
    T: np.ndarray, IFAinv: np.ndarray
) -> np.array:
  """
  Calculates the net radiative heat flux and radiosity for given surface
    temperatures.

  Args:
      T (np.ndarray): Surface temperatures in Celsius.
      IFAinv (np.ndarray): (I - F) @ A_inv.

  Returns:
      q : Net radiative heat flux [W/m^2]

  """
  sigma = 5.67 * 1e-8  # [W/m^2K^4] Stefan-Boltzmann constant

  q = sigma * IFAinv @ np.power(T, 4)
  return q


def mark_air_connected_interior_walls(
    indexed_floor_plan: np.ndarray,
    start_pos: Tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = -33,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """
  Mark all interior wall nodes that are connected to the same air space as the
      starting interior wall.
  Uses 8-directional connectivity (including diagonals) to check wall-air
      adjacency.
  All connected walls including the starting position are marked.

  Args:
    indexed_floor_plan (np.ndarray): 2D numpy array representing the floor plan
        where different values represent different types of cells (walls, air,
        etc.).
    start_pos (Tuple[int, int]): Starting position (row, col) of the interior
        wall to begin marking from.
    interior_wall_value (int, optional): Value used to represent interior walls
        in the floor plan. Defaults to -3 (from constants.py).
    marked_value (int, optional): Value used to mark connected interior walls.
        Defaults to -33.
    air_value (int, optional): Value used to represent air spaces in the floor
        plan. Defaults to 0 (from constants.py).

  Returns:
    A tuple containing:

      - `modified_floor_plan`: Copy of input floor plan with connected walls
          marked with marked_value. `None` if `start_pos` is invalid.

      - `interior_space_array`: Extracted interior space containing only air and
          marked walls, cropped to the bounding box of the connected region.
          `None` if `start_pos` is invalid or no interior space is found.

  Raises:
      ValueError: If the starting position is out of bounds of the floor plan.
  """
  # Make a copy to avoid modifying the original
  floor_plan = indexed_floor_plan.copy()

  # Check if starting position is valid
  if (
      start_pos[0] < 0
      or start_pos[0] >= floor_plan.shape[0]
      or start_pos[1] < 0
      or start_pos[1] >= floor_plan.shape[1]
  ):
    raise ValueError('Starting position is out of bounds')
  if floor_plan[start_pos[0], start_pos[1]] != interior_wall_value:
    return None, None

  # Directions for 4-connectivity (up, down, left, right)
  # - for air-to-air connections
  air_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  # Directions for 8-connectivity (including diagonals)
  # - for wall-air adjacency
  wall_air_directions = [
      (-1, -1),
      (-1, 0),
      (-1, 1),
      (0, -1),
      (0, 1),
      (1, -1),
      (1, 0),
      (1, 1),
  ]

  start_row, start_col = start_pos

  # Find all air cells that are connected to the starting wall
  # (using 8-connectivity)
  connected_air_cells = set()
  air_queue = deque()

  # Add all air cells adjacent to starting wall (including diagonals)
  for dr, dc in wall_air_directions:
    new_row, new_col = start_row + dr, start_col + dc
    if (
        0 <= new_row < floor_plan.shape[0]
        and 0 <= new_col < floor_plan.shape[1]
        and floor_plan[new_row, new_col] == air_value
    ):
      air_queue.append((new_row, new_col))
      connected_air_cells.add((new_row, new_col))

  # BFS to find all connected air cells (using 4-connectivity for air-to-air)
  while air_queue:
    current_row, current_col = air_queue.popleft()

    # Check all neighbors (4-directional for air connectivity)
    for dr, dc in air_directions:
      new_row, new_col = current_row + dr, current_col + dc

      # Skip if out of bounds
      if (
          new_row < 0
          or new_row >= floor_plan.shape[0]
          or new_col < 0
          or new_col >= floor_plan.shape[1]
      ):
        continue

      # If neighbor is air and not yet visited
      if (
          floor_plan[new_row, new_col] == air_value
          and (new_row, new_col) not in connected_air_cells
      ):
        air_queue.append((new_row, new_col))
        connected_air_cells.add((new_row, new_col))

  # Now find all interior walls that are adjacent to any of the connected air
  # cells (using 8-connectivity)
  walls_to_mark = set()

  for air_row, air_col in connected_air_cells:
    for dr, dc in wall_air_directions:
      wall_row, wall_col = air_row + dr, air_col + dc

      # Skip if out of bounds
      if (
          wall_row < 0
          or wall_row >= floor_plan.shape[0]
          or wall_col < 0
          or wall_col >= floor_plan.shape[1]
      ):
        continue

      # If it's an interior wall, mark it
      if floor_plan[wall_row, wall_col] == interior_wall_value:
        walls_to_mark.add((wall_row, wall_col))

  # Mark all the connected interior walls INCLUDING the starting position
  for wall_row, wall_col in walls_to_mark:
    if wall_row == start_row and wall_col == start_col:
      pass
    else:
      floor_plan[wall_row, wall_col] = marked_value

  # Create interior space array containing only air and marked walls
  # Find bounding box of the interior space
  all_interior_positions = connected_air_cells.union(walls_to_mark)

  if not all_interior_positions:
    return floor_plan, None

  min_row = min(pos[0] for pos in all_interior_positions)
  max_row = max(pos[0] for pos in all_interior_positions)
  min_col = min(pos[1] for pos in all_interior_positions)
  max_col = max(pos[1] for pos in all_interior_positions)

  # Extract the interior space
  interior_height = max_row - min_row + 1
  interior_width = max_col - min_col + 1
  interior_space = np.full(
      (interior_height, interior_width),
      interior_wall_value,
      dtype=floor_plan.dtype,
  )  # Use -999 as background

  # Copy air cells and marked walls to interior space
  for air_row, air_col in connected_air_cells:
    interior_space[air_row - min_row, air_col - min_col] = air_value

  for wall_row, wall_col in walls_to_mark:
    if wall_row == start_row and wall_col == start_col:
      pass
    else:
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
      See `FixViewFactors` function in [EnergyPlus](https://github.com/NREL/EnergyPlus/blob/develop/src/EnergyPlus/HeatBalanceIntRadExchange.cc)
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
          # warnings.warn(f"FixViewFactors: View factors convergence has failed and will lead to heat balance errors in zone=\"{encl_name}\".")
          pass

        # warnings.warn(f"FixViewFactors: View factors not complete. Check for bad surface descriptions or unenclosed zone=\"{encl_name}\".")
        # warnings.warn(f"Enforced reciprocity has tolerance (ideal is 0)=[{CheckConvergeTolerance:.6f}], Row Sum (ideal is {N})=[{results['row_sum']:.2f}].")
        # warnings.warn("If zone is unusual or tolerance is on the order of 0.001, view factors might be OK but results should be checked carefully.")
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


def get_VF(
    indexed_floor_plan: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = -33,
) -> np.ndarray:
  """
  Calculate view factors between interior walls in the floor plan.

  Args:
      indexed_floor_plan (np.ndarray): 2D array representing the floor plan with
          indexed values.
      interior_wall_value (int, optional): Value representing interior walls.
          Defaults to -3 (`constants.INTERIOR_WALL_VALUE_IN_FUNCTION`).
      marked_value (int, optional): Value to mark connected walls. Defaults to
          -33.

  Returns:
      View factor matrix where `VF[i,j]` represents the view factor from wall
          `i` to wall `j`.

  """
  # TODO: how to handle for non typical.. or no interior walls?
  interior_wall_mask = indexed_floor_plan == interior_wall_value
  n_interior_wall = np.sum(interior_wall_mask)
  VF = np.zeros((n_interior_wall, n_interior_wall))
  interior_wall_idx = [
      (r, c)
      for r in range(indexed_floor_plan.shape[0])
      for c in range(indexed_floor_plan.shape[1])
      if indexed_floor_plan[r, c] == interior_wall_value
  ]

  for i in range(n_interior_wall):
    result_floor_plan, _ = mark_air_connected_interior_walls(
        indexed_floor_plan, interior_wall_idx[i]
    )
    # for now, the view factor is just 1/# of seen surfaces.
    vf_ = 1 / np.sum(result_floor_plan == marked_value)

    result_floor_plan_ = result_floor_plan.copy().astype('float')
    result_floor_plan_[result_floor_plan_ == interior_wall_value] = 0
    result_floor_plan_[result_floor_plan_ == marked_value] = vf_
    VF[i, :] = result_floor_plan_[interior_wall_mask]

  VF = fix_view_factors(VF)
  return VF
