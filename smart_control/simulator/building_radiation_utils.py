"""Building Radiation Utility Functions

For computing the physical and thermal characteristics of buildings.
"""

from collections import deque
import math
from typing import Optional, Tuple

import numpy as np

from smart_control.simulator import constants

TEMPORARY_MARKED_VALUE = -33
TEMPORARY_BLOCKED_VALUE = -34
AIR_IN_LINE_OF_SIGHT = 9  # Air nodes along line of sight between wall nodes

# we are choosing to keep the mathematical notation in this file
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
  sigma = 5.67 * 1e-8  # [W/m^2K^4] Stefan-Boltzmann constant

  q = sigma * ifa_inv @ np.power(T, 4)
  return q


def calculate_exterior_lwr_for_fenestration_group(
    fenestration_group: dict,
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> float:
  r"""Calculate exterior longwave radiative heat flux for a fenestration group.

  Calculates the net longwave radiative heat flux between exterior fenestration
  surfaces and the sky/ground environment. The calculation considers:
  - Radiation exchange with ground at ambient temperature
  - Radiation exchange with sky (split between sky temperature and ambient)
  - Emission from the fenestration surface

  Equations:
  ----------
  For each exterior fenestration node, the net radiative heat flux is:

  $$q_{\text{lwr}} = \epsilon \sigma \left( F_{\text{gnd}} T_{\text{air}}^4
    + \beta F_{\text{sky}} T_{\text{sky}}^4
    + (1 - \beta) F_{\text{sky}} T_{\text{air}}^4 \right)
    - \epsilon \sigma \left( F_{\text{gnd}} + \beta F_{\text{sky}}
    + (1 - \beta) F_{\text{sky}} \right)
    T_{\text{surf}}^4$$

  where:
  - $\beta = \sqrt{0.5 (1 + \cos\phi)}$ splits sky radiation between sky
    temperature and ambient air temperature
  - $\phi$ is the tilt angle (90° for vertical surfaces)

  The total q_lwr for the group is summed over all exterior fenestration nodes,
  then distributed equally to all nodes in the group.

  Nomenclature and Units:
  -----------------------
  - $q_{\text{lwr}}$: Net exterior longwave radiative heat flux [W/m$^2$]
  - $\epsilon$: Surface emissivity [dimensionless]
  - $\sigma$: Stefan-Boltzmann constant [$\mathrm{W/(m^2 \cdot K^4)}$]
  - $F_{\text{gnd}}$: Ground view factor [dimensionless]
  - $F_{\text{sky}}$: Sky view factor [dimensionless]
  - $\beta$: Sky radiation split factor [dimensionless]
  - $T_{\text{air}}$: Ambient air temperature [K]
  - $T_{\text{sky}}$: Sky temperature [K]
  - $T_{\text{surf}}$: Surface temperature [K]
  - $\phi$: Surface tilt angle [degrees]

  Args:
    fenestration_group: Dictionary containing fenestration group properties
      including 'exterior_indices_array', 'indices_array', 'F_gnd', 'F_sky',
      'beta', and 'count'.
    surface_temperatures: 2D array of surface temperatures in K (floor_plan
      shape).
    emissivity_array: 2D array of surface emissivities (floor_plan shape).
    ambient_temperature: Ambient air temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    Average q_lwr per node in the fenestration group [W/m^2]. Positive value
    means heat gain to the surface.
  """
  sigma = 5.67e-8  # [W/m^2K^4] Stefan-Boltzmann constant

  exterior_mask = fenestration_group['exterior_indices_array']
  exterior_count = fenestration_group['exterior_count']
  total_count = fenestration_group['count']

  # If no exterior fenestration nodes, no LWR calculation needed
  if exterior_count == 0:
    return 0.0

  F_gnd = fenestration_group['F_gnd']
  F_sky = fenestration_group['F_sky']
  beta = fenestration_group['beta']

  # Get temperatures and emissivities for exterior fenestration nodes
  T_surf = surface_temperatures[exterior_mask]
  epsilon = emissivity_array[exterior_mask]

  T_air = ambient_temperature
  T_sky = sky_temperature

  # Incoming radiation from environment
  # $T_{gnd} \approx T_{air}$
  # q_in = epsilon * sigma * (F_gnd * T_air^4 + beta * F_sky * T_sky^4
  #        + (1-beta) * F_sky * T_air^4)
  q_in = (
      epsilon
      * sigma
      * (
          F_gnd * np.power(T_air, 4)
          + beta * F_sky * np.power(T_sky, 4)
          + (1 - beta) * F_sky * np.power(T_air, 4)
      )
  )

  # Outgoing radiation from surface
  # q_out = sigma * epsilon * (F_gnd + F_sky) * T_surf^4
  q_out = sigma * epsilon * (F_gnd + F_sky) * np.power(T_surf, 4)

  # Net heat flux (positive = heat gain to surface)
  q_lwr_nodes = q_in - q_out

  # Sum over all exterior fenestration nodes and distribute equally to all nodes
  total_q_lwr = np.sum(q_lwr_nodes)
  q_lwr_per_node = total_q_lwr / total_count

  return q_lwr_per_node


def net_exterior_radiative_heatflux(
    floor_plan: np.ndarray,
    fenestration_groups: dict,
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> np.ndarray:
  r"""Calculate net exterior longwave radiative heat flux for all fenestrations.

  Calculates the exterior longwave radiative heat flux (LWR) for all
  fenestration groups and returns an array with the same shape as floor_plan.

  For each fenestration group:
  1. Calculate q_lwr for exterior fenestration nodes using the formula
  2. Distribute equally to all nodes in the group
  3. Add to the output array at the group's indices

  Equations:
  ----------
  See `calculate_exterior_lwr_for_fenestration_group` for the detailed
  calculation.

  Args:
    floor_plan: 2D array representing the building floor plan.
    fenestration_groups: Dictionary of fenestration groups from
      `group_fenestrations`.
    surface_temperatures: 2D array of surface temperatures in K (floor_plan
      shape).
    emissivity_array: 2D array of surface emissivities (floor_plan shape).
    ambient_temperature: Ambient air temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    2D array (floor_plan shape) with net exterior LWR heat flux [W/m^2] at
    each fenestration position. Non-fenestration positions are 0.
  """
  q_lwr = np.zeros_like(floor_plan, dtype=float)

  # If no fenestration groups, return zero array
  if not fenestration_groups:
    return q_lwr

  for group_data in fenestration_groups.values():
    # Calculate q_lwr for this group
    q_lwr_per_node = calculate_exterior_lwr_for_fenestration_group(
        fenestration_group=group_data,
        surface_temperatures=surface_temperatures,
        emissivity_array=emissivity_array,
        ambient_temperature=ambient_temperature,
        sky_temperature=sky_temperature,
    )

    # Add q_lwr to all nodes in the group
    indices_array = group_data['indices_array']
    q_lwr[indices_array] = q_lwr_per_node

  return q_lwr


def get_exterior_wall_boundary_mask(
    floor_plan: np.ndarray,
    exterior_wall_value: int = constants.EXTERIOR_WALL_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Identify exterior wall nodes at the array boundary.

  Exterior walls at the boundary are the outermost layer of the building
  that are exposed to outdoor environment for LWR and solar radiation.

  TODO : we may need different algorithm when floor_plan has ambient nodes.

  Args:
    floor_plan: 2D array representing the indexed floor plan.
    exterior_wall_value: Value representing exterior walls (-2 by default).

  Returns:
    Boolean mask array with True at exterior wall positions that are at the
    array boundary (row 0, last row, col 0, last col).
  """
  rows, cols = floor_plan.shape
  mask = np.zeros_like(floor_plan, dtype=bool)

  # Find exterior wall positions
  exterior_wall_positions = floor_plan == exterior_wall_value

  # Mark only those at the boundary
  # Row 0 (top boundary)
  mask[0, :] = exterior_wall_positions[0, :]
  # Last row (bottom boundary)
  mask[rows - 1, :] = exterior_wall_positions[rows - 1, :]
  # Col 0 (left boundary)
  mask[:, 0] = exterior_wall_positions[:, 0]
  # Last col (right boundary)
  mask[:, cols - 1] = exterior_wall_positions[:, cols - 1]

  return mask


def calculate_exterior_lwr_for_exterior_wall(
    exterior_wall_boundary_mask: np.ndarray,
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
    phi: float = 90.0,
) -> np.ndarray:
  r"""Calculate exterior LWR heat flux for boundary exterior walls.

  Calculates the net longwave radiative heat flux between exterior wall
  surfaces at the boundary and the sky/ground environment. Currently, all nodes
  are assumed to be vertical surfaces (i.e., F_gnd, F_sky, and F_air are all
  the same).

  Equations:
  ----------
  For each exterior wall node at boundary, the net radiative heat flux is:

  $$q_{\text{lwr}} = \epsilon \sigma \left( F_{\text{gnd}} T_{\text{air}}^4
    + \beta F_{\text{sky}} T_{\text{sky}}^4
    + (1 - \beta) F_{\text{sky}} T_{\text{air}}^4 \right)
    - \epsilon \sigma \left( F_{\text{gnd}} + F_{\text{sky}} \right)
    T_{\text{surf}}^4$$

  where:
  - $\beta = \sqrt{0.5 (1 + \cos\phi)}$ splits sky radiation between sky
    temperature and ambient air temperature
  - $\phi$ is the tilt angle (90° for vertical surfaces)

  Args:
    exterior_wall_boundary_mask: Boolean mask for exterior walls at boundary.
    surface_temperatures: 2D array of surface temperatures in K.
    emissivity_array: 2D array of surface emissivities.
    ambient_temperature: Ambient air temperature in K.
    sky_temperature: Sky temperature in K.
    phi: Surface tilt angle in degrees. Defaults to 90 (vertical).

  Returns:
    2D array with net exterior LWR heat flux [W/m^2] at boundary exterior wall
    positions. Non-boundary positions are 0. Positive = heat gain.
  """
  sigma = 5.67e-8  # [W/m^2K^4] Stefan-Boltzmann constant

  q_lwr = np.zeros_like(surface_temperatures, dtype=float)

  if not np.any(exterior_wall_boundary_mask):
    return q_lwr

  # Calculate view factors for vertical surface (phi = 90 degrees)
  phi_rad = math.radians(phi)
  cos_phi = math.cos(phi_rad)

  F_gnd = 0.5 * (1 - cos_phi)  # ~0.5 for vertical
  factor = 0.5 * (1 + cos_phi)  # ~0.5 for vertical
  F_sky = factor * math.sqrt(factor)  # ~0.354 for vertical
  # F_air = factor * (1 - math.sqrt(factor))  # ~0.146 for vertical
  beta = math.sqrt(factor)  # sky radiation split factor

  # Get temperatures and emissivities for boundary exterior walls
  T_surf = surface_temperatures[exterior_wall_boundary_mask]
  epsilon = emissivity_array[exterior_wall_boundary_mask]

  T_air = ambient_temperature
  T_sky = sky_temperature

  # Incoming radiation from environment
  q_in = (
      epsilon
      * sigma
      * (
          F_gnd * np.power(T_air, 4)
          + beta * F_sky * np.power(T_sky, 4)
          + (1 - beta) * F_sky * np.power(T_air, 4)
      )
  )

  # Outgoing radiation from surface
  q_out = sigma * epsilon * (F_gnd + F_sky) * np.power(T_surf, 4)

  # Net heat flux (positive = heat gain to surface)
  q_lwr_nodes = q_in - q_out

  # Assign to output array
  q_lwr[exterior_wall_boundary_mask] = q_lwr_nodes

  return q_lwr


def calculate_solar_absorbed_for_exterior_wall(
    exterior_wall_boundary_mask: np.ndarray,
    floor_plan: np.ndarray,
    irradiance_components: dict[str, float],
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float,
    phi: float = 90.0,
) -> np.ndarray:
  r"""Calculate absorbed solar radiation for boundary exterior walls.

  For exterior walls at the boundary, calculates absorbed solar radiation
  based on the wall orientation (determined by which boundary it's on).

  Args:
    exterior_wall_boundary_mask: Boolean mask for exterior walls at boundary.
    floor_plan: 2D array representing the indexed floor plan.
    irradiance_components: Dictionary with 'ghi', 'dni', 'dhi' keys.
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.
    alpha: Solar absorptance of the wall surface (0-1).
    phi: Surface tilt angle in degrees. Defaults to 90 (vertical).

  Returns:
    2D array with absorbed solar heat flux [W/m^2] at boundary exterior wall
    positions. Non-boundary positions are 0.
  """
  q_sol_alpha = np.zeros_like(floor_plan, dtype=float)

  if not np.any(exterior_wall_boundary_mask):
    return q_sol_alpha

  rows, cols = floor_plan.shape

  # Process each boundary separately (different azimuth for each)
  boundaries = [
      (0, slice(None), constants.FENESTRATION_AZIMUTH_TOP),  # Top row -> North
      (rows - 1, slice(None), constants.FENESTRATION_AZIMUTH_BOTTOM),  # Bottom
      (slice(None), 0, constants.FENESTRATION_AZIMUTH_LEFT),  # Left col -> West
      (slice(None), cols - 1, constants.FENESTRATION_AZIMUTH_RIGHT),  # Right
  ]

  for row_idx, col_idx, azimuth in boundaries:
    # Get mask for this boundary
    boundary_mask = np.zeros_like(floor_plan, dtype=bool)
    boundary_mask[row_idx, col_idx] = True
    wall_at_boundary = exterior_wall_boundary_mask & boundary_mask

    if not np.any(wall_at_boundary):
      continue

    # Calculate POA irradiance for this orientation
    g_ts = calculate_poa_irradiance(
        irradiance_components,
        surface_tilt=phi,
        surface_azimuth=float(azimuth),
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    # Absorbed solar radiation = G_Ts * alpha
    q_sol_alpha[wall_at_boundary] = g_ts * alpha

  return q_sol_alpha


def mark_air_connected_interior_walls(
    indexed_floor_plan: np.ndarray,
    start_pos: Tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    interior_fenestration_value: int = constants.INTERIOR_FENESTRATION_VALUE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """
  Mark all interior surface nodes (walls and fenestration) that are connected
      to the same air space as the starting position.
  Uses 4-directional connectivity to check surface-air adjacency.
  All connected surfaces are marked.

  Args:
    indexed_floor_plan (np.ndarray): 2D numpy array representing the floor plan
        where different values represent different types of cells (walls, air,
        etc.).
    start_pos (Tuple[int, int]): Starting position (row, col). Can be either an
        interior wall, interior fenestration, or an air cell. If it's an
        interior surface, finds all surfaces connected to the same air space.
        If it's an air cell, finds all surfaces connected to that air space.
    interior_wall_value (int, optional): Value used to represent interior walls
        in the floor plan. Defaults to -3 (from "constants.py").
    marked_value (int, optional): Value used to mark connected interior
        surfaces. Only used internally. Defaults to -33.
    air_value (int, optional): Value used to represent air spaces in the floor
        plan. Defaults to 0 (from "constants.py").
    interior_fenestration_value (int, optional): Value used to represent
        interior fenestration in the floor plan. Defaults to -43.

  Returns:
    A tuple containing:

      - `modified_floor_plan`: Copy of input floor plan with connected surfaces
          marked with marked_value. `None` if `start_pos` is invalid.

      - `interior_space_array`: Extracted interior space containing only air and
          marked surfaces, cropped to the bounding box of the connected region.
          `None` if `start_pos` is invalid or no interior space is found.

  Raises:
      ValueError: If the starting position is out of bounds of the floor plan.

  Note:
      This function is used as the first step in radiative heat transfer
      calculations to identify all interior surface nodes (walls and
      fenestration) that are connected to the same air space. The marked_value
      (-33) indicates surfaces that can potentially participate in radiative
      heat transfer with each other.
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

  # Interior surface values include walls and fenestration
  interior_surface_values = {interior_wall_value, interior_fenestration_value}

  # Return None if start_pos is not an interior surface or air
  if (
      start_cell_value not in interior_surface_values
      and start_cell_value != air_value
  ):
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
    # If starting from an interior surface, find air cells adjacent to it
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

  # Now find all interior surfaces (walls and fenestration) that are adjacent
  # to any of the connected air cells (4-connectivity)
  surfaces_to_mark = set()
  for air_row, air_col in connected_air_cells:
    for dr, dc in directions:
      surface_row, surface_col = air_row + dr, air_col + dc
      if (
          0 <= surface_row < floor_plan.shape[0]
          and 0 <= surface_col < floor_plan.shape[1]
          and floor_plan[surface_row, surface_col] in interior_surface_values
      ):
        surfaces_to_mark.add((surface_row, surface_col))

  # Mark all the connected interior surfaces
  # If starting from an interior surface, exclude it from marking
  # (it will be marked separately)
  # If starting from an air cell, mark all surfaces found
  for surface_row, surface_col in surfaces_to_mark:
    if start_cell_value in interior_surface_values and (
        surface_row,
        surface_col,
    ) == (start_row, start_col):
      # Skip marking the starting surface here; will mark it below if any
      # surfaces were found
      continue
    floor_plan[surface_row, surface_col] = marked_value

  # If starting from an interior surface and any surfaces were found, mark the
  # starting position
  if start_cell_value in interior_surface_values and surfaces_to_mark:
    floor_plan[start_row, start_col] = marked_value

  # Create interior space array containing only air and marked surfaces
  all_interior_positions = connected_air_cells.union(surfaces_to_mark)
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

  # Mark all surfaces in interior space
  # If starting from interior surface, it will be included in surfaces_to_mark
  # and marked
  for surface_row, surface_col in surfaces_to_mark:
    interior_space[surface_row - min_row, surface_col - min_col] = marked_value

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


def mark_interior_surface_adjacent_to_air(
    arr: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    interior_fenestration_value: int = constants.INTERIOR_FENESTRATION_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Marks interior surfaces (walls and fenestration) adjacent to air spaces.

  Creates a boolean mask identifying interior walls (-3) and interior
  fenestration (-43) that share an edge with an air space (value of 0) in
  the floor plan. Checks for adjacency in four directions: up, down, left,
  and right.

  Args:
    arr: 2D array representing the floor plan with interior walls marked as
      interior_wall_value, interior fenestration as interior_fenestration_value,
      and air spaces as air_value.
    interior_wall_value: Value used to represent interior walls in the floor
      plan. Defaults to -3 (constants.INTERIOR_WALL_VALUE_IN_FUNCTION).
    interior_fenestration_value: Value used to represent interior fenestration
      in the floor plan. Default: -43 (constants.INTERIOR_FENESTRATION_VALUE).
    air_value: Value used to represent air spaces. Defaults to 0.

  Returns:
    Boolean mask array where True indicates an interior surface (wall or
    fenestration) that is adjacent to at least one air space.
  """
  # Interior surfaces include both interior walls and interior fenestration
  mask_interior_surface = (arr == interior_wall_value) | (
      arr == interior_fenestration_value
  )
  mask_air = arr == air_value

  # Find interior surfaces that have an air neighbor (up/down/left/right)
  contact = np.zeros_like(arr, dtype=bool)
  # up
  contact[1:, :] |= mask_air[:-1, :] & mask_interior_surface[1:, :]
  # down
  contact[:-1, :] |= mask_air[1:, :] & mask_interior_surface[:-1, :]
  # left
  contact[:, 1:] |= mask_air[:, :-1] & mask_interior_surface[:, 1:]
  # right
  contact[:, :-1] |= mask_air[:, 1:] & mask_interior_surface[:, :-1]

  # Only mark the interior surface cells that are adjacent to air
  marked = mask_interior_surface & contact
  return marked


# Keep the old function name as an alias for backward compatibility
mark_interior_wall_adjacent_to_air = mark_interior_surface_adjacent_to_air


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


def validate_fenestration_connectivity(
    floor_plan: np.ndarray,
    fenestration_value: int = constants.FENESTRATION_VALUE_IN_FILE_INPUT,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    exterior_wall_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
) -> bool:
  """Validate that fenestration nodes are properly connected.

  Validates that:
  1. All fenestration nodes form connected groups (4-way connectivity)
  2. Each fenestration group is connected to at least one indoor air node (0)
  3. Each fenestration group is exposed to exterior (at boundary or adjacent to
     exterior wall)
  4. Each fenestration node in a group can reach both exterior and air through
     the fenestration chain (no blocked nodes)
  5. No fenestration node is surrounded only by air (floating in interior)

  Args:
    floor_plan: 2D array representing the floor plan with fenestration marked
        as fenestration_value (default 4).
    fenestration_value: Value used to represent fenestration nodes in the
        floor plan. Defaults to 4.
    air_value: Value used to represent indoor air in the floor plan.
        Defaults to 0.
    exterior_wall_value: Value used to represent exterior walls. Defaults to 2.
    interior_wall_value: Value used to represent interior walls. Defaults to 1.

  Returns:
    True if all fenestration groups are valid.

  Raises:
    ValueError: If any fenestration group fails validation.
  """
  if not np.any(floor_plan == fenestration_value):
    return True  # No fenestration, validation passes

  # Find all fenestration groups using flood fill
  fenestration_groups = _find_connected_groups(floor_plan, fenestration_value)

  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  rows, cols = floor_plan.shape

  for group_name, group_info in fenestration_groups.items():
    group_indices = group_info['indices']

    # Track which fenestration nodes are connected to air and exterior
    nodes_adjacent_to_air = set()
    nodes_adjacent_to_exterior = set()
    nodes_surrounded_by_air = []

    for row, col in group_indices:
      adjacent_to_air = False
      adjacent_to_exterior = False
      air_neighbor_count = 0

      for dr, dc in directions:
        nr, nc = row + dr, col + dc

        # Check if fenestration node is at boundary (exposed to exterior)
        # This means the fenestration itself touches the building envelope
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          adjacent_to_exterior = True
          continue

        neighbor_val = floor_plan[nr, nc]

        if neighbor_val == air_value:
          adjacent_to_air = True
          air_neighbor_count += 1
        elif neighbor_val == exterior_wall_value:
          # Adjacent to exterior wall - this is NOT sufficient for exterior
          # exposure. Fenestration must be AT the boundary itself.
          # Being adjacent to exterior wall just means it's near the wall,
          # not that it's actually exposed to outdoors.
          # (fenestration behind exterior wall is NOT exposed to outdoor)
          pass

      if adjacent_to_air:
        nodes_adjacent_to_air.add((row, col))
      if adjacent_to_exterior:
        nodes_adjacent_to_exterior.add((row, col))

      # Check if fenestration is surrounded only by air (floating in interior)
      # A fenestration surrounded by air has 4 air neighbors and no exterior
      if air_neighbor_count == 4 and not adjacent_to_exterior:
        nodes_surrounded_by_air.append((row, col))

    # Validation 1: Group must be connected to indoor air
    if not nodes_adjacent_to_air:
      raise ValueError(
          f'Fenestration group {group_name} is not connected to indoor air. '
          'All fenestration nodes must be adjacent (4-way) to at least one '
          f'indoor air node (value {air_value}).'
      )

    # Validation 2: Group must be exposed to exterior
    if not nodes_adjacent_to_exterior:
      raise ValueError(
          f'Fenestration group {group_name} is not exposed to exterior. '
          'Fenestration must be adjacent to exterior wall or at the boundary '
          'of the floor plan.'
      )

    # Validation 3: Check for fenestration nodes surrounded by air
    if nodes_surrounded_by_air:
      raise ValueError(
          f'Fenestration group {group_name} has nodes surrounded by air at '
          f'positions {nodes_surrounded_by_air}. Fenestration cannot be '
          'floating in interior space - it must connect exterior to interior.'
      )

    # Validation 4: Check that each fenestration node can reach both exterior
    # and air through the fenestration chain
    # Use BFS to check reachability from exterior nodes to air nodes
    _validate_fenestration_chain_connectivity(
        floor_plan=floor_plan,
        group_name=group_name,
        group_indices=group_indices,
        nodes_adjacent_to_air=nodes_adjacent_to_air,
        nodes_adjacent_to_exterior=nodes_adjacent_to_exterior,
        fenestration_value=fenestration_value,
        air_value=air_value,
        interior_wall_value=interior_wall_value,
    )

  return True


def _validate_fenestration_chain_connectivity(
    floor_plan: np.ndarray,
    group_name: str,
    group_indices: list,
    nodes_adjacent_to_air: set,
    nodes_adjacent_to_exterior: set,
    fenestration_value: int,
    air_value: int,
    interior_wall_value: int,
) -> None:
  """Validate that fenestration nodes form a proper chain from exterior to air.

  Each fenestration node must be reachable from nodes adjacent to exterior
  through other fenestration nodes, eventually reaching nodes adjacent to air.
  This detects cases where interior walls block the fenestration chain.

  Additionally validates that:
  - Interior-facing edge nodes (at the edge towards air) are adjacent to air,
    not interior wall
  - No fenestration node is blocked by interior wall on its air-facing side

  Args:
    floor_plan: 2D array representing the floor plan.
    group_name: Name of the fenestration group for error messages.
    group_indices: List of (row, col) tuples for all nodes in the group.
    nodes_adjacent_to_air: Set of (row, col) tuples for nodes adjacent to air.
    nodes_adjacent_to_exterior: Set of (row, col) tuples for nodes adjacent to
        exterior.
    fenestration_value: Value used to represent fenestration nodes.
    air_value: Value used to represent indoor air.
    interior_wall_value: Value used to represent interior walls.

  Raises:
    ValueError: If any fenestration node is blocked from reaching air or
        exterior through the fenestration chain.
  """
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  direction_names = {
      (-1, 0): 'north',
      (1, 0): 'south',
      (0, -1): 'west',
      (0, 1): 'east',
  }
  rows, cols = floor_plan.shape
  group_set = set(group_indices)

  # BFS from exterior nodes to find all reachable fenestration nodes
  reachable_from_exterior = set()
  queue = deque(nodes_adjacent_to_exterior)
  reachable_from_exterior.update(nodes_adjacent_to_exterior)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_exterior:
        reachable_from_exterior.add((nr, nc))
        queue.append((nr, nc))

  # BFS from air-adjacent nodes to find all reachable fenestration nodes
  reachable_from_air = set()
  queue = deque(nodes_adjacent_to_air)
  reachable_from_air.update(nodes_adjacent_to_air)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_air:
        reachable_from_air.add((nr, nc))
        queue.append((nr, nc))

  # Check for nodes that can't reach either exterior or air
  for row, col in group_indices:
    if (row, col) not in reachable_from_exterior:
      raise ValueError(
          f'Fenestration group {group_name} has a node at ({row}, {col}) that '
          'is blocked from exterior. Each fenestration node must be reachable '
          'from exterior through the fenestration chain.'
      )

    if (row, col) not in reachable_from_air:
      raise ValueError(
          f'Fenestration group {group_name} has a node at ({row}, {col}) that '
          'is blocked from indoor air. Each fenestration node must be '
          'reachable from air through the fenestration chain.'
      )

  # Determine the interior direction (direction from exterior towards air)
  # based on where the exterior-adjacent nodes are located
  interior_direction = _determine_interior_direction(
      nodes_adjacent_to_exterior, rows, cols, floor_plan
  )

  if interior_direction is not None:
    # Find interior-facing edge nodes: nodes with no fenestration neighbor
    # in the interior direction
    interior_facing_edge_nodes = []
    for row, col in group_indices:
      dr, dc = interior_direction
      nr, nc = row + dr, col + dc
      # Check if the interior-direction neighbor is NOT fenestration
      if (nr, nc) not in group_set:
        interior_facing_edge_nodes.append((row, col))

    # Check that interior-facing edge nodes are adjacent to air, not interior
    # wall
    blocked_by_interior_wall = []
    for row, col in interior_facing_edge_nodes:
      dr, dc = interior_direction
      nr, nc = row + dr, col + dc

      # Check what's in the interior direction
      if 0 <= nr < rows and 0 <= nc < cols:
        neighbor_val = floor_plan[nr, nc]
        if neighbor_val == interior_wall_value:
          # Interior wall is blocking this fenestration from air
          blocked_by_interior_wall.append((row, col))
        elif neighbor_val != air_value and neighbor_val != fenestration_value:
          # Something other than air or fenestration in interior direction
          # This might also be a problem
          blocked_by_interior_wall.append((row, col))

    if blocked_by_interior_wall:
      dir_name = direction_names.get(
          interior_direction, str(interior_direction)
      )
      raise ValueError(
          f'Fenestration group {group_name} has nodes blocked by interior '
          f'wall at positions {blocked_by_interior_wall}. These nodes are at '
          f'the interior-facing edge (direction: {dir_name}) but are blocked '
          'by interior wall instead of connecting to indoor air. Fenestration '
          'must have a clear path to indoor air on its interior side.'
      )


def _determine_interior_direction(
    nodes_adjacent_to_exterior: set,
    rows: int,
    cols: int,
    floor_plan: np.ndarray,
) -> tuple:
  """Determine the direction from exterior towards interior for a fenestration.

  Based on where the exterior-adjacent nodes are located, determines which
  direction points towards the interior (air).

  Args:
    nodes_adjacent_to_exterior: Set of (row, col) tuples for nodes adjacent to
        exterior.
    rows: Number of rows in floor plan.
    cols: Number of columns in floor plan.
    floor_plan: 2D array representing the floor plan.

  Returns:
    Tuple (dr, dc) representing the direction towards interior, or None if
    cannot be determined.
  """
  if not nodes_adjacent_to_exterior:
    return None

  # Check if exterior is at array boundary
  for row, col in nodes_adjacent_to_exterior:
    if row == 0:
      return (1, 0)  # Exterior at top, interior is south
    if row == rows - 1:
      return (-1, 0)  # Exterior at bottom, interior is north
    if col == 0:
      return (0, 1)  # Exterior at left, interior is east
    if col == cols - 1:
      return (0, -1)  # Exterior at right, interior is west

  # Check which direction has exterior wall neighbors
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  opposite = {
      (-1, 0): (1, 0),
      (1, 0): (-1, 0),
      (0, -1): (0, 1),
      (0, 1): (0, -1),
  }
  exterior_wall_value = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT

  for row, col in nodes_adjacent_to_exterior:
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (
          0 <= nr < rows
          and 0 <= nc < cols
          and floor_plan[nr, nc] == exterior_wall_value
      ):
        # Exterior wall is in this direction, interior is opposite
        return opposite[(dr, dc)]

  return None


def _find_connected_groups(
    floor_plan: np.ndarray,
    target_value: int,
) -> dict:
  """Find all connected groups of nodes with the target value.

  Uses 4-way connectivity (not diagonal) to identify connected components.

  Args:
    floor_plan: 2D array representing the floor plan.
    target_value: Value to search for connected groups.

  Returns:
    Dictionary with group names as keys and group info as values:
    {
      'group_1': {
        'count': int,
        'indices': list of (row, col) tuples
      },
      ...
    }
  """
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups = {}
  group_count = 0
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == target_value and not visited[row, col]:
        # Start BFS for this group
        group_count += 1
        group_indices = []
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
  """Mark fenestration nodes based on their position (exterior/interior/between)

  Exterior fenestration nodes (-42): At array boundary OR adjacent to exterior
      space (value -1). This is the outermost layer of the fenestration.
  Interior fenestration nodes (-43): Adjacent to indoor air (value 0). This is
      the innermost layer of the fenestration facing the room.
  In-between fenestration nodes (-425): Neither exterior nor interior adjacent.
      These are middle layers of thick fenestration (e.g., multi-pane windows).

  The classification priority is:
  1. At array boundary -> exterior (-42)
  2. Adjacent to exterior space (-1) -> exterior (-42)
  3. Adjacent to indoor air (0) -> interior (-43)
  4. Neither -> in-between (-425)

  Args:
    floor_plan: 2D array with fenestration marked as fenestration_value.
    fenestration_value: Current fenestration marker value. Defaults to -4.
    exterior_space_value: Value for exterior space. Defaults to -1.
    air_value: Value for indoor air. Defaults to 0.
    exterior_fenestration: Value to mark exterior fenestration. Defaults to -42.
    interior_fenestration: Value to mark interior fenestration. Defaults to -43.
    inbetween_fenestration: Value for in-between fenestration. Defaults to -425.

  Returns:
    Copy of floor plan with fenestration nodes marked by position.
  """
  result = floor_plan.copy()
  rows, cols = floor_plan.shape
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  fenestration_positions = np.where(floor_plan == fenestration_value)

  for row, col in zip(fenestration_positions[0], fenestration_positions[1]):
    adjacent_to_exterior = False
    adjacent_to_air = False
    at_boundary = False

    # Check if fenestration is at the array boundary (edge/corner)
    # Fenestration at boundary is always exterior fenestration
    if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
      at_boundary = True

    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if 0 <= nr < rows and 0 <= nc < cols:
        neighbor_val = floor_plan[nr, nc]
        if neighbor_val == exterior_space_value:
          adjacent_to_exterior = True
        elif neighbor_val == air_value:
          adjacent_to_air = True

    # Determine fenestration position type
    # Priority:
    # 1. At array boundary -> exterior (-42) regardless of other adjacencies
    # 2. Adjacent to both exterior and air -> interior (-43) for single-layer
    # 3. Adjacent to exterior only -> exterior (-42)
    # 4. Adjacent to air only -> interior (-43)
    # 5. Neither -> in-between (-425)
    if at_boundary:
      # At array boundary -> always exterior fenestration
      result[row, col] = exterior_fenestration
    elif adjacent_to_air and adjacent_to_exterior:
      # Both adjacent (single-layer fenestration) -> mark as interior
      result[row, col] = interior_fenestration
    elif adjacent_to_exterior:
      # Only exterior adjacent -> exterior fenestration
      result[row, col] = exterior_fenestration
    elif adjacent_to_air:
      # Only air adjacent -> interior fenestration
      result[row, col] = interior_fenestration
    else:
      # Neither -> in-between fenestration
      result[row, col] = inbetween_fenestration

  return result


def group_fenestrations(
    floor_plan: np.ndarray,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> dict:
  """Group adjacent fenestration nodes and calculate their properties.

  Groups fenestration nodes that are 4-way adjacent and calculates:
  - count: Number of fenestration nodes in the group
  - exterior_count: Number of exterior fenestration nodes (-42) in the group
  - indices: List of (row, col) tuples for all nodes
  - indices_array: Boolean array (floor_plan shape) with True at group positions
  - phi: Tilt angle (always 90 degrees for vertical surfaces)
  - azimuth: Direction the fenestration faces:
      - 0 (top/north), 90 (right/east), 180 (bottom/south), 270 (left/west)
  - F_gnd: Ground view factor
  - F_sky: Sky view factor
  - F_air: Ambient air view factor

  View factor formulas (for vertical surface, phi = 90 degrees):
  - F_gnd = 0.5 * (1 - cos(phi))
  - F_sky = 0.5 * (1 + cos(phi)) * sqrt(0.5 * (1 + cos(phi)))
  - F_air = 0.5 * (1 + cos(phi)) * (1 - sqrt(0.5 * (1 + cos(phi))))

  Args:
    floor_plan: 2D array with marked fenestration positions.
    exterior_fenestration: Value for exterior fenestration (-42).
    interior_fenestration: Value for interior fenestration (-43).
    inbetween_fenestration: Value for in-between fenestration (-425).
    exterior_space_value: Value for exterior space (-1).

  Returns:
    Dictionary with fenestration group information:
    {
      'fenestration_1': {
        'count': int,
        'exterior_count': int,
        'indices': list of (row, col) tuples,
        'indices_array': np.ndarray of bool (floor_plan shape),
        'exterior_indices_array': np.ndarray of bool (floor_plan shape),
          True only at exterior fenestration (-42) positions,
        'phi': 90.0,
        'azimuth': float (0, 90, 180, or 270),
        'F_gnd': float,
        'F_sky': float,
        'F_air': float,
        'beta': float, sqrt(0.5 * (1 + cos(phi))), sky radiation split factor,
      },
      ...
    }
  """
  # Find all fenestration nodes (all three types)
  fenestration_mask = (
      (floor_plan == exterior_fenestration)
      | (floor_plan == interior_fenestration)
      | (floor_plan == inbetween_fenestration)
  )

  visited = np.zeros_like(floor_plan, dtype=bool)
  groups = {}
  group_count = 0
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if fenestration_mask[row, col] and not visited[row, col]:
        group_count += 1
        group_indices = []
        exterior_count = 0
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          # Count exterior fenestration nodes
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

        # Create indices_array: boolean array with True at group positions
        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        # Determine azimuth based on exterior space direction
        azimuth = _determine_fenestration_azimuth(
            floor_plan, group_indices, exterior_space_value
        )

        # Create exterior_indices_array: boolean array with True at exterior
        # fenestration positions only
        exterior_indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          if floor_plan[r, c] == exterior_fenestration:
            exterior_indices_array[r, c] = True

        # Calculate view factors
        phi = constants.FENESTRATION_TILT_ANGLE  # 90 degrees
        phi_rad = math.radians(phi)
        cos_phi = math.cos(phi_rad)

        # F_gnd = 0.5 * (1 - cos(phi))
        F_gnd = 0.5 * (1 - cos_phi)

        # F_sky = 0.5 * (1 + cos(phi)) * sqrt(0.5 * (1 + cos(phi)))
        factor = 0.5 * (1 + cos_phi)
        F_sky = factor * math.sqrt(factor)

        # F_air = 0.5 * (1 + cos(phi)) * (1 - sqrt(0.5 * (1 + cos(phi))))
        F_air = factor * (1 - math.sqrt(factor))

        # Beta factor for sky radiation split between sky temperature and
        # ambient air temperature
        beta = math.sqrt(factor)

        groups[f'fenestration_{group_count}'] = {
            'count': len(group_indices),
            'exterior_count': exterior_count,
            'indices': group_indices,
            'indices_array': indices_array,
            'exterior_indices_array': exterior_indices_array,
            'phi': float(phi),
            'azimuth': float(azimuth),
            'F_gnd': F_gnd,
            'F_sky': F_sky,
            'F_air': F_air,
            'beta': beta,
        }

  return groups


def _determine_fenestration_azimuth(
    floor_plan: np.ndarray,
    group_indices: list,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration_value: int = constants.EXTERIOR_FENESTRATION_VALUE,
) -> int:
  """Determine the azimuth (facing direction) of a fenestration group.

  Checks which direction the fenestration group faces by:
  1. First checking if exterior fenestration (-42) is at the array boundary
  2. Then checking for exterior space adjacency

  Args:
    floor_plan: 2D array with the floor plan.
    group_indices: List of (row, col) tuples for the fenestration group.
    exterior_space_value: Value for exterior space (-1).
    exterior_fenestration_value: Value for exterior fenestration (-42).

  Returns:
    Azimuth angle in degrees:
      - 0: Top (north)
      - 90: Right (east)
      - 180: Bottom (south)
      - 270: Left (west)
  """
  rows, cols = floor_plan.shape

  # First, check if any exterior fenestration node is at the array boundary
  # This handles the case where -42 is at the edge of the floor plan
  for row, col in group_indices:
    if floor_plan[row, col] == exterior_fenestration_value:
      # Check if at top boundary (row 0)
      if row == 0:
        return constants.FENESTRATION_AZIMUTH_TOP
      # Check if at bottom boundary (last row)
      if row == rows - 1:
        return constants.FENESTRATION_AZIMUTH_BOTTOM
      # Check if at left boundary (col 0)
      if col == 0:
        return constants.FENESTRATION_AZIMUTH_LEFT
      # Check if at right boundary (last col)
      if col == cols - 1:
        return constants.FENESTRATION_AZIMUTH_RIGHT

  # Direction mapping: (dr, dc) -> azimuth
  direction_azimuth = {
      (-1, 0): constants.FENESTRATION_AZIMUTH_TOP,  # Top/North
      (1, 0): constants.FENESTRATION_AZIMUTH_BOTTOM,  # Bottom/South
      (0, 1): constants.FENESTRATION_AZIMUTH_RIGHT,  # Right/East
      (0, -1): constants.FENESTRATION_AZIMUTH_LEFT,  # Left/West
  }

  # Check each node in the group for exterior adjacency
  for row, col in group_indices:
    for (dr, dc), azimuth in direction_azimuth.items():
      nr, nc = row + dr, col + dc
      if (
          0 <= nr < rows
          and 0 <= nc < cols
          and floor_plan[nr, nc] == exterior_space_value
      ):
        return azimuth

  # Default to 0 if no exterior adjacency found
  return constants.FENESTRATION_AZIMUTH_TOP


def group_air_nodes(
    floor_plan: np.ndarray,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,  # pylint: disable=unused-argument
    exterior_wall_value: int = constants.EXTERIOR_WALL_VALUE_IN_FUNCTION,  # pylint: disable=unused-argument
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    fenestration_groups: dict = None,
) -> dict:
  """Group indoor air nodes that are 4-way connected.

  Air nodes are grouped based on 4-way connectivity, where walls and
  fenestrations act as barriers. Each group represents a separate air space
  (e.g., different rooms).

  Args:
    floor_plan: 2D array with the floor plan (after fenestration marking).
    air_value: Value for indoor air. Defaults to 0.
    interior_wall_value: Value for interior walls. Defaults to -3.
    exterior_wall_value: Value for exterior walls. Defaults to -2.
    exterior_fenestration: Value for exterior fenestration. Defaults to -42.
    interior_fenestration: Value for interior fenestration. Defaults to -43.
    inbetween_fenestration: Value for in-between fenestration. Defaults to -425.
    fenestration_groups: Optional dict from group_fenestrations() to link
        adjacent fenestrations.

  Returns:
    Dictionary with air group information:
    {
      'air_1': {
        'count': int,
        'indices': list of (row, col) tuples,
        'indices_array': np.ndarray of bool (floor_plan shape),
        'fenestration_groups': list of fenestration group names adjacent,
      },
      ...
    }
  """
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups = {}
  group_count = 0
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  # Build reverse mapping: fenestration index -> group name
  fenestration_index_to_group = {}
  if fenestration_groups:
    for group_name, group_info in fenestration_groups.items():
      for idx in group_info['indices']:
        fenestration_index_to_group[idx] = group_name

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == air_value and not visited[row, col]:
        group_count += 1
        group_indices = []
        adjacent_fenestrations = set()
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < floor_plan.shape[0] and 0 <= nc < floor_plan.shape[1]:
              neighbor_val = floor_plan[nr, nc]

              # Check if neighbor is fenestration (to track adjacent fenestrations) # pylint: disable=line-too-long
              if (
                  neighbor_val == interior_fenestration
                  or neighbor_val == exterior_fenestration
                  or neighbor_val == inbetween_fenestration
              ):
                if (nr, nc) in fenestration_index_to_group:
                  adjacent_fenestrations.add(
                      fenestration_index_to_group[(nr, nc)]
                  )

              # Continue BFS only through air nodes
              if neighbor_val == air_value and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

        # Create indices_array: boolean array with True at group positions
        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        groups[f'air_{group_count}'] = {
            'count': len(group_indices),
            'indices': group_indices,
            'indices_array': indices_array,
            'fenestration_groups': sorted(list(adjacent_fenestrations)),
        }

  return groups


def calculate_poa_irradiance(
    irradiance_components: dict[str, float],
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: float,
    solar_azimuth: float,
) -> float:
  """Calculate plane-of-array (POA) global irradiance.

  Converts horizontal irradiance components (GHI, DNI, DHI) to the irradiance
  incident on a tilted surface. Uses the pvlib library's get_total_irradiance
  function.

  Args:
    irradiance_components: Dictionary with 'ghi', 'dni', and 'dhi' keys,
      containing Global Horizontal Irradiance, Direct Normal Irradiance,
      and Diffuse Horizontal Irradiance in W/m2.
    surface_tilt: Surface tilt angle from horizontal in degrees (0 = horizontal,
      90 = vertical).
    surface_azimuth: Surface azimuth angle in degrees (180 = south-facing in
      Northern Hemisphere, compass direction that the surface normal points).
    solar_zenith: Solar zenith angle in degrees (angle from vertical).
    solar_azimuth: Solar azimuth angle in degrees (compass direction of sun).

  Returns:
    POA global irradiance in W/m2.

  Example:
    >>> irrad = {'ghi': 800.0, 'dni': 700.0, 'dhi': 100.0}
    >>> poa = calculate_poa_irradiance(irrad, surface_tilt=30.0,
    ...     surface_azimuth=180.0, solar_zenith=30.0, solar_azimuth=180.0)
  """
  # Import here to avoid circular imports and keep pvlib as optional dependency
  from pvlib import irradiance as pvlib_irradiance  # pylint: disable=import-outside-toplevel

  poa_irrad = pvlib_irradiance.get_total_irradiance(
      surface_tilt=surface_tilt,
      surface_azimuth=surface_azimuth,
      dni=irradiance_components['dni'],
      ghi=irradiance_components['ghi'],
      dhi=irradiance_components['dhi'],
      solar_zenith=solar_zenith,
      solar_azimuth=solar_azimuth,
  )

  return float(poa_irrad['poa_global'])


def calculate_solar_absorbed_for_fenestration_group(
    fenestration_group: dict,
    irradiance_components: dict[str, float],
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> float:
  """Calculate total absorbed solar radiation for a fenestration group.

  Computes q_sol_alpha = G_Ts * alpha for each exterior fenestration node,
  sums them, and distributes evenly to all nodes in the fenestration group.

  Args:
    fenestration_group: Dictionary containing fenestration group properties
      including 'exterior_indices_array', 'indices_array', 'phi', 'azimuth',
      'count', and 'exterior_count'.
    irradiance_components: Dictionary with 'ghi', 'dni', 'dhi', 'solar_zenith',
      'solar_azimuth' keys.
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.
    alpha: Solar absorptance of the fenestration (0-1). Defaults to 0.1.

  Returns:
    Absorbed solar heat flux per node (W/m2) distributed evenly to all
    fenestration nodes in the group. Returns 0.0 if no exterior fenestration.
  """
  exterior_count = fenestration_group.get('exterior_count', 0)
  if exterior_count == 0:
    return 0.0

  # Calculate POA irradiance for this fenestration surface
  surface_tilt = fenestration_group['phi']
  surface_azimuth = fenestration_group['azimuth']

  g_ts = calculate_poa_irradiance(
      irradiance_components,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  # Total absorbed solar radiation = G_Ts * alpha * number_of_exterior_nodes
  total_absorbed = g_ts * alpha * exterior_count

  # Distribute evenly to all fenestration nodes in the group
  total_count = fenestration_group['count']
  return total_absorbed / total_count if total_count > 0 else 0.0


def net_solar_absorbed_heatflux(
    floor_plan: np.ndarray,
    fenestration_groups: dict,
    irradiance_components: dict[str, float],
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> np.ndarray:
  """Calculate absorbed solar radiation array for all fenestration groups.

  Creates a floor_plan-shaped array where each fenestration position contains
  the absorbed solar heat flux (q_sol_alpha) for that node.

  Args:
    floor_plan: 2D array representing the floor plan.
    fenestration_groups: Dictionary from group_fenestrations().
    irradiance_components: Dictionary with 'ghi', 'dni', 'dhi' keys.
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.
    alpha: Solar absorptance of the fenestration (0-1). Defaults to 0.1.

  Returns:
    2D array (same shape as floor_plan) with absorbed solar heat flux (W/m2)
    at fenestration positions, 0 elsewhere.
  """
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

    # Add q_sol_alpha to all nodes in this fenestration group
    indices_array = group_info['indices_array']
    q_sol_alpha_array[indices_array] += q_sol_alpha_per_node

  return q_sol_alpha_array


def calculate_solar_transmitted_for_fenestration_group(
    fenestration_group: dict,
    irradiance_components: dict[str, float],
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> float:
  """Calculate total transmitted solar radiation for a fenestration group.

  Computes total q_sol_tau = sum(G_Ts * tau) for all exterior fenestration
  nodes in the group.

  Args:
    fenestration_group: Dictionary containing fenestration group properties.
    irradiance_components: Dictionary with 'ghi', 'dni', 'dhi' keys.
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.
    tau: Solar transmittance of the fenestration (0-1). Defaults to 0.8.

  Returns:
    Total transmitted solar heat flux (W/m2 * count) for the entire group.
    Returns 0.0 if no exterior fenestration.
  """
  exterior_count = fenestration_group.get('exterior_count', 0)
  if exterior_count == 0:
    return 0.0

  # Calculate POA irradiance for this fenestration surface
  surface_tilt = fenestration_group['phi']
  surface_azimuth = fenestration_group['azimuth']

  g_ts = calculate_poa_irradiance(
      irradiance_components,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  # Total transmitted solar radiation = G_Ts * tau * number_of_exterior_nodes
  return g_ts * tau * exterior_count


def net_solar_transmitted_heatflux(
    floor_plan: np.ndarray,
    fenestration_groups: dict,
    air_groups: dict,
    irradiance_components: dict[str, float],
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> np.ndarray:
  """Calculate transmitted solar radiation array for air groups.

  For each air group, sums q_sol_tau from all connected fenestration groups,
  then distributes evenly to all air nodes in the group. This represents
  solar radiation transmitted through windows into the interior space.

  Args:
    floor_plan: 2D array representing the floor plan.
    fenestration_groups: Dictionary from group_fenestrations().
    air_groups: Dictionary from group_air_nodes().
    irradiance_components: Dictionary with 'ghi', 'dni', 'dhi' keys.
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.
    tau: Solar transmittance of the fenestration (0-1). Defaults to 0.8.

  Returns:
    2D array (same shape as floor_plan) with transmitted solar heat flux (W/m2)
    at air node positions (or interior mass positions), 0 elsewhere.
  """
  q_sol_tau_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups or not air_groups:
    return q_sol_tau_array

  # First, calculate q_sol_tau for each fenestration group
  fenestration_q_sol_tau = {}
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

  # For each air group, sum q_sol_tau from connected fenestration groups
  for air_group_info in air_groups.values():
    connected_fenestrations = air_group_info.get('fenestration_groups', [])

    # Sum q_sol_tau from all connected fenestration groups
    total_q_sol_tau = 0.0
    for fen_group_name in connected_fenestrations:
      if fen_group_name in fenestration_q_sol_tau:
        total_q_sol_tau += fenestration_q_sol_tau[fen_group_name]

    if total_q_sol_tau == 0.0:
      continue

    # Distribute evenly to all air nodes in this air group
    air_count = air_group_info['count']
    q_sol_tau_per_node = total_q_sol_tau / air_count if air_count > 0 else 0.0

    # Add to air node positions
    indices_array = air_group_info['indices_array']
    q_sol_tau_array[indices_array] += q_sol_tau_per_node

  return q_sol_tau_array
