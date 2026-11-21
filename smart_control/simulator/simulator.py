"""Simulator of a simplified building and HVAC devices."""

from typing import Mapping, Tuple

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac as hvac_py
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.utils import conversion_utils

RewardInfo = smart_control_reward_pb2.RewardInfo

CVCoordinates = Tuple[int, int]
ZoneId = Tuple[int, int]


@gin.configurable
class Simulator:
  r"""Simulates thermodynamics of a building.

  This simulator uses the finite differences / finite volume method (FDM/FVM)
  to approximate the temperature changes in each Control Volume (CV) in a
  building. This happens through an iterative process described in the
  [`finite_differences_timestep`] method.

  #### Key Governing Equations

  The building domain is discretized into square air CVs of size
  $\delta_x$ and height (floor height) $z$. Each CV exchanges heat
  by conduction with its air neighbors, and boundary CVs (corner/edge)
  additionally exchange heat by convection with the ambient air. Interior CVs
  may additionally be coupled to an interior thermal-mass node, receive a
  diffuser heat source $Q_x$, and exchange longwave radiation with other
  interior surfaces ($q_{\text{lwx}}$).

  The per-CV transient energy balance has the general form:

  ```text
  (conduction to neighbors)
  + (convection, boundary CVs only)
  + (diffuser source Q_x, interior CVs only)
  + (interior-mass coupling, if enabled)
  + (interior longwave radiation q_lwx, if enabled)
  = (energy storage)
  ```

  and is solved for the CV temperature $T_{i,j}$ at the new time step. Every
  term is a heat rate in W:

  $$\frac{\rho c\, u v z}{\Delta t} \left(T_P - T_P^{(-)}\right) = Q_P
    + \sum_{f \in I_P} A_f \Gamma_f \left(T_{N_f} - T_P\right)
    + \sum_{f \in B_P} A_f \Gamma_{\infty,f} \left(T_\infty - T_P\right)
    + \dots$$

  The storage term is $\rho c\, u v z / \Delta t$ for every CV class; the 1/4
  and 1/2 volume fractions of corner and edge CVs follow from their extents
  $u$ and $v$ (see `_get_cv_dimensions`).

  Heat crossing an interior face passes through the half cell on either side
  in series, and the two CVs share the narrower of their face widths:

  $$\Gamma_f = \left(\frac{\ell_{P,f}}{2k_P} +
      \frac{\ell_{N_f,f}}{2k_{N_f}}\right)^{-1}$$

  so both CVs agree on the heat crossing the face and the balance stays
  conservative where materials meet (`_get_face_conductance`). A face open to
  the ambient puts the exterior film in series with the outer half cell
  (`_get_ambient_conductance`):

  $$\Gamma_{\infty,f} = \left(\frac{1}{h} +
      \frac{\ell_{P,f}}{2k_P}\right)^{-1}$$

  The specific update formulas are documented in each estimator method:

  - Corner and edge CVs share one solver:
    `_get_boundary_cv_temp_estimate`
  - Corner CV (2 neighbors, 2 exposed faces):
    `_get_corner_cv_temp_estimate`
  - Edge CV (3 neighbors, 1 exposed face):
    `_get_edge_cv_temp_estimate`
  - Interior CV (4 neighbors, no exposed face):
    `_get_interior_cv_temp_estimate`
  - Interior mass node:
    `update_interior_mass_temperatures`

  #### Scope / Simplifications

  This base simulator models conduction, ambient convection at boundary CVs,
  the diffuser heat source, interior longwave radiation, and interior thermal
  mass. It does NOT apply exterior longwave radiation ($q_{\text{lwr}}$)
  or shortwave solar gains ($q_{\text{sol},\alpha}$,
  $q_{\text{sol},\tau}$) to the boundary CVs or the interior mass node;
  those terms are therefore intentionally absent from the equations below.

  #### Nomenclature

  - $T_{i,j}$: air temperature at CV (i, j) at the new time step [K]
  - $T_{i,j}^{(-)}$: air temperature at CV (i, j) at the previous step [K]
  - $T_{\text{amb}}$: ambient (external) air temperature [K]
  - $k$: thermal conductivity of the CV [W/(m K)]
  - $\rho$: density [kg/m^3]
  - $c$: specific heat capacity [J/(kg K)]
  - $h$: convection heat transfer coefficient [W/(m^2 K)]
  - $\delta_x$: spatial discretization (uniform CV size) [m]
  - $u, v$: CV extents normal to the left/right and top/bottom faces,
    $\delta_x$ or $\delta_x / 2$ [m]
  - $\ell_{P,f}$: CV extent normal to face $f$ ($u$ or $v$) [m]
  - $A_f$: area shared across face $f$ [m^2]
  - $\Gamma_f$: interior face conductance per unit area [W/(m^2 K)]
  - $\Gamma_{\infty,f}$: ambient face conductance per unit area [W/(m^2 K)]
  - $I_P$, $B_P$: interior faces and ambient-exposed faces of CV $P$
  - $z$: CV height (floor height) [m]
  - $\Delta t$: time step [s]
  """

  def __init__(
      self,
      building: building_py.Building,
      hvac: hvac_py.Hvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
      relative_convergence_threshold: float | None = 1e-6,
      relative_convergence_streak: int = 20,
  ):
    """Simulator init.

    Args:
      building: Building object controlling the control volumes.
      hvac: Hvac for the building.
      weather_controller: Weather Controller for modelling ambient temperature.
      time_step_sec: Size of each time step in seconds.
      convergence_threshold: Minimum iteration temperature change to finish a
        FDM time step.
      iteration_limit: Maximum number of iterations for FDM per time step.
      iteration_warning: Number of iterations for FDM after which a warning will
        be logged.
      start_timestamp: Pandas timestamp representing start time for simulation.
      relative_convergence_threshold: If not None, also converge when the
        change in max_delta is <= this value for
        relative_convergence_streak consecutive iterations. Change is
        |max_delta(n) - max_delta(n-1)|. Default 1e-6.
        Set to None to disable early stopping.
      relative_convergence_streak: Consecutive iterations required for early
        stopping when relative_convergence_threshold is set. Default 20.
    """
    self.building = building
    self._hvac = hvac
    self._weather_controller = weather_controller
    self._time_step_sec = time_step_sec
    self._convergence_threshold = convergence_threshold
    self._iteration_limit = iteration_limit
    self._iteration_warning = iteration_warning
    self._start_timestamp = start_timestamp
    self._relative_convergence_threshold = relative_convergence_threshold
    self._relative_convergence_streak = relative_convergence_streak
    self.reset()

  def reset(self):
    """Resets the simulation to its initial configuration."""
    self.building.reset()
    self._hvac.reset()
    self._current_timestamp = self._start_timestamp

  @property
  def time_step_sec(self) -> float:
    return self._time_step_sec

  @property
  def hvac(self) -> hvac_py.Hvac:
    return self._hvac

  @property
  def current_timestamp(self) -> pd.Timestamp:
    return self._current_timestamp

  def _get_cv_dimensions(
      self, cv_coordinates: CVCoordinates
  ) -> tuple[float, float]:
    r"""Returns the extents $(u, v)$ of a CV in m.

    A boundary CV (corner or edge) is truncated by the ambient, so it spans
    only half a grid spacing along every axis on which it is missing a
    neighbor. An edge CV is therefore half width along one axis and a corner
    CV along both, which gives the 1/2 and 1/4 volume fractions.

    $u$ is the extent along the column (second) index, i.e. normal to the
    left/right faces, and $v$ the extent along the row (first) index, normal
    to the top/bottom faces. This matches
    `tf_simulator.get_cv_dimension_tensors`.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.

    Returns:
      2-Tuple $(u, v)$ of the CV extents in m.
    """
    x, y = cv_coordinates
    delta_x = self.building.cv_size_cm / 100.0
    neighbors = {tuple(n) for n in self.building.neighbors[x][y]}
    u = delta_x
    v = delta_x
    # Only boundary CVs are truncated; interior and exterior CVs are full size.
    if len(neighbors) in (2, 3):
      if (x, y - 1) not in neighbors or (x, y + 1) not in neighbors:
        u = delta_x / 2.0
      if (x - 1, y) not in neighbors or (x + 1, y) not in neighbors:
        v = delta_x / 2.0
    return u, v

  def _get_face_conductance(
      self,
      cv_coordinates: CVCoordinates,
      neighbor_coordinates: CVCoordinates,
  ) -> float:
    r"""Returns the conductance $A_f \Gamma_f$ in W/K of an interior face.

    The heat crossing a face shared by CV $P$ and its neighbor $N_f$ passes
    through the half cell on either side in series:

    $$\Gamma_f = \left(\frac{\ell_{P,f}}{2k_P} +
        \frac{\ell_{N_f,f}}{2k_{N_f}}\right)^{-1}$$

    where $\ell_{\cdot,f}$ is the CV extent normal to the face ($u$ for a
    left/right face, $v$ for a top/bottom face). The two CVs only touch over
    the narrower of their widths, so the shared area is

    $$A_f = \min(v_P, v_{N_f})\, z \;\text{(left/right)}, \qquad
      A_f = \min(u_P, u_{N_f})\, z \;\text{(top/bottom)}.$$

    When does $\min(\cdot)$ matter? Only at a reentrant corner. Everywhere
    else the two CVs sharing a face have the same width along it:

    ```text
      convex corner: min() not needed      reentrant corner: min() needed

         E   E   E   E                      E   E   E   E   E
         E   2 - 3 - 3                      E   E   E   N - 4
         E   |   |   |                      E   E   E   |   |
         E   3 - 4 - 4                      3 - 3 - 3 - P - 4

      2-3 along a wall: dx/2 / dx/2            P-N:  dx / dx/2
      3-4 behind it:    dx   / dx              P-3:  dx / dx/2

      E: exterior CV (T_amb)    2: corner CV (2 neighbors, 1/4 volume)
      3: edge CV (1/2 volume)   4: interior CV (4 neighbors, full size)
      N: edge CV truncated on its left    P: interior CV, full width
    ```

    A boundary CV is truncated only toward the ambient, so along a wall
    both edge CVs (or the corner and its edge) are half width, and across the
    wall the edge CV and the interior CV behind it are both full width. At a
    reentrant corner the exterior CV that truncates $N$ is only a diagonal
    neighbor of $P$, so $P$ stays a full interior CV while $N$ and the edge CV
    to its left are half width. Zoomed in (`....` = cut-away half):

    ```text
           +---------+----+----+
           |    E    |....| N  |      ====  shared, width dx/2
           +---------+~~~~+====+      ~~~~  P's face with no CV behind it
           |.........~         |
           +---------#    P    |      #     shared, width dx/2
           |    3    #         |
           +---------+---------+
    ```

    Seen from $P$ the face is $\delta_x z$, seen from $N$ it is
    $\delta_x z / 2$: without $\min(\cdot)$ the two disagree by 2x on the heat
    crossing it and energy is created or destroyed there. With it both use
    $\delta_x z / 2$. On a rectangular building $\min(w_P, w_N) = w_P$ on
    every face; the sb1 floor plan has about 1,200 such faces out of 2
    million.

    The `~` halves of $P$'s faces belong to no face conductance and $P$ has no
    ambient conductance, so they are adiabatic, although the building outline
    physically passes there. This is a geometric approximation, not a
    conservation error: drawing the two exterior walls through $P$ would cut
    $P$ to a 3/4 L-shaped CV with the same shared areas ($\delta_x z / 2$ to
    $N$ and to the edge CV, $\delta_x z$ to the two interior CVs) but a 1/4
    smaller volume and two exterior half faces instead of adiabatic ones
    (about 2.5% of the envelope conductance on sb1). One $(u, v)$ pair per CV
    cannot describe that L shape, which is why $P$ keeps its full size and
    $\min(\cdot)$ supplies the shared width face by face.

    Both CVs therefore compute the same conductance for the same face, which
    keeps the balance conservative where materials (e.g. air and wall) or CV
    sizes (e.g. at a reentrant corner) differ. For a uniform material between
    two full CVs it reduces to $k z$.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      neighbor_coordinates: 2-Tuple of coordinates of the neighboring CV.

    Returns:
      The face conductance $A_f \Gamma_f$ in W/K.
    """
    x, y = cv_coordinates
    nx, ny = neighbor_coordinates
    z = self.building.floor_height_cm / 100.0
    u, v = self._get_cv_dimensions(cv_coordinates)
    neighbor_u, neighbor_v = self._get_cv_dimensions(neighbor_coordinates)

    # A left/right neighbor shares a face of area v*z normal to u; a top/bottom
    # neighbor shares a face of area u*z normal to v.
    horizontal = ny != y
    length = u if horizontal else v
    neighbor_length = neighbor_u if horizontal else neighbor_v
    area = (min(v, neighbor_v) if horizontal else min(u, neighbor_u)) * z

    return area / (
        length / (2.0 * self.building.conductivity[x][y])
        + neighbor_length / (2.0 * self.building.conductivity[nx][ny])
    )

  def _get_ambient_conductance(
      self, cv_coordinates: CVCoordinates, convection_coefficient: float
  ) -> float:
    r"""Returns the conductance to the ambient, in W/K.

    The returned value is $\sum_f A_f \Gamma_{\infty,f}$.

    On a face that opens onto the ambient the heat passes through the outer
    half cell of the CV and the exterior convective film in series:

    $$\Gamma_{\infty,f} = \left(\frac{1}{h} +
        \frac{\ell_{P,f}}{2k_P}\right)^{-1}$$

    with $A_f = v z$ for a left/right face and $A_f = u z$ for a top/bottom
    face. Every direction without a neighbor counts as an exterior face, so a
    corner CV contributes two faces and an edge CV one.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      The summed ambient conductance in W/K.
    """
    x, y = cv_coordinates
    z = self.building.floor_height_cm / 100.0
    u, v = self._get_cv_dimensions(cv_coordinates)
    conductivity = self.building.conductivity[x][y]
    neighbors = {tuple(n) for n in self.building.neighbors[x][y]}

    ambient_conductance = 0.0
    for coordinates, length, area in (
        ((x, y - 1), u, v * z),
        ((x, y + 1), u, v * z),
        ((x - 1, y), v, u * z),
        ((x + 1, y), v, u * z),
    ):
      if coordinates not in neighbors:
        ambient_conductance += area / (
            1.0 / convection_coefficient + length / (2.0 * conductivity)
        )
    return ambient_conductance

  def _get_boundary_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    r"""Returns temperature estimate in K for a corner or edge CV.

    Solves the boundary CV energy balance for $T_P$:

    $$T_P = \frac{\sum_{f \in I_P} A_f \Gamma_f T_{N_f} +
        \sum_{f \in B_P} A_f \Gamma_{\infty,f} T_\infty +
        t_0 T_P^{(-)}}
        {\sum_{f \in I_P} A_f \Gamma_f +
        \sum_{f \in B_P} A_f \Gamma_{\infty,f} + t_0},
      \qquad t_0 = \frac{\rho c\, u v z}{\Delta t}$$

    Corner and edge CVs differ only in their extents $u, v$ and in how many
    faces open onto the ambient, both of which are derived from the neighbors.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    delta_t = self._time_step_sec
    z = self.building.floor_height_cm / 100.0
    density = self.building.density[x][y]
    heat_capacity = self.building.heat_capacity[x][y]
    last_temp = self.building.temp[x][y]
    neighbors = self.building.neighbors[x][y]
    u, v = self._get_cv_dimensions(cv_coordinates)

    t0 = density * heat_capacity * u * v * z / delta_t
    retained_heat = t0 * last_temp

    neighbor_transfer = 0.0
    neighbor_conductance = 0.0
    for nx, ny in neighbors:
      conductance = self._get_face_conductance(cv_coordinates, (nx, ny))
      neighbor_transfer += conductance * temperature_estimates[nx][ny]
      neighbor_conductance += conductance

    ambient_conductance = self._get_ambient_conductance(
        cv_coordinates, convection_coefficient
    )
    convection_transfer = ambient_conductance * ambient_temperature

    denominator = neighbor_conductance + ambient_conductance + t0

    return (
        neighbor_transfer + convection_transfer + retained_heat
    ) / denominator

  def _get_corner_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    r"""Returns temperature estimate for corner CV in K for next time step.

    A corner CV has two air neighbors and two faces exposed to the ambient
    air. It is half width along both axes, $u = v = \delta_x / 2$, so it
    represents one quarter of a full interior CV volume.

    **Energy Balance (corner CV)**

    Conduction from the two neighbors, convection from the two exposed faces,
    and transient storage over the 1/4 CV volume:

    $$\sum_{n=1}^{2} A_n \Gamma_n (T_n - T_{i,j})
      + \sum_{f \in B_P} A_f \Gamma_{\infty,f} (T_{\text{amb}} - T_{i,j})
      = \frac{\rho c\, u v z}{\Delta t}
        \left( T_{i,j} - T_{i,j}^{(-)} \right)$$

    with the series face conductances (see `_get_face_conductance` and
    `_get_ambient_conductance`)

    $$\Gamma_n = \left(\frac{\ell_{P,n}}{2k_P} +
        \frac{\ell_{N_n,n}}{2k_{N_n}}\right)^{-1}, \qquad
      \Gamma_{\infty,f} = \left(\frac{1}{h} +
        \frac{\ell_{P,f}}{2k_P}\right)^{-1}.$$

    Solving for $T_{i,j}$ gives the expression evaluated by
    `_get_boundary_cv_temp_estimate`. For a uniform material
    ($A_n \Gamma_n = \tfrac{2}{3} k z$ toward an edge neighbor):

    $$T_{i,j} = \frac{\tfrac{2}{3} k z (T_{n1} + T_{n2})
        + \frac{\delta_x z}{1/h + \delta_x/(4k)} T_{\text{amb}}
        + t_0 T_{i,j}^{(-)}}
        {\tfrac{4}{3} k z + \frac{\delta_x z}{1/h + \delta_x/(4k)} + t_0},
      \qquad t_0 = \frac{\rho c \delta_x^2 z}{4 \Delta t}$$

    Note:
      Exterior longwave radiation ($q_{\text{lwr}}$) and solar gains
      ($q_{\text{sol}}$) are not modeled by this base simulator, so those
      terms do not appear.

    **Nomenclature and Units**

    - $T_{i,j}$: corner CV air temperature at new time step [K]
    - $T_{i,j}^{(-)}$: corner CV air temperature at previous time step [K]
    - $T_n$: neighbor CV temperatures [K]
    - $T_{\text{amb}}$: ambient (external) air temperature [K]
    - $A_n, A_f$: face area, $vz$ for a left/right face and $uz$ for a
      top/bottom face (shared area $\min$ of both CVs for $A_n$)
      [$\mathrm{m^2}$]
    - $\Gamma_n$: interior face conductance per unit area
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $\Gamma_{\infty,f}$: ambient face conductance per unit area
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $B_P$: faces of the CV open to the ambient
    - $\ell_{P,f}$: CV extent normal to face $f$ ($u$ or $v$) [$\mathrm{m}$]
    - $k$: thermal conductivity [$\mathrm{W/(m \cdot K)}$]
    - $h$: convection coefficient [$\mathrm{W/(m^2 \cdot K)}$]
    - $\rho$: density [$\mathrm{kg/m^3}$]
    - $c$: specific heat capacity [$\mathrm{J/(kg \cdot K)}$]
    - $u, v$: CV extents normal to the left/right and top/bottom faces
      [$\mathrm{m}$]
    - $\delta_x$: spatial discretization (uniform CV size) [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $\Delta t$: time step [$\mathrm{s}$]
    - $t_0$: storage conductance [$\mathrm{W/K}$]

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    neighbors = self.building.neighbors[x][y]

    # Ensure corner CV.
    if len(neighbors) != 2:
      raise ValueError(
          f'Expected 2 neighbors for a corner CV, but found {len(neighbors)} '
          f'at coordinates {cv_coordinates}. '
          'This indicates an invalid building structure.'
      )

    return self._get_boundary_cv_temp_estimate(
        cv_coordinates,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

  def _get_edge_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    r"""Returns temperature estimate for edge CV in K for next time step.

    An edge CV has three air neighbors and one face exposed to the ambient
    air. It is half width along the axis normal to the exposed face, so it
    represents one half of a full interior CV volume.

    **Energy Balance (edge CV)**

    Conduction from the three neighbors, convection from the single exposed
    face, and transient storage over the 1/2 CV volume:

    $$\sum_{n=1}^{3} A_n \Gamma_n (T_n - T_{i,j})
      + A_f \Gamma_{\infty,f} (T_{\text{amb}} - T_{i,j})
      = \frac{\rho c\, u v z}{\Delta t}
        \left( T_{i,j} - T_{i,j}^{(-)} \right)$$

    with the series face conductances (see `_get_face_conductance` and
    `_get_ambient_conductance`)

    $$\Gamma_n = \left(\frac{\ell_{P,n}}{2k_P} +
        \frac{\ell_{N_n,n}}{2k_{N_n}}\right)^{-1}, \qquad
      \Gamma_{\infty,f} = \left(\frac{1}{h} +
        \frac{\ell_{P,f}}{2k_P}\right)^{-1}.$$

    The two faces along the exterior are only half as wide ($A_n =
    \delta_x z / 2$), while the face toward the interior neighbor is full
    width ($A_n = \delta_x z$). This replaces the former ad hoc 0.5 weight on
    boundary neighbors, and the half-cell resistances replace the owner-cell
    $k_P/\delta_x$, so both CVs sharing a face see the same flux even where
    the materials differ.

    Solving for $T_{i,j}$ gives the expression evaluated by
    `_get_boundary_cv_temp_estimate`, with
    $t_0 = \rho c\, u v z / \Delta t = \rho c \delta_x^2 z / (2 \Delta t)$.

    Note:
      Exterior longwave radiation ($q_{\text{lwr}}$) and solar gains
      ($q_{\text{sol}}$) are not modeled by this base simulator, so those
      terms do not appear.

    **Nomenclature and Units**

    - $T_{i,j}$: edge CV air temperature at new time step [K]
    - $T_{i,j}^{(-)}$: edge CV air temperature at previous time step [K]
    - $T_n$: neighbor CV temperatures [K]
    - $T_{\text{amb}}$: ambient (external) air temperature [K]
    - $A_n, A_f$: face area, $vz$ for a left/right face and $uz$ for a
      top/bottom face (shared area $\min$ of both CVs for $A_n$)
      [$\mathrm{m^2}$]
    - $\Gamma_n$: interior face conductance per unit area
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $\Gamma_{\infty,f}$: ambient face conductance per unit area
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $\ell_{P,f}$: CV extent normal to face $f$ ($u$ or $v$) [$\mathrm{m}$]
    - $k$: thermal conductivity [$\mathrm{W/(m \cdot K)}$]
    - $h$: convection coefficient [$\mathrm{W/(m^2 \cdot K)}$]
    - $\rho$: density [$\mathrm{kg/m^3}$]
    - $c$: specific heat capacity [$\mathrm{J/(kg \cdot K)}$]
    - $u, v$: CV extents normal to the left/right and top/bottom faces
      [$\mathrm{m}$]
    - $\delta_x$: spatial discretization (uniform CV size) [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $\Delta t$: time step [$\mathrm{s}$]
    - $t_0$: storage conductance [$\mathrm{W/K}$]

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    neighbors = self.building.neighbors[x][y]

    # Ensure edge CV.
    if len(neighbors) != 3:
      raise ValueError(
          f'Expected 3 neighbors for an edge CV, but found {len(neighbors)} '
          f'at coordinates {cv_coordinates}. '
          'This indicates an invalid building structure.'
      )

    return self._get_boundary_cv_temp_estimate(
        cv_coordinates,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

  def _get_interior_cv_temp_estimate(
      self, cv_coordinates: CVCoordinates, temperature_estimates: np.ndarray
  ) -> float:
    r"""Returns temperature estimate for interior CV in K for next time step.

    This function calculates the solution to an equation involving the energy
    transfer by conduction to neighboring air CVs, heat input from a diffuser,
    radiative exchange with interior surfaces, and heat exchange with interior
    mass nodes (if present).

    **Equations**

    The energy balance for an interior control volume (CV) with interior mass
    is given by:

    $$\begin{multline}
      \sum_{f \in I_P} A_f \Gamma_f \left( T_{N_f} - T_{i,j} \right)
      + Q_x + \frac{k_{\text{mass}} u v}{z}
      (T_{\text{mass},i,j} - T_{i,j}) + q_{\text{lwx}} (u z) =
      \frac{\rho c u v z}{\Delta t} \left( T_{i,j} - T_{i,j}^{(-)} \right)
      \end{multline}$$

    Here $q_{\text{lwx}}$ is a heat flux [W/m^2] and $q_{\text{lwx}} (u z)$ is
    the corresponding power [W] entering the CV through the wall face of area
    $u z$.

    Each face conductance is the series resistance of the two half cells that
    meet at the face, see `_get_face_conductance`:

    $$\Gamma_f = \left(\frac{\ell_{P,f}}{2k_P} +
        \frac{\ell_{N_f,f}}{2k_{N_f}}\right)^{-1}$$

    so both CVs sharing a face agree on the heat crossing it, even where the
    materials differ (e.g. air next to a wall). Because $\Gamma_f$ is no longer
    the owner CV's $k/\delta_x$, the balance is kept in W instead of being
    divided through by $k z$.

    Solving for $T_{i,j}$ with uniform spacing ($u = v = \delta_x$):

    $$T_{i,j} = \frac{\sum_{f \in I_P} A_f \Gamma_f T_{N_f} + Q_x
      + \frac{k_{\text{mass}} \delta_x^2}{z} T_{\text{mass},i,j}
      + q_\text{lwx}\, \delta_x z + t_0 T_{i,j}^{(-)}}
      {\sum_{f \in I_P} A_f \Gamma_f
      + \frac{k_{\text{mass}} \delta_x^2}{z} + t_0}$$

    where the storage term is:

    $$t_0 = \frac{\rho c \delta_x^2 z}{\Delta t}$$

    For a uniform material surrounded by interior CVs, $A_f \Gamma_f = k z$
    and dividing through by $k z$ recovers the former form
    $\left(\sum T_{N_f} + \dots\right) / (4 + \dots)$.

    **Nomenclature and Units**

    - $T_{i,j}$: Air temperature at CV $(i,j)$ at new time step [K]
    - $T_{i,j}^{(-)}$: Air temperature at CV $(i,j)$ at previous time step [K]
    - $T_{\text{mass},i,j}$: Interior mass temperature at CV $(i,j)$ [K]
    - $T_{N_f}$: Neighbor CV temperature across face $f$ [K]
    - $I_P$: The four interior faces of CV $(i,j)$
    - $A_f$: Area shared across face $f$ [$\mathrm{m^2}$]
    - $\Gamma_f$: Interior face conductance per unit area
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $\ell_{P,f}$: CV extent normal to face $f$ [$\mathrm{m}$]
    - $k_P, k_{N_f}$: Thermal conductivity of the CV and of its neighbor
      [$\mathrm{W/(m \cdot K)}$]
    - $k_{\text{mass}}$: Thermal conductivity of interior mass
      [$\mathrm{W/(m \cdot K)}$]
    - $Q_x$: External heat source (e.g., diffuser) [$\mathrm{W}$]
    - $q_{\text{lwx}}$: Interior longwave radiative heat flux
      [$\mathrm{W/m^2}$]
    - $u, v$: CV dimensions in x and y directions [$\mathrm{m}$]
    - $\delta_x$: Spatial discretization (uniform CV size) [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $\rho$: Density [$\mathrm{kg/m^3}$]
    - $c$: Specific heat capacity [$\mathrm{J/(kg \cdot K)}$]
    - $\Delta t$: Time step [$\mathrm{s}$]
    - $t_0$: Storage conductance [$\mathrm{W/K}$]

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
    """
    x, y = cv_coordinates
    delta_x = self.building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    z = self.building.floor_height_cm / 100.0
    density = self.building.density[x][y]

    heat_capacity = self.building.heat_capacity[x][y]
    last_temp = self.building.temp[x][y]
    input_q = self.building.input_q[x][y]
    neighbors = self.building.neighbors[x][y]
    # Ensure interior CV.
    if len(neighbors) != 4:
      raise ValueError(
          'Expected 4 neighbors for an interior CV, but found'
          f' {len(neighbors)} at coordinates {cv_coordinates}. This indicates'
          ' an invalid building structure.'
      )

    t0 = density * heat_capacity * delta_x**2 * z / delta_t

    # Conduction through the series half-cell conductance of each face.
    neighbor_transfer = 0.0
    denominator = t0
    for nx, ny in neighbors:
      conductance = self._get_face_conductance(cv_coordinates, (nx, ny))
      neighbor_transfer += conductance * temperature_estimates[nx][ny]
      denominator += conductance

    retained_heat = t0 * last_temp

    thermal_source = input_q

    # Interior mass heat transfer (adiabatic node connected only to air CV)
    if (
        hasattr(self.building, 'include_interior_mass')
        and self.building.include_interior_mass
        and self.building.interior_mass_mask[x, y]
    ):
      interior_mass_conductivity = self.building.interior_mass_conductivity[x][
          y
      ]
      interior_mass_conductance = interior_mass_conductivity * delta_x**2 / z
      denominator += interior_mass_conductance

      # Heat transfer between air CV and its interior mass node
      interior_mass_temp = self.building.interior_mass_temp[x, y]
      # Heat flux from interior mass to air CV
      neighbor_transfer += interior_mass_temp * interior_mass_conductance

    # checking for implementation of `include_radiative_heat_transfer` because
    # the `FloorPlanBasedBuilding` implements it, but the `Building` doesn't
    if (
        hasattr(self.building, 'include_radiative_heat_transfer')
        and self.building.include_radiative_heat_transfer
    ):
      # Radiative heat transfer. Within a sweep the exchange is evaluated once
      # (see update_temperature_estimates); called on its own, recompute it.
      q_lwx_array = getattr(self, '_q_lwx_sweep', None)
      if q_lwx_array is None:
        q_lwx_array = (
            self.building.apply_longwave_interior_radiative_heat_transfer(
                temperature_estimates
            )
        )
      # q_lwx_idx is -1 if the CV does not have LWX
      q_lwx_idx = self.building.lwx_index[x, y]
      q_lwx = (q_lwx_array[q_lwx_idx] * delta_x * z) if q_lwx_idx != -1 else 0.0
    else:
      q_lwx = 0.0

    return (
        neighbor_transfer + thermal_source + retained_heat + q_lwx
    ) / denominator

  def _get_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Returns temperature estimate for CV for next time step.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    neighbors = self.building.neighbors[x][y]
    if len(neighbors) <= 1:
      # Exterior CVs should always return ambient air temps.
      return ambient_temperature
    if len(neighbors) == 2:
      return self._get_corner_cv_temp_estimate(
          cv_coordinates,
          temperature_estimates,
          ambient_temperature,
          convection_coefficient,
      )
    elif len(neighbors) == 3:
      return self._get_edge_cv_temp_estimate(
          cv_coordinates,
          temperature_estimates,
          ambient_temperature,
          convection_coefficient,
      )
    else:
      return self._get_interior_cv_temp_estimate(
          cv_coordinates, temperature_estimates
      )

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    """Iterates across all CVs and updates the temperature estimate.

    Corner and edge CVs are exposed to thermal exchange with the ambient air
    through convection.

    Args:
      temperature_estimates: Current temperature estimate for each CV, will be
        updated with new values.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      Maximum difference in temperture_estimates across all CVs before and after
      operation.
    """
    nrows, ncols = temperature_estimates.shape
    max_delta = 0.0

    # The interior longwave exchange couples every surface of an enclosure to
    # every other, so it costs a dense matrix product over the enclosure.
    # Evaluate it once per sweep, from the field at the start of the sweep,
    # instead of once per CV. Radiation is then updated Jacobi-style while
    # conduction stays Gauss-Seidel; the converged field is the same, since at
    # convergence the field no longer changes between sweeps. TFSimulator also
    # evaluates the exchange once per sweep.
    self._q_lwx_sweep = None
    if getattr(self.building, 'include_radiative_heat_transfer', False):
      self._q_lwx_sweep = (
          self.building.apply_longwave_interior_radiative_heat_transfer(
              temperature_estimates
          )
      )

    try:
      for x in range(nrows):
        for y in range(ncols):
          temp_estimate = self._get_cv_temp_estimate(
              (x, y),
              temperature_estimates,
              ambient_temperature,
              convection_coefficient,
          )

          delta = abs(temp_estimate - temperature_estimates[x][y])
          max_delta = max(delta, max_delta)

          temperature_estimates[x][y] = temp_estimate
    finally:
      # Only valid for this sweep; never let a stale field be read back.
      self._q_lwx_sweep = None

    return temperature_estimates, max_delta

  def update_interior_mass_temperatures(
      self, air_temperature_estimates: np.ndarray
  ) -> tuple[np.ndarray, float]:
    r"""Updates interior mass node temperatures from air-CV heat transfer.

    Interior mass nodes are adiabatic (no interaction with each other) and only
    exchange heat with their corresponding air CV. The heat exchange occurs
    through the vertical direction (height z) of the control volume.

    **Equations**

    The energy balance for the interior mass node exchanging heat only with its
    corresponding air CV through a characteristic length z is:

    $$\frac{k_{\text{mass}} u v}{z} (T_{i,j} - T_{\text{mass},i,j}) =
      \rho_{\text{mass}} c_{\text{mass}} u v z
      \frac{T_{\text{mass},i,j} - T_{\text{mass},i,j}^{(-)}}{\Delta t}$$

    Dividing both sides by $(u v)$ and rearranging:

    $$\frac{k_{\text{mass}}}{z} (T_{i,j} - T_{\text{mass},i,j}) =
      \rho_{\text{mass}} c_{\text{mass}} z
      \frac{T_{\text{mass},i,j} - T_{\text{mass},i,j}^{(-)}}{\Delta t}$$

    Multiplying both sides by $z$:

    $$k_{\text{mass}} (T_{i,j} - T_{\text{mass},i,j}) =
      \rho_{\text{mass}} c_{\text{mass}} z^2
      \frac{T_{\text{mass},i,j} - T_{\text{mass},i,j}^{(-)}}{\Delta t}$$

    Expanding and collecting terms with $T_{\text{mass},i,j}$:

    $$k_{\text{mass}} T_{i,j} +
      \rho_{\text{mass}} c_{\text{mass}} \frac{z^2}{\Delta t}
      T_{\text{mass},i,j}^{(-)} =
      \left( k_{\text{mass}} +
      \rho_{\text{mass}} c_{\text{mass}} \frac{z^2}{\Delta t} \right)
      T_{\text{mass},i,j}$$

    Dividing both sides by $k_{\text{mass}}$ and defining the temporal
    parameter:

    $$t_{0,\text{mass}} = \frac{\rho_{\text{mass}} c_{\text{mass}} z^2}
      {k_{\text{mass}} \Delta t} =
      \frac{z^2}{\Delta t \cdot \alpha_{\text{mass}}}$$

    where $\alpha_{\text{mass}} = \frac{k_{\text{mass}}}
      {\rho_{\text{mass}} c_{\text{mass}}}$ is the thermal diffusivity of the
      interior mass.

    The final solution for the interior mass temperature update is:

    $$T_{\text{mass},i,j} =
      \frac{T_{i,j} + t_{0,\text{mass}} \cdot T_{\text{mass},i,j}^{(-)}}
      {1 + t_{0,\text{mass}}}$$

    This formulation is consistent with the air CV energy balance where the
    interior mass coupling term is $\frac{k_{\text{mass}} u v}{z}
    (T_{\text{mass},i,j} - T_{i,j})$.

    **Nomenclature and Units**

    - $T_{i,j}$: Converged air temperature at new time step [K]
    - $T_{\text{mass},i,j}$: Interior mass temperature at new time step
       (unknown) [$\mathrm{K}$]
    - $T_{\text{mass},i,j}^{(-)}$: Interior mass temperature at previous
      time step (known) [$\mathrm{K}$]
    - $k_{\text{mass}}$: Thermal conductivity of interior mass
      [$\mathrm{W/(m \cdot K)}$]
    - $\rho_{\text{mass}}$: Density of interior mass [$\mathrm{kg/m^3}$]
    - $c_{\text{mass}}$: Specific heat capacity of interior mass
      [$\mathrm{J/(kg \cdot K)}$]
    - $\alpha_{\text{mass}}$: Thermal diffusivity of interior mass
      [$\mathrm{m^2/s}$]
    - $u, v$: CV dimensions in x and y directions [$\mathrm{m}$]
    - $z$: CV height (floor height), characteristic length for heat exchange
      [$\mathrm{m}$]
    - $\Delta t$: Time step [$\mathrm{s}$]
    - $t_{0,\text{mass}}$: Temporal parameter for interior mass [dimensionless]

    Args:
      air_temperature_estimates: Current air temperature estimates for each CV.

    Returns:
      Tuple of (updated interior mass temperatures, maximum temperature change)
    """
    if not (
        hasattr(self.building, 'include_interior_mass')
        and self.building.include_interior_mass
    ):
      return self.building.interior_mass_temp.copy(), 0.0

    z = self.building.floor_height_cm / 100.0
    delta_t = self._time_step_sec

    # Copy current interior mass temperatures for updates
    interior_mass_temp_estimates = self.building.interior_mass_temp.copy()
    max_delta = 0.0

    # Iterate over all CVs that have interior mass nodes
    for x in range(self.building.interior_mass_mask.shape[0]):
      for y in range(self.building.interior_mass_mask.shape[1]):
        if not self.building.interior_mass_mask[x, y]:
          continue

        # Get properties
        air_temp = air_temperature_estimates[x, y]
        interior_mass_temp = self.building.interior_mass_temp[x, y]
        interior_mass_conductivity = self.building.interior_mass_conductivity[
            x
        ][y]
        interior_mass_density = self.building.interior_mass_density[x][y]
        interior_mass_heat_capacity = self.building.interior_mass_heat_capacity[
            x
        ][y]

        # Calculate thermal diffusivity for interior mass
        alpha_mass = (
            interior_mass_conductivity
            / interior_mass_density
            / interior_mass_heat_capacity
        )

        # Temperature update using finite difference with z as characteristic
        # length. Heat exchange with air CV occurs through height z, consistent
        # with the air CV energy balance coupling term k_mass * u * v / z.
        t0_mass = z**2 / (delta_t * alpha_mass)
        denominator = 1.0 + t0_mass

        # New interior mass temperature
        new_temp = (air_temp + t0_mass * interior_mass_temp) / denominator

        # Track maximum change
        delta = abs(new_temp - interior_mass_temp)
        max_delta = max(delta, max_delta)

        interior_mass_temp_estimates[x, y] = new_temp

    return interior_mass_temp_estimates, max_delta

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    r"""Calculates the temperature for each Control Volume (CV) after a step.

    To find the temperature after conduction/convection for each CV, we set
    up a system of linear equations. To approximate the solution:

    1.   Create a starting estimate temperature for each CV.
    2.   For each CV, solve for temperature T, based on the current estimate
         for neighboring CVs and known thermal losses/gains.
    3.   Calculate the difference between previous T and new T.
    4.   If interior mass is enabled, update interior mass temperatures and
         check their convergence as well. The mass update always starts from
         the previous time step's mass temperature $T_{\text{mass}}^{(-)}$,
         not from the previous iteration, so the converged pair solves the
         coupled air/mass balance of one time step (Picard iteration).

    If the maximum difference in the grid is less than some small constant,
    conversion_threshold, then quit. Otherwise, return to step 2.

    The update_temperature_estimates function performs steps 2, and 3.

    Args:
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      Whether or not there was convergence before iteration_limit was reached.
    """
    # Initialize estimates with the last update.
    # TODO(gusatb): Please provide a unit test for convergence.
    temp_estimate = self.building.temp.copy()

    # Check if interior mass is enabled
    include_interior_mass = (
        hasattr(self.building, 'include_interior_mass')
        and self.building.include_interior_mass
    )

    if include_interior_mass:
      # Interior mass temperature at the previous time step, T_mass^(-). The
      # Picard iteration below re-solves the mass node every iteration, but
      # always from this value, so that at convergence air and mass satisfy
      # both backward Euler balances of the same time step. Starting each
      # iteration from the previous iterate instead would advance the mass by
      # one time step per iteration and drive it to the air temperature.
      interior_mass_temp_last_step = self.building.interior_mass_temp.copy()
      interior_mass_temp_estimate = interior_mass_temp_last_step.copy()

    converged_successfully = False
    for iteration_count in range(self._iteration_limit):
      # Update air CV temperatures
      temp_estimate, max_delta_air = self.update_temperature_estimates(
          temp_estimate,
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )

      # Update interior mass temperatures if enabled
      if include_interior_mass:
        # Update interior mass temperatures based on current air temperature
        # estimates, always starting from the previous time step's value.
        self.building.interior_mass_temp = interior_mass_temp_last_step
        interior_mass_temp_update, _ = self.update_interior_mass_temperatures(
            temp_estimate
        )
        # The convergence check tracks how much the estimate still moves from
        # one iteration to the next, not how much the mass moves over the step.
        max_delta_mass = float(
            np.max(
                np.abs(interior_mass_temp_update - interior_mass_temp_estimate)
            )
        )
        interior_mass_temp_estimate = interior_mass_temp_update
        # Store the current iterate; the air balance reads it as T_mass.
        self.building.interior_mass_temp = interior_mass_temp_estimate

        # Combined convergence check
        max_delta = max(max_delta_air, max_delta_mass)
      else:
        max_delta = max_delta_air

      if iteration_count + 1 == self._iteration_warning:
        if include_interior_mass:
          logging.warning(
              'Step %d, not converged in %d steps, '
              'max_delta_air = %3.3f, max_delta_mass = %3.3f',
              iteration_count,
              self._iteration_warning,
              max_delta_air,
              max_delta_mass,
          )
        else:
          logging.warning(
              'Step %d, not converged in %d steps, max_delta = %3.3f',
              iteration_count,
              self._iteration_warning,
              max_delta,
          )

      if max_delta <= self._convergence_threshold:
        converged_successfully = True
        break
    else:
      if include_interior_mass:
        logging.warning(
            'Max iteration count reached, max_delta_air = %3.3f, '
            'max_delta_mass = %3.3f',
            max_delta_air,
            max_delta_mass,
        )
      else:
        logging.warning(
            'Max iteration count reached, max_delta = %3.3f', max_delta
        )

    # Final update of building temperatures
    self.building.temp = temp_estimate

    # Interior mass temperatures are already updated in the loop
    # No need for additional update here

    return converged_successfully

  def _calculate_return_water_temperature(
      self, zone_temps: Mapping[ZoneId, float]
  ) -> float:
    numerator = 0.0
    denominator = 0.0
    for zone_id, vav in self._hvac.vavs.items():
      numerator += vav.reheat_valve_setting * zone_temps[zone_id]
      denominator += vav.reheat_valve_setting
    return numerator / (denominator + 1e-6)

  def setup_step_sim(self) -> None:
    """This method should not change the state of the building."""
    current_ts = self._current_timestamp
    hvac = self._hvac

    # Get the average temps in each zone. Assumes that the thermostat reads
    # the average room temperatures.
    avg_temps = self.building.get_zone_average_temps()

    for zone, zone_temp in avg_temps.items():
      vav = hvac.vavs[zone]

      # VAV update_setting handles the thermostat internally.
      vav.update_settings(zone_temp, current_ts)

  def execute_step_sim(self) -> None:
    """This method should not change any actions set on smart devices."""
    current_ts = self._current_timestamp
    hvac = self._hvac

    # Get the average temps in each zone. Assumes that the thermostat reads
    # the average room temperatures.
    avg_temps = self.building.get_zone_average_temps()

    # Recirculation temperature at the air handler is the global average.
    recirculation_temp = self.building.temp.mean()

    ambient_temperature = self._weather_controller.get_current_temp(current_ts)

    supply_air_temp = hvac.air_handler.get_supply_air_temp(
        recirculation_temp, ambient_temperature
    )

    convection_coefficient = (
        self._weather_controller.get_air_convection_coefficient(current_ts)
    )

    # Update each control volume.
    self.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
    )

    # Reset the air handler and boiler flow rate demand before accumulating.
    hvac.air_handler.reset_demand()
    hvac.boiler.reset_demand()

    zone_supply_temp_map = {}

    # Iterate through each zone.
    for zone, zone_temp in avg_temps.items():
      vav = hvac.vavs[zone]

      q_zone, zone_supply_temp = vav.output(zone_temp, supply_air_temp)
      zone_supply_temp_map[zone] = zone_supply_temp

      # Update the air handler airflow demand by summing from all VAVs.
      if vav.flow_rate_demand > 0:
        hvac.air_handler.add_demand(vav.flow_rate_demand)

      # Update the boiler demand for hot water as the sum of each VAV's demand.
      if vav.reheat_demand > 0:
        hvac.boiler.add_demand(vav.reheat_demand)

      # Apply the thermal energy to the zone.
      self.building.apply_thermal_power_zone(zone, q_zone)

    hvac.boiler.return_water_temperature_sensor = (
        self._calculate_return_water_temperature(zone_supply_temp_map)
    )

    # Increment the timestamp.
    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit='s')

  def _get_zone_reward_info(
      self,
      occupancy_function: BaseOccupancy,
      zone_coords: Tuple[int, int],
      zone_id: str,
      zone_air_temperature: float,
  ) -> RewardInfo.ZoneRewardInfo:
    """Returns a messagde with zone data to compute the instantaneous reward."""
    schedule = self._hvac.vavs[zone_coords].thermostat.get_setpoint_schedule()
    heating_setpoint_temperature, cooling_setpoint_temperature = (
        schedule.get_temperature_window(self._current_timestamp)
    )
    air_flow_rate_setpoint = self._hvac.vavs[zone_coords].max_air_flow_rate
    air_flow_rate = self._hvac.air_handler.air_flow_rate
    average_occupancy = occupancy_function.average_zone_occupancy(
        zone_id,
        self._current_timestamp,
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit='s'),
    )
    zone_info = RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heating_setpoint_temperature,
        cooling_setpoint_temperature=cooling_setpoint_temperature,
        zone_air_temperature=zone_air_temperature,
        air_flow_rate_setpoint=air_flow_rate_setpoint,
        air_flow_rate=air_flow_rate,
        average_occupancy=average_occupancy,
    )
    return zone_info

  def _get_zone_reward_infos(
      self, occupancy_function: BaseOccupancy
  ) -> Mapping[str, RewardInfo.ZoneRewardInfo]:
    """Returns a map of messages with zone data.

    This data is used to compute the instantaneous reward.

    Args:
      occupancy_function: An occupancy function.
    """
    zone_reward_infos = {}
    for (
        zone_coords,
        zone_air_temperature,
    ) in self.building.get_zone_average_temps().items():
      zone_id = conversion_utils.zone_coordinates_to_id(zone_coords)
      zone_reward_infos[zone_id] = self._get_zone_reward_info(
          occupancy_function, zone_coords, zone_id, zone_air_temperature
      )
    return zone_reward_infos

  def _get_air_handler_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.AirHandlerRewardInfo]:
    """Returns a map of messages with air handler data.

    This data is used to compute the instantaneous reward.
    """
    air_handler_reward_infos = {}
    air_handler_id = self._hvac.air_handler.device_id()
    blower_electrical_energy_rate = (
        self._hvac.air_handler.compute_intake_fan_energy_rate()
        + self._hvac.air_handler.compute_exhaust_fan_energy_rate()
    )
    recirculation_temp = self.building.temp.mean()
    ambient_temp = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    air_conditioning_electrical_energy_rate = (
        self._hvac.air_handler.compute_thermal_energy_rate(
            recirculation_temp, ambient_temp
        )
    )
    air_handler_reward_info = RewardInfo.AirHandlerRewardInfo(
        blower_electrical_energy_rate=blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate,  # pylint: disable=line-too-long
    )
    air_handler_reward_infos[air_handler_id] = air_handler_reward_info
    return air_handler_reward_infos

  def _get_boiler_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.BoilerRewardInfo]:
    """Returns a map of messages with boiler data.

    This data is used to compute the instantaneous reward.
    """
    boiler_reward_infos = {}
    boiler_id = self._hvac.boiler.device_id()
    return_water_temp = self._hvac.boiler.return_water_temperature_sensor
    natural_gas_heating_energy_rate = (
        self._hvac.boiler.compute_thermal_energy_rate(
            return_water_temp,
            self._weather_controller.get_current_temp(self._current_timestamp),
        )
    )
    pump_electrical_energy_rate = self._hvac.boiler.compute_pump_power()
    boiler_reward_info = RewardInfo.BoilerRewardInfo(
        natural_gas_heating_energy_rate=natural_gas_heating_energy_rate,
        pump_electrical_energy_rate=pump_electrical_energy_rate,
    )
    boiler_reward_infos[boiler_id] = boiler_reward_info
    return boiler_reward_infos

  def reward_info(self, occupancy_function: BaseOccupancy) -> RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    start_time_stamp = self._current_timestamp
    end_time_stamp = start_time_stamp + pd.Timedelta(
        self._time_step_sec, unit='s'
    )

    # get zone data
    zone_reward_infos = self._get_zone_reward_infos(occupancy_function)

    # get air handler info
    air_handler_reward_infos = self._get_air_handler_reward_infos()

    # get boiler info
    boiler_reward_infos = self._get_boiler_reward_infos()

    return RewardInfo(
        start_timestamp=conversion_utils.pandas_to_proto_timestamp(
            start_time_stamp
        ),
        end_timestamp=conversion_utils.pandas_to_proto_timestamp(
            end_time_stamp
        ),
        zone_reward_infos=zone_reward_infos,
        air_handler_reward_infos=air_handler_reward_infos,
        boiler_reward_infos=boiler_reward_infos,
    )

  def step_sim(self) -> None:
    """Steps the simulation by a small amount of time.

    The following steps are completed in order to proceed to the next time step:

    1. Get external temperature.
    2. Update temperatures for each CV using FDM.
    3. Reset HVAC reheat and flow demands.
    4. For each zone/VAV:

        a. Update the VAV using the zone's temperature.

        b. Apply thermal energy from VAV to the zone.

        c. Accumulate HVAC reheat and flow demands from VAV.

    Note: There is a one step delay in application of current vav
    settings/heating.
    """
    self.setup_step_sim()
    self.execute_step_sim()
