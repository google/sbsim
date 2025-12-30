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
  """Simulates thermodynamics of a building.

  This simulator uses finite differences method (FDM) to approximate the
  temperature changes in each Control Volume (CV) in a building. This happens
  through an iterative process described in the finite_differences_timestep
  method.
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
    """
    self.building = building
    self._hvac = hvac
    self._weather_controller = weather_controller
    self._time_step_sec = time_step_sec
    self._convergence_threshold = convergence_threshold
    self._iteration_limit = iteration_limit
    self._iteration_warning = iteration_warning
    self._start_timestamp = start_timestamp
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

  def _get_corner_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Returns temperature estimate for corner CV in K for next time step.

    This function calculates the solution to an equation involving the energy
    transfer by conduction to neighoring air CVs as well as energy transfer by
    convection from the external ambient air.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    delta_x = self.building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    density = self.building.density[x][y]
    conductivity = self.building.conductivity[x][y]
    heat_capacity = self.building.heat_capacity[x][y]
    last_temp = self.building.temp[x][y]
    neighbors = self.building.neighbors[x][y]
    neighbor_temps = [temperature_estimates[nx][ny] for nx, ny in neighbors]

    # Ensure corner CV.
    assert len(neighbors) == 2

    t0 = density * delta_x**2 * heat_capacity / delta_t / 2.0
    retained_heat = t0 * last_temp
    neighbor_transfer = conductivity * sum(neighbor_temps)
    convection_transfer = (
        2.0 * convection_coefficient * delta_x * ambient_temperature
    )
    denominator = (
        2.0 * conductivity + 2.0 * convection_coefficient * delta_x + t0
    )

    return (
        neighbor_transfer + convection_transfer + retained_heat
    ) / denominator

  def _get_edge_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Returns temperature estimate for edge CV in K for next time step.

    This function calculates the solution to an equation involving the energy
    transfer by conduction to neighoring air CVs as well as energy transfer by
    convection from the external ambient air.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    x, y = cv_coordinates
    delta_x = self.building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    density = self.building.density[x][y]
    conductivity = self.building.conductivity[x][y]
    heat_capacity = self.building.heat_capacity[x][y]
    last_temp = self.building.temp[x][y]
    neighbors = self.building.neighbors[x][y]
    neighbor_temps = [temperature_estimates[nx][ny] for nx, ny in neighbors]

    # Ensure edge CV.
    assert len(neighbors) == 3

    t0 = density * delta_x**2 / 2 * heat_capacity / delta_t
    retained_heat = t0 * last_temp

    # Edges and corners are multiplied by 0.5, others by 1.0
    edge_factor = [
        0.5 if len(self.building.neighbors[nx][ny]) < 4 else 1.0
        for nx, ny in neighbors
    ]

    neighbor_transfer = conductivity * sum(
        [f * n for f, n in zip(edge_factor, neighbor_temps)]
    )

    convection_transfer = convection_coefficient * delta_x * ambient_temperature

    denominator = 2.0 * conductivity + convection_coefficient * delta_x + t0

    return (
        neighbor_transfer + convection_transfer + retained_heat
    ) / denominator

  def _get_interior_cv_temp_estimate(
      self, cv_coordinates: CVCoordinates, temperature_estimates: np.ndarray
  ) -> float:
    r"""Returns temperature estimate for interior CV in K for next time step.

    This function calculates the solution to an equation involving the energy
    transfer by conduction to neighboring air CVs, heat input from a diffuser,
    radiative exchange with interior surfaces, and heat exchange with interior
    mass nodes (if present).

    Equations:
    --------------------
    The energy balance for an interior control volume (CV) with interior mass
    is given by:

    $$\begin{multline}
      k_1 (v z) \frac{T_{i-1,j} - T_{i,j}}{u} +
      k_2 (u z) \frac{T_{i,j-1} - T_{i,j}}{v} +
      k_3 (v z) \frac{T_{i+1,j} - T_{i,j}}{u} +
      k_4 (u z) \frac{T_{i,j+1} - T_{i,j}}{v} \\
      + Q_x + \frac{k_{\text{mass}} u v}{z}
      (T_{\text{mass},i,j} - T_{i,j}) + q_{\text{lwx}} =
      \frac{\rho c u v z}{\Delta t} \left( T_{i,j} - T_{i,j}^{(-)} \right)
      \end{multline}$$

    Solving for $T_{i,j}$ with uniform spacing ($u = v = \delta_x$) and uniform
    conductivity ($k_1 = k_2 = k_3 = k_4 = k$):

    $$T_{i,j} = \frac{\sum_{\text{neighbors}} T_{\text{neighbor}} +
      \frac{Q_x}{z k} + \frac{k_{\text{mass}} \delta_x^2}{z^2 k}
      T_{\text{mass},i,j}+\frac{q_\text{lwx}}{zk} + t_0 T_{i,j}^{(-)}}
      {4 + \frac{k_{\text{mass}} \delta_x^2}{z^2 k} + t_0}$$

    where the temporal parameter is:

    $$t_0 = \frac{\rho c \delta_x^2}{k \Delta t} =
      \frac{\delta_x^2}{\Delta t \cdot \alpha}$$

    and the thermal diffusivity is:

    $$\alpha = \frac{k}{\rho c}$$

    Nomenclature and Units:
    -----------------------
    - $T_{i,j}$: Air temperature at CV $(i,j)$ at new time step [K]
    - $T_{i,j}^{(-)}$: Air temperature at CV $(i,j)$ at previous time step [K]
    - $T_{\text{mass},i,j}$: Interior mass temperature at CV $(i,j)$ [K]
    - $T_{i-1,j}, T_{i+1,j}, T_{i,j-1}, T_{i,j+1}$: Neighbor CV temperatures
      (left, right, bottom, top) [K]
    - $k_1, k_2, k_3, k_4$: Thermal conductivity for left, bottom, right, top
      faces [$\mathrm{W/(m \cdot K)}$]
    - $k$: Thermal conductivity (uniform assumption) [$\mathrm{W/(m \cdot K)}$]
    - $k_{\text{mass}}$: Thermal conductivity of interior mass
      [$\mathrm{W/(m \cdot K)}$]
    - $Q_x$: External heat source (e.g., diffuser) [$\mathrm{W}$]
    - $q_{\text{lwx}}$: Longwave radiative exchange [$\mathrm{W}$]
    - $u, v$: CV dimensions in x and y directions [$\mathrm{m}$]
    - $\delta_x$: Spatial discretization (uniform CV size) [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $\rho$: Density [$\mathrm{kg/m^3}$]
    - $c$: Specific heat capacity [$\mathrm{J/(kg \cdot K)}$]
    - $\alpha$: Thermal diffusivity [$\mathrm{m^2/s}$]
    - $\Delta t$: Time step [$\mathrm{s}$]
    - $t_0$: Temporal parameter [dimensionless]

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
    """
    x, y = cv_coordinates
    delta_x = self.building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    z = self.building.floor_height_cm / 100.0
    density = self.building.density[x][y]
    conductivity = self.building.conductivity[x][y]

    heat_capacity = self.building.heat_capacity[x][y]
    last_temp = self.building.temp[x][y]
    input_q = self.building.input_q[x][y]
    neighbors = self.building.neighbors[x][y]
    neighbor_temps = [temperature_estimates[nx][ny] for nx, ny in neighbors]
    # Ensure interior CV.
    assert len(neighbors) == 4

    alpha = conductivity / density / heat_capacity

    t0 = delta_x**2 / delta_t / alpha

    neighbor_transfer = sum(neighbor_temps)

    retained_heat = t0 * last_temp

    thermal_source = input_q / conductivity / z

    # Interior mass heat transfer (adiabatic node connected only to air CV)
    if (
        hasattr(self.building, 'include_interior_mass')
        and self.building.include_interior_mass
        and self.building.interior_mass_mask[x, y]
    ):
      interior_mass_conductivity = self.building.interior_mass_conductivity[x][
          y
      ]
      denominator = (
          4.0
          + interior_mass_conductivity * delta_x**2 / conductivity / z**2
          + t0
      )

      # Heat transfer between air CV and its interior mass node
      interior_mass_temp = self.building.interior_mass_temp[x, y]
      # Heat flux from interior mass to air CV
      neighbor_transfer += (
          interior_mass_temp
          * delta_x**2
          * interior_mass_conductivity
          / conductivity
          / z**2
      )
    else:
      denominator = 4.0 + t0

    # checking for implementation of `include_radiative_heat_transfer` because
    # the `FloorPlanBasedBuilding` implements it, but the `Building` doesn't
    if (
        hasattr(self.building, 'include_radiative_heat_transfer')
        and self.building.include_radiative_heat_transfer
    ):
      # Radiative heat transfer
      q_lwx_array = (
          self.building.apply_longwave_interior_radiative_heat_transfer(
              temperature_estimates
          )
      )
      # q_lwx_idx is -1 if the CV does not have LWX
      q_lwx_idx = self.building.lwx_index[x, y]
      q_lwx = (
          (q_lwx_array[q_lwx_idx] / conductivity / z)
          if q_lwx_idx != -1
          else 0.0
      )
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

    return temperature_estimates, max_delta

  def update_interior_mass_temperatures(
      self, air_temperature_estimates: np.ndarray
  ) -> tuple[np.ndarray, float]:
    r"""Updates interior mass node temperatures based on heat transfer with air
       CVs.

    Interior mass nodes are adiabatic (no interaction with each other) and only
    exchange heat with their corresponding air CV. The heat exchange occurs
    through the vertical direction (height z) of the control volume.

    Equations:
    --------------------
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

    Nomenclature and Units:
    -----------------------
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
    """Calculates the temperature for each Control Volume (CV) after a step.

    To find the temperature after conduction/convection for each CV, we set
    up a system of linear equations. To approximate the solution:

    1.   Create a starting estimate temperature for each CV.
    2.   For each CV, solve for temperature T, based on the current estimate
         for neighboring CVs and known thermal losses/gains.
    3.   Calculate the difference between previous T and new T.
    4.   If interior mass is enabled, update interior mass temperatures and
         check their convergence as well.

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
        # estimates
        interior_mass_temp_estimate, max_delta_mass = (
            self.update_interior_mass_temperatures(temp_estimate)
        )
        # Store the updated interior mass temperatures
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
