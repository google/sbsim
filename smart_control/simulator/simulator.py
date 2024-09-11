"""Simulator of a simplified building and HVAC devices.

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
    self._building = building
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
    self._building.reset()
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
    delta_x = self._building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    density = self._building.density[x][y]
    conductivity = self._building.conductivity[x][y]
    heat_capacity = self._building.heat_capacity[x][y]
    last_temp = self._building.temp[x][y]
    neighbors = self._building.neighbors[x][y]
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
    delta_x = self._building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    density = self._building.density[x][y]
    conductivity = self._building.conductivity[x][y]
    heat_capacity = self._building.heat_capacity[x][y]
    last_temp = self._building.temp[x][y]
    neighbors = self._building.neighbors[x][y]
    neighbor_temps = [temperature_estimates[nx][ny] for nx, ny in neighbors]

    # Ensure edge CV.
    assert len(neighbors) == 3

    t0 = density * delta_x**2 / 2 * heat_capacity / delta_t
    retained_heat = t0 * last_temp

    # Edges and corners are multiplied by 0.5, others by 1.0
    edge_factor = [
        0.5 if len(self._building.neighbors[nx][ny]) < 4 else 1.0
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
    """Returns temperature estimate for interior CV in K for next time step.

    This function calculates the solution to an equation involving the energy
    transfer by conduction to neighoring air CVs as well as energy transfer
    from heat input to the CV from a diffuser.

    Args:
      cv_coordinates: 2-Tuple representing coordinates in building of CV.
      temperature_estimates: Current temperature estimate for each CV.
    """
    x, y = cv_coordinates
    delta_x = self._building.cv_size_cm / 100.0
    delta_t = self._time_step_sec
    z = self._building.floor_height_cm / 100.0
    density = self._building.density[x][y]
    conductivity = self._building.conductivity[x][y]
    heat_capacity = self._building.heat_capacity[x][y]
    last_temp = self._building.temp[x][y]
    input_q = self._building.input_q[x][y]
    neighbors = self._building.neighbors[x][y]
    neighbor_temps = [temperature_estimates[nx][ny] for nx, ny in neighbors]

    # Ensure interior CV.
    assert len(neighbors) == 4

    alpha = conductivity / density / heat_capacity

    t0 = delta_x**2 / delta_t / alpha

    denominator = 4.0 + t0

    neighbor_transfer = sum(neighbor_temps)

    retained_heat = t0 * last_temp

    thermal_source = input_q / conductivity / z

    return (neighbor_transfer + thermal_source + retained_heat) / denominator

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
    neighbors = self._building.neighbors[x][y]
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
    temp_estimate = self._building.temp.copy()

    converged_successfully = False
    for iteration_count in range(self._iteration_limit):
      temp_estimate, max_delta = self.update_temperature_estimates(
          temp_estimate,
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
      if iteration_count + 1 == self._iteration_warning:
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
      logging.warning(
          'Max iteration count reached, max_delta = %3.3f', max_delta
      )
    self._building.temp = temp_estimate

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
    avg_temps = self._building.get_zone_average_temps()

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
    avg_temps = self._building.get_zone_average_temps()

    # Recirculation temperature at the air handler is the global average.
    recirculation_temp = self._building.temp.mean()

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
      self._building.apply_thermal_power_zone(zone, q_zone)

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
  ) -> smart_control_reward_pb2.RewardInfo.ZoneRewardInfo:
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
    zone_info = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
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
  ) -> Mapping[str, smart_control_reward_pb2.RewardInfo.ZoneRewardInfo]:
    """Returns a map of messages with zone data to compute the instantaneous reward."""
    zone_reward_infos = {}
    for (
        zone_coords,
        zone_air_temperature,
    ) in self._building.get_zone_average_temps().items():
      zone_id = conversion_utils.zone_coordinates_to_id(zone_coords)
      zone_reward_infos[zone_id] = self._get_zone_reward_info(
          occupancy_function, zone_coords, zone_id, zone_air_temperature
      )
    return zone_reward_infos

  def _get_air_handler_reward_infos(
      self,
  ) -> Mapping[str, smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo]:
    """Returns a map of messages with air handler data to compute the instantaneous reward."""
    air_handler_reward_infos = {}
    air_handler_id = self._hvac.air_handler.device_id()
    blower_electrical_energy_rate = (
        self._hvac.air_handler.compute_intake_fan_energy_rate()
        + self._hvac.air_handler.compute_exhaust_fan_energy_rate()
    )
    recirculation_temp = self._building.temp.mean()
    ambient_temp = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    air_conditioning_electrical_energy_rate = (
        self._hvac.air_handler.compute_thermal_energy_rate(
            recirculation_temp, ambient_temp
        )
    )
    air_handler_reward_info = smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
        blower_electrical_energy_rate=blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate,
    )
    air_handler_reward_infos[air_handler_id] = air_handler_reward_info
    return air_handler_reward_infos

  def _get_boiler_reward_infos(
      self,
  ) -> Mapping[str, smart_control_reward_pb2.RewardInfo.BoilerRewardInfo]:
    """Returns a map of messages with boiler data to compute the instantaneous reward."""
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
    boiler_reward_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
        natural_gas_heating_energy_rate=natural_gas_heating_energy_rate,
        pump_electrical_energy_rate=pump_electrical_energy_rate,
    )
    boiler_reward_infos[boiler_id] = boiler_reward_info
    return boiler_reward_infos

  def reward_info(
      self, occupancy_function: BaseOccupancy
  ) -> smart_control_reward_pb2.RewardInfo:
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

    return smart_control_reward_pb2.RewardInfo(
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
      1: Get external temperature.
      2: Update temperatures for each CV using FDM.
      3: Reset HVAC reheat and flow demands.
      4: For each zone/VAV:
        a: Update the VAV using the zone's temperature.
        b: Apply thermal energy from VAV to the zone.
        c: Accumulate HVAC reheat and flow demands from VAV

      Note: There is a one step delay in application of current vav
      settings/heating.
    """
    self.setup_step_sim()
    self.execute_step_sim()
