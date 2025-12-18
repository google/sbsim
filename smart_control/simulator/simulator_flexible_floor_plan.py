"""Simulator of a simplified thermodynamic system for flexible geometries.

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

from typing import Mapping, Optional, Tuple

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_buildings.smart_control.models.base_occupancy import BaseOccupancy
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import constants
from smart_buildings.smart_control.simulator import hvac_floorplan_based
from smart_buildings.smart_control.simulator import simulator
from smart_buildings.smart_control.simulator import weather_controller as weather_controller_py
from smart_buildings.smart_control.utils import building_renderer
from smart_buildings.smart_control.utils import conversion_utils
from smart_buildings.smart_control.utils import visual_logger

RewardInfo = smart_control_reward_pb2.RewardInfo

CVCoordinates = Tuple[int, int]
ZoneId = Tuple[int, int]


@gin.configurable
class SimulatorFlexibleGeometries(simulator.Simulator):
  """Simulates thermodynamics of a building with flexible geometries.

  NOTE: post-refector

  This simulator uses finite differences method (FDM) to approximate the
  temperature changes in each Control Volume (CV) in a building. This happens
  through an iterative process described in the finite_differences_timestep
  method.
  """

  def __init__(
      self,
      building: building_py.FloorPlanBasedBuilding,
      hvac: hvac_floorplan_based.FloorPlanBasedHvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
  ):
    """Simulator init.

    Args:
      building: Refactored flexible FloorPlanBasedBuilding object controlling
        the control volumes.
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

    logging.info("Constructing the floorplan based simulation.")

    if self._hvac.fill_zone_identifier_exogenously:
      logging.info("Filling zones exogenously")
      self._hvac.initialize_zone_identifier(self.building._room_dict.keys())

    super().__init__(
        self.building,
        self._hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    logging.info("Constructing the floorplan based simulation.")

    render_zones = np.copy(self.building.floor_plan)
    render_zones[render_zones == 2] = 0

    renderer = building_renderer.BuildingRenderer(render_zones, 1)

    self._log_and_plotter = visual_logger.VisualLogger(renderer)
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
  def hvac(self) -> hvac_floorplan_based.FloorPlanBasedHvac:
    return self._hvac

  @property
  def current_timestamp(self) -> pd.Timestamp:
    return self._current_timestamp

  def execute_step_sim(
      self, video_filename: Optional[str] = "sample.mp4"
  ) -> None:
    """This method should not change any actions set on smart devices."""

    current_ts = self._current_timestamp
    hvac = self._hvac

    # Get the average temps in each zone. Assumes that the thermostat reads
    # the average room temperatures.
    avg_temps = self.building.get_zone_average_temps()

    # Recirculation temperature at the air handler is the global average.
    recirculation_temp = sum(list(avg_temps.values())) / len(
        list(avg_temps.values())
    )

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

    # Simulate airflow
    self.building.apply_convection()

    # Reset the air handler and hws flow rate demand before accumulating.
    hvac.air_handler.reset_demand()
    hvac.hot_water_system.reset_demand()

    # sum up all the VAV hot waterdemands for the current timestep
    # this needs to be calculated first before the output function is called,
    # since the flow rate of the entire system can only be determined if we know
    # the total demand from all VAVs
    for vav in hvac.vavs.values():
      hvac.hot_water_system.add_demand(vav.reheat_flow_factor)

    zone_supply_temp_map = {}

    # Iterate through each zone.
    for zone, zone_temp in avg_temps.items():
      vav = hvac.vavs[zone]

      if isinstance(supply_air_temp, dict):
        q_zone, zone_supply_temp = vav.output(
            zone_temp, supply_air_temp[vav.air_handler.device_id()]
        )
      else:
        q_zone, zone_supply_temp = vav.output(
            zone_temp, supply_air_temp
        )
      zone_supply_temp_map[zone] = zone_supply_temp
      # Update the air handler airflow demand by summing from all VAVs.
      if vav.flow_rate_demand > 0:
        vav.air_handler.add_demand(vav.flow_rate_demand)

      # Apply the thermal energy to the zone.
      self.building.apply_thermal_power_zone(zone, q_zone)

    hvac.hot_water_system.return_water_temperature_sensor = (
        self._calculate_return_water_temperature(zone_supply_temp_map)
    )

    # stabilize
    self.building.temp = np.clip(
        self.building.temp,
        250,
        450,
    )

    # Increment the timestamp.
    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit="s")
    self._log_and_plotter.log(self.building.temp)

    if self.current_timestamp == self._start_timestamp + pd.Timedelta(days=4):
      self.get_video(path=constants.VIDEO_PATH_ROOT + video_filename)

  def _get_zone_reward_info(
      self,
      occupancy_function: BaseOccupancy,
      zone_coords: str,
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
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit="s"),
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
      zone_id = conversion_utils.floor_plan_based_zone_identifier_to_id(
          zone_coords
      )
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

  def _get_hws_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.BoilerRewardInfo]:
    """Returns a map of messages with hws data.

    This data is used to compute the instantaneous reward.
    """
    hws_reward_infos = {}
    hws_id = self._hvac.hot_water_system.device_id()
    return_water_temp = (
        self._hvac.hot_water_system.return_water_temperature_sensor
    )
    natural_gas_heating_energy_rate = (
        self._hvac.hot_water_system.compute_thermal_energy_rate(
            return_water_temp,
            self._weather_controller.get_current_temp(self._current_timestamp),
        )
    )
    pump_electrical_energy_rate = (
        self._hvac.hot_water_system.compute_pump_power()
    )
    hws_reward_info = RewardInfo.BoilerRewardInfo(
        natural_gas_heating_energy_rate=natural_gas_heating_energy_rate,
        pump_electrical_energy_rate=pump_electrical_energy_rate,
    )
    hws_reward_infos[hws_id] = hws_reward_info
    return hws_reward_infos

  def reward_info(self, occupancy_function: BaseOccupancy) -> RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    start_time_stamp = self._current_timestamp
    end_time_stamp = start_time_stamp + pd.Timedelta(
        self._time_step_sec, unit="s"
    )

    # get zone data
    zone_reward_infos = self._get_zone_reward_infos(occupancy_function)

    # get air handler info
    air_handler_reward_infos = self._get_air_handler_reward_infos()

    # get hot water system info
    hws_reward_infos = self._get_hws_reward_infos()

    return RewardInfo(
        start_timestamp=conversion_utils.pandas_to_proto_timestamp(
            start_time_stamp
        ),
        end_timestamp=conversion_utils.pandas_to_proto_timestamp(
            end_time_stamp
        ),
        zone_reward_infos=zone_reward_infos,
        air_handler_reward_infos=air_handler_reward_infos,
        boiler_reward_infos=hws_reward_infos,
    )

  def get_video(self, path: str) -> None:
    """Wraps the get_video function from the visual_logger.

    Args:
      path: path to desired video directory (cns, etc.).

    Returns:
      None
    """

    self._log_and_plotter.get_video(
        file_path=path, fps=12, vmin=280, vmax=300, cmap="rainbow"
    )
