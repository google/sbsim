"""Simulator of a simplified thermodynamic system for flexible geometries.

This simulator uses finite differences method (FDM) to approximate the
temperature changes in each Control Volume (CV) in a building.

The `execute_step_sim` method is responsible for calculating the supply air
temperature from the air handler(s) for the current simulation time step.

First, it calculates the recirculation air temperature for each air handler
unit (AHU). This is the area-weighted average temperature of all zones serviced
by that AHU.

It iterates through each Variable Air Volume (VAV) unit to find the average
temperature of the zones it serves (v_sensing_temp), weighted by the area of
each zone. It then calculates the recirculation temperature for each AHU by
taking an area-weighted average of the v_sensing_temp of all VAVs connected
to it. Finally, it uses this per-AHU recirculation temperature and the outside
ambient temperature to calculate the temperature of the air that will be
supplied by each AHU. This logic handles both single and multiple AHU
configurations.
"""

import collections
from typing import Mapping, Optional, Tuple

from absl import logging
import gin
import numpy as np
import pandas as pd
from smart_buildings.smart_control.models.base_occupancy import BaseOccupancy
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
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

    zone_identifiers = [
        z
        for z in self.building.room_dict.keys()
        if not constants.is_non_physical_space(z)
    ]
    self._hvac.initialize_zone_identifier(
        zone_identifiers, self.building.custom_zone_to_vavs
    )

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

  def setup_step_sim(self) -> None:
    """Disables the base Simulator setup step.

    This method is overridden with a `pass` because SimulatorFlexibleGeometries
    handles VAV updates more robustly in `execute_step_sim` using weighted
    averages for shared zones and custom VAV IDs. The base class's
    `setup_step_sim` assumes a 1:1 mapping between room names and VAV IDs,
    which causes a KeyError when custom mappings are used.
    """
    pass

  def step_sim(self, video_filename: Optional[str] = "sample.mp4") -> None:
    """Steps simulation, executing one time step."""
    self.execute_step_sim(video_filename)

  def execute_step_sim(
      self, video_filename: Optional[str] = "sample.mp4"
  ) -> None:
    """This method should not change any actions set on smart devices."""

    current_ts = self._current_timestamp
    hvac = self._hvac
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}
    avg_temps = self.building.get_zone_average_temps()
    ambient_temperature = self._weather_controller.get_current_temp(current_ts)

    ahu_temp_weighted_sum = collections.defaultdict(float)
    ahu_area_total = collections.defaultdict(float)
    vav_cached_data = {}
    vav_sensing_temp_map = {}
    vav_supply_temp_map = {}

    for v_id, vav in hvac.vavs.items():
      ahu_id = vav.air_handler.device_id()
      assigned_zones = [
          z for z in hvac.get_zones_for_vav(v_id) if z in avg_temps
      ]
      if not assigned_zones:
        continue

      v_area = sum(room_areas[z] for z in assigned_zones)
      v_sensing_temp = (
          sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
      )
      vav.update_settings(v_sensing_temp, current_ts)

      vav_cached_data[v_id] = {
          "assigned_zones": assigned_zones,
          "v_area": v_area,
          "v_sensing_temp": v_sensing_temp,
      }
      vav_sensing_temp_map[v_id] = v_sensing_temp

      ahu_temp_weighted_sum[ahu_id] += v_sensing_temp * v_area
      ahu_area_total[ahu_id] += v_area

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
      ahus = hvac.air_handler.ahus
    else:
      ahus = [hvac.air_handler]

    recirculation_temps = {}
    for ahu in ahus:
      ahu_id = ahu.device_id()
      if ahu_area_total.get(ahu_id, 0) > 0:
        recirculation_temps[ahu_id] = (
            ahu_temp_weighted_sum[ahu_id] / ahu_area_total[ahu_id]
        )
      else:
        recirculation_temps[ahu_id] = self.building.temp.mean()

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
      supply_air_temp = hvac.air_handler.get_supply_air_temp(
          recirculation_temps, ambient_temperature
      )
    else:
      # Single AHU case
      supply_air_temp = hvac.air_handler.get_supply_air_temp(
          recirculation_temps[hvac.air_handler.device_id()], ambient_temperature
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

    # Stores the total thermal energy to be applied to each physical zone
    zone_q_aggregate = collections.defaultdict(float)

    for v_id, vav in hvac.vavs.items():
      if v_id not in vav_cached_data:
        continue
      cached_data = vav_cached_data[v_id]
      assigned_zones = cached_data["assigned_zones"]
      total_vav_area = cached_data["v_area"]
      weighted_sensing_temp = cached_data["v_sensing_temp"]

      # 2) thermal computation
      if isinstance(supply_air_temp, dict):
        current_supply_temp = supply_air_temp[vav.air_handler.device_id()]
      else:
        current_supply_temp = supply_air_temp
      q_vav, temp_vav_supply = vav.output(
          weighted_sensing_temp, current_supply_temp
      )

      vav_supply_temp_map[v_id] = temp_vav_supply

      # 3) virtual vav distribution to zones proportional to area
      for z in assigned_zones:
        area_share = room_areas[z] / total_vav_area
        zone_q_aggregate[z] += q_vav * area_share

      # Update the air handler airflow demand by summing from all VAVs.
      if vav.flow_rate_demand > 0:
        vav.air_handler.add_demand(vav.flow_rate_demand)

      # apply aggregated power to building
    for z, q_total in zone_q_aggregate.items():
      self.building.apply_thermal_power_zone(z, q_total)

    hvac.hot_water_system.return_water_temperature_sensor = (
        self._calculate_return_water_temperature(vav_supply_temp_map)
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

  def _calculate_return_water_temperature(
      self, vav_supply_temps: Mapping[str, float]
  ) -> float:
    """Calculates return water temperature based on VAV supply air temperatures.

    This assumes that the water returning from each VAV is at the same
    temperature as the VAV supply air it provides.

    Args:
      vav_supply_temps: A mapping from VAV id to its supply air temperature.

    Returns:
      The calculated return water temperature.
    """
    numerator = 0.0
    denominator = 0.0
    for v_id, vav in self._hvac.vavs.items():
      if v_id in vav_supply_temps:
        numerator += vav.reheat_flow_factor * vav_supply_temps[v_id]
        denominator += vav.reheat_flow_factor
    if denominator > 1e-6:
      return numerator / denominator
    else:
      # If there is no reheat flow, the return water temperature does not
      # change. Return the current sensor value.
      return self._hvac.hot_water_system.return_water_temperature_sensor

  def _get_zone_reward_info(
      self,
      occupancy_function: BaseOccupancy,
      zone_coords: str,
      zone_id: str,
      zone_air_temperature: float,
  ) -> RewardInfo.ZoneRewardInfo:
    """Returns a message with zone data to compute the instantaneous reward.

    This assumes all VAVs for a zone share the same setpoint schedule,
    so the schedule is taken from the first VAV. If this assumption
    changes with local control, this method will need updating. Airflow from all
    VAVs feeding the zone is aggregated into a single virtual VAV to compute
    the correct airflow setpoint and actual airflow.

    Args:
      occupancy_function: The occupancy function to use for the zone.
      zone_coords: The coordinates of the zone.
      zone_id: The identifier of the zone.
      zone_air_temperature: The air temperature in the zone.

    Returns:
      A ZoneRewardInfo message.
    """
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}
    vav_ids = self._hvac.get_vav_ids_for_zone(zone_coords)
    vavs = [self._hvac.vavs[v_id] for v_id in vav_ids]
    if not vavs:
      logging.warning("Zone %s has no servicing VAVs.", zone_id)
      return RewardInfo.ZoneRewardInfo()
    schedule = vavs[0].thermostat.get_setpoint_schedule()
    heating_setpoint_temperature, cooling_setpoint_temperature = (
        schedule.get_temperature_window(self._current_timestamp)
    )
    # Aggregate Capacity and Demand with Area Weighting
    zone_total_flow_setpoint = 0.0
    zone_total_flow_actual = 0.0
    for v in vavs:
      # How much of this specific VAV's total footprint belongs to this room?
      serviced_zones = self._hvac.get_zones_for_vav(v.device_id())
      total_vav_area = sum(room_areas[z] for z in serviced_zones)
      area_share = room_areas[zone_coords] / total_vav_area
      zone_total_flow_setpoint += v.max_air_flow_rate * area_share
      zone_total_flow_actual += v.flow_rate_demand * area_share

    average_occupancy = occupancy_function.average_zone_occupancy(
        zone_id,
        self._current_timestamp,
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit="s"),
    )
    return RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heating_setpoint_temperature,
        cooling_setpoint_temperature=cooling_setpoint_temperature,
        zone_air_temperature=zone_air_temperature,
        air_flow_rate_setpoint=zone_total_flow_setpoint,
        air_flow_rate=zone_total_flow_actual,
        average_occupancy=average_occupancy,
    )

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
      zone_id = zone_coords
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
    ambient_temp = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    avg_temps = self.building.get_zone_average_temps()
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}

    ahu_temp_weighted_sum = collections.defaultdict(float)
    ahu_area_total = collections.defaultdict(float)

    for v_id, vav in self._hvac.vavs.items():
      ahu_id = vav.air_handler.device_id()
      assigned_zones = [
          z for z in self._hvac.get_zones_for_vav(v_id) if z in avg_temps
      ]
      if not assigned_zones:
        continue

      v_area = sum(room_areas[z] for z in assigned_zones)
      if v_area > 0:
        v_sensing_temp = (
            sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
        )
        ahu_temp_weighted_sum[ahu_id] += v_sensing_temp * v_area
        ahu_area_total[ahu_id] += v_area

    if isinstance(self._hvac.air_handler, air_handler_py.AirHandlerSystem):
      ahus = self._hvac.air_handler.ahus
    else:
      ahus = [self._hvac.air_handler]

    recirculation_temps = {}
    for ahu in ahus:
      ahu_id = ahu.device_id()
      if ahu_area_total.get(ahu_id, 0) > 0:
        recirculation_temps[ahu_id] = (
            ahu_temp_weighted_sum[ahu_id] / ahu_area_total[ahu_id]
        )
      else:
        recirculation_temps[ahu_id] = self.building.temp.mean()

    if isinstance(self._hvac.air_handler, air_handler_py.AirHandlerSystem):
      recirculation_input = recirculation_temps
    else:
      # Single AHU case
      recirculation_input = recirculation_temps[
          self._hvac.air_handler.device_id()
      ]

    air_conditioning_electrical_energy_rate = (
        self._hvac.air_handler.compute_thermal_energy_rate(
            recirculation_input, ambient_temp
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
  ) -> tuple[
      Mapping[str, RewardInfo.BoilerRewardInfo | RewardInfo.HeatPumpRewardInfo],
      smart_control_building_pb2.DeviceInfo.DeviceType,
  ]:
    """Returns a map of messages with hot water system data.

    This data is used to compute the instantaneous reward.
    """
    hws_reward_infos = {}
    hws_id = self._hvac.hot_water_system.device_id()
    return_water_temp = (
        self._hvac.hot_water_system.return_water_temperature_sensor
    )
    heating_energy_rate = (
        self._hvac.hot_water_system.compute_thermal_energy_rate(
            return_water_temp,
            self._weather_controller.get_current_temp(self._current_timestamp),
        )
    )
    pump_electrical_energy_rate = (
        self._hvac.hot_water_system.compute_pump_power()
    )
    hws_device_type = self._hvac.hot_water_system.heat_source_device_type
    device_type = smart_control_building_pb2.DeviceInfo.DeviceType
    if hws_device_type == device_type.BLR:
      hws_reward_info = RewardInfo.BoilerRewardInfo(
          natural_gas_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    elif hws_device_type == device_type.ASHP:
      hws_reward_info = RewardInfo.HeatPumpRewardInfo(
          electricity_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    else:
      raise ValueError(
          f"Unsupported heat source device type: {hws_device_type}"
      )

    hws_reward_infos[hws_id] = hws_reward_info
    return hws_reward_infos, hws_device_type

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
    hws_reward_infos, hws_device_type = self._get_hws_reward_infos()

    reward_info_args = {
        "start_timestamp": conversion_utils.pandas_to_proto_timestamp(
            start_time_stamp
        ),
        "end_timestamp": conversion_utils.pandas_to_proto_timestamp(
            end_time_stamp
        ),
        "zone_reward_infos": zone_reward_infos,
        "air_handler_reward_infos": air_handler_reward_infos,
    }

    if hws_device_type == smart_control_building_pb2.DeviceInfo.DeviceType.BLR:
      reward_info_args["boiler_reward_infos"] = hws_reward_infos
    elif (
        hws_device_type == smart_control_building_pb2.DeviceInfo.DeviceType.ASHP
    ):
      reward_info_args["heat_pump_reward_infos"] = hws_reward_infos
    else:
      raise ValueError(
          f"Unsupported heat source device type: {hws_device_type}"
      )
    return RewardInfo(**reward_info_args)

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
