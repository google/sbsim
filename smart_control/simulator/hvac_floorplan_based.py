"""Models HVAC for simulation post refactor for flexible floorplan geometries.

The model assumes a hot water system with a single boiler, and a singleair
handler, with one VAV per zone in the
building.

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

from typing import List, Mapping, Optional
from typing import Union

import gin
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import constants
from smart_buildings.smart_control.simulator import hot_water_system as hot_water_system_py
from smart_buildings.smart_control.simulator import setpoint_schedule
from smart_buildings.smart_control.simulator import thermostat
from smart_buildings.smart_control.simulator import vav
from smart_buildings.smart_control.utils import conversion_utils


@gin.configurable
class FloorPlanBasedHvac:
  """Model for the HVAC components of the building.

  Creates a single hot water system and air handler, along with one vav for each
  zone.

  Attributes:
    vavs: Mapping from zone_identifier to VAV.
    air_handler: AirHandler
    hot_water_system: HotWaterSystem
    zone_infos: information about each zone in the building.
    fill_zone_identifier_exogenously: flag to tell simulator to fill the zone
      coordinates exogenously or not.
  """

  def __init__(
      self,
      air_handler: Union[
          air_handler_py.AirHandler, air_handler_py.AirHandlerSystem
      ],
      hot_water_system: hot_water_system_py.HotWaterSystem,
      schedule: setpoint_schedule.SetpointSchedule,
      vav_max_air_flow_rate: float,
      vav_reheat_max_water_flow_factor: float,
      vav_max_air_flow_static_pressure: float = 20000.0,
      zone_identifier: Optional[List[str]] = None,
  ):
    """Initialize HVAC.

    Args:
      air_handler: the air handler for the HVAC
      hot_water_system: the hot water system for the HVAC
      schedule: the setpoint_schedule for the thermostats
      vav_max_air_flow_rate: the max airflow rate for the vavs
      vav_reheat_max_water_flow_factor: the max water reheat flow factor for the
        vavs
      vav_max_air_flow_static_pressure: the  air flow static pressure for the
        vavs at which the max air flow rate can be reached.
      zone_identifier: List of strings containing zone coordinates to service.
        If None, then the Simulator which calls the hvac must have a list of
        rooms that it plans on passing.
    """
    self.fill_zone_identifier_exogenously = True
    self._air_handler = air_handler
    self._hot_water_system = hot_water_system
    self._vav_max_air_flow_rate = vav_max_air_flow_rate
    self._vav_reheat_max_water_flow_factor = vav_reheat_max_water_flow_factor
    self._vav_max_air_flow_static_pressure = vav_max_air_flow_static_pressure
    self._vavs = {}
    self._schedule = schedule
    self._zone_infos = {}

    if zone_identifier is not None:
      self.initialize_zone_identifier(zone_identifier)
      self.fill_zone_identifier_exogenously = False

  @property
  def schedule(self) -> setpoint_schedule.SetpointSchedule:
    """Returns the building operational schedule for the HVAC."""
    return self._schedule

  @property
  def vav_max_air_flow_rate(self) -> float:
    """Returns the max air flow rate for the vavs."""
    return self._vav_max_air_flow_rate

  @property
  def vav_reheat_max_water_flow_factor(self) -> float:
    """Returns the max water reheat flow factor for the vavs."""
    return self._vav_reheat_max_water_flow_factor

  @property
  def vav_max_air_flow_static_pressure(self) -> float:
    """Returns the max air flow static pressure for the vavs."""
    return self._vav_max_air_flow_static_pressure

  def initialize_zone_identifier(self, zone_identifier: List[str]):
    """Initializes the zone devices with zone coordinates passed in.

    Args:
      zone_identifier: list of strings with the room names.
    """

    filtered_zone_identifier = []

    for z in zone_identifier:
      if (
          z == constants.INTERIOR_WALL_NAME_IN_ROOM_DICT
          or z == constants.EXTERIOR_SPACE_NAME_IN_ROOM_DICT
      ):
        continue
      zone_id = conversion_utils.floor_plan_based_zone_identifier_to_id(
          identifier=z
      )
      therm = thermostat.Thermostat(self._schedule)
      device_id = f"vav_{z}"
      self._vavs[z] = vav.Vav(
          self._vav_max_air_flow_rate,
          self._vav_reheat_max_water_flow_factor,
          therm,
          self._hot_water_system,
          self._air_handler.get_vav_air_handler(zone_id),
          device_id=device_id,
          zone_id=zone_id,
          max_air_flow_static_pressure=self._vav_max_air_flow_static_pressure,
      )
      self._zone_infos[z] = smart_control_building_pb2.ZoneInfo(
          zone_id=zone_id,
          building_id="US-SIM-001",
          zone_description="Simulated zone",
          devices=[device_id],
          zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
          floor=0,
      )
      filtered_zone_identifier.append(z)
    self._zone_identifier = filtered_zone_identifier
    self.reset()

  def reset(self):
    self.air_handler.reset()
    self.hot_water_system.reset()
    for z in self._zone_identifier:
      self._vavs[z].reset()

  @property
  def vavs(self) -> Mapping[str, vav.Vav]:
    return self._vavs

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    return self._air_handler

  @property
  def hot_water_system(self) -> hot_water_system_py.HotWaterSystem:
    return self._hot_water_system

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""
    return self._schedule.is_comfort_mode(current_time)

  @property
  def zone_infos(self) -> Mapping[str, smart_control_building_pb2.ZoneInfo]:
    return self._zone_infos
