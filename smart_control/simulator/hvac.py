"""Models HVAC for simulation.

The model assumes a single boiler and air handler, with one VAV per zone in the
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

from typing import List, Mapping, Tuple

import gin
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import thermostat
from smart_control.simulator import vav
from smart_control.utils import conversion_utils


@gin.configurable
class Hvac:
  """Model for the HVAC components of the building.

  Creates a single boiler and air handler, along with one vav for each zone.

  Attributes:
    vavs: Mapping from zone_coordinates to VAV.
    air_handler: AirHandler
    boiler: Boiler
    zone_infos: information about each zone in the building.
  """

  def __init__(
      self,
      zone_coordinates: List[Tuple[int, int]],
      air_handler: air_handler_py.AirHandler,
      boiler: boiler_py.Boiler,
      schedule: setpoint_schedule.SetpointSchedule,
      vav_max_air_flow_rate: float,
      vav_reheat_max_water_flow_rate: float,
  ):
    """Initialize HVAC.

    Args:
      zone_coordinates: List of 2-tuple containing zone coordinates to service.
      air_handler: the air handler for hte HVAC
      boiler: the boiler for the HVAC
      schedule: the setpoint_schedule for the thermostats
      vav_max_air_flow_rate: the max airflow rate for the vavs
      vav_reheat_max_water_flow_rate: the max water reheat flowrate for the vavs
    """
    self._air_handler = air_handler
    self._boiler = boiler
    self._vav_max_air_flow_rate = vav_max_air_flow_rate
    self._vav_reheat_max_water_flow_rate = vav_reheat_max_water_flow_rate
    self._zone_coordinates = zone_coordinates
    self._vavs = {}
    self._schedule = schedule
    self._zone_infos = {}

    for z in self._zone_coordinates:
      zone_id = conversion_utils.zone_coordinates_to_id(z)
      therm = thermostat.Thermostat(self._schedule)
      device_id = f'vav_{z[0]}_{z[1]}'
      self._vavs[z] = vav.Vav(
          self._vav_max_air_flow_rate,
          self._vav_reheat_max_water_flow_rate,
          therm,
          self._boiler,
          device_id=device_id,
          zone_id=zone_id,
      )
      self._zone_infos[z] = smart_control_building_pb2.ZoneInfo(
          zone_id=zone_id,
          building_id='US-SIM-001',
          zone_description='Simulated zone',
          devices=[device_id],
          zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
          floor=0,
      )
    self.reset()

  def reset(self):
    self.air_handler.reset()
    self.boiler.reset()
    for z in self._zone_coordinates:
      self._vavs[z].reset()

  @property
  def vavs(self) -> Mapping[Tuple[int, int], vav.Vav]:
    return self._vavs

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    return self._air_handler

  @property
  def boiler(self) -> boiler_py.Boiler:
    return self._boiler

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""
    return self._schedule.is_comfort_mode(current_time)

  @property
  def zone_infos(
      self,
  ) -> Mapping[Tuple[int, int], smart_control_building_pb2.ZoneInfo]:
    return self._zone_infos
