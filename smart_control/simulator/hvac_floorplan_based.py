"""Models HVAC for simulation post refactor for flexible floorplan geometries.

The model assumes a hot water system with a single boiler, and a singleair
handler, with one VAV per zone in the
building.
"""

import collections
from typing import Mapping
from typing import Union

import gin
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import hot_water_system as hot_water_system_py
from smart_buildings.smart_control.simulator import setpoint_schedule
from smart_buildings.smart_control.simulator import thermostat
from smart_buildings.smart_control.simulator import vav


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
      zone_to_vavs: Mapping[str, list[str]] | None = None,
      vav_max_air_flow_static_pressure: float = 20000.0,
      zone_identifier: list[str] | None = None,
  ):
    """Initialize HVAC.

    Args:
      air_handler: the air handler for the HVAC
      hot_water_system: the hot water system for the HVAC
      schedule: the setpoint_schedule for the thermostats
      vav_max_air_flow_rate: the max airflow rate for the vavs
      vav_reheat_max_water_flow_factor: the max water reheat flow factor for the
        vavs in (m^3/s)/sqrt(Pa).
      zone_to_vavs: mapping of custom zone to vavs. For example:
        {'zone_a': ['vav_1', 'vav_2'], 'zone_b': ['vav_3']}.
      vav_max_air_flow_static_pressure: the  air flow static pressure for the
        vavs at which the max air flow rate can be reached.
      zone_identifier: list of strings containing zone coordinates to service.
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
      self.initialize_zone_identifier(zone_identifier, zone_to_vavs)
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

  def get_zones_for_vav(self, vav_id: str) -> list[str]:
    """Returns the list of zone identifiers serviced by a specific VAV."""
    return self._vav_id_to_zones.get(vav_id, [])

  def get_vav_ids_for_zone(self, zone_id: str) -> list[str]:
    """Returns the list of VAV identifiers servicing a specific zone."""
    return self._zone_to_vav_ids.get(zone_id, [])

  def initialize_zone_identifier(
      self,
      zone_identifier: list[str],
      zone_to_vavs: Mapping[str, list[str]] | None = None,
  ):
    """Initializes the zone devices with zone coordinates passed in.

    Args:
      zone_identifier: list of strings with the room names.
      zone_to_vavs: mapping of custom zone to vavs.
    """

    if zone_to_vavs is None:
      zone_to_vavs = {z: [z.replace("room", "VAV")] for z in zone_identifier}

    self._zone_to_vav_ids = zone_to_vavs
    self._vav_id_to_zones = collections.defaultdict(list)
    for zone, v_ids in self._zone_to_vav_ids.items():
      for v_id in v_ids:
        self._vav_id_to_zones[v_id].append(zone)

    self._create_vavs()

    for z, v_ids in self._zone_to_vav_ids.items():
      self._zone_infos[z] = building_pb2.ZoneInfo(
          zone_id=z,
          building_id="US-SIM-001",
          zone_description="Simulated zone",
          devices=v_ids,
          zone_type=building_pb2.ZoneInfo.ROOM,
          floor=0,
      )

  def set_override_zones(self, zones: list[building_pb2.ZoneInfo]):
    """Overrides the zones in the HVAC system.

    Args:
      zones: A list of ZoneInfo objects.
    """
    self._zone_infos.clear()
    self._zone_to_vav_ids = {}
    self._vav_id_to_zones = collections.defaultdict(list)

    for zone in zones:
      self._zone_infos[zone.zone_id] = zone
      self._zone_to_vav_ids[zone.zone_id] = list(zone.devices)
      for v_id in zone.devices:
        self._vav_id_to_zones[v_id].append(zone.zone_id)

    self._create_vavs()

  def _create_vavs(self) -> None:
    """Creates VAV devices for the HVAC system."""
    self._vavs = {}
    for v_id, affected_zones in sorted(self._vav_id_to_zones.items()):
      # Vav constructor requires a single zone as id
      rep_zone = affected_zones[0]
      therm = thermostat.Thermostat(self._schedule)
      vav_device = vav.Vav(
          max_air_flow_rate=self._vav_max_air_flow_rate,
          reheat_max_water_flow_factor=self._vav_reheat_max_water_flow_factor,
          therm=therm,
          hot_water_system=self._hot_water_system,
          air_handler=self._air_handler.get_vav_air_handler(rep_zone),
          device_id=v_id,
          zone_id=rep_zone,
          max_air_flow_static_pressure=self._vav_max_air_flow_static_pressure,
      )
      self._vavs[v_id] = vav_device

  def reset(self):
    self.air_handler.reset()
    self.hot_water_system.reset()
    for v_id in self._vavs:
      self._vavs[v_id].reset()

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
  def zone_infos(self) -> Mapping[str, building_pb2.ZoneInfo]:
    return self._zone_infos
