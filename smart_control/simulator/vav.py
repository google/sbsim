"""Models a Variable Air Volume device for the simulation.

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

from typing import Optional, Tuple
import uuid

import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import smart_device
from smart_control.simulator import thermostat
from smart_control.utils import constants


class Vav(smart_device.SmartDevice):
  """Models a Variable Air Volume device with damper and reheat.

  Attributes:
    max_air_flow_rate: Air flow rate when damper is fully open.
    reheat_max_water_flow_rate: Water flow rate when valve is fully open.
    reheat_valve_setting: Proportion of water the valve is allowing through [0,
      1].
    damper_setting: Proportion of air the damper is allowing through [0, 1].
    thermostat: Thermostat which controls VAV.
    boiler: Boiler supplying hot water to the VAV.
    flow_rate_demand: the flow rate demand
    reheat_demand: the reheat demand
    zone_air_temperature: the average temperature in the zone
  """

  def __init__(
      self,
      max_air_flow_rate: float,
      reheat_max_water_flow_rate: float,
      therm: thermostat.Thermostat,
      boiler: boiler_py.Boiler,
      device_id: Optional[str] = None,
      zone_id: Optional[str] = None,
  ):
    observable_fields = {
        'supply_air_damper_percentage_command': smart_device.AttributeInfo(
            'damper_setting', float
        ),
        'supply_air_flowrate_setpoint': smart_device.AttributeInfo(
            'max_air_flow_rate', float
        ),
        'zone_air_temperature_sensor': smart_device.AttributeInfo(
            'zone_air_temperature', float
        ),
    }
    action_fields = {
        'supply_air_damper_percentage_command': smart_device.AttributeInfo(
            'damper_setting', float
        ),
    }

    if device_id is None:
      device_id = f'vav_id_{uuid.uuid4()}'
    if zone_id is None:
      zone_id = f'zone_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.VAV,
        device_id=device_id,
        zone_id=zone_id,
    )

    self._init_max_air_flow_rate = max_air_flow_rate
    self._init_reheat_max_water_flow_rate = reheat_max_water_flow_rate
    self._init_reheat_valve_setting = 0.0
    self._init_damper_setting = 0.1
    self._init_thermostat = therm
    self._init_zone_air_temperature = 0
    self.reset()
    self._boiler = boiler

  def reset(self):
    self._max_air_flow_rate = self._init_max_air_flow_rate
    self._reheat_max_water_flow_rate = self._init_reheat_max_water_flow_rate
    self._reheat_valve_setting = self._init_reheat_valve_setting
    self._damper_setting = self._init_damper_setting
    self._thermostat = self._init_thermostat
    self._zone_air_temperature = self._init_zone_air_temperature

  @property
  def thermostat(self) -> thermostat.Thermostat:
    return self._thermostat

  @property
  def boiler(self) -> boiler_py.Boiler:
    return self._boiler

  @property
  def reheat_valve_setting(self) -> float:
    return self._reheat_valve_setting

  @reheat_valve_setting.setter
  def reheat_valve_setting(self, value: float):
    if value < 0 or value > 1:
      raise ValueError('reheat_valve_setting must be in [0 ,1]')
    self._reheat_valve_setting = value

  @property
  def max_air_flow_rate(self) -> float:
    return self._max_air_flow_rate

  @max_air_flow_rate.setter
  def max_air_flow_rate(self, value: float):
    assert value > 0
    self._max_air_flow_rate = value

  @property
  def damper_setting(self) -> float:
    return self._damper_setting

  @damper_setting.setter
  def damper_setting(self, value: float):
    if value < 0 or value > 1:
      raise ValueError('damper_setting must be in [0 ,1]')
    self._damper_setting = value

  @property
  def flow_rate_demand(self) -> float:
    return self._damper_setting * self._max_air_flow_rate

  @property
  def reheat_demand(self) -> float:
    return self._reheat_valve_setting * self._reheat_max_water_flow_rate

  @property
  def zone_air_temperature(self) -> float:
    return self._zone_air_temperature

  def compute_reheat_energy_rate(
      self, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns energy consumption in W due to heating the air.

    Args:
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    reheat_flow_rate = (
        self._reheat_valve_setting * self._reheat_max_water_flow_rate
    )
    return (
        reheat_flow_rate
        * constants.WATER_HEAT_CAPACITY
        * (input_water_temp - supply_air_temp)
    )

  def compute_zone_supply_temp(
      self, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns temperature in K of air output from the VAV, supplied to the zone.

    Args:
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    assert self.damper_setting > 0
    assert self._max_air_flow_rate > 0
    reheat_flow_rate = (
        self._reheat_valve_setting * self._reheat_max_water_flow_rate
    )
    air_flow_rate = self._damper_setting * self._max_air_flow_rate

    heat_difference = (
        constants.AIR_HEAT_CAPACITY * air_flow_rate
        - constants.WATER_HEAT_CAPACITY * reheat_flow_rate
    )
    input_water_heat = (
        input_water_temp * constants.WATER_HEAT_CAPACITY * reheat_flow_rate
    )
    return (
        (supply_air_temp * heat_difference + input_water_heat)
        / air_flow_rate
        / constants.AIR_HEAT_CAPACITY
    )

  def compute_energy_applied_to_zone(
      self, zone_temp: float, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns thermal energy in W to apply to the zone.

    Args:
      zone_temp: Current temperature in K of the zone.
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    if self.damper_setting == 0 or self._max_air_flow_rate == 0:
      return 0
    zone_supply_temp = self.compute_zone_supply_temp(
        supply_air_temp, input_water_temp
    )
    air_flow_rate = self._damper_setting * self._max_air_flow_rate
    return (
        air_flow_rate
        * constants.AIR_HEAT_CAPACITY
        * (zone_supply_temp - zone_temp)
    )

  def update_settings(
      self, zone_temp: float, current_timestamp: pd.Timestamp
  ) -> None:
    """Adjusts the VAV configuration based on thermostat mode.

    Args:
      zone_temp: Current temperature in K of zone.
      current_timestamp: Pandas timestamp representing current time.
    """
    self._zone_air_temperature = zone_temp
    mode = self._thermostat.update(zone_temp, current_timestamp)
    if mode == thermostat.Thermostat.Mode.HEAT:
      self.damper_setting = 1.0
      self.reheat_valve_setting = 1.0
    elif mode == thermostat.Thermostat.Mode.COOL:
      self.damper_setting = 1.0
      self.reheat_valve_setting = 0.0
    elif mode == thermostat.Thermostat.Mode.OFF:
      self.damper_setting = 0.1  # Allow for ventilation
      self.reheat_valve_setting = 0.0
    elif mode == thermostat.Thermostat.Mode.PASSIVE_COOL:
      self.damper_setting = 0.1  # Allow for ventilation
      self.reheat_valve_setting = 0.0
    else:  # Do nothing - keep existing configuration
      pass

  def output(
      self, zone_temp: float, supply_air_temp: float
  ) -> Tuple[float, float]:
    """Returns values corresponding to current output.

    Args:
      zone_temp: Current temperature in K of zone.
      supply_air_temp: Temperature in K of air being supplied to VAV.

    Returns:
      Tuple containing energy to apply to zone and temperature applied to zone.
    """
    self._zone_air_temperature = zone_temp
    q_zone = self.compute_energy_applied_to_zone(
        zone_temp, supply_air_temp, self.boiler.reheat_water_setpoint
    )
    temp_vav_supply = self.compute_zone_supply_temp(
        supply_air_temp, self.boiler.reheat_water_setpoint
    )
    return q_zone, temp_vav_supply

  def update(
      self,
      zone_temp: float,
      current_timestamp: pd.Timestamp,
      supply_air_temp: float,
  ) -> Tuple[float, float]:
    """Returns values corresponding to current output.

    Adjusts the VAV configuration based on thermostat mode.

    Args:
      zone_temp: Current temperature in K of zone.
      current_timestamp: Pandas timestamp representing current time.
      supply_air_temp: Temperature in K of air being supplied to VAV.

    Returns:
      Tuple containing energy to apply to zone, temperature applied to zone, and
      flow rate demand.
    """
    self.update_settings(zone_temp, current_timestamp)
    return self.output(zone_temp, supply_air_temp)
