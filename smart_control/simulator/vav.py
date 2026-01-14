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

import math
from typing import Optional, Tuple
import uuid
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import hot_water_system as hot_water_system_py
from smart_buildings.smart_control.simulator import smart_device
from smart_buildings.smart_control.simulator import thermostat
from smart_buildings.smart_control.utils import constants


class Vav(smart_device.SmartDevice):
  """Models a Variable Air Volume device with damper and reheat.

  Attributes:
    max_air_flow_rate: Air flow rate when damper is fully open.
    reheat_max_water_flow_factor: Water flow factor when valve is fully open
      (m^3/h).
    reheat_valve_setting: Proportion of water the valve is allowing through [0,
      1].
    damper_setting: Proportion of air the damper is allowing through [0, 1].
    thermostat: Thermostat which controls VAV.
    hot_water_system: Hot water system supplying hot water to the VAV.
    flow_rate_demand: the flow rate demand
    reheat_flow_factor: the reheat demand
    zone_air_temperature: the average temperature in the zone
    max_air_flow_static_pressure: The minimum  pressure at which the design max
      air flow rate can be reached.
  """

  def __init__(
      self,
      max_air_flow_rate: float,
      reheat_max_water_flow_factor: float,
      therm: thermostat.Thermostat,
      hot_water_system: hot_water_system_py.HotWaterSystem,
      air_handler: air_handler_py.AirHandler,
      device_id: Optional[str] = None,
      zone_id: Optional[str] = None,
      max_air_flow_static_pressure: Optional[float] = 20000.0,
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
    self._reheat_max_water_flow_factor = reheat_max_water_flow_factor
    self._init_reheat_valve_setting = 0.0
    self._init_damper_setting = 0.1
    self._init_thermostat = therm
    self._init_zone_air_temperature = 0
    self.reset()
    self._hot_water_system = hot_water_system
    self._air_handler = air_handler
    self._max_air_flow_static_pressure = max_air_flow_static_pressure

  def reset(self):
    self._max_air_flow_rate = self._init_max_air_flow_rate
    self._reheat_valve_setting = self._init_reheat_valve_setting
    self._damper_setting = self._init_damper_setting
    self._thermostat = self._init_thermostat
    self._zone_air_temperature = self._init_zone_air_temperature

  @property
  def thermostat(self) -> thermostat.Thermostat:
    return self._thermostat

  @property
  def hot_water_system(self) -> hot_water_system_py.HotWaterSystem:
    return self._hot_water_system

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
    return self._compute_flow_rate_demand()

  def _compute_flow_rate_demand(self) -> float:
    """Returns the flow rate demand of the VAV.

      This assumes that the flow rate of each VAV is not impacted by the flow
    from other VAVs. This allows us to not consider the damper positions of the
    other VAVs, which simplifies things considerably.

    Thus, we can compute the flow rate demand as: first getting the ratio of
    static pressure to design static pressure. Then using the ratio to adjust
    the max air flow rate, so that when the static pressure is at the design
    value, the max air flow rate is reached. The sqrt is used because pressure
    and airflow have a quadratic relationship.
    """

    flow_rate_demand = (
        self._damper_setting
        * self._max_air_flow_rate
        * math.sqrt(
            self._air_handler.fan_static_pressure
            / self._max_air_flow_static_pressure
        )
    )

    # we will assume that there is always a minimal flow rate demand, even when
    # the damper is closed. This is to avoid dividing by zero.
    return max(flow_rate_demand, 0.00001)

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    return self._air_handler

  @property
  def reheat_flow_factor(self) -> float:
    return self._reheat_valve_setting * self._reheat_max_water_flow_factor

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
    if self._hot_water_system.flow_factor_sum == 0:
      reheat_flow_rate = 0.0
    else:
      reheat_flow_rate = self._hot_water_system.total_flow_rate * (
          self.reheat_flow_factor / self._hot_water_system.flow_factor_sum
      )
    return (
        reheat_flow_rate
        * constants.WATER_HEAT_CAPACITY
        * (input_water_temp - supply_air_temp)
    )

  def compute_zone_supply_temp(
      self, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns temperature of air output from the VAV, supplied to the zone.

    Temperatures are measured in Kelvin.

    Args:
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    assert self.damper_setting > 0
    assert self._max_air_flow_rate > 0

    if self._hot_water_system.flow_factor_sum == 0:
      reheat_flow_rate = 0.0
    else:
      reheat_flow_rate = self._hot_water_system.total_flow_rate * (
          self.reheat_flow_factor / self._hot_water_system.flow_factor_sum
      )
    air_flow_rate = self.flow_rate_demand

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
    air_flow_rate = self.flow_rate_demand
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
        zone_temp,
        supply_air_temp,
        self.hot_water_system.supply_water_temperature_sensor,
    )
    temp_vav_supply = self.compute_zone_supply_temp(
        supply_air_temp, self.hot_water_system.supply_water_temperature_sensor
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
