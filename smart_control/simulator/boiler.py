"""Models a boiler for the simulation.

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

from typing import Optional
import uuid

import gin
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device
from smart_control.utils import constants


@gin.configurable
class Boiler(smart_device.SmartDevice):
  """Models a central boiler with water pump.

  Attributes:
    _total_flow_rate: Flow rate of water in m3/s.
    reheat_water_setpoint: Temperature in K that the boiler will maintain.
    _water_pump_differential_head: Length in meters of pump head.
    _water_pump_efficiency: Electrical efficiency of water pump [0,1].
    device_code: unique name of the device.
    heating_request_count: count of VAVs that have requested heat in this cycle.
    supply_water_temperature_sensor: temp [K] of water being supplied to VAVs.
    supply_water_setpoint: setpoint [K] of the supply water.
    return_water_temperature_sensor: temp [K] of return water
    heating_rate: degrees C / minute a boiler can heat
    cooling_rate: degrees C / minute the boiler temp will drop
    convection_coefficient: convection (airflow) loss external to the boiler
      [W/m^2/K]
    water_capacity: boiler water capacity [m^3]
    tank_length: tank interior length [m]
    tank_radius: tank interior radius [m]
    insulation_conductivity: conductivity of the insulating walls [W/m/K]
    insulation_thickness: thickness of the cylindrical insulating wall [m]
  """

  def __init__(
      self,
      reheat_water_setpoint: float,
      water_pump_differential_head: float,
      water_pump_efficiency: float,
      device_id: Optional[str] = None,
      heating_rate: Optional[float] = 0,
      cooling_rate: Optional[float] = 0,
      convection_coefficient: Optional[float] = 5.6,
      tank_length: Optional[float] = 2.0,
      tank_radius: Optional[float] = 0.5,
      water_capacity: Optional[float] = 1.5,
      insulation_conductivity: Optional[float] = 0.067,
      insulation_thickness: Optional[float] = 0.06,
  ):
    observable_fields = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            'reheat_water_setpoint', float
        ),
        'supply_water_temperature_sensor': smart_device.AttributeInfo(
            'supply_water_temperature_sensor', float
        ),
        'heating_request_count': smart_device.AttributeInfo(
            'heating_request_count', int
        ),
    }

    action_fields = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            'reheat_water_setpoint', float
        )
    }

    if device_id is None:
      device_id = f'boiler_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.BLR,
        device_id=device_id,
    )

    self._init_reheat_water_setpoint = reheat_water_setpoint
    self._init_water_pump_differential_head = water_pump_differential_head
    self._init_water_pump_efficiency = water_pump_efficiency
    self._init_heating_request_count = 0
    self._init_return_water_temperature_sensor = 0.0
    self._heating_rate = heating_rate
    self._cooling_rate = cooling_rate
    self._convection_coefficient = convection_coefficient
    self._tank_length = tank_length
    self._tank_radius = tank_radius
    self._water_capacity = water_capacity
    self._insulation_conductivity = insulation_conductivity
    self._insulation_thickness = insulation_thickness
    self.reset()

  def reset(self):
    self.reset_demand()
    self._reheat_water_setpoint = self._init_reheat_water_setpoint
    self._water_pump_differential_head = self._init_water_pump_differential_head
    self._water_pump_efficiency = self._init_water_pump_efficiency
    self._heating_request_count = self._init_heating_request_count
    self._return_water_temperature_sensor = (
        self._init_return_water_temperature_sensor
    )
    self._current_temperature = self._init_reheat_water_setpoint
    self._step_tank_temperature_change = 0.0
    self._last_step_duration = pd.Timedelta(0, unit='second')

  @property
  def return_water_temperature_sensor(self) -> float:
    return self._return_water_temperature_sensor

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    self._return_water_temperature_sensor = value

  @property
  def reheat_water_setpoint(self) -> float:
    return self._reheat_water_setpoint

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    self._reheat_water_setpoint = value

  @property
  def heating_request_count(self) -> int:
    return self._heating_request_count

  @property
  def supply_water_temperature_sensor(self) -> float:
    self._set_current_temperature()
    return self._current_temperature

  @property
  def supply_water_setpoint(self) -> float:
    return self._reheat_water_setpoint

  def reset_demand(self) -> None:
    self._total_flow_rate = 0.0
    self._heating_request_count = 0

  def _set_current_temperature(self):
    """Adjusts the temperature based on time elapsed after setpoint change."""

    # Retain instantaneous behavior if rates aren't set.
    # If no action has been applied, setpoint and measured temps are equal.
    if self._action_timestamp:
      self._last_step_duration = (
          self._observation_timestamp - self._action_timestamp
      )
    else:
      self._action_timestamp = self._observation_timestamp
    if (
        self._action_timestamp
        and self._cooling_rate > 0.0
        and self._heating_rate > 0.0
    ):
      begin_step_temp = self._current_temperature
      self._current_temperature = self._adjust_temperature(
          self._reheat_water_setpoint, begin_step_temp, self._last_step_duration
      )

      self._step_tank_temperature_change = (
          self._current_temperature - begin_step_temp
      )
    else:
      self._current_temperature = self._reheat_water_setpoint

  def _adjust_temperature(
      self,
      setpoint_temperature: float,
      actual_temperature: float,
      time_difference: pd.Timedelta,
  ) -> float:
    """Adjusts the current temp to match setpoints at a heating/cooling rate.

    Args:
      setpoint_temperature: setpoint temp
      actual_temperature: temperature of the boiling water
      time_difference: amount of time the setpoint has been applied

    Returns:
      temperature with linear heating and cooling.
    """

    if setpoint_temperature > actual_temperature:
      return min(
          actual_temperature
          + self._heating_rate * time_difference.total_seconds() / 60.0,
          setpoint_temperature,
      )

    elif setpoint_temperature < actual_temperature:
      return max(
          actual_temperature
          - self._cooling_rate * time_difference.total_seconds() / 60.0,
          setpoint_temperature,
      )

    else:
      return setpoint_temperature

  def add_demand(self, flow_rate: float):
    """Adds to current flow rate demand.

    Args:
      flow_rate: Flow rate to add.

    Raises:
      ValueError: If flow_rate is not positive.
    """
    if flow_rate <= 0:
      raise ValueError('Flow rate must be positive')
    self._total_flow_rate += flow_rate
    self._heating_request_count += 1

  def compute_thermal_energy_rate(
      self, return_water_temp: float, outside_temp: float
  ) -> float:
    """Returns energy rate in W consumed by boiler to heat water.

    Args:
      return_water_temp: Temperature in K that water is received at.
      outside_temp: Temperature in K that the water tank is in.
    """
    # If return_water_temp is greater than the setpoint,
    # the boiler should not be cooling.
    if self._reheat_water_setpoint > return_water_temp:
      supply_water_temp = self._reheat_water_setpoint
    else:
      supply_water_temp = return_water_temp

    flow_heating_energy_rate = (
        constants.WATER_HEAT_CAPACITY
        * self._total_flow_rate
        * (supply_water_temp - return_water_temp)
    )

    dissipation_energy_rate = self.compute_thermal_dissipation_rate(
        supply_water_temp, outside_temp
    )

    if self._last_step_duration.total_seconds() > 0:
      tank_heating_energy_rate = (
          constants.WATER_HEAT_CAPACITY
          * self._water_capacity
          * self._step_tank_temperature_change
          / self._last_step_duration.total_seconds()
      )
    else:
      tank_heating_energy_rate = 0

    return (
        flow_heating_energy_rate
        + dissipation_energy_rate
        + tank_heating_energy_rate
    )

  def compute_thermal_dissipation_rate(
      self, water_temp: float, outside_temp: float
  ) -> float:
    """Returns the amount of thermal loss in W from a boiler tank.

    Thermal dissipation is the loss of heat due from the tank to the environment
    due to imperfect insulation, measured in Watts.

    The tank is assumed to be a cylindrical annulus, with an internal radius
    internal length, and an insulation thickness. Heat is dissapated only
    through the cylinder walls, and no heat is lost through the ends/caps.

    The equation is computed by applying an energy balance of:
    Q = Q_conduction = Q_convection, where Q_condition is the heat transferred
    from water at water_temp (T1) to the exterior surface temp (T2), and
    Q_convection is the transfer of the heat from the external surface at T2
    to the outside at T_inf.

    Q_conduction = (T1 - T2) / (ln(R2/R1) / 2 x pi x L x k), and
    Q_convection = (T2 - Tinf) / (1 / h x A), where
      A = the area of the annulus = 2 x pi x R2 x L

    Solving for Q:
    Q = (T1 - Tinf) / [ln(R2/R1) / 2pi / L / k + 1 / h / A]

    Source: Heat Transfer, E. M. Sparrow, 1993

    Args:
      water_temp: average temperature of the water [K]
      outside_temp: temperature outside of the tank, can be ambient [K]

    Returns:
      thermal loss rate of the tank in Watts
    """

    assert water_temp >= outside_temp
    delta_temp = water_temp - outside_temp
    numerator = self._tank_length * 2.0 * np.pi * delta_temp
    interior_radius = self._tank_radius
    exterior_radius = interior_radius + self._insulation_thickness
    conduction_factor = (
        np.log(exterior_radius / interior_radius)
        / self._insulation_conductivity
    )
    convection_factor = 1.0 / self._convection_coefficient / exterior_radius
    return numerator / (conduction_factor + convection_factor)

  def compute_pump_power(self) -> float:
    """Returns power consumed by pump in W to move water to VAVs.

    derived from: https://www.engineeringtoolbox.com/pumps-power-d_505.html
    """
    return (
        self._total_flow_rate
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * self._water_pump_differential_head
        / self._water_pump_efficiency
    )
