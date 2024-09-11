"""Models an air handler in an HVAC system.

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
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device
from smart_control.simulator import weather_controller
from smart_control.utils import constants


@gin.configurable
class AirHandler(smart_device.SmartDevice):
  """Models an air hander with heating/cooling, input/exhaust and recirculation.

  Attributes:
    recirculation: Proportion of air recirculated.
    air_flow_rate: Flow rate produced by fan in m^3/s.
    heating_air_temp_setpoint: Minimum temperature in K until air will need to
      be heated.
    cooling_air_temp_setpoint: Maximum temperature in K until air will be
      cooled.
    fan_differential_pressure: Amount of pressure in Pa needed to push air
      effectively.
    fan_efficiency: Electrical efficiency of fan (0 - 1).
    cooling_request_count: count of VAVs that have requested cooling in this
      cycle.
    max_air_flow_rate: max air flow rate in kg/s
  """

  def __init__(
      self,
      recirculation: float,
      heating_air_temp_setpoint: int,
      cooling_air_temp_setpoint: int,
      fan_differential_pressure: float,
      fan_efficiency: float,
      max_air_flow_rate: float = 8.67,
      device_id: Optional[str] = None,
      sim_weather_controller: Optional[
          weather_controller.WeatherController
      ] = None,
  ):
    if cooling_air_temp_setpoint <= heating_air_temp_setpoint:
      raise ValueError(
          'cooling_air_temp_setpoint must greater than'
          ' heating_air_temp_setpoint'
      )

    observable_fields = {
        'differential_pressure_setpoint': smart_device.AttributeInfo(
            'fan_differential_pressure', float
        ),
        'supply_air_flowrate_sensor': smart_device.AttributeInfo(
            'air_flow_rate', float
        ),
        'supply_air_heating_temperature_setpoint': smart_device.AttributeInfo(
            'heating_air_temp_setpoint', float
        ),
        'supply_air_cooling_temperature_setpoint': smart_device.AttributeInfo(
            'cooling_air_temp_setpoint', float
        ),
        'supply_fan_speed_percentage_command': smart_device.AttributeInfo(
            'supply_fan_speed_percentage', float
        ),
        'discharge_fan_speed_percentage_command': smart_device.AttributeInfo(
            'supply_fan_speed_percentage', float
        ),
        'outside_air_flowrate_sensor': smart_device.AttributeInfo(
            'ambient_flow_rate', float
        ),
        'cooling_request_count': smart_device.AttributeInfo(
            'cooling_request_count', float
        ),
    }
    if sim_weather_controller:
      observable_fields['outside_air_temperature_sensor'] = (
          smart_device.AttributeInfo('outside_air_temperature_sensor', float)
      )

    action_fields = {
        'supply_air_heating_temperature_setpoint': smart_device.AttributeInfo(
            'heating_air_temp_setpoint', float
        ),
        'supply_air_cooling_temperature_setpoint': smart_device.AttributeInfo(
            'cooling_air_temp_setpoint', float
        ),
    }

    if device_id is None:
      device_id = f'air_handler_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.AHU,
        device_id=device_id,
    )

    self._init_recirculation = recirculation
    self._init_air_flow_rate = 0.0
    self._init_heating_air_temp_setpoint = heating_air_temp_setpoint
    self._init_cooling_air_temp_setpoint = cooling_air_temp_setpoint
    self._init_fan_differential_pressure = fan_differential_pressure
    self._init_fan_efficiency = fan_efficiency
    self._init_cooling_request_count = 0
    self._init_max_air_flow_rate = max_air_flow_rate
    self._sim_weather_controller = sim_weather_controller
    self.reset()

  def reset(self):
    self._recirculation = self._init_recirculation
    self._air_flow_rate = self._init_air_flow_rate
    self._heating_air_temp_setpoint = self._init_heating_air_temp_setpoint
    self._cooling_air_temp_setpoint = self._init_cooling_air_temp_setpoint
    self._fan_differential_pressure = self._init_fan_differential_pressure
    self._fan_efficiency = self._init_fan_efficiency
    self._cooling_request_count = self._init_cooling_request_count
    self._max_air_flow_rate = self._init_max_air_flow_rate

  @property
  def outside_air_temperature_sensor(self) -> float:
    if not self._sim_weather_controller:
      raise RuntimeError(
          'Outside air temperature requested, but air handler has no weather'
          ' controller.'
      )
    return self._sim_weather_controller.get_current_temp(
        self._observation_timestamp
    )

  @property
  def recirculation(self) -> float:
    return self._recirculation

  @recirculation.setter
  def recirculation(self, value: float):
    self._recirculation = value

  @property
  def air_flow_rate(self) -> float:
    return self._air_flow_rate

  @air_flow_rate.setter
  def air_flow_rate(self, value: float):
    self._air_flow_rate = value

  @property
  def cooling_air_temp_setpoint(self) -> int:
    return self._cooling_air_temp_setpoint  # pytype: disable=bad-return-type  # trace-all-classes

  @cooling_air_temp_setpoint.setter
  def cooling_air_temp_setpoint(self, value: float):
    self._cooling_air_temp_setpoint = value

  @property
  def heating_air_temp_setpoint(self) -> int:
    return self._heating_air_temp_setpoint  # pytype: disable=bad-return-type  # trace-all-classes

  @heating_air_temp_setpoint.setter
  def heating_air_temp_setpoint(self, value: float):
    self._heating_air_temp_setpoint = value

  @property
  def fan_differential_pressure(self) -> float:
    return self._fan_differential_pressure

  @fan_differential_pressure.setter
  def fan_differential_pressure(self, value: float):
    self._fan_differential_pressure = value

  @property
  def fan_efficiency(self) -> float:
    return self._fan_efficiency

  @fan_efficiency.setter
  def fan_efficiency(self, value: float):
    self._fan_efficiency = value

  @property
  def cooling_request_count(self) -> int:
    return self._cooling_request_count

  @property
  def max_air_flow_rate(self) -> float:
    return self._max_air_flow_rate

  def get_mixed_air_temp(
      self, recirculation_temp: float, ambient_temp: float
  ) -> float:
    """Returns temperature in K of air after recirculation.

    Args:
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of ambient/outside air.
    """
    return (
        self._recirculation * recirculation_temp
        + (1 - self._recirculation) * ambient_temp
    )

  def get_supply_air_temp(
      self, recirculation_temp: float, ambient_temp: float
  ) -> float:
    """Returns temperature in K of air output from air handler after A/C or heat.

    Args:
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of ambient/outside air.
    """
    mixed_air_temp = self.get_mixed_air_temp(recirculation_temp, ambient_temp)
    if mixed_air_temp > self._cooling_air_temp_setpoint:
      return self._cooling_air_temp_setpoint
    elif mixed_air_temp < self._heating_air_temp_setpoint:
      return self._heating_air_temp_setpoint
    else:
      return mixed_air_temp

  @property
  def ambient_flow_rate(self) -> float:
    """Returns rate of flow coming from outside."""
    return (1.0 - self._recirculation) * self._air_flow_rate

  @property
  def recirculation_flow_rate(self) -> float:
    """Returns rate of flow from recirculated air."""
    return self._recirculation * self._air_flow_rate

  @property
  def supply_fan_speed_percentage(self) -> float:
    """Returns supply fan speed percentage."""
    return self._air_flow_rate / self.max_air_flow_rate

  def reset_demand(self):
    self._air_flow_rate = 0.0
    self._cooling_request_count = 0

  def add_demand(self, flow_rate: float):
    """Adds to current flow rate demand.

    Args:
      flow_rate: Flow rate to add.

    Raises:
      ValueError: If flow_rate is not positive.
    """
    if flow_rate <= 0:
      raise ValueError('Flow rate must be positive')
    self._air_flow_rate += flow_rate
    if self._air_flow_rate > self.max_air_flow_rate:
      self._air_flow_rate = self.max_air_flow_rate
    self._cooling_request_count += 1

  def compute_thermal_energy_rate(
      self, recirculation_temp: float, ambient_temp: float
  ) -> float:
    """Returns energy in W needed by the air handler to meet supply temp.

    Args:
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of outside air.
    """
    mixed_air_temp = self.get_mixed_air_temp(recirculation_temp, ambient_temp)
    supply_air_temp = self.get_supply_air_temp(recirculation_temp, ambient_temp)
    return (
        self._air_flow_rate
        * constants.AIR_HEAT_CAPACITY
        * (supply_air_temp - mixed_air_temp)
    )

  def compute_fan_power(
      self,
      flow_rate: float,
      fan_differential_pressure: float,
      fan_efficiency: float,
  ) -> float:
    """Returns power in W consumed by fan.

    Derived from:
    https://www.engineeringtoolbox.com/fans-efficiency-power-consumption-d_197.html

    Args:
      flow_rate: Rate of air flow in m^3/s.
      fan_differential_pressure: Pressure difference in Pa between fan intake
        and fan output.
      fan_efficiency: Electrical efficiency of fan (0-1).
    """
    return flow_rate * fan_differential_pressure / fan_efficiency

  def compute_intake_fan_energy_rate(self) -> float:
    """Returns power in W consumed by the intake fan."""
    return self.compute_fan_power(
        self._air_flow_rate,
        self._fan_differential_pressure,
        self._fan_efficiency,
    )

  def compute_exhaust_fan_energy_rate(self) -> float:
    """Returns power in W consumed by the exhaust fan."""
    return self.compute_fan_power(
        self._air_flow_rate * (1.0 - self._recirculation),
        self._fan_differential_pressure,
        self._fan_efficiency,
    )
