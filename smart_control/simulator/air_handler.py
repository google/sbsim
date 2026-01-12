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

import enum
from typing import Optional
import uuid

import gin
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import smart_device
from smart_buildings.smart_control.simulator import weather_controller
from smart_buildings.smart_control.utils import constants

RunStatus = enum.IntEnum('RunStatus', [('On', 1), ('Off', 0)])


@gin.configurable
class AirHandler(smart_device.SmartDevice):
  """Models an air hander with heating/cooling, input/exhaust and recirculation.

  Attributes:
    recirculation: Proportion of air recirculated.
    air_flow_rate: Flow rate produced by fan in m^3/s.
    heating_air_temp_setpoint: Minimum temperature in K until air will need to
      be heated. Deprecated, use supply_air_temperature_setpoint instead.
    cooling_air_temp_setpoint: Maximum temperature in K until air will be
      cooled. Deprecated, use supply_air_temperature_setpoint instead.
    supply_air_temperature_setpoint: Average temperature in K of air output from
      the air handler. This is reduced to the heating or cooling setpoint.
      Either this, or those two, can be used. For backwards compatibility, the
      heating and cooling setpoints are here, but in the real building, only the
      supply_air_temperature_setpoint exists.
    fan_static_pressure: Amount of pressure in Pa needed to push air
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
      fan_static_pressure: float,
      fan_efficiency: float,
      max_air_flow_rate: float = 8.67,
      device_id: Optional[str] = None,
      sim_weather_controller: Optional[
          weather_controller.WeatherController
      ] = None,
      run_command=RunStatus.On,
  ):
    if cooling_air_temp_setpoint <= heating_air_temp_setpoint:
      raise ValueError(
          'cooling_air_temp_setpoint must greater than'
          ' heating_air_temp_setpoint'
      )

    observable_fields = {
        'static_pressure_setpoint': smart_device.AttributeInfo(
            'fan_static_pressure', float
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
        'supervisor_run_command': smart_device.AttributeInfo(
            'run_command', int
        ),
        'supply_air_temperature_setpoint': smart_device.AttributeInfo(
            'supply_air_temperature_setpoint', float
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
        'supervisor_run_command': smart_device.AttributeInfo(
            'run_command', int
        ),
        'static_pressure_setpoint': smart_device.AttributeInfo(
            'fan_static_pressure', float
        ),
        'supply_air_temperature_setpoint': smart_device.AttributeInfo(
            'supply_air_temperature_setpoint', float
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
    self._init_fan_static_pressure = fan_static_pressure
    self._init_fan_efficiency = fan_efficiency
    self._init_cooling_request_count = 0
    self._init_max_air_flow_rate = max_air_flow_rate
    self._sim_weather_controller = sim_weather_controller
    self._init_run_command = run_command
    self.reset()

  def reset(self):
    self._recirculation = self._init_recirculation
    self._air_flow_rate = self._init_air_flow_rate
    self._heating_air_temp_setpoint = self._init_heating_air_temp_setpoint
    self._cooling_air_temp_setpoint = self._init_cooling_air_temp_setpoint
    self._fan_static_pressure = self._init_fan_static_pressure
    self._fan_efficiency = self._init_fan_efficiency
    self._cooling_request_count = self._init_cooling_request_count
    self._max_air_flow_rate = self._init_max_air_flow_rate
    self._run_command = self._init_run_command

  def get_vav_air_handler(self, _):
    return self

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

  @property
  def run_command(self) -> RunStatus:
    return self._run_command

  @run_command.setter
  def run_command(self, value: RunStatus):
    self._run_command = value

  @recirculation.setter
  def recirculation(self, value: float):
    self._recirculation = value

  @property
  def air_flow_rate(self) -> float:
    if self._run_command == RunStatus.Off:
      return 0.0
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
  def supply_air_temperature_setpoint(self) -> float:
    return (self.heating_air_temp_setpoint + self.cooling_air_temp_setpoint) / 2

  @supply_air_temperature_setpoint.setter
  def supply_air_temperature_setpoint(self, value: float):
    temperature_band = (
        self.cooling_air_temp_setpoint - self.heating_air_temp_setpoint
    ) / 2
    self._supply_air_temperature_setpoint = value
    self.cooling_air_temp_setpoint = value + temperature_band
    self.heating_air_temp_setpoint = value - temperature_band

  @property
  def fan_static_pressure(self) -> float:
    if self._run_command == RunStatus.Off:
      return 0.0
    return self._fan_static_pressure

  @fan_static_pressure.setter
  def fan_static_pressure(self, value: float):
    self._fan_static_pressure = value

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
    """Returns temperature of air output from air handler after A/C or heat.

    Temperatures are measured in Kelvin.

    Args:
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of ambient/outside air.
    """
    mixed_air_temp = self.get_mixed_air_temp(recirculation_temp, ambient_temp)
    if (
        mixed_air_temp > self.supply_air_temperature_setpoint
        and self._run_command == RunStatus.On
    ):
      return self.supply_air_temperature_setpoint
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
      fan_static_pressure: float,
      fan_efficiency: float,
  ) -> float:
    """Returns power in W consumed by fan.

    Derived from:
    https://www.engineeringtoolbox.com/fans-efficiency-power-consumption-d_197.html

    Args:
      flow_rate: Rate of air flow in m^3/s.
      fan_static_pressure: Pressure difference in Pa between fan intake and fan
        output.
      fan_efficiency: Electrical efficiency of fan (0-1).
    """
    return flow_rate * fan_static_pressure / fan_efficiency

  def compute_intake_fan_energy_rate(self) -> float:
    """Returns power in W consumed by the intake fan."""
    return self.compute_fan_power(
        self.air_flow_rate,
        self.fan_static_pressure,
        self._fan_efficiency,
    )

  def compute_exhaust_fan_energy_rate(self) -> float:
    """Returns power in W consumed by the exhaust fan."""
    return self.compute_fan_power(
        self.air_flow_rate * (1.0 - self._recirculation),
        self.fan_static_pressure,
        self._fan_efficiency,
    )

  def set_action(self, action_field_name, value, action_timestamp):
    if 'supervisor_run_command' in action_field_name:
      if value == 1:
        value = RunStatus.On
      else:
        value = RunStatus.Off
    super().set_action(action_field_name, value, action_timestamp)


@gin.configurable
class AirHandlerSystem(smart_device.SmartDevice):
  """A system controller that manages multiple AirHandler units."""

  def __init__(
      self,
      ahus,
      device_id: Optional[str] = None,
  ):
    if not ahus:
      raise ValueError('AirHandlerSystem requires at least one AirHandler.')

    self._ahus = list(ahus.keys())
    self._map = ahus

    if device_id is None:
      device_id = f'ahu_system_{uuid.uuid4()}'

    system_actions = {}
    system_observables = {}
    for i, ahu in enumerate(self._ahus):
      prefix = f'ahu_{i+1}_'

      for key, info in ahu._action_fields.items():
        system_actions[prefix + key] = smart_device.AttributeInfo(
            prefix + key, info.clazz
        )
        setattr(self, prefix + key, None)
      for key, info in ahu._observable_fields.items():
        system_observables[prefix + key] = smart_device.AttributeInfo(
            prefix + key, info.clazz
        )
        setattr(self, prefix + key, None)

    super().__init__(
        observable_fields=system_observables,
        action_fields=system_actions,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.AHU,
        device_id=device_id or f'ahu_system_{uuid.uuid4()}',
    )

  def _get_target(self, name):
    """Parses 'ahu_1_run_command' -> returns (ahu_object, 'run_command')."""
    parts = name.split('_', 2)  # Splits into ['ahu', '1', 'run_command']

    if len(parts) >= 3 and parts[0] == 'ahu':
      index = int(parts[1]) - 1  # Convert '1' to 0 (list index)
      field_name = parts[2]  # 'run_command'
      return self._ahus[index], field_name

    raise ValueError(f'Could not find child for field: {name}')

  @property
  def ahus(self) -> list[AirHandler]:
    return self._ahus

  def set_action(self, action_field_name, value, action_timestamp):
    """Send an action to the target AHU.

    Args:
      action_field_name: The name of the action field to set (e.g.,
        'ahu_1_supervisor_run_command').
      value: The value to set for the action field.
      action_timestamp: The timestamp of when the action is being set.
    """
    if 'supervisor_run_command' in action_field_name:
      if value == 1:
        value = RunStatus.On
      else:
        value = RunStatus.Off
    target_ahu, target_field = self._get_target(action_field_name)
    target_ahu.set_action(target_field, value, action_timestamp)

  def get_observation(self, observable_field_name, observation_timestamp):
    """Gets an observation from a specific AirHandler unit.

    Args:
      observable_field_name: The name of the observable field (e.g.,
        'ahu_1_supply_air_flowrate_sensor').
      observation_timestamp: The timestamp of when the observation is requested.

    Returns:
      The value of the requested observable field from the target AHU.
    """
    target_ahu, target_field = self._get_target(observable_field_name)
    return target_ahu.get_observation(target_field, observation_timestamp)

  def reset(self):
    for ahu in self._ahus:
      ahu.reset()

  def reset_demand(self):
    for ahu in self._ahus:
      ahu.reset_demand()

  @property
  def air_flow_rate(self) -> float:
    return sum(ahu.air_flow_rate for ahu in self._ahus)

  @property
  def cooling_request_count(self) -> int:
    return sum(ahu.cooling_request_count for ahu in self._ahus)

  def compute_thermal_energy_rate(
      self, recirculation_temp: float, ambient_temp: float
  ) -> float:
    return sum(
        ahu.compute_thermal_energy_rate(recirculation_temp, ambient_temp)
        for ahu in self._ahus
    )

  def compute_intake_fan_energy_rate(self) -> float:
    return sum(ahu.compute_intake_fan_energy_rate() for ahu in self._ahus)

  def compute_exhaust_fan_energy_rate(self) -> float:
    return sum(ahu.compute_exhaust_fan_energy_rate() for ahu in self._ahus)

  def get_supply_air_temp(self, recirculation_temp, ambient_temperature):
    temps = {}
    for ahu in self._ahus:
      temps[ahu.device_id()] = ahu.get_supply_air_temp(
          recirculation_temp, ambient_temperature
      )
    return temps

  def get_vav_air_handler(self, zone_id):
    """Gets the AirHandler associated with a given zone.

    Args:
      zone_id: The ID of the zone.

    Returns:
      The AirHandler instance responsible for the given zone.

    Raises:
      ValueError: If no AirHandler is found for the specified zone.
    """
    for ahu, zones in self._map.items():
      if zone_id in zones:
        return ahu
    raise ValueError(f'No VAV found for zone {zone_id}')
