"""Models a pump for the simulation.

Copyright 2025 Google LLC

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
from smart_buildings.smart_control.utils import constants

IntEnum = enum.IntEnum

RunStatus = IntEnum('RunStatus', [('On', 1), ('Off', 0)])


@gin.configurable
class WaterPump(smart_device.SmartDevice):
  """Models a water pump.

  Attributes:
    differential_pressure: Differential pressure of the pump in bars.
    run_command: Command indicating if the pump is on or off
    _water_pump_differential_head: Length in meters of pump head.
    _water_pump_efficiency: Electrical efficiency of water pump [0,1].
  """

  def __init__(
      self,
      water_pump_differential_head: float,
      water_pump_efficiency: float,
      device_id: Optional[str] = None,
  ):
    observable_fields = {
        'differential_pressure': smart_device.AttributeInfo(
            'differential_pressure', float
        ),
        'supervisor_run_command': smart_device.AttributeInfo(
            'run_command', RunStatus
        ),
        'run_status': smart_device.AttributeInfo('run_command', RunStatus),
    }

    action_fields = {
        'differential_pressure': smart_device.AttributeInfo(
            'differential_pressure', float
        ),
        'supervisor_run_command': smart_device.AttributeInfo(
            'run_command', RunStatus
        ),
        'run_status': smart_device.AttributeInfo('run_status', RunStatus),
    }

    if device_id is None:
      device_id = f'pump_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.PMP,
        device_id=device_id,
    )

    self._init_water_pump_differential_head = water_pump_differential_head
    self._init_water_pump_efficiency = water_pump_efficiency
    self._init_run_command = RunStatus.On
    self.reset()

  def reset(self):
    self._water_pump_differential_head = self._init_water_pump_differential_head
    self._water_pump_efficiency = self._init_water_pump_efficiency
    self._run_command = self._init_run_command

  def compute_pump_power(self, total_flow_rate_demand) -> float:
    """Returns power consumed by pump in W to move water to VAVs.

    derived from: https://www.engineeringtoolbox.com/pumps-power-d_505.html

    Args:
      total_flow_rate_demand: The total flow rate of water through the pump in
        m3/s.
    """
    return (
        total_flow_rate_demand
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * self.water_pump_differential_head
        / self._water_pump_efficiency
    )

  def _convert_differential_head_to_pressure(
      self, differential_head: float
  ) -> float:
    """Converts a differential head (m) to differential pressure (bar).

    formula derived from:
    https://www.engineeringtoolbox.com/pump-head-pressure-d_663.html
    pressure (pa) = fluid_density * gravity * differential_head
    water density = 1000 kg/m^3, gravity = 9.81 m/s^2,
    so we get p= 9810 * differential_head
    now, to convert to bar, 1 bar = 100,000 Pa, so we get:
    pressure (bar) = differential_head * 0.0981
    (specific gravity of water is 1 so we leave it as is)

    Args:
      differential_head: The differential head of the pump in meters.

    Returns:
      The differential pressure of the pump in bars.
    """
    return (
        constants.GRAVITY
        * constants.WATER_DENSITY
        * differential_head
        / constants.PASCALS_PER_BAR
    )

  def _convert_pressure_to_differential_head(self, pressure: float) -> float:
    """Converts a differential pressure (bar) to differential head (m).

    We simmple reverse the conversion above:
    differential_head = pressure / 0.0981
    This simplifies to differential_head = pressure * 10.1937

    Args:
      pressure: The differential pressure of the pump in bars.

    Returns:
      The differential head of the pump in meters.
    """
    return pressure / (
        constants.GRAVITY * constants.WATER_DENSITY / constants.PASCALS_PER_BAR
    )

  @property
  def differential_pressure(self) -> float:
    if self._run_command == RunStatus.Off:
      return 0.0
    return self._convert_differential_head_to_pressure(
        self._water_pump_differential_head
    )

  @differential_pressure.setter
  def differential_pressure(self, value: float) -> None:
    self._water_pump_differential_head = (
        self._convert_pressure_to_differential_head(value)
    )

  @property
  def water_pump_differential_head(self) -> float:
    if self._run_command == RunStatus.Off:
      return 0.0
    return self._water_pump_differential_head

  @water_pump_differential_head.setter
  def water_pump_differential_head(self, value: float) -> None:
    self._water_pump_differential_head = value

  @property
  def run_command(self) -> RunStatus:
    return self._run_command

  @run_command.setter
  def run_command(self, value: RunStatus) -> None:
    self._run_command = value

  @property
  def run_status(self) -> RunStatus:
    return self._run_command  # in simulation, these are equivalent
