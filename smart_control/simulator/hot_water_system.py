"""Models a hot water system for the simulation.

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
import math
from typing import List, Optional, Union
import uuid
import gin
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import boiler as boiler_py
from smart_buildings.smart_control.simulator import pump as pump_py
from smart_buildings.smart_control.simulator import smart_device

IntEnum = enum.IntEnum
ZoneID = str

RunStatus = IntEnum("RunStatus", [("On", 1), ("Off", 0)])


@gin.configurable
class HotWaterSystem(smart_device.SmartDevice):
  """Models a hot water system.

  For simplicity, we model a single pump and boiler. We can view multiple
  boilers/pumps as a single big boiler/pump that has their combined capacity.

  Attributes:
    boiler: a boiler responsible for heating water
    pump: a pump respnsible for circulating water to the VAVs
    device_id: unique name of the device.
    run_command: Run command of the hot water system.
    differential_pressure: Differential pressure setpoint of the hot water
      system in bars.
    header_resistance: Resistance of the header in bar/(m^3/h)^2
  """

  def __init__(
      self,
      boiler: boiler_py.Boiler,
      pump: pump_py.WaterPump,
      device_id: Optional[str] = None,
      header_resistance: float = 0.0,
  ):
    observable_fields = {
        "supply_water_setpoint": smart_device.AttributeInfo(
            "reheat_water_setpoint", float
        ),
        "supply_water_temperature_sensor": smart_device.AttributeInfo(
            "supply_water_temperature_sensor", float
        ),
        "heating_request_count": smart_device.AttributeInfo(
            "heating_request_count", int
        ),
        "supervisor_run_command": smart_device.AttributeInfo(
            "run_command", RunStatus
        ),
        "run_status": smart_device.AttributeInfo("run_status", RunStatus),
        "differential_pressure": smart_device.AttributeInfo(
            "differential_pressure", float
        ),
    }

    action_fields = {
        "supply_water_setpoint": smart_device.AttributeInfo(
            "reheat_water_setpoint", float
        ),
        "supervisor_run_command": smart_device.AttributeInfo(
            "run_command", RunStatus
        ),
        "differential_pressure": smart_device.AttributeInfo(
            "differential_pressure", float
        ),
    }

    if device_id is None:
      device_id = f"hot_water_system_id_{uuid.uuid4()}"

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.HWS,
        device_id=device_id,
    )

    self._boiler = boiler
    self._pump = pump
    self._header_resistance = header_resistance
    self.reset()

  def reset(self):
    self.reset_demand()
    self._boiler.reset()
    self._pump.reset()
    self._run_command = RunStatus.On

  def reset_demand(self) -> None:
    self.flow_rate = 0.0
    self._heating_request_count = 0
    self._flow_factor_sum = 0.0

  @property
  def return_water_temperature_sensor(self) -> float:
    return self._boiler.return_water_temperature_sensor

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    self._boiler.return_water_temperature_sensor = value

  @property
  def reheat_water_setpoint(self) -> float:
    return self._boiler.get_observation(
        "supply_water_setpoint", self._observation_timestamp
    )

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    self._boiler.set_action(
        "supply_water_setpoint", value, self._action_timestamp
    )

  @property
  def heating_request_count(self) -> int:
    return self._heating_request_count

  @property
  def supply_water_temperature_sensor(self) -> float:
    return self._boiler.get_observation(
        "supply_water_temperature_sensor", self._observation_timestamp
    )

  @property
  def supply_water_setpoint(self) -> float:
    return self._boiler.get_observation(
        "supply_water_setpoint", self._observation_timestamp
    )

  @property
  def run_status(self) -> RunStatus:
    return self._run_command

  @property
  def run_command(self) -> RunStatus:
    return self._run_command

  @run_command.setter
  def run_command(self, value: RunStatus) -> None:
    self._run_command = value
    self._boiler.run_command = value
    self._pump.run_command = value

  @property
  def water_pump_differential_head(self) -> float:
    return self._pump.water_pump_differential_head

  @water_pump_differential_head.setter
  def water_pump_differential_head(self, value: float) -> None:
    self._pump.water_pump_differential_head = value

  @property
  def differential_pressure(self) -> float:
    return self._pump.differential_pressure

  @differential_pressure.setter
  def differential_pressure(self, value: float) -> None:
    self._pump.differential_pressure = value

  def add_demand(self, flow_factor: float):
    """Adds to current flow rate demand.

    Args:
      flow_factor: The flow factor of the VAV.

    Raises:
      ValueError: If flow_rate is less than 0.
    """
    if flow_factor < 0.0:
      raise ValueError("Flow factor cannot be less than 0.")

    self._flow_factor_sum += flow_factor
    self._heating_request_count += 1

  @property
  def flow_factor_sum(self) -> float:
    return self._flow_factor_sum

  def compute_thermal_energy_rate(
      self, return_water_temp: Union[float, List[float]], outside_temp: float
  ) -> float:
    """Returns energy rate in W consumed by boiler to heat water.

    Args:
      return_water_temp: Temperature in K that water is received at.
      outside_temp: Temperature in K that the water tank is in.
    """
    return self._boiler.compute_thermal_energy_rate(
        return_water_temp, outside_temp, self.total_flow_rate
    )

  def compute_thermal_dissipation_rate(
      self, water_temp: Union[float, List[float]], outside_temp: float
  ) -> float:
    """Returns the amount of thermal loss in W from a boiler tank.

    Args:
      water_temp: average temperature of the water [K]
      outside_temp: temperature outside of the tank, can be ambient [K]

    Returns:
      thermal loss rate of the tank in Watts
    """
    return self._boiler.compute_thermal_dissipation_rate(
        water_temp, outside_temp
    )

  def compute_pump_power(self):
    """Returns power consumed by pump in W to move water to VAVs."""
    return self._pump.compute_pump_power(self.total_flow_rate)

  def _calculate_flow_rate(
      self, differential_pressure: float, flow_factor: float
  ) -> float:
    """Calculates the total water flow rate based on pressure and demand.

    Args:
      differential_pressure: The pressure from the pump in  bar.
      flow_factor: The flow factor of the VAV.

    Returns:
      The calculated total flow rate in cubic meters per second (m^3/s).
    """

    flow_rate = flow_factor * math.sqrt(
        differential_pressure / (1 + flow_factor**2 * self._header_resistance)
    )
    assert not math.isnan(flow_rate)
    return flow_rate

  @property
  def total_flow_rate(self) -> float:
    return self._calculate_flow_rate(
        self.differential_pressure, self._flow_factor_sum
    )

  def set_action(self, action_field_name, value, action_timestamp):
    if "supervisor_run_command" in action_field_name:
      if value == 1:
        value = RunStatus.On
      else:
        value = RunStatus.Off
      self._pump.run_command = value
    super().set_action(action_field_name, value, action_timestamp)


@gin.configurable
def construct_hot_water_system(
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
    init_return_water_temperature_sensor: float = 295.0,
    boiler_device_id: Optional[str] = None,
    pump_device_id: Optional[str] = None,
) -> "HotWaterSystem":
  """Constructs a hot water system."""
  boiler = boiler_py.Boiler(
      reheat_water_setpoint,
      boiler_device_id,
      heating_rate,
      cooling_rate,
      convection_coefficient,
      tank_length,
      tank_radius,
      water_capacity,
      insulation_conductivity,
      insulation_thickness,
      init_return_water_temperature_sensor,
  )
  pump = pump_py.WaterPump(
      water_pump_differential_head,
      water_pump_efficiency,
      pump_device_id,
  )
  return HotWaterSystem(boiler, pump, device_id)
