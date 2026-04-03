"""Models a commercial hot water system for building simulations.

This module provides a flexible, discrete-time thermodynamic simulation
environment for hot water plants. It is designed to evaluate smart control
strategies, including Reinforcement Learning (RL) agents or LLM agents,
by exposing standard Building Management System (BMS) telemetry (observables)
and control points (actions).

Architecture:
    The simulation relies on a component-based, dependency-injected architecture
    to separate the physics of heat generation from fluid distribution:

    * HotWaterHeatSource (Abstract Base Class): Defines the standard interface
      for any hot water generating equipment. It mandates methods for
      calculating thermal energy consumption, ambient thermal dissipation, and
      state management.
    * Concrete Heat Sources: Physical implementations of the
    `HotWaterHeatSource`such as `Boiler` (traditional gas/electric heating) and
      `AirSourceHeatPump` (dynamic, environment-dependent COP-based heating).
    * WaterPump: Models the fluid flow and differential pressure dynamics
    required to distribute the heated water to the building's terminal units
    (e.g., VAVs).
    * HotWaterSystem: The system orchestrator. It acts as the primary device
      interface, coupling a `HotWaterHeatSource` and a `WaterPump`. It
      aggregates building-level flow demand, calculates total system power draw,
      and manages the mixed return water temperatures.

Thermodynamics & Physics:
    * Energy calculations rely on steady-state heat transfer equations
      (Q = m * c_p * dT), dynamically constrained by the physical capacity
      limits and efficiencies of the instantiated equipment.
    * Flow calculations utilize square-root pressure-to-flow relationships
      adjusted by the specified piping header resistance.

Configuration:
    Major physical parameters, setpoints, and equipment types are exposed via
    the `gin` configuration framework. This allows researchers to seamlessly
    switch a simulation from a legacy boiler system to a modernized heat pump
    array strictly through configuration files, without altering the
    underlying Python code.
"""

import enum
import math
import uuid
import gin
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import air_source_heat_pump
from smart_buildings.smart_control.simulator import boiler as boiler_py
from smart_buildings.smart_control.simulator import hot_water_heat_source
from smart_buildings.smart_control.simulator import pump as pump_py
from smart_buildings.smart_control.simulator import smart_device

ZoneID = str

# Default ASHP parameters.
DEFAULT_ASHP_MAX_CAPACITY_W: float = 180000.0
DEFAULT_ASHP_NOMINAL_COP: float = 3.2


@gin.constants_from_enum
class HeatSourceType(enum.Enum):
  BOILER = 1
  ASHP = 2


@gin.configurable
class HotWaterSystem(smart_device.SmartDevice):
  """A model of a commercial hot water system.

  For simplicity, we model a single pump and heat source. We can view multiple
  sources/pumps as a single big boiler/pump that has their combined capacity.

  TODO(sipple): Add support for multiple pumps and multiple heat sources.

  Attributes:
    heat_source: a boiler or ASHP responsible for heating water
    pump: a pump respnsible for circulating water to the VAVs
    device_id: unique name of the device.
    run_command: Run command of the hot water system.
    differential_pressure: Differential pressure setpoint of the hot water
      system in bars.
    header_resistance: Resistance of the header in bar/(m^3/h)^2
  """

  def __init__(
      self,
      heat_source: hot_water_heat_source.HotWaterHeatSource,
      pump: pump_py.WaterPump,
      device_id: str | None = None,
      header_resistance: float = 0.0,
  ):
    observable_fields = {
        smart_device.SUPPLY_WATER_SETPOINT: smart_device.AttributeInfo(
            smart_device.SUPPLY_WATER_SETPOINT, float
        ),
        smart_device.SUPPLY_WATER_TEMPERATURE_SENSOR: (
            smart_device.AttributeInfo(
                smart_device.SUPPLY_WATER_TEMPERATURE_SENSOR, float
            )
        ),
        smart_device.HEATING_REQUEST_COUNT: smart_device.AttributeInfo(
            smart_device.HEATING_REQUEST_COUNT, int
        ),
        smart_device.SUPERVISOR_RUN_COMMAND: smart_device.AttributeInfo(
            smart_device.RUN_COMMAND, smart_device.RunStatus
        ),
        smart_device.RUN_STATUS: smart_device.AttributeInfo(
            smart_device.RUN_STATUS, smart_device.RunStatus
        ),
        smart_device.DIFFERENTIAL_PRESSURE: smart_device.AttributeInfo(
            smart_device.DIFFERENTIAL_PRESSURE, float
        ),
    }

    action_fields = {
        smart_device.SUPPLY_WATER_SETPOINT: smart_device.AttributeInfo(
            smart_device.REHEAT_WATER_SETPOINT, float
        ),
        smart_device.SUPERVISOR_RUN_COMMAND: smart_device.AttributeInfo(
            smart_device.RUN_COMMAND, smart_device.RunStatus
        ),
        smart_device.DIFFERENTIAL_PRESSURE: smart_device.AttributeInfo(
            smart_device.DIFFERENTIAL_PRESSURE, float
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

    self._heat_source = heat_source
    self._pump = pump
    self._header_resistance = header_resistance
    self.reset()

  def reset(self):
    self.reset_demand()
    self._heat_source.reset()
    self._pump.reset()
    self._run_command = smart_device.RunStatus.ON

  def reset_demand(self) -> None:
    self.flow_rate = 0.0
    self._heating_request_count = 0
    self._flow_factor_sum = 0.0

  @property
  def heat_source_device_type(
      self,
  ) -> smart_control_building_pb2.DeviceInfo.DeviceType:
    return self._heat_source.device_type()

  @property
  def return_water_temperature_sensor(self) -> float:
    return self._heat_source.return_water_temperature_sensor

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    self._heat_source.return_water_temperature_sensor = value

  @property
  def reheat_water_setpoint(self) -> float:
    return self._heat_source.get_observation(
        smart_device.SUPPLY_WATER_SETPOINT, self._observation_timestamp
    )

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    self._heat_source.set_action(
        smart_device.SUPPLY_WATER_SETPOINT, value, self._action_timestamp
    )

  @property
  def heating_request_count(self) -> int:
    return self._heating_request_count

  @property
  def supply_water_temperature_sensor(self) -> float:
    return self._heat_source.get_observation(
        smart_device.SUPPLY_WATER_TEMPERATURE_SENSOR,
        self._observation_timestamp,
    )

  @property
  def supply_water_setpoint(self) -> float:
    return self._heat_source.get_observation(
        smart_device.SUPPLY_WATER_SETPOINT, self._observation_timestamp
    )

  @property
  def run_status(self) -> smart_device.RunStatus:
    return self._run_command

  @property
  def run_command(self) -> smart_device.RunStatus:
    return self._run_command

  @run_command.setter
  def run_command(self, value: smart_device.RunStatus) -> None:
    self._run_command = value
    self._heat_source.run_command = value
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
      self, return_water_temp: float | list[float], outside_temp: float
  ) -> float:
    """Returns energy rate in W consumed by boiler to heat water.

    Args:
      return_water_temp: Temperature in K that water is received at.
      outside_temp: Temperature in K that the water tank is in.
    """
    return self._heat_source.compute_thermal_energy_rate(
        return_water_temp, outside_temp, self.total_flow_rate
    )

  def compute_thermal_dissipation_rate(
      self, water_temp: float | list[float], outside_temp: float
  ) -> float:
    """Returns the amount of thermal loss in W from a boiler tank.

    Args:
      water_temp: average temperature of the water [K]
      outside_temp: temperature outside of the tank, can be ambient [K]

    Returns:
      thermal loss rate of the tank in Watts
    """
    return self._heat_source.compute_thermal_dissipation_rate(
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
        value = smart_device.RunStatus.ON
      else:
        value = smart_device.RunStatus.OFF
      self._pump.run_command = value
    super().set_action(action_field_name, value, action_timestamp)


# TODO(sipple): Add keword arguments for all parameters in construct_hot_water_
# system.
@gin.configurable
def construct_hot_water_system(
    # --- Shared System & Pump Parameters ---
    water_pump_differential_head: float,
    water_pump_efficiency: float,
    reheat_water_setpoint: float,
    device_id: str | None = None,
    pump_device_id: str | None = None,
    heat_source_device_id: str | None = None,
    header_resistance: float = 0.0,
    # --- Heat Source Toggle ---
    heat_source_type: HeatSourceType = HeatSourceType.BOILER,
    init_return_water_temperature_sensor: float = 295.0,
    # --- Boiler-Specific Parameters ---
    heating_rate: float | None = 0,
    cooling_rate: float | None = 0,
    convection_coefficient: float | None = 5.6,
    tank_length: float | None = 2.0,
    tank_radius: float | None = 0.5,
    water_capacity: float | None = 1.5,
    insulation_conductivity: float | None = 0.067,
    insulation_thickness: float | None = 0.06,
    # --- ASHP-Specific Parameters ---
    ashp_max_capacity_w: float = DEFAULT_ASHP_MAX_CAPACITY_W,
    ashp_nominal_cop: float = DEFAULT_ASHP_NOMINAL_COP,
) -> HotWaterSystem:
  """Constructs a hot water system based on the requested heat source.

  How to use this in .gin configuration files:

  Example 1: A traditional building with a Boiler
  ```gin
  construct_hot_water_system.heat_source_type = "boiler"
  construct_hot_water_system.reheat_water_setpoint = 338.0  # 65C / 150F
  construct_hot_water_system.water_pump_differential_head = 15.0
  construct_hot_water_system.water_pump_efficiency = 0.75
  # Boiler specific configs
  construct_hot_water_system.water_capacity = 2.0
  construct_hot_water_system.insulation_thickness = 0.08
  ```

  Example 2: A building with an Air Source Heat Pump
  ```gin
  construct_hot_water_system.heat_source_type = "ashp"
  # 40C / 104F (Lower temp for ASHP efficiency)
  construct_hot_water_system.water_pump_differential_head = 15.0
  construct_hot_water_system.reheat_water_setpoint = 313.0
  construct_hot_water_system.water_pump_efficiency = 0.85
  # ASHP specific configs
  construct_hot_water_system.ashp_max_capacity_w = 250000.0
  construct_hot_water_system.ashp_nominal_cop = 3.5
  ```

  Args:
    water_pump_differential_head: The differential head of the water pump in
      meters of water column.
    water_pump_efficiency: The efficiency of the water pump.
    reheat_water_setpoint: The desired water temperature setpoint in Kelvin.
    device_id: The unique identifier for the hot water system.
    pump_device_id: The unique identifier for the water pump.
    heat_source_device_id: The unique identifier for the heat source.
    header_resistance: The resistance of the header in bar/(m^3/h)^2.
    heat_source_type: The type of heat source to use ("boiler" or "ashp").
    init_return_water_temperature_sensor: The initial return water temperature
      in Kelvin.
    heating_rate: The heating rate of the boiler in Watts per Kelvin.
    cooling_rate: The cooling rate of the boiler in Watts per Kelvin.
    convection_coefficient: The convection coefficient of the boiler in
      W/(m^2*K).
    tank_length: The length of the boiler tank in meters.
    tank_radius: The radius of the boiler tank in meters.
    water_capacity: The volume of water the boiler tank can hold in m^3.
    insulation_conductivity: The thermal conductivity of the boiler tank
      insulation in W/(m*K).
    insulation_thickness: The thickness of the boiler tank insulation in meters.
    ashp_max_capacity_w: The maximum heating capacity of the ASHP in Watts.
    ashp_nominal_cop: The nominal Coefficient of Performance of the ASHP.
  """

  # 1. Instantiate the requested Heat Source
  if heat_source_type == HeatSourceType.ASHP:
    heat_source = air_source_heat_pump.AirSourceHeatPump(
        reheat_water_setpoint=reheat_water_setpoint,
        device_id=heat_source_device_id,
        max_heating_capacity_w=ashp_max_capacity_w,
        nominal_cop=ashp_nominal_cop,
        init_return_water_temperature_sensor=init_return_water_temperature_sensor,
    )
  elif heat_source_type == HeatSourceType.BOILER:
    heat_source = boiler_py.Boiler(
        reheat_water_setpoint=reheat_water_setpoint,
        device_id=heat_source_device_id,
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
        convection_coefficient=convection_coefficient,
        tank_length=tank_length,
        tank_radius=tank_radius,
        water_capacity=water_capacity,
        insulation_conductivity=insulation_conductivity,
        insulation_thickness=insulation_thickness,
        init_return_water_temperature_sensor=init_return_water_temperature_sensor,
    )
  else:
    raise ValueError(
        f"Unknown heat_source_type: '{heat_source_type}'. "
        "Must be HeatSourceType.BOILER or HeatSourceType.ASHP."
    )

  # 2. Instantiate the Pump
  pump = pump_py.WaterPump(
      water_pump_differential_head=water_pump_differential_head,
      water_pump_efficiency=water_pump_efficiency,
      device_id=pump_device_id,
  )

  # 3. Assemble and return the System
  return HotWaterSystem(
      heat_source=heat_source,
      pump=pump,
      device_id=device_id,
      header_resistance=header_resistance,
  )
