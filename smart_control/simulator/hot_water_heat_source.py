"""Defines the base class for hot water heating sources in sbsim.

This module establishes the `HotWaterHeatSource` abstract base class,
which serves as the foundational contract for any equipment responsible for
generating thermal energy within a hydronic heating system. By decoupling the
heat generation physics from the broader distribution system, this architecture
allows for modular, plug-and-play building simulations.

Overview:
    In commercial HVAC systems, the method of generating hot water drastically
    changes the thermodynamic math, energy consumption profiles, and efficiency
    curves of the plant. A traditional gas boiler operates differently than an
    Air Source Heat Pump (ASHP).

    This module standardizes how the orchestrating `HotWaterSystem` interacts
    with these distinct physical units. It ensures that regardless of the
    underlying technology, the system can reliably request thermal
    calculations, update telemetry, and actuate controls.

Core Abstractions:
    Any concrete heat source inheriting from this class must implement the
    following thermodynamic and state-management behaviors:

    * Thermal Generation (`compute_thermal_energy_rate`): Calculates the actual
      power (electrical or fuel) consumed by the equipment to raise the supplied
      return water to the target setpoint, factoring in mass flow rates and
      environmental conditions (e.g., dynamic COP for heat pumps).
    * Thermal Dissipation (`compute_thermal_dissipation_rate`): Calculates the
      passive heat loss from the equipment to its surrounding ambient
      environment.
    * State Management: Standard properties for BMS telemetry, including
      `return_water_temperature_sensor`, `run_command`, and equipment `reset()`.

Extensibility:
    To introduce a new heating technology (e.g., an Electric Resistance Boiler
    or a District Hot Water Heat Exchanger) to the simulation ecosystem,
    developers only need to create a new concrete class that inherits from
    `HotWaterHeatSource` and implements its abstract methods.
    The `gin` configuration factory will
    then be able to seamlessly inject it into the overarching building model.
"""

import abc
import enum

from smart_buildings.smart_control.simulator import smart_device

IntEnum = enum.IntEnum
RunStatus = IntEnum("RunStatus", [("On", 1), ("Off", 0)])


class HotWaterHeatSource(smart_device.SmartDevice, abc.ABC):
  """Base class for any hot water heating source (Boiler, ASHP, etc.)."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the heat source to its initial state."""

  @property
  @abc.abstractmethod
  def return_water_temperature_sensor(self) -> float:
    """Gets the return water temperature."""

  @property
  @abc.abstractmethod
  def reheat_water_setpoint(self) -> float:
    """Gets the reheat water setpoint."""

  @reheat_water_setpoint.setter
  @abc.abstractmethod
  def reheat_water_setpoint(self, value: float) -> None:
    """Sets the reheat water setpoint."""

  @property
  @abc.abstractmethod
  def run_command(self) -> RunStatus:
    """Gets the run command."""

  @run_command.setter
  @abc.abstractmethod
  def run_command(self, value: RunStatus) -> None:
    """Sets the run command."""

  @abc.abstractmethod
  def compute_thermal_energy_rate(
      self,
      return_water_temp: float,
      outside_temp: float,
      flow_rate: float,
  ) -> float:
    """Computes energy rate in W consumed to heat the water."""

  @abc.abstractmethod
  def compute_thermal_dissipation_rate(
      self, water_temp: float, outside_temp: float
  ) -> float:
    """Computes the thermal loss rate in W to the environment."""
