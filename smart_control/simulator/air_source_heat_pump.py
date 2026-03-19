r"""Models commercial Air Source Heat Pump (ASHP), patterned after Aermec 0700.

Overview:
  This class provides a discrete-time thermodynamic simulation of an ASHP acting
  as a hot water heat source. Unlike traditional gas or electric resistance
  boilers that operate with a relatively constant efficiency, an ASHP operates
  by absorbing heat from the ambient outside air and transferring it to a
  hydronic water loop using a vapor-compression refrigeration cycle.
  Consequently, its heating capacity and electrical efficiency are
  highly dependent on environmental conditions.

Thermodynamic Modeling Approach:
  The simulation calculates the electrical power consumption based on the
  thermal demand of the building and the real-time
  Coefficient of Performance (COP) of the heat pump.

  1. Thermal Demand: The required thermal power $Q$ to reach the supply setpoint
  is calculated using the mass flow rate and specific heat capacity of water:
     $$Q = \dot{m} \cdot c_p \cdot \max(0, T_{setpoint} - T_{return})$$

  2. Dynamic Efficiency (COP): The COP represents the ratio of heat delivered to
  electrical work consumed. This model implements a simplified dynamic COP
  that degrades linearly as the ambient outside air temperature $T_{ambient}$
  decreases, reflecting the reduced thermal energy available in colder air and
  the increased lift required by the compressor.

  3. Electrical Power: The final electrical power $P_{elec}$ consumed by the
  unit is:
     $$P_{elec} = \frac{\min(Q, Q_{max})}{COP(T_{ambient})}$$

Strengths of this Implementation:
  * Computational Efficiency: By utilizing a simplified linear COP degradation
  model and steady-state thermal equations, this simulation is extremely fast,
  making it suitable for training Reinforcement Learning (RL) agents or running
  multi-year energy simulations.
  * Environmental Responsiveness: Accurately reflects the most critical behavior
  of an ASHP—that its electrical efficiency drops significantly as outside air
  gets colder.
  * Interoperability: Adheres to the abstract `HotWaterHeatSource` interface,
  allowing it to be seamlessly swapped with `Boiler` models in `gin`
  configurations without breaking the broader building simulation.

Limitations and Simplifications:
  * Linear COP Curve: Real-world compressor efficiency curves are non-linear and
  rely on bi-variate performance maps (dependent on both ambient air temperature
  AND leaving water temperature). This model simplifies that relationship.
  * Constant Maximum Capacity: In physical ASHPs, maximum heating capacity
  $Q_{max}$ drops alongside COP in cold weather. This model assumes a
  constant maximum capacity.
  * Defrost Cycles Excluded: At ambient temperatures near freezing (e.g., 30°F -
  40°F) with high humidity, ASHPs must periodically reverse their cycle to melt
  frost off the evaporator coils. This causes significant, transient drops in
  efficiency and capacity that are not captured in this model.
  * Part-Load Ratios (PLR): The model does not explicitly account for compressor
  staging, inverter efficiencies, or part-load performance degradation, assuming
  linea power scaling regardless of the load fraction.
"""

from typing import Final
import uuid

import gin
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import hot_water_heat_source
from smart_buildings.smart_control.simulator import smart_device
from smart_buildings.smart_control.utils import constants
from smart_buildings.smart_control.utils import temperature_conversion

# approx surface area of ASHP heat exchanger/buffer
_EXTERIOR_SURFACE_AREA_M2: Final[float] = 4.0
# Fixed convection coefficient for exterior surface of ASHP due to ambient air
# currents or wind.
_EXTERIOR_CONVECTION_COEFF_W_M2_K: Final[float] = 5.6


@gin.configurable
class AirSourceHeatPump(hot_water_heat_source.HotWaterHeatSource):
  """Simulation of an Aermec 0700 Air Source Heat Pump."""

  def __init__(
      self,
      reheat_water_setpoint: float,
      device_id: str | None = None,
      # ~600k BTU/hr, typical for large commercial ASHP
      max_heating_capacity_w: float = 180000.0,
      nominal_cop: float = 3.2,
      init_return_water_temperature_sensor: float = 295.0,
  ):
    observable_fields = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            'reheat_water_setpoint', float
        ),
        'supply_water_temperature_sensor': smart_device.AttributeInfo(
            'supply_water_temperature_sensor', float
        ),
    }

    action_fields = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            'reheat_water_setpoint', float
        ),
    }

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.ASHP,
        device_id=device_id or f'ashp_id_{uuid.uuid4()}',
    )

    self._max_capacity_w = max_heating_capacity_w
    self._nominal_cop = nominal_cop
    self._init_return_water_temp = init_return_water_temperature_sensor
    self._return_water_temp = init_return_water_temperature_sensor
    self._run_command = hot_water_heat_source.RunStatus.On
    self._supply_water_setpoint = reheat_water_setpoint

  def reset(self) -> None:
    self._return_water_temp = self._init_return_water_temp
    self._run_command = hot_water_heat_source.RunStatus.On

  @property
  def supply_water_temperature_sensor(self) -> float:
    # ASHP does not have a tank, so supply water temperature is the setpoint.
    return self._supply_water_setpoint

  @property
  def reheat_water_setpoint(self) -> float:
    return self._supply_water_setpoint

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    self._supply_water_setpoint = value

  @property
  def return_water_temperature_sensor(self) -> float:
    return self._return_water_temp

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    self._return_water_temp = value

  @property
  def run_command(self) -> hot_water_heat_source.RunStatus:
    return self._run_command

  @run_command.setter
  def run_command(self, value: hot_water_heat_source.RunStatus) -> None:
    self._run_command = value

  def _calculate_dynamic_cop(self, outside_temp_k: float) -> float:
    """Calculates COP based on outside air temperature.

    ASHP efficiency drops as outside air gets colder.

    Args:
      outside_temp_k: Outside air temperature in Kelvin.

    Returns:
      The dynamic COP at the given outside temperature.
    """
    outside_temp_c = temperature_conversion.kelvin_to_celsius(outside_temp_k)

    # Simplified linear COP degradation model for Aermec units.
    # Assumes nominal COP is rated at 7C (45F) outside air.
    cop = self._nominal_cop + (outside_temp_c - 7.0) * 0.05

    # Clamp COP to realistic physical bounds for an ASHP
    return max(1.0, min(cop, 5.0))

  def compute_thermal_energy_rate(
      self, return_water_temp: float, outside_temp: float, flow_rate: float
  ) -> float:
    """Returns the electrical power (W) consumed by the ASHP to heat water.

    Args:
      return_water_temp: The temperature of the water returning to the ASHP (K).
      outside_temp: The ambient outside air temperature (K).
      flow_rate: The mass flow rate of water through the ASHP (kg/s).
    """
    if (
        self._run_command == hot_water_heat_source.RunStatus.Off
        or flow_rate <= 0
    ):
      return 0.0

    # Calculate required thermal energy to reach setpoint (Q = m * c * dT)
    # Using placeholder constants for density and specific heat capacity
    specific_heat_water = constants.WATER_HEAT_CAPACITY  # J/(kg*K)
    target_temp = self.get_observation(
        'supply_water_setpoint', self._observation_timestamp
    )
    self._supply_water_setpoint = target_temp

    delta_t = max(0.0, target_temp - return_water_temp)
    required_thermal_power_w = (
        flow_rate * constants.WATER_DENSITY * specific_heat_water * delta_t
    )

    # Limit by the maximum physical capacity of the Aermec unit
    actual_thermal_power_w = min(required_thermal_power_w, self._max_capacity_w)

    # Electrical power = Thermal Power / COP
    cop = self._calculate_dynamic_cop(outside_temp)
    return actual_thermal_power_w / cop

  def compute_thermal_dissipation_rate(
      self, water_temp: float, outside_temp: float
  ) -> float:
    """Computes thermal loss rate from the ASHP unit to the ambient air."""

    surface_area = _EXTERIOR_SURFACE_AREA_M2
    return (
        _EXTERIOR_CONVECTION_COEFF_W_M2_K
        * surface_area
        * max(0.0, water_temp - outside_temp)
    )
