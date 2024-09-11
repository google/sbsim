"""A collection of utility functions for Smart Building energy problems.

Copyright 2022 Google LLC

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

from typing import Optional, Sequence
import numpy as np
from smart_control.utils import constants

# Source: Thermodynamik, (1992), Hans Dieter Baehr, 8. Auflage, Springer Verlag
# Tabelle 5.4, p. 213
_WATER_SATURATION_TEMPS_REF = [i + 273.0 for i in range(-40, 80, 10)]
_WATER_SATURATION_PRESSURES_REF = [
    0.1285,
    0.3802,
    1.0328,
    2.5992,
    6.1115,
    12.279,
    23.385,
    42.452,
    73.813,
    123.448,
    199.33,
    311.77,
]

_FAN_SPEED_PERCENTAGE_OPERATIONAL_THRESH = 5.0
_SUPPLY_STATIC_PRESSURE_OPERATIONAL_THRESH = 0.2
_DEFAULT_EER = 12.0


def get_water_vapor_partial_pressure(temps: Sequence[float]) -> Sequence[float]:
  """Returns the partial pressure of moist air.

  Source: Thermodynamik, (1992), Hans Dieter Baehr, 8. Auflage, Springer Verlag
  Tabelle 5.4, p. 213

  Args:
    temps: list of temps [K]

  Returns:
    approximate water vapor partial pressure of water in mbar
  """
  return np.interp(
      temps, _WATER_SATURATION_TEMPS_REF, _WATER_SATURATION_PRESSURES_REF
  )


def get_humidity_ratio(
    temps: Sequence[float],
    relative_humidities: Sequence[float],
    pressures: Sequence[float],
) -> Sequence[float]:
  """Returns the humidity ratio of moist air.

  Source: Thermodynamik, (1992), Hans Dieter Baehr, 8. Auflage, Springer Verlag
  Gleichung 5.26, p. 215

  Args:
    temps: list of air temperatures in K
    relative_humidities: relative humidity [0.0 (dry) - 1.0 (saturated)]
    pressures: atmospheric pressure in bar, defaults to 1.0 (standard atm)

  Returns: water mass to air mass ratio in kg Water / kg Air
  """
  assert len(temps) == len(relative_humidities) == len(pressures)
  psat = [p / 1000.0 for p in get_water_vapor_partial_pressure(temps)]
  return [
      0.622 * psat[i] / (pressures[i] / relative_humidities[i] - psat[i])
      for i in range(len(temps))
  ]


def get_air_conditioning_energy_rate(
    *,
    air_flow_rates: Sequence[float],
    outside_temps: Sequence[float],
    outside_relative_humidities: Sequence[float],
    supply_temps: Sequence[float],
    ambient_pressures: Sequence[float],
) -> Sequence[float]:
  """Returns the energy rate for heating/cooling moist outside air.

  Source: Thermodynamik, (1992), Hans Dieter Baehr, 8. Auflage, Springer Verlag
  Beispiel 5.6, p. 219

  Assumes isobaric conditions, with no additional (de-) humidification, and
  outside air is not saturated.

  Args:
    air_flow_rates: list of mass flow of outside air [kg/s]
    outside_temps: list of outside air temperature [K]
    outside_relative_humidities: relative humidity [0.0 (dry) - 1.0 (saturated)]
    supply_temps: temperatures of the supply air [K]
    ambient_pressures: lost of pressures [bar]

  Returns: Thermal power applied to heat the air to supply temp [W]
  """

  assert (
      len(air_flow_rates)
      == len(outside_temps)
      == len(outside_relative_humidities)
      == len(supply_temps)
      == len(ambient_pressures)
  ), 'All input vectors must be of the same length.'

  x = get_humidity_ratio(
      temps=outside_temps,
      relative_humidities=outside_relative_humidities,
      pressures=ambient_pressures,
  )
  return [
      air_flow_rates[i]
      * (
          constants.AIR_HEAT_CAPACITY
          + x[i] * constants.WATER_VAPOR_HEAT_CAPACITY
      )
      * (supply_temps[i] - outside_temps[i])
      for i in range(len(air_flow_rates))
  ]


def get_fan_power(
    *,
    design_hp: Optional[float] = None,
    brake_hp: Optional[float] = None,
    fan_speed_percentage: Optional[float] = None,
    supply_static_pressure: Optional[float] = None,
    motor_factor: Optional[float] = None,
    num_fans: Optional[int] = 1,
) -> float:
  """Calculates fan power using design motor horsepower, speed, and duty cycle.

  If fan speed is not available, and the fan is constant volume, assume
  whenever the fan is running that it is at 100% speed.
  If only horsepower is available, multiple by a Motor Factor of 0.85
  (which accounts for losses to friction). If brake horsepower is available,
  no Motor Factor is needed. Horsepower (HP) or brake horsepower (BHP)
  can typically be found in AHU manufacturer documentation, or directly on
  the equipment nameplate (on-site). If there are multiple fans in the AHU, m
  ultiply by the total number of fans running (integer) to obtain total fan
  power. Source: go/sb-energy-calculations.

  Args:
    design_hp: Design horsepower of the fan.
    brake_hp: Brake horsepower of the fan.
    fan_speed_percentage: Fan speed percentage, 0 - 100.
    supply_static_pressure: Static pressure in psi.
    motor_factor: Fan's efficiency coefficient when uing the design horsepower.
    num_fans: Number of fans that are included in the total calculation.

  Returns:
    The fan power in Watts.

  Raises:
    ValueError if neither design_hp or break_hp are provided.
  """

  if design_hp is None and brake_hp is None:
    raise ValueError(
        'Must provide either design horseposer or brake horsepower.'
    )

  if fan_speed_percentage is None:
    fan_speed_percentage = 100.0

  if motor_factor is None:
    motor_factor = 0.85

  if brake_hp:
    hp = brake_hp
  else:
    hp = motor_factor * design_hp

  # Fan is operational if the supply_static_pressure > threshold for
  # supply fan. Exhaust fan doesn't report static pressure, so assume on
  # when fan_speed_percentage is > 0.
  is_operational = float(
      supply_static_pressure is None
      or supply_static_pressure >= _SUPPLY_STATIC_PRESSURE_OPERATIONAL_THRESH
  )

  return (
      hp
      * 0.746
      * (fan_speed_percentage / 100.0) ** 2.5
      * is_operational
      * num_fans
  )


def get_air_volumetric_flowrate(
    *, average_fan_speed_percentage: float, design_cfm: float
) -> float:
  """Calculates the air handler volumetric flow rate.

  To calculate the volumetric flow rate of air in the system, multiply the
  average fan speed (in percentage) and the design flow rate for the AHU.
  The design volumetric flow rate for the fan(s) can typically be found on
  the equipment nameplate (on-site) or within manufacturer fan curve
  documentation. Source: go/sb-energy-calculations.

  Args:
    average_fan_speed_percentage: The fan average percentage (0-100).
    design_cfm: Air handler design flow rate in cubic feet per minute (cfm).

  Returns:
    volumetric flow rate in cfm
  """

  return design_cfm * average_fan_speed_percentage / 100.0


def get_compressor_power_thermal(
    *,
    mixed_air_temp: float,
    supply_air_temp: float,
    volumetric_flow_rate: float,
    fan_speed_percentage: float = 100.0,
    eer: float = 12.0,
    fan_heat_temp: float = 0.0,
) -> float:
  """Computes power from the air conditioner compressor, using thermal method.

  Calculated based on the volumetric air flow rate, cooling ΔT, and fan duty
  cycle when the compressor command is ‘ON’.  Note that cp is assumed to be
  1.08 and the kW/Ton Efficiency is 12/EER or 12/IEER of the AHU where
  available.

  EER is the ratio of energy capacity [BTUs] / power input [W]. A higher EER
  means a more efficient unit, in general. The conversion of 12/EER calculates
  the kW/Ton Efficiency (where a lower value is more efficient).

    Example: If my unit has a capacity of 13,000 [BTU] and requires
    1,000 [W] of power --> my EER is 13

    Converting to kW/Ton:
    EER = 12/[kW/Ton] --> kW/Ton = 12/EER

    kW/Ton = 12/13 = 0.92 (which means for every ton of cooling provided to the
    building, the unit consumes 0.92 kW of power)

    If my EER is < 12, the kW/Ton value will be greater than 1, meaning the
    unit is quite inefficient and will consume more than 1 kW per ton of
    cooling provided to the building.


  Source: go/sb-energy-calculations.

  Args:
    mixed_air_temp: Air handler mixed air (recirc + fresh) temperature in °F.
    supply_air_temp: Air handler supply air temperature in °F.
    volumetric_flow_rate: Amount of air flow in cfm.
    fan_speed_percentage: The fan speed percentage, 0 - 100.
    eer: The system EER, see explanation above.
    fan_heat_temp: Additional temp in °F (1 - 2°F) induced by fan motor heat.

  Returns:
    Compressor power in kW.
  """

  fan_operational = float(
      fan_speed_percentage >= _FAN_SPEED_PERCENTAGE_OPERATIONAL_THRESH
  )
  kw_ton_eff = 12.0 / eer
  return (
      1.08
      * volumetric_flow_rate
      * (mixed_air_temp - supply_air_temp + fan_heat_temp)
      * fan_operational
      / 12000.0
      * kw_ton_eff
  )


def get_compressor_power_utilization(
    *,
    design_capacity: float,
    cooling_percentage: Optional[float] = None,
    count_stages_on: Optional[int] = None,
    total_stages: Optional[int] = None,
    eer: Optional[float] = None,
) -> float:
  """Computes power from the air conditioner compressor based on utilization.

  Calculated based on the design capacity of the compressors and the compressor
  utilization ratio. If a compressor has four (4) cooling stages (assuming
  all stages are equal in cooling capacity) and two (2) stages are running,
  the compressor utilization ratio would be 0.5. Occasionally the BMS may have
  a cooling percentage sensor (which basically calculates the utilization ratio
  for us, this is preferred wherever possible).

  The total number of cooling stages can typically be inferred from the BMS, or
  manufacturer design metadata. The design capacity of the compressor can also
  be found in manufacturer documentation (or on equipment nameplates on-site).

  EER is the ratio of energy capacity [BTUs] / power input [W]. A higher EER
  means a more efficient unit, in general. The conversion of 12/EER calculates
  the kW/Ton Efficiency (where a lower value is more efficient).

    Example: If my unit has a capacity of 13,000 [BTU] and requires
    1,000 [W] of power --> my EER is 13

    Converting to kW/Ton:
    EER = 12/[kW/Ton] --> kW/Ton = 12/EER

    kW/Ton = 12/13 = 0.92 (which means for every ton of cooling provided to the
    building, the unit consumes 0.92 kW of power)

    If my EER is < 12, the kW/Ton value will be greater than 1, meaning the
    unit is quite inefficient and will consume more than 1 kW per ton of
    cooling provided to the building.


  Source: go/sb-energy-calculations.

  Args:
    design_capacity: Total air handler design capacity in tons.
    cooling_percentage: Air handler cooling percentage [0 - 100].
    count_stages_on: Number of stages that were in ON configuration.
    total_stages: Maximum run stages: 0 <= count_stages_on <= total_stages.
    eer: The system EER, see explanation above.

  Returns:
    Compressor power in kW.

  Raises:
    ValueError for invalid cooling percentage, or invalid count/total stages.
  """

  if eer is None:
    eer = _DEFAULT_EER

  if cooling_percentage is not None:  # Preferred approach.
    if cooling_percentage < 0.0 or cooling_percentage > 100.0:
      raise ValueError('cooling_percentage must be between 0 and 100.')
    utilization_ratio = cooling_percentage / 100.0

  elif total_stages is not None and count_stages_on is not None:
    if total_stages <= 0:
      raise ValueError('Total stages must be greater than 0.')

    if count_stages_on < 0:
      raise ValueError('Stages on must be not be negative.')

    if count_stages_on > total_stages:
      raise ValueError('Total stages must be greater than count_stages_on.')

    utilization_ratio = count_stages_on / total_stages

  else:
    raise ValueError(
        'Both cooling_percentage and total_stages, count_stages_on are None.'
    )
  kw_ton_eff = 12.0 / eer
  return utilization_ratio * design_capacity * kw_ton_eff


def get_water_pump_power(
    *,
    pump_duty_cycle: float,
    pump_speed_percentage: Optional[float] = 100.0,
    brake_horse_power: Optional[float] = None,
    design_motor_horse_power: Optional[float] = None,
    motor_factor: float = 0.85,
    num_pumps: int = 1,
) -> float:
  """Calculates the hot water pump power in kW.

  Calculate the pump power using the design motor horsepower, pump speed, and
  pump duty cycle. If pump speed is not available, and the pump is constant
  volume, assume whenever the pump is running that it is at 100% speed.
  If only horsepower is available, multiple by a Motor Factor of 0.85
  (which accounts for losses to friction). If brake horsepower is available,
  no Motor Factor is needed. Horsepower (HP) or brake horsepower (BHP) can
  typically be found in pump manufacturer documentation, or directly on the
  equipment nameplate (on-site).  If there are multiple pumps in the system,
  multiply by the total number of pumps running (integer) to obtain total pump
  power. Source: go/sb-energy-calculations.

  Args:
    pump_duty_cycle: Duty cycle, ranging in [0 - 1.0]
    pump_speed_percentage: Percentage of pump speed [0 - 100].
    brake_horse_power: Motor horse power if available.
    design_motor_horse_power: Pump motor horsepower.
    motor_factor: Motor efficiency coefficient [0.0 - 1.0].
    num_pumps: Total number of pumps affected.

  Returns:
    Pump power in kW.

  Raises:
    ValueError if neither brake_horse_power nor design_motor_horse_power are
    provided.
  """

  if brake_horse_power is None and design_motor_horse_power is None:
    raise ValueError('Must provide either brake_ or design_motor_horse_power.')
  if brake_horse_power:
    total_motor_hp = brake_horse_power
  else:
    total_motor_hp = design_motor_horse_power * motor_factor

  return (
      total_motor_hp
      * 0.746
      * (pump_speed_percentage / 100) ** 2.5
      * pump_duty_cycle
      * num_pumps
  )


def get_water_volumetric_flow_rate(
    *,
    design_flow_rate: float,
    pump_speed_percentage: float,
    num_pumps_on: int = 1,
) -> float:
  """Calculates the water pump flow rate in gallons per minute.

  To calculate the volumetric flow rate of water in the system, multiply the
  number of pumps running (integer) by the average pump speed (in percentage)
  and the design flow rate for a single pump. The design volumetric flow rate
  for the pumps can typically be found on the equipment nameplate (on-site) or
  within manufacturer pump curve documentation.
  Source: go/sb-energy-calculations.

  Args:
    design_flow_rate: Design flow rate of the pump in gallons per minute (gpm).
    pump_speed_percentage: Percentage of pump speed [0 - 100].
    num_pumps_on: Number of pumps running.

  Returns:
    Water volumetric flow rate in gpm.
  """
  return num_pumps_on * (pump_speed_percentage / 100.0) * design_flow_rate


def get_water_heating_energy_rate(
    *,
    volumetric_flow_rate: float,
    supply_water_temperature: float,
    return_water_temperature: float,
) -> float:
  """Computes the water heating energy rate in BTU/hr.

  If the system is equipped with a physical flow rate sensor, use the sensor
  reading, if not use the above calculated volumetric flow rate. Then,
  calculate the system load by multiplying the flow rate (in GPM) by the
  system delta T (SWT – RWT). In the equation below, 500 is a combined value
  representing the specific heat of water (1 BTU/lbm*°F) multiplied by the
  specific density of water (8.34 lbm/gal) and 60 (min/hr).
  Source: go/sb-energy-calculations.

  Args:
    volumetric_flow_rate: Water flow rate in gpm.
    supply_water_temperature: Heated supply temperature in °F.
    return_water_temperature: Cooled return temperature in °F.

  Returns:
    Heating energy rate in BTU/hr.
  """

  return max(
      0,
      500.0
      * volumetric_flow_rate
      * (supply_water_temperature - return_water_temperature),
  )


def get_water_heating_energy_rate_primary(
    *,
    design_boiler_flow_rate: float,
    boiler_outlet_temperature: float,
    return_water_temperature: float,
    num_active_boilers: int = 1,
) -> float:
  """Computes Primary/Secondary water heating rates.

  This method is only used for primary-secondary systems. It estimates the
  flow through each individual boiler using the design flow rate of the
  boiler’s onboard circulation pump and the number of active boilers.
  Next, the temperature rise across each boiler can be calculated using the
  difference between the discharge water temperature at the boiler outlet and
  the system return water temperature.
  Source: go/sb-energy-calculations.

  Args:
    design_boiler_flow_rate: Water flow rate in gpm.
    boiler_outlet_temperature: Heated supply temperature in °F.
    return_water_temperature: Cooled return temperature in °F.
    num_active_boilers: Number of boilers currently running.

  Returns:
    Heating energy rate in BTU/hr.
  """
  primary_flow_rate = design_boiler_flow_rate * num_active_boilers
  return get_water_heating_energy_rate(
      volumetric_flow_rate=primary_flow_rate,
      supply_water_temperature=boiler_outlet_temperature,
      return_water_temperature=return_water_temperature,
  )


def get_water_heating_energy_rate_primary_secondary(
    *,
    design_primary_boiler_flow_rate: float,
    design_secondary_boiler_flow_rate: float,
    boiler_outlet_temperature: float,
    return_water_temperature: float,
    num_active_boilers: int = 1,
    num_active_secondary_pumps: int = 0,
    avg_secondary_pump_speed_percentage: float = 0.0,
) -> float:
  """Computes Primary/Secondary water heating rates.

  Similar to get_water_heating_energy_rate_primary(), this method only applies
  to primary-secondary systems.  The flow through the primary system loop is
  calculated using the design flow rate of the boiler’s onboard circulation
  pump multiplied by the number of active boilers.

  The estimated flow through the secondary loop is also calculated in this way
  using the secondary HWP speeds and number of active secondary HWPs.

  In cases when the primary loop flow is greater than the secondary loop flow,
  not all the heated discharge water from the boilers will make it to the
  secondary loop; the excess primary flow will take the common piping,
  blend with the system return water, and re-enter the boilers at their intake.
  In order to accurately represent the temperature rise across the boilers,
  this blended return water temperature needs to be calculated.
  Source: go/sb-energy-calculations.

  Args:
    design_primary_boiler_flow_rate: Water flow rate in gpm of primary cycle.
    design_secondary_boiler_flow_rate: Water flow rate in gpm of sec. cycle.
    boiler_outlet_temperature: Heated supply temperature in F.
    return_water_temperature: Cooled return temperature in F.
    num_active_boilers: Number of boilers currently running.
    num_active_secondary_pumps: Number of active secondary pumps.
    avg_secondary_pump_speed_percentage: Pecentage [0 - 100] opf sec. pumps.

  Returns:
    Heating energy rate in BTU/hr.

  Raises:
    ValueError if either primary_ or secondary_flow_rate are negative.
  """

  primary_flow_rate = design_primary_boiler_flow_rate * num_active_boilers
  if primary_flow_rate < 0.0:
    raise ValueError('Primary flow rate must be non-negative.')

  secondary_flow_rate = (
      design_secondary_boiler_flow_rate
      * num_active_secondary_pumps
      * avg_secondary_pump_speed_percentage
      / 100.0
  )
  if secondary_flow_rate < 0.0:
    raise ValueError('Secondary flow rate must be non-negative.')

  diff_flow_rate = primary_flow_rate - secondary_flow_rate

  blended_return_temperature = (
      secondary_flow_rate * return_water_temperature
      + diff_flow_rate * boiler_outlet_temperature
  ) / primary_flow_rate

  return get_water_heating_energy_rate(
      volumetric_flow_rate=primary_flow_rate,
      supply_water_temperature=boiler_outlet_temperature,
      return_water_temperature=blended_return_temperature,
  )
