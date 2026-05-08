"""Thermal utilities, including constants and conversion functions.

This is the place for all logic related to temperature, pressure, volume,
and other thermodynamic physical quantities and their units.
"""

# TODO: b/505380216 - Rename this file to be called "thermal_utils.py"

import enum
from typing import Callable, Final, Mapping

TempConversionFunction = Callable[[float], float]


@enum.unique
class TempUnit(str, enum.Enum):
  """Temperature units."""

  KELVIN: Final[str] = 'Kelvin'
  CELSIUS: Final[str] = 'Celsius'
  FAHRENHEIT: Final[str] = 'Fahrenheit'

  @property
  def abbrev(self) -> str:
    """The single letter abbreviation for the temperature unit."""
    return self.value[0].upper()

  @property
  def deg_symbol(self) -> str:
    """The degree symbol for the temperature unit. Kelvin does not use one."""
    return '' if self == TempUnit.KELVIN else '°'


TEMP_UNITS: Final = tuple(TempUnit)
TEMP_UNIT_ABBREVIATIONS_MAP: Final[Mapping[str, TempUnit]] = {
    unit.value[0].upper(): unit for unit in TEMP_UNITS
}

CELSIUS_TO_KELVIN_OFFSET = 273.15

ABSOLUTE_ZERO_KELVIN = 0.0
ABSOLUTE_ZERO_CELSIUS = -CELSIUS_TO_KELVIN_OFFSET
ABSOLUTE_ZERO_FAHRENHEIT = -459.67


def assign_temp_unit(temp_unit: TempUnit | str) -> TempUnit:
  """Assigns and validates a given temperature display unit.

  Args:
    temp_unit: The desired temperature unit (e.g. "Kelvin", "Celsius", or
      "Fahrenheit"). Alternatively, you can use just the first letter, or a
      TempUnit enum. This provides a more flexible experience, because
      "Fahrenheit" is easily misspelled, and some services only provide the
      first letter of the unit.

  Returns:
    A valid official long-form temperature display unit.
  """
  try:
    first_letter = temp_unit[0].upper()
    return TEMP_UNIT_ABBREVIATIONS_MAP[first_letter]
  except (IndexError, KeyError) as e:
    raise ValueError(
        f'Unable to assign a valid temperature unit from: {temp_unit}.'
    ) from e


#
# FROM KELVIN
#


def kelvin_to_celsius(temp_k: float) -> float:
  """Converts Kelvin temperature to Celsius.

  Args:
    temp_k: Temperature in Kelvin.

  Returns:
    The corresponding temperature in Celsius.

  Raises:
    A ValueError if the input value is less than or equal to absolute zero.
  """
  if temp_k <= ABSOLUTE_ZERO_KELVIN:
    raise ValueError('Temperature must be greater than absolute zero.')
  return temp_k - CELSIUS_TO_KELVIN_OFFSET


def kelvin_to_fahrenheit(temp_k: float) -> float:
  """Converts Kelvin temperature to Fahrenheit.

  Args:
    temp_k: Temperature in Kelvin.

  Returns:
    The corresponding temperature in Fahrenheit.

  Raises:
    A ValueError if the input value is less than or equal to absolute zero.
  """
  if temp_k <= ABSOLUTE_ZERO_KELVIN:
    raise ValueError('Temperature must be greater than absolute zero.')
  temp_c = temp_k - CELSIUS_TO_KELVIN_OFFSET
  return temp_c * 9.0 / 5.0 + 32.0


KELVIN_CONVERSION_FUNCTIONS_MAP: Final[
    Mapping[TempUnit, TempConversionFunction | None]
] = {
    TempUnit.KELVIN: None,
    TempUnit.CELSIUS: kelvin_to_celsius,
    TempUnit.FAHRENHEIT: kelvin_to_fahrenheit,
}


def assign_kelvin_conversion_function(
    temp_unit: str,
) -> TempConversionFunction | None:
  """Assigns an appropriate temperature conversion function, from Kelvin.

  The conversion function converts temperatures from Kelvin to the specified
  display unit.

  Args:
    temp_unit: The temperature unit to be converted to (e.g. "Kelvin",
      "Celsius", or "Fahrenheit"), or just the first letter.

  Returns:
    The temperature conversion function (or None, if no conversion is needed).
  """
  return KELVIN_CONVERSION_FUNCTIONS_MAP[assign_temp_unit(temp_unit)]


def from_kelvin(temp_k: float, temp_unit: str) -> float:
  """Converts temperature from Kelvin to the specified display unit.

  Args:
    temp_k: Temperature in Kelvin.
    temp_unit: The temperature unit to be converted to (e.g. "Kelvin",
      "Celsius", or "Fahrenheit"), or just the first letter.

  Returns:
    The corresponding temperature in the specified display unit.
  """
  conversion_function = assign_kelvin_conversion_function(temp_unit)
  if conversion_function:
    return conversion_function(temp_k)
  return temp_k


#
# FROM FAHRENHEIT
#


def fahrenheit_to_kelvin(temp_f: float) -> float:
  """Converts Fahrenheit temperature to Kelvin.

  Args:
    temp_f: Temperature in Fahrenheit.

  Returns:
    The corresponding temperature in Kelvin.

  Raises:
    A ValueError if the input value is less than or equal to absolute zero.
  """
  if temp_f <= ABSOLUTE_ZERO_FAHRENHEIT:
    raise ValueError('Temperature must be greater than absolute zero.')
  temp_c = (temp_f - 32.0) * 5.0 / 9.0
  return temp_c + CELSIUS_TO_KELVIN_OFFSET


def fahrenheit_to_celsius(temp_f: float) -> float:
  """Converts Fahrenheit temperature to Celsius.

  Args:
    temp_f: Temperature in Fahrenheit.

  Returns:
    The corresponding temperature in Celsius.

  Raises:
    A ValueError if the input value is less than or equal to absolute zero.
  """
  if temp_f <= ABSOLUTE_ZERO_FAHRENHEIT:
    raise ValueError('Temperature must be greater than absolute zero.')
  return (temp_f - 32.0) * 5.0 / 9.0
