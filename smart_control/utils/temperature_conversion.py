"""Temperature-related utilities, including conversion functions."""

DISPLAY_UNITS = ('Kelvin', 'Celsius', 'Fahrenheit')

CELSIUS_TO_KELVIN_OFFSET = 273.15

ABSOLUTE_ZERO_KELVIN = 0.0
ABSOLUTE_ZERO_CELSIUS = -CELSIUS_TO_KELVIN_OFFSET
ABSOLUTE_ZERO_FAHRENHEIT = -459.67

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


def assign_temp_display_and_conversion(display_unit: str):
  """Assigns a temperature display unit and corresponding conversion function.

  Assumes temperature inputs are in Kelvin.

  Args:
    display_unit: The temperature display unit to use (e.g. "Kelvin", "Celsius",
      or "Fahrenheit"). Alternatively, you can use just the first letter (e.g.
      "K", "C", or "F"). This provides a more convenient developer experience,
      because "Fahrenheit" is easily misspelled.

  Returns:
    A tuple of the temperature display unit and corresponding temperature
    conversion function (or None, if no conversion is needed).
  """
  display_unit = display_unit.title()
  if display_unit.startswith('F'):
    display_unit = 'Fahrenheit'
    conversion_function = kelvin_to_fahrenheit
  elif display_unit.startswith('C'):
    display_unit = 'Celsius'
    conversion_function = kelvin_to_celsius
  elif display_unit.startswith('K'):
    display_unit = 'Kelvin'
    conversion_function = None
  else:
    raise ValueError(
        f'Invalid display temperature unit: {display_unit}.'
        f' Expecting one of: {DISPLAY_UNITS}'
    )
  return display_unit, conversion_function


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
