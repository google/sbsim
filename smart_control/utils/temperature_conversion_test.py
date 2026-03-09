"""Tests for temperature conversion functions."""

from absl.testing import absltest
from absl.testing import parameterized

from smart_buildings.smart_control.utils import temperature_conversion

k_to_c = temperature_conversion.kelvin_to_celsius
k_to_f = temperature_conversion.kelvin_to_fahrenheit
f_to_c = temperature_conversion.fahrenheit_to_celsius
f_to_k = temperature_conversion.fahrenheit_to_kelvin

assign = temperature_conversion.assign_temp_display_and_conversion


class TemperatureConversionsTest(parameterized.TestCase):

  # FROM KELVIN

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_k_to_f(self, temp_f, temp_k):
    self.assertAlmostEqual(k_to_f(temp_k), temp_f, places=2)

  @parameterized.parameters((0.0), (-1.0))
  def test_k_to_f_invalid(self, temp_k):
    with self.assertRaises(ValueError):
      _ = k_to_f(temp_k)

  @parameterized.parameters(
      (0.0, 273.15), (-23.33, 249.817), (21.11, 294.261), (43.33, 316.483)
  )
  def test_k_to_c(self, temp_c, temp_k):
    self.assertAlmostEqual(k_to_c(temp_k), temp_c, places=2)

  @parameterized.parameters((0.0), (-1.0))
  def test_k_to_c_invalid(self, temp_k):
    with self.assertRaises(ValueError):
      _ = k_to_c(temp_k)

  # FROM FAHRENHEIT

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_f_to_k(self, temp_f, temp_k):
    self.assertAlmostEqual(f_to_k(temp_f), temp_k, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_k_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      _ = f_to_k(temp_f)

  @parameterized.parameters(
      (32.0, 0.0), (-10.0, -23.33), (70.0, 21.11), (110.0, 43.33)
  )
  def test_f_to_c(self, temp_f, temp_c):
    self.assertAlmostEqual(f_to_c(temp_f), temp_c, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_c_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      _ = f_to_c(temp_f)


class TemperatureConversionFunctionAssignmentTest(parameterized.TestCase):

  @parameterized.parameters('Kelvin', 'K')
  def test_conversion_function_assignment_kelvin(self, input_unit):
    """Tests temperature conversion for Kelvin."""
    display_unit, conversion_function = assign(display_unit=input_unit)
    self.assertEqual(display_unit, 'Kelvin')
    self.assertIsNone(conversion_function)

  @parameterized.parameters(
      ('Celsius', 'Celsius', k_to_c, 26.85),
      ('C', 'Celsius', k_to_c, 26.85),
      ('Fahrenheit', 'Fahrenheit', k_to_f, 80.33),
      ('F', 'Fahrenheit', k_to_f, 80.33),
  )
  def test_conversion_function_assignment_non_kelvin(
      self,
      input_unit,
      expected_display_unit,
      expected_conversion_function,
      expected_display_temp,
  ):
    """Tests temperature conversion for non-Kelvin units."""
    display_unit, conversion_function = assign(display_unit=input_unit)
    self.assertEqual(conversion_function, expected_conversion_function)
    self.assertEqual(display_unit, expected_display_unit)
    self.assertIsNotNone(conversion_function)
    self.assertAlmostEqual(
        conversion_function(300), expected_display_temp, places=2
    )

  def test_invalid_temp_unit_raises_error(self):
    """Tests that an invalid temp unit raises a ValueError."""
    with self.assertRaises(ValueError):
      assign(display_unit='OOPS')

if __name__ == '__main__':
  absltest.main()
