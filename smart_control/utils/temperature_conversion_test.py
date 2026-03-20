"""Tests for temperature conversion functions."""

from absl.testing import absltest
from absl.testing import parameterized

from smart_buildings.smart_control.utils import temperature_conversion

k_to_c = temperature_conversion.kelvin_to_celsius
k_to_f = temperature_conversion.kelvin_to_fahrenheit
f_to_c = temperature_conversion.fahrenheit_to_celsius
f_to_k = temperature_conversion.fahrenheit_to_kelvin

get_kelvin_converter = temperature_conversion.assign_kelvin_conversion_function


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
      k_to_f(temp_k)

  @parameterized.parameters(
      (0.0, 273.15), (-23.33, 249.817), (21.11, 294.261), (43.33, 316.483)
  )
  def test_k_to_c(self, temp_c, temp_k):
    self.assertAlmostEqual(k_to_c(temp_k), temp_c, places=2)

  @parameterized.parameters((0.0), (-1.0))
  def test_k_to_c_invalid(self, temp_k):
    with self.assertRaises(ValueError):
      k_to_c(temp_k)

  @parameterized.parameters(
      (273.15, 'Fahrenheit', 32.0),
      (273.15, 'F', 32.0),
      (273.15, 'f', 32.0),
      (273.15, 'Celsius', 0.0),
      (273.15, 'C', 0.0),
      (273.15, 'c', 0.0),
      (273.15, 'Kelvin', 273.15),
      (273.15, 'K', 273.15),
      (273.15, 'k', 273.15),
  )
  def test_from_kelvin(self, temp_k, temp_unit, expected_temp):
    display_temp = temperature_conversion.from_kelvin(
        temp_k=temp_k, temp_unit=temp_unit
    )
    self.assertAlmostEqual(display_temp, expected_temp)

  def test_from_kelvin_invalid_unit(self):
    with self.assertRaisesRegex(
        ValueError,
        'Unable to assign a valid temperature unit from: OOPS'
    ):
      temperature_conversion.from_kelvin(temp_k=273.15, temp_unit='OOPS')

  @parameterized.parameters(
      ('Kelvin', None),
      ('K', None),
      ('k', None),
      ('Celsius', k_to_c),
      ('C', k_to_c),
      ('c', k_to_c),
      ('Fahrenheit', k_to_f),
      ('F', k_to_f),
      ('f', k_to_f),
  )
  def test_kelvin_conversion_function_assignment(self, unit, expected_function):
    self.assertEqual(get_kelvin_converter(temp_unit=unit), expected_function)

  def test_kelvin_conversion_function_assignment_invalid_unit(self):
    with self.assertRaisesRegex(
        ValueError,
        'Unable to assign a valid temperature unit from: OOPS'
    ):
      get_kelvin_converter(temp_unit='OOPS')

  # FROM FAHRENHEIT

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_f_to_k(self, temp_f, temp_k):
    self.assertAlmostEqual(f_to_k(temp_f), temp_k, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_k_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      f_to_k(temp_f)

  @parameterized.parameters(
      (32.0, 0.0), (-10.0, -23.33), (70.0, 21.11), (110.0, 43.33)
  )
  def test_f_to_c(self, temp_f, temp_c):
    self.assertAlmostEqual(f_to_c(temp_f), temp_c, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_c_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      f_to_c(temp_f)


if __name__ == '__main__':
  absltest.main()
