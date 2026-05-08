"""Tests for temperature conversion functions."""

from absl.testing import absltest
from absl.testing import parameterized

from smart_buildings.smart_control.utils import temperature_conversion as tc


class TempUnitTest(parameterized.TestCase):

  @parameterized.parameters(
      (tc.TempUnit.KELVIN, 'K'),
      (tc.TempUnit.CELSIUS, 'C'),
      (tc.TempUnit.FAHRENHEIT, 'F'),
  )
  def test_abbrev(self, temp_unit, expected_abbrev):
    self.assertEqual(temp_unit.abbrev, expected_abbrev)

  @parameterized.parameters(
      (tc.TempUnit.KELVIN, ''),
      (tc.TempUnit.CELSIUS, '°'),
      (tc.TempUnit.FAHRENHEIT, '°'),
  )
  def test_deg_symbol(self, temp_unit, expected_symbol):
    self.assertEqual(temp_unit.deg_symbol, expected_symbol)


class AssignTempUnitTest(parameterized.TestCase):

  @parameterized.parameters(
      ('Kelvin', tc.TempUnit.KELVIN),
      ('Celsius', tc.TempUnit.CELSIUS),
      ('Fahrenheit', tc.TempUnit.FAHRENHEIT),
      ('K', tc.TempUnit.KELVIN),
      ('C', tc.TempUnit.CELSIUS),
      ('F', tc.TempUnit.FAHRENHEIT),
      ('k', tc.TempUnit.KELVIN),
      ('c', tc.TempUnit.CELSIUS),
      ('f', tc.TempUnit.FAHRENHEIT),
      (tc.TempUnit.KELVIN, tc.TempUnit.KELVIN),
      (tc.TempUnit.CELSIUS, tc.TempUnit.CELSIUS),
      (tc.TempUnit.FAHRENHEIT, tc.TempUnit.FAHRENHEIT),
  )
  def test_assign_temp_unit_valid(self, temp_unit_input, expected_temp_unit):
    self.assertEqual(tc.assign_temp_unit(temp_unit_input), expected_temp_unit)

  @parameterized.parameters(('X'), (''))
  def test_assign_temp_unit_raises_value_error(self, temp_unit_input):
    with self.assertRaises(ValueError):
      tc.assign_temp_unit(temp_unit_input)

  @parameterized.parameters((None), (123))
  def test_assign_temp_unit_raises_type_error(self, temp_unit_input):
    with self.assertRaises(TypeError):
      tc.assign_temp_unit(temp_unit_input)


class TemperatureConversionsTest(parameterized.TestCase):

  # FROM KELVIN

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_k_to_f(self, temp_f, temp_k):
    self.assertAlmostEqual(tc.kelvin_to_fahrenheit(temp_k), temp_f, places=2)

  @parameterized.parameters((0.0), (-1.0))
  def test_k_to_f_invalid(self, temp_k):
    with self.assertRaises(ValueError):
      tc.kelvin_to_fahrenheit(temp_k)

  @parameterized.parameters(
      (0.0, 273.15), (-23.33, 249.817), (21.11, 294.261), (43.33, 316.483)
  )
  def test_k_to_c(self, temp_c, temp_k):
    self.assertAlmostEqual(tc.kelvin_to_celsius(temp_k), temp_c, places=2)

  @parameterized.parameters((0.0), (-1.0))
  def test_k_to_c_invalid(self, temp_k):
    with self.assertRaises(ValueError):
      tc.kelvin_to_celsius(temp_k)

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
    display_temp = tc.from_kelvin(temp_k=temp_k, temp_unit=temp_unit)
    self.assertAlmostEqual(display_temp, expected_temp)

  def test_from_kelvin_invalid_unit(self):
    with self.assertRaisesRegex(
        ValueError,
        'Unable to assign a valid temperature unit from: OOPS'
    ):
      tc.from_kelvin(temp_k=273.15, temp_unit='OOPS')

  @parameterized.parameters(
      ('Kelvin', None),
      ('K', None),
      ('k', None),
      ('Celsius', tc.kelvin_to_celsius),
      ('C', tc.kelvin_to_celsius),
      ('c', tc.kelvin_to_celsius),
      ('Fahrenheit', tc.kelvin_to_fahrenheit),
      ('F', tc.kelvin_to_fahrenheit),
      ('f', tc.kelvin_to_fahrenheit),
  )
  def test_kelvin_conversion_function_assignment(self, unit, expected_function):
    self.assertEqual(
        tc.assign_kelvin_conversion_function(temp_unit=unit), expected_function
    )

  def test_kelvin_conversion_function_assignment_invalid_unit(self):
    with self.assertRaisesRegex(
        ValueError, 'Unable to assign a valid temperature unit from: OOPS'
    ):
      tc.assign_kelvin_conversion_function(temp_unit='OOPS')

  # FROM FAHRENHEIT

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_f_to_k(self, temp_f, temp_k):
    self.assertAlmostEqual(tc.fahrenheit_to_kelvin(temp_f), temp_k, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_k_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      tc.fahrenheit_to_kelvin(temp_f)

  @parameterized.parameters(
      (32.0, 0.0), (-10.0, -23.33), (70.0, 21.11), (110.0, 43.33)
  )
  def test_f_to_c(self, temp_f, temp_c):
    self.assertAlmostEqual(tc.fahrenheit_to_celsius(temp_f), temp_c, places=2)

  @parameterized.parameters((-495.67), (-500.0))
  def test_f_to_c_invalid(self, temp_f):
    with self.assertRaises(ValueError):
      tc.fahrenheit_to_celsius(temp_f)


if __name__ == '__main__':
  absltest.main()
