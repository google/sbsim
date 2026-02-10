"""Tests for reinforcement learning data processing utils."""

from absl.testing import absltest

from smart_control.reinforcement_learning.utils.data_processing import convert_celsius_to_kelvin
from smart_control.reinforcement_learning.utils.data_processing import convert_kelvin_to_celsius


class TestTempConversions(absltest.TestCase):

  def test_c_to_k(self):
    self.assertEqual(convert_celsius_to_kelvin(0), 273.15)

  def test_k_to_c(self):
    self.assertEqual(convert_kelvin_to_celsius(273.15), 0)


if __name__ == '__main__':
  absltest.main()
