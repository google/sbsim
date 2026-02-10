"""Tests for Dbo compliant pump."""

from absl.testing import absltest
from absl.testing import parameterized

from smart_buildings.smart_control.simulator import dbo_pump
from smart_buildings.smart_control.utils import constants


class DboPumpTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.5, 3, 0.9),
      (0.2, 7, 0.5),
      (0.5, 8, 0.23),
      (0.5, 9, 0.7),
  )
  def test_compute_pump_power(
      self, total_flow_rate, water_pump_differential_head, water_pump_efficiency
  ):
    pump = dbo_pump.Pump(
        water_pump_differential_head=water_pump_differential_head,
        water_pump_efficiency=water_pump_efficiency,
    )

    expected = (
        total_flow_rate
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * water_pump_differential_head
        / water_pump_efficiency
    )
    self.assertEqual(pump.compute_pump_power(total_flow_rate), expected)

  def test_run_command(self):
    pump = dbo_pump.Pump(
        water_pump_differential_head=3,
        water_pump_efficiency=0.9,
    )
    self.assertEqual(pump.run_command, dbo_pump.RunStatus.Off)
    pump.run_command = dbo_pump.RunStatus.On
    self.assertEqual(pump.run_command, dbo_pump.RunStatus.On)

  def test_differential_pressure(self):
    water_pump_differential_head = 3
    pump = dbo_pump.Pump(
        water_pump_differential_head=water_pump_differential_head,
        water_pump_efficiency=0.9,
    )
    expected_dp = (
        constants.GRAVITY
        * constants.WATER_DENSITY
        * water_pump_differential_head
        / constants.PASCALS_PER_BAR
    )
    self.assertAlmostEqual(pump.differential_pressure, expected_dp)
    pump.differential_pressure = 30
    self.assertAlmostEqual(pump.differential_pressure, 30)
    expected_head = 30 / (
        constants.GRAVITY * constants.WATER_DENSITY / constants.PASCALS_PER_BAR
    )
    self.assertAlmostEqual(pump._water_pump_differential_head, expected_head)

  def test_pressure_conversion(self):
    pump = dbo_pump.Pump(
        water_pump_differential_head=3,
        water_pump_efficiency=0.9,
    )
    self.assertAlmostEqual(
        pump._convert_pressure_to_differential_head(
            pump._convert_differential_head_to_pressure(3)
        ),
        3,
    )
    self.assertAlmostEqual(
        pump._convert_differential_head_to_pressure(
            pump._convert_pressure_to_differential_head(30)
        ),
        30,
    )

  def test_reset(self):
    pump = dbo_pump.Pump(
        water_pump_differential_head=3,
        water_pump_efficiency=0.9,
    )
    pump.run_command = dbo_pump.RunStatus.On
    pump._water_pump_differential_head = 4
    pump._water_pump_efficiency = 0.1
    pump.reset()
    self.assertEqual(pump.run_command, dbo_pump.RunStatus.Off)
    self.assertEqual(pump._water_pump_differential_head, 3)
    self.assertEqual(pump._water_pump_efficiency, 0.9)


if __name__ == '__main__':
  absltest.main()
