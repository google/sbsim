from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from smart_buildings.smart_control.simulator import air_source_heat_pump
from smart_buildings.smart_control.simulator import hot_water_heat_source
from smart_buildings.smart_control.utils import constants


class AirSourceHeatPumpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Ensure constants are defined for consistent testing
    self.water_density = 1000.0  # kg/m^3
    self.enter_context(
        mock.patch.object(constants, "WATER_DENSITY", new=self.water_density)
    )
    self.specific_heat = 4184.0  # J/(kg*K)

    self.reheat_setpoint_k = 313.15  # 40°C
    self.max_capacity_w = 180000.0
    self.nominal_cop = 3.2

    self.ashp = air_source_heat_pump.AirSourceHeatPump(
        reheat_water_setpoint=self.reheat_setpoint_k,
        device_id="test_ashp_01",
        max_heating_capacity_w=self.max_capacity_w,
        nominal_cop=self.nominal_cop,
        init_return_water_temperature_sensor=295.15,  # 22°C
    )

    # Mock the observation state to return our test setpoint
    self.ashp.get_observation = mock.MagicMock(
        return_value=self.reheat_setpoint_k
    )

  def test_initialization(self):
    self.assertEqual(self.ashp.device_id(), "test_ashp_01")
    self.assertEqual(self.ashp.run_command, hot_water_heat_source.RunStatus.On)
    self.assertEqual(self.ashp.return_water_temperature_sensor, 295.15)
    self.assertEqual(self.ashp._max_capacity_w, self.max_capacity_w)

  def test_reset(self):
    """Verifies reset restores the initial return temperature and run state."""
    self.ashp.return_water_temperature_sensor = 300.0
    self.ashp.run_command = hot_water_heat_source.RunStatus.Off

    self.ashp.reset()

    self.assertEqual(self.ashp.return_water_temperature_sensor, 295.15)
    self.assertEqual(self.ashp.run_command, hot_water_heat_source.RunStatus.On)

  def test_reheat_water_setpoint_setter(self):
    """Tests we can change the reheat setpoint."""
    new_setpoint = 310.0
    self.ashp.reheat_water_setpoint = new_setpoint
    self.assertEqual(self.ashp.reheat_water_setpoint, new_setpoint)

  @parameterized.parameters(
      (280.15, 3.2),  # 7°C Outside -> Nominal COP
      (273.15, 2.85),  # 0°C Outside -> Degraded COP (3.2 - 0.35)
      (253.15, 1.85),  # -20°C Outside -> Highly Degraded COP (3.2 - 1.35)
      (200.00, 1.0),  # Extreme Cold -> Clamped to Minimum COP of 1.0
      (330.00, 5.0),  # Extreme Heat -> Clamped to Maximum COP of 5.0
  )
  def test_calculate_dynamic_cop(self, outside_temp_k, expected_cop):
    """Tests linear degradation and clamping of the COP curve."""
    calculated_cop = self.ashp._calculate_dynamic_cop(outside_temp_k)
    self.assertAlmostEqual(calculated_cop, expected_cop, places=2)

  @parameterized.parameters(
      # Status, Flow, Return Temp, Outside Temp, Expected Electrical Watts
      (
          hot_water_heat_source.RunStatus.Off,
          0.05,
          295.15,
          280.15,
          0.0,
      ),  # Off -> 0W
      (
          hot_water_heat_source.RunStatus.On,
          0.0,
          295.15,
          280.15,
          0.0,
      ),  # No flow -> 0W
      (
          hot_water_heat_source.RunStatus.On,
          0.05,
          315.00,
          280.15,
          0.0,
      ),  # Return > Setpoint (313.15K) -> 0W
  )
  def test_compute_thermal_energy_rate_zero_conditions(
      self, run_status, flow_rate, return_temp, outside_temp, expected_w
  ):
    """Tests conditions where the heat pump should consume zero power."""
    self.ashp.run_command = run_status
    power_w = self.ashp.compute_thermal_energy_rate(
        return_temp, outside_temp, flow_rate
    )
    self.assertEqual(power_w, expected_w)

  def test_compute_thermal_energy_rate_normal_operation(self):
    """Tests standard heating calculations well within max capacity bounds."""
    flow_rate = 0.002  # m^3/s (approx 31 GPM)
    return_temp = 308.15  # 35°C (delta T of 5°C to the 40°C setpoint)
    outside_temp = 280.15  # 7°C (COP = 3.2)

    # Expected Thermal Q = mass_flow * density * Cp * delta_T
    # Q = 0.002 * 1000 * 4184 * 5.0 = 41,840 W
    # Expected Electrical = Q / COP = 41840 / 3.2 = 13,063 W
    expected_electrical_power = 13062.5

    power_w = self.ashp.compute_thermal_energy_rate(
        return_temp, outside_temp, flow_rate
    )
    self.assertAlmostEqual(power_w, expected_electrical_power, places=1)

  def test_compute_thermal_energy_rate_capacity_clamped(self):
    """Tests that electrical power is bounded when demand above max capacity."""
    flow_rate = 0.05  # Huge flow
    return_temp = 280.15  # Very cold return (large delta T)
    outside_temp = 280.15  # 7°C (COP = 3.2)

    # Unbounded demand would be massive. Should clamp to max capacity.
    # Max capacity = 180,000 W
    # Expected Electrical = 180,000 / 3.2 = 56,250 W
    expected_electrical_power = 180000.0 / 3.2

    power_w = self.ashp.compute_thermal_energy_rate(
        return_temp, outside_temp, flow_rate
    )
    self.assertAlmostEqual(power_w, expected_electrical_power, places=1)

  @parameterized.parameters(
      (
          320.0,
          290.0,
          5.6 * 4.0 * 30.0,
      ),  # Water warmer than outside -> Heat loss
      (290.0, 320.0, 0.0),  # Water cooler than outside -> Clamped to 0
  )
  def test_compute_thermal_dissipation_rate(
      self, water_temp, outside_temp, expected_loss
  ):
    """Tests the simplified convection heat loss calculation."""
    loss_w = self.ashp.compute_thermal_dissipation_rate(
        water_temp, outside_temp
    )
    self.assertAlmostEqual(loss_w, expected_loss, places=1)


if __name__ == "__main__":
  absltest.main()
