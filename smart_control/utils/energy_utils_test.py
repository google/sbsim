from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from smart_buildings.smart_control.utils import energy_utils


class EnergyUtilsTest(parameterized.TestCase):

  def test_get_water_vapor_partial_pressure(self):
    temps = [t + 273.0 for t in [-40, -35, 0, 5, 10, 15, 20, 30, 40, 50, 60]]
    pressures = energy_utils.get_water_vapor_partial_pressure(temps)
    expected_pressures = [
        1.28500e-01,
        2.54350e-01,
        6.11150e00,
        9.19525e00,
        1.22790e01,
        1.78320e01,
        2.33850e01,
        4.24520e01,
        7.38130e01,
        1.23448e02,
        1.99330e02,
    ]

    for i in range(len(expected_pressures)):
      self.assertAlmostEqual(expected_pressures[i], pressures[i], 5)

  def test_get_humidity_ratio(self):
    expected = 0.00868
    actual = energy_utils.get_humidity_ratio([293], [0.6], [1.02])
    self.assertAlmostEqual(expected, actual[0], 4)

  def test_get_air_conditioning_energy_rate(self):
    power = energy_utils.get_air_conditioning_energy_rate(
        air_flow_rates=[0.170],
        outside_temps=[15 + 273.0],
        outside_relative_humidities=[0.75],
        supply_temps=[120 + 273.0],
        ambient_pressures=[1.025],
    )
    self.assertAlmostEqual(18230.6705, power[0], 4)

  @parameterized.named_parameters(
      ('brake_hp', None, 8.0, 100.0, 0.8, 0.85, 3, 17.904),
      ('design_hp', 10.0, None, None, None, None, 1, 6.3410),
      ('design_hp_motor_factor', 10.0, None, None, None, 0.6, 1, 4.4760),
  )
  def test_get_fan_power_valid(
      self,
      design_hp,
      brake_hp,
      fan_speed_percentage,
      supply_static_pressure,
      motor_factor,
      num_fans,
      expected,
  ):
    power = energy_utils.get_fan_power(
        design_hp=design_hp,
        brake_hp=brake_hp,
        fan_speed_percentage=fan_speed_percentage,
        supply_static_pressure=supply_static_pressure,
        motor_factor=motor_factor,
        num_fans=num_fans,
    )
    self.assertAlmostEqual(expected, power, places=2)

  def test_get_fan_power_invalid(self):
    with self.assertRaises(ValueError):
      _ = energy_utils.get_fan_power(
          fan_speed_percentage=20,
          supply_static_pressure=0.9,
          motor_factor=0.85,
          num_fans=3,
      )

  def test_get_air_volumetric_flowrate(self):
    flowrate = energy_utils.get_air_volumetric_flowrate(
        average_fan_speed_percentage=80.0, design_cfm=250.0
    )
    self.assertAlmostEqual(200.0, flowrate, places=4)

  @parameterized.named_parameters(
      ('fully_defined', 82.0, 68.0, 150.0, 80.0, 14.0, 2.0, 0.18514),
      ('no_air_flow', 82.0, 68.0, 0.0, 80.0, 9.0, 2.0, 0.0),
      ('no_fan_percentage', 82.0, 68.0, 150.0, 1.0, 12.0, 2.0, 0.0),
  )
  def test_get_compressor_power_thermal(
      self,
      mixed_air_temp,
      supply_air_temp,
      volumetric_flow_rate,
      fan_speed_percentage,
      eer,
      fan_heat_temp,
      expected,
  ):
    power = energy_utils.get_compressor_power_thermal(
        mixed_air_temp=mixed_air_temp,
        supply_air_temp=supply_air_temp,
        volumetric_flow_rate=volumetric_flow_rate,
        fan_speed_percentage=fan_speed_percentage,
        fan_heat_temp=fan_heat_temp,
        eer=eer,
    )
    self.assertAlmostEqual(expected, power, places=5)

  @parameterized.named_parameters(
      ('cooling_percentage', 12.0, 80.0, None, None, 13.0, 8.86153),
      ('stages', 12.0, None, 8, 10, 13.0, 8.86153),
      ('no_eer', 12.0, None, 8, 10, None, 9.600),
  )
  def test_get_compressor_power_utilization_valid(
      self,
      design_capacity,
      cooling_percentage,
      count_stages_on,
      total_stages,
      eer,
      expected,
  ):
    power = energy_utils.get_compressor_power_utilization(
        design_capacity=design_capacity,
        cooling_percentage=cooling_percentage,
        count_stages_on=count_stages_on,
        total_stages=total_stages,
        eer=eer,
    )
    self.assertAlmostEqual(expected, power, places=4)

  @parameterized.named_parameters(
      ('no_cooling_percentage', None, None, None),
      ('neg_cooling_percentage', -5.0, None, None),
      ('neg_stages', None, -8, 10),
      ('bigger_stages', None, 12, 10),
      ('zero_stages', None, 0, 0),
  )
  def test_get_compressor_power_utilization_invalid(
      self, cooling_percentage, count_stages_on, total_stages
  ):
    with self.assertRaises(ValueError):
      _ = energy_utils.get_compressor_power_utilization(
          design_capacity=12.0,
          cooling_percentage=cooling_percentage,
          count_stages_on=count_stages_on,
          total_stages=total_stages,
      )

  @parameterized.named_parameters(
      ('brake_horse_power', 0.8, 95.0, 6.0, None, 0.85, 3, 9.44953271),
      ('design_horse_power', 0.8, 95.0, None, 10.0, 0.85, 3, 13.386838),
  )
  def test_get_water_pump_power_valid(
      self,
      pump_duty_cycle,
      pump_speed_percantage,
      brake_horse_power,
      design_horse_power,
      motor_factor,
      num_pumps,
      expected,
  ):
    p1 = energy_utils.get_water_pump_power(
        pump_duty_cycle=pump_duty_cycle,
        pump_speed_percentage=pump_speed_percantage,
        brake_horse_power=brake_horse_power,
        design_motor_horse_power=design_horse_power,
        motor_factor=motor_factor,
        num_pumps=num_pumps,
    )
    self.assertAlmostEqual(expected, p1, places=4)

  def test_get_water_pump_power_invalid(self):
    with self.assertRaises(ValueError):
      _ = energy_utils.get_water_pump_power(
          pump_duty_cycle=0.8,
          pump_speed_percentage=95.0,
          motor_factor=0.6,
          num_pumps=3,
      )

  @parameterized.named_parameters(
      ('two_pumps', 3.0, 75.0, 2, 4.5), ('one_pump', 8.0, 30.0, 1, 2.4)
  )
  def test_get_water_volumetric_flow_rate(
      self, design_flow_rate, pump_speed_percentage, num_pumps_on, expected
  ):
    v_dot = energy_utils.get_water_volumetric_flow_rate(
        design_flow_rate=design_flow_rate,
        pump_speed_percentage=pump_speed_percentage,
        num_pumps_on=num_pumps_on,
    )
    self.assertAlmostEqual(expected, v_dot, places=4)

  def test_get_water_heating_energy_rate_negative(self):
    power = energy_utils.get_water_heating_energy_rate(
        volumetric_flow_rate=8.0,
        supply_water_temperature=99.0,
        return_water_temperature=100.0,
    )
    self.assertAlmostEqual(0.0, power, places=4)

  def test_get_water_heating_energy_rate(self):
    power = energy_utils.get_water_heating_energy_rate(
        volumetric_flow_rate=8.0,
        supply_water_temperature=131,
        return_water_temperature=100.0,
    )
    self.assertAlmostEqual(124000.0, power, places=4)

  def test_get_water_heating_energy_rate_primary(self):
    power = energy_utils.get_water_heating_energy_rate_primary(
        design_boiler_flow_rate=6.0,
        boiler_outlet_temperature=123.0,
        return_water_temperature=98.0,
        num_active_boilers=2,
    )
    self.assertAlmostEqual(150000.0, power, places=4)

  def test_get_water_heating_energy_rate_primary_secondary_valid(self):
    power = energy_utils.get_water_heating_energy_rate_primary_secondary(
        design_primary_boiler_flow_rate=8.0,
        design_secondary_boiler_flow_rate=10.0,
        boiler_outlet_temperature=131.0,
        return_water_temperature=90.0,
        num_active_boilers=2,
        num_active_secondary_pumps=2,
        avg_secondary_pump_speed_percentage=35,
    )
    self.assertAlmostEqual(143500.0, power, places=4)

  @parameterized.named_parameters(
      ('invalid_primary', -8.0, 10.0), ('invalid_secondary', 8.0, -10.0)
  )
  def test_get_water_heating_energy_rate_primary_secondary_invalid(
      self, design_primary_flow_rate, design_secondary_flow_rate
  ):
    with self.assertRaises(ValueError):
      _ = energy_utils.get_water_heating_energy_rate_primary_secondary(
          design_primary_boiler_flow_rate=design_primary_flow_rate,
          design_secondary_boiler_flow_rate=design_secondary_flow_rate,
          boiler_outlet_temperature=131.0,
          return_water_temperature=90.0,
          num_active_boilers=2,
          num_active_secondary_pumps=2,
          avg_secondary_pump_speed_percentage=35,
      )


class ASHPSystemTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.estimator = energy_utils.ASHPSystemEstimator(hp_cop=3.4)
    self.p_bhp = 5.0
    self.p_gpm = 100.0

  @parameterized.parameters(
      (100.0, 3728.5),  # 100% speed -> Full BHP in Watts
      (50.0, 466.06),  # 50% speed -> 1/8th power (Affinity Law)
      (0.0, 0.0),  # 0% speed -> 0 Watts
      (-10.0, 0.0),  # Negative input safety
  )
  def test_pump_power_scaling(self, speed, expected_watts):
    calc = energy_utils.calculate_pump_power(self.p_bhp, speed)
    self.assertAlmostEqual(calc, expected_watts, places=1)

  @parameterized.named_parameters([
      ('normal_heating', 130.0, 120.0, 100.0, 43098.7),
      ('stagnant_water', 130.0, 130.0, 100.0, 0.0),
      ('pumps_off', 130.0, 120.0, 0.0, 0.0),
  ])
  def test_hp_consumption_logic(self, hws, hwr, speed, expected_hp_w):
    # Testing HP electrical draw based on deltaT and Flow
    result = self.estimator.estimate_interval_power(
        hws, hwr, speed, 0, self.p_bhp, self.p_bhp, self.p_gpm, self.p_gpm
    )
    self.assertAlmostEqual(result.hp_watts, expected_hp_w, places=1)

  def test_numpy_array_support(self):
    """Verify the library handles time-series arrays correctly."""
    hws_series = np.array([130.0, 130.0, 130.0])
    hwr_series = np.array([120.0, 125.0, 130.0])  # Decreasing deltaT
    speeds = np.array([100.0, 100.0, 100.0])

    results = self.estimator.estimate_interval_power(
        hws_series,
        hwr_series,
        speeds,
        0,
        self.p_bhp,
        self.p_bhp,
        self.p_gpm,
        self.p_gpm,
    )

    # Check that the output is also a numpy array of the same length
    self.assertIsInstance(results.total_watts, np.ndarray)
    self.assertLen(results.total_watts, 3)
    # Verify the third interval (deltaT=0) is just pump power (~3728W)
    self.assertAlmostEqual(results.total_watts[2], 3728.5, places=1)


if __name__ == '__main__':
  absltest.main()
