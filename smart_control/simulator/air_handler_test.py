from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_buildings.smart_control.simulator import air_handler
from smart_buildings.smart_control.simulator import smart_device
from smart_buildings.smart_control.simulator import weather_controller
from smart_buildings.smart_control.simulator import constants


class AirHandlerTest(parameterized.TestCase):
  recirculation = 0.3
  supply_air_temperature_setpoint = 288
  fan_static_pressure = 20000.0
  fan_efficiency = 0.8
  max_air_flow_rate = 10

  def test_init(self):
    handler = air_handler.AirHandler(
        recirculation=self.recirculation,
        supply_air_temperature_setpoint=self.supply_air_temperature_setpoint,
        fan_static_pressure=self.fan_static_pressure,
        fan_efficiency=self.fan_efficiency,
        max_air_flow_rate=self.max_air_flow_rate,
        device_id='device_id',
    )

    self.assertEqual(handler.recirculation, self.recirculation)

    self.assertEqual(
        handler.supply_air_temperature_setpoint,
        self.supply_air_temperature_setpoint,
    )
    self.assertEqual(
        handler.supply_air_static_pressure_setpoint, self.fan_static_pressure
    )
    self.assertEqual(handler.fan_efficiency, self.fan_efficiency)
    self.assertEqual(handler.air_flow_rate, 0)
    self.assertEqual(handler.cooling_request_count, 0)
    self.assertEqual(handler.max_air_flow_rate, self.max_air_flow_rate)
    self.assertEqual(handler._device_id, 'device_id')

  def test_init_default(self):
    handler = air_handler.AirHandler(
        recirculation=self.recirculation,
        supply_air_temperature_setpoint=self.supply_air_temperature_setpoint,
        fan_static_pressure=self.fan_static_pressure,
        fan_efficiency=self.fan_efficiency,
    )
    self.assertEqual(handler.max_air_flow_rate, 8.67)
    self.assertIsNotNone(handler._device_id)

  def test_setters(self):
    handler = air_handler.AirHandler(
        recirculation=self.recirculation,
        supply_air_temperature_setpoint=self.supply_air_temperature_setpoint,
        fan_static_pressure=self.fan_static_pressure,
        fan_efficiency=self.fan_efficiency,
    )
    handler.recirculation = self.recirculation + 0.2

    handler.supply_air_temperature_setpoint = (
        self.supply_air_temperature_setpoint + 10
    )
    handler.supply_air_static_pressure_setpoint = (
        self.fan_static_pressure + 1000
    )
    handler.fan_efficiency = self.fan_efficiency + 0.1
    handler.air_flow_rate = 30

    self.assertEqual(handler.recirculation, self.recirculation + 0.2)

    self.assertEqual(
        handler.supply_air_temperature_setpoint,
        self.supply_air_temperature_setpoint + 10,
    )
    self.assertEqual(
        handler.supply_air_static_pressure_setpoint,
        self.fan_static_pressure + 1000,
    )
    self.assertEqual(handler.fan_efficiency, self.fan_efficiency + 0.1)
    self.assertEqual(handler.air_flow_rate, 30)

  @parameterized.parameters(
      (0.3, 280, 240, 0.3 * 280 + 0.7 * 240),
      (0.6, 244, 270, 0.6 * 244 + 0.4 * 270),
      (0.1, 210, 316, 0.1 * 210 + 0.9 * 316),
      (0.4, 250, 316, 0.4 * 250 + 0.6 * 316),
      (0.4, 286, 266, 0.4 * 286 + 0.6 * 266),
      (0.12, 198, 290, 0.12 * 198 + 0.88 * 290),
  )
  def test_get_mixed_air_temp(
      self, recirculation, recirculation_temp, ambient_temp, expected
  ):
    """Calculates the mixed air temperature.

    This function is calculated by muliplying the recirculation_temp by
    the recirculation factor, and the ambient_temp
    by 1 minus the recirculation factor, and adding the two.

    Args:
      recirculation: the recirculation coefficient
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of ambient/outside air.
      expected: the expected value
    """
    handler = air_handler.AirHandler(
        recirculation=recirculation,
        supply_air_temperature_setpoint=self.supply_air_temperature_setpoint,
        fan_static_pressure=self.fan_static_pressure,
        fan_efficiency=self.fan_efficiency,
    )
    self.assertEqual(
        handler.get_mixed_air_temp(recirculation_temp, ambient_temp), expected
    )

  @parameterized.named_parameters(
      ('below setpoint window case 1', 0.3, 280, 240, 252),
      ('below setpount window case 2', 0.6, 244, 270, 254.4),
      ('above setpoint window case 1', 0.1, 210, 316, 288),
      ('above setpoint window case 2', 0.4, 250, 316, 288),
  )
  def test_get_supply_air_temp(
      self, recirculation, recirculation_temp, ambient_temp, expected
  ):
    """Calculates the supply air temperature.

    This function returns the mixed_air_temp if it is within the setpoints,
    otherwise returns the closest setpoint.

    Args:
      recirculation: the recirculation coefficient
      recirculation_temp: Temperature in K of recirculated air.
      ambient_temp: Temperature in K of ambient/outside air.
      expected: the expected value
    """
    handler = air_handler.AirHandler(
        recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(
        handler.get_supply_air_temp(recirculation_temp, ambient_temp), expected
    )

  @parameterized.parameters(
      (0.3, 10),
      (0.8, 45),
      (0.7, 1000),
      (0.1, 5000),
      (0.4, 2545),
  )
  def test_ambient_flow_rate(self, recirculation, air_flow_rate):
    handler = air_handler.AirHandler(
        recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = air_flow_rate

    self.assertEqual(
        (1 - recirculation) * air_flow_rate, handler.ambient_flow_rate
    )

  @parameterized.parameters(
      (0.3, 10),
      (0.8, 45),
      (0.7, 1000),
      (0.1, 5000),
      (0.4, 2545),
  )
  def test_recirculation_flow_rate(self, recirculation, air_flow_rate):
    handler = air_handler.AirHandler(
        recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = air_flow_rate

    self.assertEqual(
        recirculation * air_flow_rate, handler.recirculation_flow_rate
    )

  def test_reset_demand(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.add_demand(5)
    self.assertEqual(handler.air_flow_rate, 5)
    handler.reset_demand()
    self.assertEqual(handler.air_flow_rate, 0)
    self.assertEqual(handler.cooling_request_count, 0)

  def test_add_demand(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
        10,
    )
    self.assertEqual(handler.air_flow_rate, 0)
    handler.add_demand(10)
    self.assertEqual(handler.air_flow_rate, 10)
    self.assertEqual(handler.cooling_request_count, 1)

  def test_add_demand_above_max(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(handler.air_flow_rate, 0)
    handler.add_demand(10)
    self.assertEqual(handler.air_flow_rate, 8.67)
    self.assertEqual(handler.cooling_request_count, 1)

  def test_add_demand_raises_value_error(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )

    with self.assertRaises(ValueError):
      handler.add_demand(0)

  def test_reset(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.recirculation += 1.0
    handler.supply_air_temperature_setpoint += 1.0
    handler.supply_air_static_pressure_setpoint += 0.1
    handler.fan_efficiency = 0.1

    handler.reset()

    self.assertEqual(handler.recirculation, self.recirculation)

    self.assertEqual(
        handler.supply_air_temperature_setpoint,
        self.supply_air_temperature_setpoint,
    )
    self.assertEqual(
        handler.supply_air_static_pressure_setpoint, self.fan_static_pressure
    )
    self.assertEqual(handler.fan_efficiency, self.fan_efficiency)

  @parameterized.parameters(
      (100, 250, 210),
      (0.5, 280, 320),
      (1000, 155, 134),
      (2, 246, 290),
      (900, 50, 270),
  )
  def test_compute_thermal_energy_rate(
      self, air_flow_rate, ambient_temp, recirculation_temp
  ):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = air_flow_rate

    # This test is based on mixed_air_temp and supply_air_temp
    # and relies on their correctness as verified in the above tests.

    mixed_air_temp = handler.get_mixed_air_temp(
        recirculation_temp, ambient_temp
    )
    supply_air_temp = handler.get_supply_air_temp(
        recirculation_temp, ambient_temp
    )
    expected = (
        handler.air_flow_rate
        * constants.AIR_DENSITY
        * constants.AIR_HEAT_CAPACITY
        * (supply_air_temp - mixed_air_temp)
    )
    self.assertEqual(
        handler.compute_thermal_energy_rate(recirculation_temp, ambient_temp),
        expected,
    )

  @parameterized.parameters(
      (100, 2000.0, 0.8),
      (205, 2300.0, 0.3),
      (1, 4000.0, 0.4),
  )
  def test_compute_fan_power(
      self, flow_rate, fan_static_pressure, fan_efficiency
  ):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(
        handler.compute_fan_power(
            flow_rate, fan_static_pressure, fan_efficiency
        ),
        flow_rate * fan_static_pressure / fan_efficiency,
    )

  def test_invalid_outside_air_temperature_sensor(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    with self.assertRaises(RuntimeError):
      _ = handler.outside_air_temperature_sensor

  @parameterized.parameters(
      (pd.Timestamp('2021-09-01 00:00'), 0.0),
      (pd.Timestamp('2021-09-01 12:00'), 10.0),
      (pd.Timestamp('2021-09-01 06:00'), 5.0),
  )
  def test_valid_outside_air_handler_temperature_sensor(
      self, timestamp, expected_temp
  ):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
        sim_weather_controller=weather_controller.WeatherController(0.0, 10.0),
    )
    handler._observation_timestamp = timestamp
    self.assertAlmostEqual(
        handler.outside_air_temperature_sensor, expected_temp
    )

  def test_compute_intake_fan_energy_rate(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = 5

    # This test is based on compute_fan_power,
    # and relies on its correctness as verified in the above test.

    self.assertEqual(
        handler.compute_intake_fan_energy_rate(),
        handler.compute_fan_power(
            handler.air_flow_rate,
            handler.supply_air_static_pressure_setpoint,
            handler.fan_efficiency,
        ),
    )

  def test_compute_exhaust_fan_energy_rate(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = 5

    # This should return the same value as compute_intake_fan_energy_rate,
    # except airflow rate is multiplied by (1 - recirculation).
    # This test is based on compute_fan_power,
    # and relies on its correctness as verified in an above test.

    self.assertEqual(
        handler.compute_exhaust_fan_energy_rate(),
        handler.compute_fan_power(
            handler.air_flow_rate * (1 - self.recirculation),
            handler.supply_air_static_pressure_setpoint,
            handler.fan_efficiency,
        ),
    )

  def test_supply_fan_speed_percentage(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
        10,
    )
    self.assertEqual(handler.supply_fan_speed_percentage, 0)
    handler.add_demand(5)
    self.assertEqual(handler.supply_fan_speed_percentage, 0.5)

  def test_observable_field_names(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )

    self.assertSameElements(
        handler.observable_field_names(),
        [
            'supply_air_static_pressure_setpoint',
            'supply_air_static_pressure_sensor',
            'supply_air_flowrate_sensor',
            'supply_air_flowrate_setpoint',
            'supply_air_temperature_setpoint',
            'supply_air_temperature_sensor',
            'supply_fan_run_command',
            'exhaust_fan_run_command',
            'discharge_fan_speed_percentage_command',
            'supply_fan_speed_percentage_command',
            'outside_air_flowrate_sensor',
            'cooling_request_count',
            'supervisor_run_command',
        ],
    )

  @parameterized.parameters(
      (
          'supply_air_static_pressure_setpoint',
          'supply_air_static_pressure_setpoint',
      ),
      ('supply_air_temperature_setpoint', 'supply_air_temperature_setpoint'),
      ('supervisor_run_command', 'run_command'),
  )
  def test_observations(self, observation_name, attribute_name):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    observed_value = handler.get_observation(
        observation_name, pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, getattr(handler, attribute_name))

  def test_observe_cooling_request_count(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )

    observed_value = handler.get_observation(
        'cooling_request_count', pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, handler.cooling_request_count)
    handler.add_demand(5)
    observed_value = handler.get_observation(
        'cooling_request_count', pd.Timestamp('2021-09-01 10:15:00')
    )
    self.assertEqual(observed_value, handler.cooling_request_count)

  def test_action_field_names(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )
    self.assertSameElements(
        handler.action_field_names(),
        [
            'supply_air_temperature_setpoint',
            'supervisor_run_command',
            'supply_air_static_pressure_setpoint',
        ],
    )

  @parameterized.parameters(
      (
          290.0,
          'supply_air_temperature_setpoint',
          'supply_air_temperature_setpoint',
      ),
      (
          120.0,
          'supply_air_static_pressure_setpoint',
          'supply_air_static_pressure_setpoint',
      ),
      (
          smart_device.RunStatus.OFF,
          'supervisor_run_command',
          'run_command',
      ),
  )
  def test_actions(self, new_value, action_name, attribute_name):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
    )

    handler.set_action(
        action_name, new_value, pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(getattr(handler, attribute_name), new_value)

  @parameterized.parameters(
      (smart_device.RunStatus.OFF),
      (smart_device.RunStatus.ON),
  )
  def test_run_command(self, run_command):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.supply_air_temperature_setpoint,
        self.fan_static_pressure,
        self.fan_efficiency,
        run_command=run_command,
    )
    handler.add_demand(5)
    self.assertEqual(handler.run_command, run_command)
    if run_command == smart_device.RunStatus.OFF:
      self.assertEqual(handler.air_flow_rate, 0.0)
      self.assertEqual(handler.supply_air_static_pressure_setpoint, 0.0)
    else:
      self.assertEqual(handler.air_flow_rate, 5.0)
      self.assertEqual(
          handler.supply_air_static_pressure_setpoint, self.fan_static_pressure
      )

    handler.set_action(
        'supervisor_run_command',
        smart_device.RunStatus.OFF,
        pd.Timestamp('2021-09-01 10:10:00'),
    )
    self.assertEqual(handler.run_command, smart_device.RunStatus.OFF)
    self.assertEqual(handler.air_flow_rate, 0.0)
    self.assertEqual(handler.supply_air_static_pressure_setpoint, 0.0)

    handler.set_action(
        'supervisor_run_command',
        smart_device.RunStatus.ON,
        pd.Timestamp('2021-09-01 10:10:00'),
    )
    self.assertEqual(handler.run_command, smart_device.RunStatus.ON)
    self.assertEqual(handler.air_flow_rate, 5.0)
    self.assertEqual(
        handler.supply_air_static_pressure_setpoint, self.fan_static_pressure
    )

    handler.reset()
    self.assertEqual(handler.run_command, run_command)

  def test_air_flow_units(self):
    # A flow of 1.0 m^3/s through 1.0 Pascal of pressure at 100% efficiency
    # should equal exactly 1.0 Watt of power.
    # This verifies that air_flow_rate is in m^3/s and pressure is in Pascals.
    handler = air_handler.AirHandler(
        recirculation=0.5,
        supply_air_temperature_setpoint=300,
        fan_static_pressure=1.0,  # 1 Pascal
        fan_efficiency=1.0,  # 100% efficiency
    )
    power = handler.compute_fan_power(
        flow_rate=1.0,
        fan_static_pressure=1.0,
        fan_efficiency=1.0,
    )
    self.assertAlmostEqual(power, 1.0)


class AirHandlerSystemTest(parameterized.TestCase):
  """System-level tests for AirHandlerSystem and multiple AirHandlers."""

  def setUp(self):
    super().setUp()
    self.ahu1 = self._create_air_handler('ahu1')
    self.ahu2 = self._create_air_handler('ahu2')
    self.ahu_map = {self.ahu1: ['zone1'], self.ahu2: ['zone2']}
    self.system = air_handler.AirHandlerSystem(self.ahu_map)

  def _create_air_handler(
      self,
      device_id: str,
      recirculation: float = 0.3,
      supply_air_temperature_setpoint: float = 279,
      fan_static_pressure: float = 20000.0,
      fan_efficiency: float = 0.8,
      max_air_flow_rate: float = 10,
  ) -> air_handler.AirHandler:
    """Helper to create an AirHandler with standard test parameters."""
    return air_handler.AirHandler(
        recirculation=recirculation,
        supply_air_temperature_setpoint=supply_air_temperature_setpoint,
        fan_static_pressure=fan_static_pressure,
        fan_efficiency=fan_efficiency,
        max_air_flow_rate=max_air_flow_rate,
        device_id=device_id,
    )

  def test_ahu_zones_map(self):
    """Tests that ahu_zones_map returns the correct mapping."""
    self.assertEqual(self.system.ahu_zones_map, self.ahu_map)

  @parameterized.named_parameters(
      dict(
          testcase_name='both_below_setpoint',
          recirculation_temps={'ahu1': 280, 'ahu2': 244},
          ambient_temp=240,
          # In this case, mixed air temp is below the supply setpoint (279K).
          # ahu1: 0.3 * 280 + 0.7 * 240 = 252.0. Since 252 < 279,
          # supply is 252.
          # ahu2: 0.3 * 244 + 0.7 * 240 = 241.2. Since 241.2 < 279,
          # supply is 241.2.
          expected_temps={'ahu1': 252.0, 'ahu2': 241.2},
      ),
      dict(
          testcase_name='both_above_setpoint',
          recirculation_temps={'ahu1': 300, 'ahu2': 310},
          ambient_temp=320,
          # Mixed air temp is above the supply setpoint (279K), so it cools
          # to 279.
          # ahu1: 0.3 * 300 + 0.7 * 320 = 314.0. Since 314 > 279, supply is 279.
          # ahu2: 0.3 * 310 + 0.7 * 320 = 317.0. Since 317 > 279, supply is 279.
          expected_temps={'ahu1': 279.0, 'ahu2': 279.0},
      ),
      dict(
          testcase_name='one_above_one_below',
          recirculation_temps={'ahu1': 380, 'ahu2': 244},
          ambient_temp=240,
          # ahu1: 0.3 * 380 + 0.7 * 240 = 114 + 168 = 282. Since 282 > 279,
          # supply is 279.
          # ahu2: 0.3 * 244 + 0.7 * 240 = 241.2. Since 241.2 < 279,
          # supply is 241.2.
          expected_temps={'ahu1': 279.0, 'ahu2': 241.2},
      ),
      dict(
          testcase_name='at_setpoint_threshold',
          recirculation_temps={'ahu1': 279, 'ahu2': 300},
          ambient_temp=279,
          # ahu1: 0.3 * 279 + 0.7 * 279 = 279.0. Supply is 279.
          # ahu2: 0.3 * 300 + 0.7 * 279 = 90 + 195.3 = 285.3. Supply is 279.
          expected_temps={'ahu1': 279.0, 'ahu2': 279.0},
      ),
  )
  def test_get_supply_air_temp(
      self, recirculation_temps, ambient_temp, expected_temps
  ):
    """Verifies supply air temperature calculation for a system of AHUs.

    The expected supply temperature is derived from:
    1. Mixed Air Temp = (recirculation * recirculation_temp) +
                        ((1 - recirculation) * ambient_temp)
    2. Supply Air Setpoint
    3. If Mixed Air Temp > Setpoint: Supply Temp = Setpoint
       Else: Supply Temp = Mixed Air Temp

    Args:
      recirculation_temps: Mapping of AHU device IDs to recirculation temps.
      ambient_temp: The ambient air temperature.
      expected_temps: Mapping of AHU device IDs to expected supply temps.
    """
    temps = self.system.get_supply_air_temp(recirculation_temps, ambient_temp)
    self.assertLen(temps, 2)
    self.assertAlmostEqual(temps['ahu1'], expected_temps['ahu1'], places=4)
    self.assertAlmostEqual(temps['ahu2'], expected_temps['ahu2'], places=4)

  def test_compute_thermal_energy_rate(self):
    """Verifies combined thermal energy rate for a system of AHUs.

    Energy Rate (W) = air_flow_rate (m^3/s) * density (kg/m^3) *
                      heat_capacity (J/kg*K) * (T_supply - T_mixed)
    """
    # Set air flow rates for each AHU
    self.ahu1.air_flow_rate = 1.0
    self.ahu2.air_flow_rate = 2.0

    recirculation_temps = {'ahu1': 290.0, 'ahu2': 260.0}
    ambient_temp = 300.0

    # Expected energy rate calculations:
    # Setpoint = (270 + 288) / 2 = 279

    # AHU1:
    # T_mixed = 0.3 * 290 + 0.7 * 300 = 87 + 210 = 297
    # T_supply = 279
    # Delta_T = 279 - 297 = -18
    # density = 1.2
    # energy_ahu1 = 1.0 * 1.2 * 1005 * -18 = -21708
    expected_energy_ahu1 = (
        1.0
        * constants.AIR_DENSITY
        * constants.AIR_HEAT_CAPACITY
        * (279.0 - 297.0)
    )

    # AHU2:
    # T_mixed = 0.3 * 260 + 0.7 * 300 = 78 + 210 = 288
    # T_supply = 279
    # Delta_T = 279 - 288 = -9
    # energy_ahu2 = 2.0 * 1.2 * 1005 * -9 = -21708
    expected_energy_ahu2 = (
        2.0
        * constants.AIR_DENSITY
        * constants.AIR_HEAT_CAPACITY
        * (279.0 - 288.0)
    )

    expected_total_energy = expected_energy_ahu1 + expected_energy_ahu2

    actual_total_energy = self.system.compute_thermal_energy_rate(
        recirculation_temps, ambient_temp
    )

    self.assertAlmostEqual(actual_total_energy, expected_total_energy, places=4)


if __name__ == '__main__':
  absltest.main()
