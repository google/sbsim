"""Tests for air_handler.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_control.simulator import air_handler
from smart_control.simulator import weather_controller
from smart_control.utils import constants


class AirHandlerTest(parameterized.TestCase):
  recirculation = 0.3
  heating_air_temp_setpoint = 270
  cooling_air_temp_setpoint = 288
  fan_differential_pressure = 20000.0
  fan_efficiency = 0.8
  max_air_flow_rate = 10

  def test_init(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
        self.max_air_flow_rate,
        'device_id',
    )

    self.assertEqual(handler.recirculation, self.recirculation)
    self.assertEqual(
        handler.heating_air_temp_setpoint, self.heating_air_temp_setpoint
    )
    self.assertEqual(
        handler.cooling_air_temp_setpoint, self.cooling_air_temp_setpoint
    )
    self.assertEqual(
        handler.fan_differential_pressure, self.fan_differential_pressure
    )
    self.assertEqual(handler.fan_efficiency, self.fan_efficiency)
    self.assertEqual(handler.air_flow_rate, 0)
    self.assertEqual(handler.cooling_request_count, 0)
    self.assertEqual(handler.max_air_flow_rate, self.max_air_flow_rate)
    self.assertEqual(handler._device_id, 'device_id')

  def test_init_default(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(handler.max_air_flow_rate, 8.67)
    self.assertIsNotNone(handler._device_id)

  def test_init_invalid_setpoints(self):
    with self.assertRaises(ValueError):
      air_handler.AirHandler(
          self.recirculation,
          self.cooling_air_temp_setpoint,
          self.heating_air_temp_setpoint,
          self.fan_differential_pressure,
          self.fan_efficiency,
      )

  def test_setters(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    handler.recirculation = self.recirculation + 0.2
    handler.heating_air_temp_setpoint = self.heating_air_temp_setpoint + 10
    handler.cooling_air_temp_setpoint = self.cooling_air_temp_setpoint + 10
    handler.fan_differential_pressure = self.fan_differential_pressure + 1000
    handler.fan_efficiency = self.fan_efficiency + 0.1
    handler.air_flow_rate = 30

    self.assertEqual(handler.recirculation, self.recirculation + 0.2)
    self.assertEqual(
        handler.heating_air_temp_setpoint, self.heating_air_temp_setpoint + 10
    )
    self.assertEqual(
        handler.cooling_air_temp_setpoint, self.cooling_air_temp_setpoint + 10
    )
    self.assertEqual(
        handler.fan_differential_pressure, self.fan_differential_pressure + 1000
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
        recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(
        handler.get_mixed_air_temp(recirculation_temp, ambient_temp), expected
    )

  @parameterized.named_parameters(
      ('below setpoint window case 1', 0.3, 280, 240, 270),
      ('below setpount window case 2', 0.6, 244, 270, 270),
      ('above setpoint window case 1', 0.1, 210, 316, 288),
      ('above setpoint window case 2', 0.4, 250, 316, 288),
      ('in setpoint window case 1', 0.4, 286, 266, 0.4 * 286 + 0.6 * 266),
      ('in setpoint window case 2', 0.12, 198, 290, 0.12 * 198 + 0.88 * 290),
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = air_flow_rate

    self.assertEqual(
        recirculation * air_flow_rate, handler.recirculation_flow_rate
    )

  def test_reset_demand(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(handler.air_flow_rate, 0)
    handler.add_demand(10)
    self.assertEqual(handler.air_flow_rate, 8.67)
    self.assertEqual(handler.cooling_request_count, 1)

  def test_add_demand_raises_value_error(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )

    with self.assertRaises(ValueError):
      handler.add_demand(0)

  def test_reset(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    handler.recirculation += 1.0
    handler.heating_air_temp_setpoint += 1.0
    handler.cooling_air_temp_setpoint += 1.0
    handler.fan_differential_pressure += 0.1
    handler.fan_efficiency = 0.1

    handler.reset()

    self.assertEqual(handler.recirculation, self.recirculation)
    self.assertEqual(
        handler.heating_air_temp_setpoint, self.heating_air_temp_setpoint
    )
    self.assertEqual(
        handler.cooling_air_temp_setpoint, self.cooling_air_temp_setpoint
    )
    self.assertEqual(
        handler.fan_differential_pressure, self.fan_differential_pressure
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
      self, flow_rate, fan_differential_pressure, fan_efficiency
  ):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    self.assertEqual(
        handler.compute_fan_power(
            flow_rate, fan_differential_pressure, fan_efficiency
        ),
        flow_rate * fan_differential_pressure / fan_efficiency,
    )

  def test_invalid_outside_air_temperature_sensor(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    handler.air_flow_rate = 5

    # This test is based on compute_fan_power,
    # and relies on its correctness as verified in the above test.

    self.assertEqual(
        handler.compute_intake_fan_energy_rate(),
        handler.compute_fan_power(
            handler.air_flow_rate,
            handler.fan_differential_pressure,
            handler.fan_efficiency,
        ),
    )

  def test_compute_exhaust_fan_energy_rate(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
            handler.fan_differential_pressure,
            handler.fan_efficiency,
        ),
    )

  def test_supply_fan_speed_percentage(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
        10,
    )
    self.assertEqual(handler.supply_fan_speed_percentage, 0)
    handler.add_demand(5)
    self.assertEqual(handler.supply_fan_speed_percentage, 0.5)

  def test_observable_field_names(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )

    self.assertSameElements(
        handler.observable_field_names(),
        [
            'differential_pressure_setpoint',
            'supply_air_flowrate_sensor',
            'supply_air_heating_temperature_setpoint',
            'supply_air_cooling_temperature_setpoint',
            'supply_fan_speed_percentage_command',
            'discharge_fan_speed_percentage_command',
            'outside_air_flowrate_sensor',
            'cooling_request_count',
        ],
    )

  @parameterized.parameters(
      ('differential_pressure_setpoint', 'fan_differential_pressure'),
      ('supply_air_heating_temperature_setpoint', 'heating_air_temp_setpoint'),
      ('supply_air_cooling_temperature_setpoint', 'cooling_air_temp_setpoint'),
      ('supply_fan_speed_percentage_command', 'supply_fan_speed_percentage'),
      ('discharge_fan_speed_percentage_command', 'supply_fan_speed_percentage'),
      ('outside_air_flowrate_sensor', 'ambient_flow_rate'),
      ('supply_air_flowrate_sensor', 'air_flow_rate'),
  )
  def test_observations(self, observation_name, attribute_name):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    observed_value = handler.get_observation(
        observation_name, pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, getattr(handler, attribute_name))

  def test_observe_cooling_request_count(self):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
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
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )
    self.assertSameElements(
        handler.action_field_names(),
        [
            'supply_air_heating_temperature_setpoint',
            'supply_air_cooling_temperature_setpoint',
        ],
    )

  @parameterized.parameters(
      (
          280.0,
          'supply_air_heating_temperature_setpoint',
          'heating_air_temp_setpoint',
      ),
      (
          280.0,
          'supply_air_cooling_temperature_setpoint',
          'cooling_air_temp_setpoint',
      ),
  )
  def test_actions(self, new_value, action_name, attribute_name):
    handler = air_handler.AirHandler(
        self.recirculation,
        self.heating_air_temp_setpoint,
        self.cooling_air_temp_setpoint,
        self.fan_differential_pressure,
        self.fan_efficiency,
    )

    handler.set_action(
        action_name, new_value, pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(getattr(handler, attribute_name), new_value)


if __name__ == '__main__':
  absltest.main()
