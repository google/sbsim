"""Tests for boiler.

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
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import boiler
from smart_control.utils import constants


class BoilerTest(parameterized.TestCase):

  def get_default_boiler(self):
    reheat_water_setpoint = 360
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    return boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
        heating_rate=0.0,
        cooling_rate=0.0,
        convection_coefficient=5.6,
        tank_length=2.0,
        tank_radius=0.5,
        water_capacity=1.5,
        insulation_conductivity=0.067,
        insulation_thickness=0.06,
    )

  def test_init(self):
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    self.assertEqual(b.reheat_water_setpoint, reheat_water_setpoint)
    self.assertEqual(
        b._water_pump_differential_head, water_pump_differential_head
    )
    self.assertEqual(b._water_pump_efficiency, water_pump_efficiency)
    self.assertEqual(b._total_flow_rate, 0)

  def test_reset(self):
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    b._reheat_water_setpoint += 1.0
    b._water_pump_differential_head = 4.0
    b._water_pump_efficiency = 0.1
    b._heating_request_count = 10
    b._return_water_temperature_sensor = 310.0
    b._total_flow_rate = 10.0

    b.reset()

    self.assertEqual(b.reheat_water_setpoint, reheat_water_setpoint)
    self.assertEqual(
        b._water_pump_differential_head, water_pump_differential_head
    )
    self.assertEqual(b._water_pump_efficiency, water_pump_efficiency)
    self.assertEqual(b._total_flow_rate, 0)

  def test_init_default_id(self):
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
    )
    self.assertIsNotNone(b._device_id)

  def test_setters(self):
    b = self.get_default_boiler()

    b.reheat_water_setpoint = 300
    self.assertEqual(b.reheat_water_setpoint, 300)

  def test_reset_demand(self):
    b = self.get_default_boiler()

    b.add_demand(5)
    b.reset_demand()

    self.assertEqual(b._total_flow_rate, 0)
    self.assertEqual(b.heating_request_count, 0)

  def test_add_demand(self):
    b = self.get_default_boiler()

    b.add_demand(5)
    self.assertEqual(b._total_flow_rate, 5)
    self.assertEqual(b.heating_request_count, 1)

  def test_add_demand_raises_value_error(self):
    b = self.get_default_boiler()

    with self.assertRaises(ValueError):
      b.add_demand(0.0)

  def test_compute_thermal_energy_rate_heating(self):
    b = self.get_default_boiler()
    setpoint_temperature = 370
    return_water_temp = 300
    outside_temp = 280
    q0 = b.compute_thermal_energy_rate(return_water_temp, outside_temp)
    b.reheat_water_setpoint = setpoint_temperature
    _ = b._adjust_temperature(
        setpoint_temperature, outside_temp, pd.Timedelta(5, unit='minute')
    )
    b._last_step_duration = pd.Timedelta(5, unit='minute')
    q1 = b.compute_thermal_energy_rate(return_water_temp, outside_temp)

    self.assertAlmostEqual(500.066862, q0, places=4)
    self.assertAlmostEqual(562.57521, q1, places=4)

  @parameterized.parameters(
      (340.0, 300.0, 280.0, 0.6, 100695.0501),
      (300.0, 300.0, 280.0, 0.6, 125.0167),
      (300.0, 300.0, 280.0, 0.01, 125.0167),
      (300.0, 300.0, 300.0, 0.01, 0.0),
  )
  def test_compute_thermal_energy_rate(
      self,
      water_temp_setpoint,
      return_water_temp,
      outside_temp,
      total_flow_rate,
      expected_energy_rate,
  ):
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        water_temp_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    b.add_demand(total_flow_rate)

    self.assertAlmostEqual(
        b.compute_thermal_energy_rate(return_water_temp, outside_temp),
        expected_energy_rate,
        places=3,
    )

  def test_compute_thermal_energy_rate_raises_assertion_error(self):
    return_water_temp = 200
    total_flow_rate = 0.5
    reheat_water_setpoint = 100
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    outside_temp = 293
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    b.add_demand(total_flow_rate)

    with self.assertRaises(AssertionError):
      _ = b.compute_thermal_energy_rate(return_water_temp, outside_temp)

  @parameterized.parameters(
      (330.0, 290.0, pd.Timedelta(60, unit='second'), 0.0, 0.0, 290.0),
      (330.0, 290.0, pd.Timedelta(60, unit='second'), 2.0, 0.0, 292.0),
      (300.0, 290.0, pd.Timedelta(600, unit='second'), 2.0, 0.0, 300.0),
      (320.0, 330.0, pd.Timedelta(60, unit='second'), 0.0, 0.5, 329.5),
      (320.0, 330.0, pd.Timedelta(600, unit='second'), 0.0, 2.0, 320.0),
  )
  def test_adjust_temperature(
      self,
      setpoint_temperature,
      actual_temperature,
      time_difference,
      heating_rate,
      cooling_rate,
      expected_temperature,
  ):
    reheat_water_setpoint = 310
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
    )

    self.assertAlmostEqual(
        expected_temperature,
        b._adjust_temperature(
            setpoint_temperature, actual_temperature, time_difference
        ),
    )

  @parameterized.parameters(
      (0.5, 3, 0.9),
      (0.2, 7, 0.5),
      (0.5, 8, 0.23),
      (0.5, 9, 0.7),
  )
  def test_compute_pump_power(
      self, total_flow_rate, water_pump_differential_head, water_pump_efficiency
  ):
    reheat_water_setpoint = 100
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
    )

    b.add_demand(total_flow_rate)

    expected = (
        total_flow_rate
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * water_pump_differential_head
        / water_pump_efficiency
    )
    self.assertEqual(b.compute_pump_power(), expected)

  def test_observable_field_names(self):
    b = self.get_default_boiler()

    self.assertSameElements(
        b.observable_field_names(),
        [
            'supply_water_setpoint',
            'supply_water_temperature_sensor',
            'heating_request_count',
        ],
    )

  def test_observe_supply_water_setpoint(self):
    reheat_water_setpoint = 360
    b = self.get_default_boiler()

    observed_value = b.get_observation(
        'supply_water_setpoint', pd.Timestamp('2021-09-01 10:00')
    )

    self.assertEqual(observed_value, reheat_water_setpoint)

  def test_observe_supply_water_temperature_sensor(self):
    reheat_water_setpoint = 360
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    heating_rate = 2.0
    cooling_rate = 0.5
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
    )

    # Start with a temp & setpoint at 360.
    observed_value = b.get_observation(
        'supply_water_temperature_sensor', pd.Timestamp('2021-09-01 10:00')
    )
    self.assertEqual(observed_value, reheat_water_setpoint)

    # Up the setpoint to 365, one minute later, the temp should go to 362.
    b.set_action(
        'supply_water_setpoint', 365.0, pd.Timestamp('2021-09-01 10:00:00')
    )
    observed_value = b.get_observation(
        'supply_water_temperature_sensor', pd.Timestamp('2021-09-01 10:01')
    )
    self.assertAlmostEqual(observed_value, 362.0)

    # At 10 min after the change, the temp should be at set point, 365.
    observed_value = b.get_observation(
        'supply_water_temperature_sensor', pd.Timestamp('2021-09-01 10:10')
    )
    self.assertAlmostEqual(observed_value, 365.0)

    # Drop the setpoint to 350; after 20 min, should drop to 355.
    b.set_action(
        'supply_water_setpoint', 350.0, pd.Timestamp('2021-09-01 10:10:00')
    )

    observed_value = b.get_observation(
        'supply_water_temperature_sensor', pd.Timestamp('2021-09-01 10:30')
    )
    self.assertAlmostEqual(observed_value, 355.0)

    # And 50 min later should be at setpoint (no lower).
    observed_value = b.get_observation(
        'supply_water_temperature_sensor', pd.Timestamp('2021-09-01 11:00')
    )
    self.assertAlmostEqual(observed_value, 350.0)

  @parameterized.parameters(
      (
          300.0,
          310.0,
          pd.Timestamp('2021-09-01 11:00'),
          pd.Timestamp('2021-09-01 11:10'),
          0.0,
          0.0,
          310.0,
          137.518,
      ),
      (
          300.0,
          310.0,
          pd.Timestamp('2021-09-01 11:00'),
          pd.Timestamp('2021-09-01 11:05'),
          1.0,
          1.0,
          305.0,
          242.0183,
      ),
      (
          300.0,
          320.0,
          pd.Timestamp('2021-09-01 11:00'),
          pd.Timestamp('2021-09-01 11:30'),
          5.0,
          1.0,
          320.0,
          269.6934,
      ),
      (
          320.0,
          300.0,
          pd.Timestamp('2021-09-01 11:00'),
          pd.Timestamp('2021-09-01 11:05'),
          5.0,
          2.0,
          310.0,
          -133.9899,
      ),
      (
          320.0,
          300.0,
          pd.Timestamp('2021-09-01 11:00'),
          pd.Timestamp('2021-09-01 11:30'),
          5.0,
          1.0,
          300.0,
          5.3433,
      ),
  )
  def test_set_current_temperature_default(
      self,
      current_temp,
      setpoint_temp,
      action_timestamp,
      observation_timestamp,
      heating_rate,
      cooling_rate,
      expected_temp,
      expected_energy_rate,
  ):
    reheat_water_setpoint = current_temp
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        device_id='boiler_id',
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
    )

    b.set_action('supply_water_setpoint', setpoint_temp, action_timestamp)

    observed_temp = b.get_observation(
        'supply_water_temperature_sensor', observation_timestamp
    )

    self.assertAlmostEqual(expected_temp, observed_temp)
    energy_rate = b.compute_thermal_energy_rate(300, 288)
    self.assertAlmostEqual(expected_energy_rate, energy_rate, places=3)

  def test_observe_heating_request_count(self):
    b = self.get_default_boiler()

    b.add_demand(1.5)
    b.add_demand(1.5)

    observed_value = b.get_observation(
        'heating_request_count', pd.Timestamp('2021-09-01 10:00')
    )

    self.assertEqual(observed_value, 2)

  def test_compute_thermal_dissipation_rate_valid(self):
    b = self.get_default_boiler()
    q = b.compute_thermal_dissipation_rate(340.0, 290.0)
    self.assertAlmostEqual(q, 312.5418, places=4)

  def test_compute_thermal_dissipation_rate_zero(self):
    b = self.get_default_boiler()
    q = b.compute_thermal_dissipation_rate(290.0, 290.0)
    self.assertAlmostEqual(q, 0.0, places=4)

  def test_compute_thermal_dissipation_rate_invalid(self):
    b = self.get_default_boiler()
    with self.assertRaises(AssertionError):
      _ = b.compute_thermal_dissipation_rate(240.0, 290.0)

  def test_action_field_names(self):
    b = self.get_default_boiler()

    self.assertSameElements(b.action_field_names(), ['supply_water_setpoint'])

  def test_action_supply_water_setpoint(self):
    b = self.get_default_boiler()

    new_value = 280.0
    b.set_action(
        'supply_water_setpoint', new_value, pd.Timestamp('2021-09-01 10:00')
    )

    self.assertEqual(b.reheat_water_setpoint, new_value)

  def test_device_type(self):
    b = self.get_default_boiler()

    device_type = b.device_type()

    self.assertEqual(
        device_type, smart_control_building_pb2.DeviceInfo.DeviceType.BLR
    )

  def test_device_id(self):
    b = self.get_default_boiler()

    device_id = b.device_id()

    self.assertEqual(device_id, 'boiler_id')


if __name__ == '__main__':
  absltest.main()
