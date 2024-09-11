"""Tests for vav.

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
from smart_control.simulator import boiler
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import thermostat
from smart_control.simulator import vav
from smart_control.utils import constants


def _get_default_thermostat():
  morning_start_hour = 9
  evening_start_hour = 18
  comfort_temp_window = (292, 295)
  eco_temp_window = (290, 297)

  schedule = setpoint_schedule.SetpointSchedule(
      morning_start_hour,
      evening_start_hour,
      comfort_temp_window,
      eco_temp_window,
  )
  t = thermostat.Thermostat(schedule)
  return t


def _get_default_boiler():
  reheat_water_setpoint = 260
  water_pump_differential_head = 3
  water_pump_efficiency = 0.6
  b = boiler.Boiler(
      reheat_water_setpoint,
      water_pump_differential_head,
      water_pump_efficiency,
      'boiler_id',
  )
  return b


def compute_zone_supply_temp(
    reheat_valve_setting,
    reheat_max_water_flow_rate,
    damper_setting,
    max_air_flow_rate,
    supply_air_temp,
    input_water_temp,
):
  reheat_flow_rate = reheat_valve_setting * reheat_max_water_flow_rate
  air_flow_rate = damper_setting * max_air_flow_rate
  return (
      (
          supply_air_temp
          * (
              constants.AIR_HEAT_CAPACITY * air_flow_rate
              - constants.WATER_HEAT_CAPACITY * reheat_flow_rate
          )
          + input_water_temp * constants.WATER_HEAT_CAPACITY * reheat_flow_rate
      )
      / air_flow_rate
      / constants.AIR_HEAT_CAPACITY
  )


class VavTest(parameterized.TestCase):

  def test_init(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(
        max_air_flow_rate,
        reheat_max_water_flow_rate,
        t,
        b,
        'device_id',
        'zone_id',
    )

    self.assertEqual(v.max_air_flow_rate, max_air_flow_rate)
    self.assertEqual(v._reheat_max_water_flow_rate, reheat_max_water_flow_rate)
    self.assertEqual(v.thermostat, t)
    self.assertEqual(v.boiler, b)
    self.assertEqual(v.reheat_valve_setting, 0)
    self.assertEqual(v.damper_setting, 0.1)
    self.assertEqual(v.zone_air_temperature, 0)
    self.assertEqual(v._device_id, 'device_id')
    self.assertEqual(v._zone_id, 'zone_id')

  def test_init_default(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    self.assertIsNotNone(v._device_id)
    self.assertIsNotNone(v._zone_id)

  def test_setters(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    v.reheat_valve_setting += 0.1
    v.max_air_flow_rate += 0.1
    v.damper_setting += 0.1

    self.assertEqual(v.reheat_valve_setting, 0.1)
    self.assertEqual(v.damper_setting, 0.2)
    self.assertEqual(v.max_air_flow_rate, max_air_flow_rate + 0.1)

  def test_setters_raise_error(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    with self.assertRaises(ValueError):
      v.reheat_valve_setting = 1.1
    with self.assertRaises(ValueError):
      v.damper_setting = 1.1
    with self.assertRaises(ValueError):
      v.reheat_valve_setting = -0.1
    with self.assertRaises(ValueError):
      v.damper_setting = -0.1

  @parameterized.parameters(
      (pd.Timestamp('2021-05-09 14:00'), 293, 0.1, 0.0),
      (pd.Timestamp('2021-05-10 09:00'), 296, 1.0, 0.0),
      (pd.Timestamp('2021-05-12 09:00'), 291, 1.0, 1.0),
      (pd.Timestamp('2021-05-12 17:59'), 291, 1.0, 1.0),
      (pd.Timestamp('2021-05-11 03:00'), 288, 1.0, 1.0),
      (pd.Timestamp('2021-05-11 03:00'), 291, 0.1, 0.0),
      (pd.Timestamp('2021-05-11 22:00'), 298, 1.0, 0.0),
      (pd.Timestamp('2021-05-11 22:00'), 297, 0.1, 0.0),
  )
  def test_update_settings(
      self,
      current_timestamp,
      zone_temp,
      expected_damper_setting,
      expected_reheat_valve_setting,
  ):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    t._previous_timestamp = current_timestamp - pd.Timedelta(
        60.0, unit='minute'
    )
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)
    v.update_settings(zone_temp, current_timestamp)
    self.assertEqual(expected_damper_setting, v._damper_setting)
    self.assertEqual(expected_reheat_valve_setting, v._reheat_valve_setting)

  @parameterized.parameters(
      (0.5, 0.4, 270, 260),
      (0.3, 0.5, 280, 240),
      (0.2, 0.87, 120, 260),
      (0.6, 0.32, 23, 26),
  )
  def test_compute_reheat_energy_rate(
      self,
      reheat_valve_setting,
      reheat_max_water_flow_rate,
      input_water_temp,
      supply_air_temp,
  ):
    max_air_flow_rate = 0.6
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)
    v.reheat_valve_setting = reheat_valve_setting

    expected = (
        reheat_valve_setting
        * reheat_max_water_flow_rate
        * constants.WATER_HEAT_CAPACITY
        * (input_water_temp - supply_air_temp)
    )

    self.assertEqual(
        v.compute_reheat_energy_rate(supply_air_temp, input_water_temp),
        expected,
    )

  @parameterized.parameters(
      (0.5, 0.8, 0.3, 0.3, 270, 360),
      (0.1, 0.1, 0.4, 0.4, 210, 32),
      (0, 0.2, 0.2, 0.9, 260, 270),
      (0.9, 0.4, 0.1, 0.6, 270, 430),
  )
  def test_compute_zone_supply_temp(
      self,
      reheat_valve_setting,
      damper_setting,
      max_air_flow_rate,
      reheat_max_water_flow_rate,
      input_water_temp,
      supply_air_temp,
  ):
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)
    v.reheat_valve_setting = reheat_valve_setting
    v.damper_setting = damper_setting

    expected = compute_zone_supply_temp(
        reheat_valve_setting,
        reheat_max_water_flow_rate,
        damper_setting,
        max_air_flow_rate,
        supply_air_temp,
        input_water_temp,
    )

    self.assertEqual(
        v.compute_zone_supply_temp(supply_air_temp, input_water_temp), expected
    )

  def test_compute_zone_supply_temp_asserts_error(self):
    reheat_valve_setting = 0.5
    max_air_flow_rate = 0.3
    reheat_max_water_flow_rate = 0.4
    input_water_temp = 270
    supply_air_temp = 260
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)
    v.reheat_valve_setting = reheat_valve_setting
    v.damper_setting = 0

    with self.assertRaises(AssertionError):
      v.compute_zone_supply_temp(supply_air_temp, input_water_temp)

    v.damper_setting = 0.5
    v._max_air_flow_rate = 0
    with self.assertRaises(AssertionError):
      v.compute_zone_supply_temp(supply_air_temp, input_water_temp)

  @parameterized.parameters(
      (560, 250, 400, 0, 0.4),
      (120, 260, 130, 0.9, 0),
      (250, 80, 80, 0.2, 0.1),
      (270, 160, 420, 0.7, 0.2),
  )
  def test_compute_energy_applied_to_zone(
      self,
      zone_temp,
      supply_air_temp,
      input_water_temp,
      damper_setting,
      max_air_flow_rate,
  ):
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)
    v.damper_setting = damper_setting

    expected = 0
    if damper_setting != 0 and max_air_flow_rate != 0:
      # This test relies on the accuracy of this function, tested above.
      zone_supply_temp = v.compute_zone_supply_temp(
          supply_air_temp, input_water_temp
      )

      expected = (
          damper_setting
          * max_air_flow_rate
          * constants.AIR_HEAT_CAPACITY
          * (zone_supply_temp - zone_temp)
      )
    self.assertEqual(
        v.compute_energy_applied_to_zone(
            zone_temp, supply_air_temp, input_water_temp
        ),
        expected,
    )

  @parameterized.parameters(
      (255, 250),
      (120, 260),
      (250, 80),
      (270, 160),
  )
  def test_update_heat(self, zone_temp, supply_air_temp):
    # This test relies on the accuracy of compute_energy_applied_to_zone and
    # compute_zone_supply_temp as tested above.

    # This should produce a different result depending on the mode the
    # thermostat is in.
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)

    self.assertEqual(t.update(zone_temp, time), thermostat.Thermostat.Mode.HEAT)
    damper_setting = 1.0
    reheat_valve_setting = 1.0

    zone_supply_temp = compute_zone_supply_temp(
        reheat_valve_setting,
        reheat_max_water_flow_rate,
        damper_setting,
        max_air_flow_rate,
        supply_air_temp,
        b.reheat_water_setpoint,
    )

    q_zone = (
        damper_setting
        * max_air_flow_rate
        * constants.AIR_HEAT_CAPACITY
        * (zone_supply_temp - zone_temp)
    )

    expected = (q_zone, zone_supply_temp)
    self.assertEqual(v.update(zone_temp, time, supply_air_temp), expected)

  @parameterized.parameters(
      (299, 250),
      (400, 660),
      (450, 80),
      (399, 450),
  )
  def test_update_cool(self, zone_temp, supply_air_temp):
    # This test relies on the accuracy of compute_energy_applied_to_zone and
    # compute_zone_supply_temp as tested above.

    # This should produce a different result depending on the mode the
    # thermostat is in.
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)

    self.assertEqual(t.update(zone_temp, time), thermostat.Thermostat.Mode.COOL)
    damper_setting = 1.0
    reheat_valve_setting = 0

    zone_supply_temp = compute_zone_supply_temp(
        reheat_valve_setting,
        reheat_max_water_flow_rate,
        damper_setting,
        max_air_flow_rate,
        supply_air_temp,
        b.reheat_water_setpoint,
    )

    q_zone = (
        damper_setting
        * max_air_flow_rate
        * constants.AIR_HEAT_CAPACITY
        * (zone_supply_temp - zone_temp)
    )

    expected = (q_zone, zone_supply_temp)
    self.assertEqual(v.update(zone_temp, time, supply_air_temp), expected)

  @parameterized.parameters(
      (292, 250),
      (295, 660),
      (294, 80),
      (293, 450),
  )
  def test_update_off(self, zone_temp, supply_air_temp):
    # This test relies on the accuracy of compute_energy_applied_to_zone and
    # compute_zone_supply_temp as tested above.

    # This should produce a different result depending on the mode the
    # thermostat is in.
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)

    self.assertEqual(t.update(zone_temp, time), thermostat.Thermostat.Mode.OFF)
    damper_setting = 0.1
    reheat_valve_setting = 0

    # change the VAV damper setting, since otherwise this test passes since the
    # default setting is the same as the one update switches too in Off mode
    v.damper_setting = 0.6
    zone_supply_temp = compute_zone_supply_temp(
        reheat_valve_setting,
        reheat_max_water_flow_rate,
        damper_setting,
        max_air_flow_rate,
        supply_air_temp,
        b.reheat_water_setpoint,
    )

    q_zone = (
        damper_setting
        * max_air_flow_rate
        * constants.AIR_HEAT_CAPACITY
        * (zone_supply_temp - zone_temp)
    )

    expected = (q_zone, zone_supply_temp)
    self.assertEqual(v.update(zone_temp, time, supply_air_temp), expected)

  def test_observable_field_names(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    self.assertSameElements(
        v.observable_field_names(),
        [
            'supply_air_damper_percentage_command',
            'supply_air_flowrate_setpoint',
            'zone_air_temperature_sensor',
        ],
    )

  @parameterized.parameters(
      ('supply_air_damper_percentage_command', 'damper_setting'),
      ('supply_air_flowrate_setpoint', 'max_air_flow_rate'),
  )
  def test_observations(self, observation_name, attribute_name):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    observed_value = v.get_observation(
        observation_name, pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, getattr(v, attribute_name))

  def test_zone_air_temperature_sensor(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    observed_value = v.get_observation(
        'zone_air_temperature_sensor', pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, 0)
    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)

    v.update(300, time, 200)

    observed_value = v.get_observation(
        'zone_air_temperature_sensor', pd.Timestamp('2021-09-01 10:10:00')
    )
    self.assertEqual(observed_value, 300)

  def test_action_field_names(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    self.assertSameElements(
        v.action_field_names(), ['supply_air_damper_percentage_command']
    )

  def test_action_supply_air_flowrate_setpoint(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    new_value = 0.8
    v.set_action(
        'supply_air_damper_percentage_command',
        new_value,
        pd.Timestamp('2021-09-01 10:10:00'),
    )

    self.assertEqual(v.damper_setting, new_value)

  def test_output_does_not_change_settings(self):
    max_air_flow_rate = 0.6
    reheat_max_water_flow_rate = 0.4
    t = _get_default_thermostat()
    b = _get_default_boiler()
    v = vav.Vav(max_air_flow_rate, reheat_max_water_flow_rate, t, b)

    v.damper_setting = 0.6
    v.reheat_valve_setting = 0.7

    v.output(200, 205)

    self.assertEqual(v.damper_setting, 0.6)
    self.assertEqual(v.reheat_valve_setting, 0.7)


if __name__ == '__main__':
  absltest.main()
