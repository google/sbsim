"""Tests for hvac.

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
import pandas as pd
from smart_control.simulator import air_handler
from smart_control.simulator import boiler
from smart_control.simulator import hvac
from smart_control.simulator import setpoint_schedule
from smart_control.utils import conversion_utils


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


def _get_default_air_handler():
  recirculation = 0.3
  heating_air_temp_setpoint = 270
  cooling_air_temp_setpoint = 288
  fan_differential_pressure = 20000.0
  fan_efficiency = 0.8

  handler = air_handler.AirHandler(
      recirculation,
      heating_air_temp_setpoint,
      cooling_air_temp_setpoint,
      fan_differential_pressure,
      fan_efficiency,
  )
  return handler


def _get_default_setpoint_schedule():
  morning_start_hour = 9
  evening_start_hour = 18
  comfort_temp_window = (292, 295)
  eco_temp_window = (290, 297)
  holidays = set([7, 223, 245])

  schedule = setpoint_schedule.SetpointSchedule(
      morning_start_hour,
      evening_start_hour,
      comfort_temp_window,
      eco_temp_window,
      holidays,
  )
  return schedule


class HvacTest(absltest.TestCase):

  def test_init(self):
    zone_coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    handler = _get_default_air_handler()
    b = _get_default_boiler()
    schedule = _get_default_setpoint_schedule()
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4
    h = hvac.Hvac(
        zone_coordinates,
        handler,
        b,
        schedule,
        vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate,
    )
    self.assertEqual(h.air_handler, handler)
    self.assertEqual(h.boiler, b)

    self.assertCountEqual(h.vavs.keys(), zone_coordinates)

    for coord in zone_coordinates:
      vav = h.vavs[coord]
      self.assertEqual(vav.thermostat._setpoint_schedule, schedule)
      self.assertEqual(vav.boiler, b)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_rate, vav_reheat_max_water_flow_rate
      )
      self.assertEqual(
          vav._zone_id, conversion_utils.zone_coordinates_to_id(coord)
      )

  def test_reset(self):
    zone_coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    handler = _get_default_air_handler()
    b = _get_default_boiler()
    schedule = _get_default_setpoint_schedule()
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4
    h = hvac.Hvac(
        zone_coordinates,
        handler,
        b,
        schedule,
        vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate,
    )

    h.boiler._return_water_temperature_sensor += 10.0
    h.boiler._water_pump_differential_head += 100.0
    h.boiler._reheat_water_setpoint += 2.0

    h.air_handler._air_flow_rate += 0.1
    h.air_handler._fan_differential_pressure = 0.1

    for coord in zone_coordinates:
      vav = h.vavs[coord]
      vav.thermostat._setpoint_schedule.morning_start_hour += 1.0
      vav.thermostat._setpoint_schedule.comfort_temp_window = (280, 310)

      vav.max_air_flow_rate += 0.1
      vav._reheat_max_water_flow_rate += 0.1

    h.reset()

    expected_air_handler = _get_default_air_handler()
    self.assertEqual(
        h.air_handler.recirculation, expected_air_handler.recirculation
    )
    self.assertEqual(
        h.air_handler.heating_air_temp_setpoint,
        expected_air_handler.heating_air_temp_setpoint,
    )
    self.assertEqual(
        h.air_handler.cooling_air_temp_setpoint,
        expected_air_handler.cooling_air_temp_setpoint,
    )
    self.assertEqual(
        h.air_handler.fan_differential_pressure,
        expected_air_handler.fan_differential_pressure,
    )
    self.assertEqual(
        h.air_handler.fan_efficiency, expected_air_handler.fan_efficiency
    )

    expected_boiler = _get_default_boiler()
    self.assertEqual(
        h.boiler.reheat_water_setpoint, expected_boiler._reheat_water_setpoint
    )
    self.assertEqual(
        h.boiler._water_pump_differential_head,
        expected_boiler._water_pump_differential_head,
    )
    self.assertEqual(
        h.boiler._water_pump_efficiency, expected_boiler._water_pump_efficiency
    )
    self.assertEqual(h.boiler._total_flow_rate, 0)

    for coord in zone_coordinates:
      vav = h.vavs[coord]
      self.assertEqual(vav.thermostat._setpoint_schedule, schedule)
      self.assertEqual(vav.boiler, b)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_rate, vav_reheat_max_water_flow_rate
      )
      self.assertEqual(
          vav._zone_id, conversion_utils.zone_coordinates_to_id(coord)
      )

  def test_vav_device_ids(self):
    expected_vav_ids = [
        'vav_0_0',
        'vav_1_0',
        'vav_1_1',
        'vav_0_1',
    ]

    zone_coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    handler = _get_default_air_handler()
    b = _get_default_boiler()
    schedule = _get_default_setpoint_schedule()
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4
    h = hvac.Hvac(
        zone_coordinates,
        handler,
        b,
        schedule,
        vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate,
    )

    vav_ids = []
    for coord in zone_coordinates:
      vav = h.vavs[coord]
      vav_ids.append(vav._device_id)

    self.assertListEqual(vav_ids, expected_vav_ids)

  def test_id_comfort_mode(self):
    zone_coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    handler = _get_default_air_handler()
    b = _get_default_boiler()
    schedule = _get_default_setpoint_schedule()
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4
    h = hvac.Hvac(
        zone_coordinates,
        handler,
        b,
        schedule,
        vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate,
    )
    self.assertFalse(h.is_comfort_mode(pd.Timestamp('2021-10-31 10:00')))
    self.assertFalse(h.is_comfort_mode(pd.Timestamp('2021-11-01 03:00')))
    self.assertTrue(h.is_comfort_mode(pd.Timestamp('2021-11-01 13:00')))
    self.assertFalse(h.is_comfort_mode(pd.Timestamp('2021-11-01 23:00')))


if __name__ == '__main__':
  absltest.main()
