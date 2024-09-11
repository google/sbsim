"""Tests for floor plan based hvac.

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

from typing import Collection, Optional

from absl.testing import absltest
import pandas as pd
from smart_control.simulator import air_handler
from smart_control.simulator import boiler
from smart_control.simulator import hvac_floorplan_based
from smart_control.simulator import setpoint_schedule
from smart_control.utils import conversion_utils


class FloorPlanBasedHvacTest(absltest.TestCase):

  def setUp(self):
    super(FloorPlanBasedHvacTest, self).setUp()
    self._zone_identifier = ["room_0", "room_1", "room_2"]
    self._global_boiler = self._get_default_boiler()
    self._global_handler = self._get_default_air_handler()
    self._global_setpoint_schedule = self._get_default_setpoint_schedule()
    self._hvac = self._create_default_hvac(self._zone_identifier)

  def _create_default_hvac(
      self, zone_identifier: Optional[Collection[str]] = None
  ) -> hvac_floorplan_based.FloorPlanBasedHvac:
    handler = self._global_handler
    b = self._global_boiler
    schedule = self._global_setpoint_schedule
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4
    h = hvac_floorplan_based.FloorPlanBasedHvac(
        zone_identifier=zone_identifier,
        air_handler=handler,
        boiler=b,
        schedule=schedule,
        vav_max_air_flow_rate=vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate=vav_reheat_max_water_flow_rate,
    )
    return h

  def _get_default_boiler(self):
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    b = boiler.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        "boiler_id",
    )
    return b

  def _get_default_air_handler(self):
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

  def _get_default_setpoint_schedule(self):
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

  def test_init(self):
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4

    h = self._create_default_hvac(self._zone_identifier)
    self.assertEqual(h.air_handler, self._global_handler)
    self.assertEqual(h.boiler, self._global_boiler)

    self.assertCountEqual(h.vavs.keys(), self._zone_identifier)

    for coord in self._zone_identifier:
      vav = h.vavs[coord]
      self.assertEqual(
          vav.thermostat._setpoint_schedule, self._global_setpoint_schedule
      )
      self.assertEqual(vav.boiler, self._global_boiler)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_rate, vav_reheat_max_water_flow_rate
      )
      self.assertEqual(
          vav._zone_id,
          conversion_utils.floor_plan_based_zone_identifier_to_id(coord),
      )

  def test_reset(self):
    self._hvac.boiler._return_water_temperature_sensor += 10.0
    self._hvac.boiler._water_pump_differential_head += 100.0
    self._hvac.boiler._reheat_water_setpoint += 2.0

    self._hvac.air_handler._air_flow_rate += 0.1
    self._hvac.air_handler._fan_differential_pressure = 0.1

    for coord in self._zone_identifier:
      vav = self._hvac.vavs[coord]
      vav.thermostat._setpoint_schedule.morning_start_hour += 1.0
      vav.thermostat._setpoint_schedule.comfort_temp_window = (280, 310)

      vav.max_air_flow_rate += 0.1
      vav._reheat_max_water_flow_rate += 0.1

    self._hvac.reset()

    expected_air_handler = self._global_handler
    self.assertEqual(
        self._hvac.air_handler.recirculation, expected_air_handler.recirculation
    )
    self.assertEqual(
        self._hvac.air_handler.heating_air_temp_setpoint,
        expected_air_handler.heating_air_temp_setpoint,
    )
    self.assertEqual(
        self._hvac.air_handler.cooling_air_temp_setpoint,
        expected_air_handler.cooling_air_temp_setpoint,
    )
    self.assertEqual(
        self._hvac.air_handler.fan_differential_pressure,
        expected_air_handler.fan_differential_pressure,
    )
    self.assertEqual(
        self._hvac.air_handler.fan_efficiency,
        expected_air_handler.fan_efficiency,
    )

    expected_boiler = self._global_boiler
    self.assertEqual(
        self._hvac.boiler.reheat_water_setpoint,
        expected_boiler._reheat_water_setpoint,
    )
    self.assertEqual(
        self._hvac.boiler._water_pump_differential_head,
        expected_boiler._water_pump_differential_head,
    )
    self.assertEqual(
        self._hvac.boiler._water_pump_efficiency,
        expected_boiler._water_pump_efficiency,
    )
    self.assertEqual(self._hvac.boiler._total_flow_rate, 0)

    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_rate = 0.4

    for coord in self._zone_identifier:
      vav = self._hvac.vavs[coord]
      self.assertEqual(
          vav.thermostat._setpoint_schedule, self._global_setpoint_schedule
      )
      self.assertEqual(vav.boiler, self._global_boiler)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_rate, vav_reheat_max_water_flow_rate
      )
      self.assertEqual(
          vav._zone_id,
          conversion_utils.floor_plan_based_zone_identifier_to_id(coord),
      )

  def test_vav_device_ids(self):
    expected_vav_ids = [
        "vav_room_0",
        "vav_room_1",
        "vav_room_2",
    ]

    vav_ids = []
    for coord in self._zone_identifier:
      vav = self._hvac.vavs[coord]
      vav_ids.append(vav._device_id)

    self.assertListEqual(vav_ids, expected_vav_ids)

  def test_id_comfort_mode(self):
    self.assertFalse(
        self._hvac.is_comfort_mode(pd.Timestamp("2021-10-31 10:00"))
    )
    self.assertFalse(
        self._hvac.is_comfort_mode(pd.Timestamp("2021-11-01 03:00"))
    )
    self.assertTrue(
        self._hvac.is_comfort_mode(pd.Timestamp("2021-11-01 13:00"))
    )
    self.assertFalse(
        self._hvac.is_comfort_mode(pd.Timestamp("2021-11-01 23:00"))
    )

  def test_hvac_init_without_zone_identifier(self):
    test_hvac = self._create_default_hvac()
    with self.subTest("check_fill_zone_identifier_flag"):
      self.assertTrue(test_hvac.fill_zone_identifier_exogenously)

    zones = self._hvac.vavs.keys()
    test_hvac.initialize_zone_identifier(zones)

    with self.subTest("check_zone_assignment_is_equal"):
      self.assertEqual(test_hvac._vavs.keys(), self._hvac.vavs.keys())


if __name__ == "__main__":
  absltest.main()
