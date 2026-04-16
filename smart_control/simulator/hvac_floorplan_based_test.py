from typing import Collection, Optional

from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.simulator import air_handler
from smart_buildings.smart_control.simulator import hot_water_system as hot_water_system_py
from smart_buildings.smart_control.simulator import hvac_floorplan_based
from smart_buildings.smart_control.simulator import setpoint_schedule


class FloorPlanBasedHvacTest(absltest.TestCase):

  def setUp(self):
    super(FloorPlanBasedHvacTest, self).setUp()
    self._zone_identifier = ["room_0", "room_1", "room_2"]
    self._global_hot_water_system = self._get_default_hot_water_system()
    self._global_handler = self._get_default_air_handler()
    self._global_setpoint_schedule = self._get_default_setpoint_schedule()
    self._hvac = self._create_default_hvac(self._zone_identifier)

  def _create_default_hvac(
      self, zone_identifier: Optional[Collection[str]] = None
  ) -> hvac_floorplan_based.FloorPlanBasedHvac:
    handler = self._global_handler
    hws = self._global_hot_water_system
    schedule = self._global_setpoint_schedule
    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_factor = 0.4
    zone_to_vavs = (
        {z: [f"vav_{z}"] for z in zone_identifier} if zone_identifier else {}
    )
    h = hvac_floorplan_based.FloorPlanBasedHvac(
        zone_identifier=zone_identifier,
        air_handler=handler,
        hot_water_system=hws,
        schedule=schedule,
        vav_max_air_flow_rate=vav_max_air_flow_rate,
        vav_reheat_max_water_flow_factor=vav_reheat_max_water_flow_factor,
        zone_to_vavs=zone_to_vavs,
    )
    return h

  def _get_default_hot_water_system(self):
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    hot_water_system = hot_water_system_py.construct_hot_water_system(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        "hws_id",
    )
    return hot_water_system

  def _get_default_air_handler(self):
    recirculation = 0.3
    heating_air_temp_setpoint = 270
    cooling_air_temp_setpoint = 288
    fan_static_pressure = 20000.0
    fan_efficiency = 0.8

    handler = air_handler.AirHandler(
        recirculation,
        heating_air_temp_setpoint,
        cooling_air_temp_setpoint,
        fan_static_pressure,
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
    vav_reheat_max_water_flow_factor = 0.4

    h = self._create_default_hvac(self._zone_identifier)
    self.assertEqual(h.air_handler, self._global_handler)
    self.assertEqual(h.hot_water_system, self._global_hot_water_system)

    expected_vav_ids = [f"vav_{z}" for z in self._zone_identifier]
    self.assertCountEqual(h.vavs.keys(), expected_vav_ids)

    for coord in self._zone_identifier:
      v_id = f"vav_{coord}"
      vav = h.vavs[v_id]  # vavs are not indexed by zone_id anymore
      self.assertEqual(
          vav.thermostat._setpoint_schedule, self._global_setpoint_schedule
      )
      self.assertEqual(vav.hot_water_system, self._global_hot_water_system)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_factor, vav_reheat_max_water_flow_factor
      )
      self.assertEqual(
          vav._zone_id,
          coord,
      )

  def test_reset(self):
    self._hvac.hot_water_system.return_water_temperature_sensor += 10.0
    self._hvac.hot_water_system.water_pump_differential_head += 100.0
    self._hvac.hot_water_system.reheat_water_setpoint += 2.0

    self._hvac.air_handler._air_flow_rate += 0.1
    self._hvac.air_handler._fan_static_pressure = 0.1

    for coord in self._zone_identifier:
      vav = self._hvac.vavs[f"vav_{coord}"]
      vav.thermostat._setpoint_schedule.morning_start_hour += 1.0
      vav.thermostat._setpoint_schedule.comfort_temp_window = (280, 310)

      vav.max_air_flow_rate += 0.1

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
        self._hvac.air_handler.fan_static_pressure,
        expected_air_handler.fan_static_pressure,
    )
    self.assertEqual(
        self._hvac.air_handler.fan_efficiency,
        expected_air_handler.fan_efficiency,
    )

    expected_hot_water_system = self._global_hot_water_system
    self.assertEqual(
        self._hvac.hot_water_system.reheat_water_setpoint,
        expected_hot_water_system.reheat_water_setpoint,
    )
    self.assertEqual(
        self._hvac.hot_water_system._pump._water_pump_differential_head,
        expected_hot_water_system._pump._water_pump_differential_head,
    )
    self.assertEqual(
        self._hvac.hot_water_system._pump._water_pump_efficiency,
        expected_hot_water_system._pump._water_pump_efficiency,
    )
    self.assertEqual(self._hvac.hot_water_system.total_flow_rate, 0)

    vav_max_air_flow_rate = 0.2
    vav_reheat_max_water_flow_factor = 0.4

    for coord in self._zone_identifier:
      vav = self._hvac.vavs[f"vav_{coord}"]
      self.assertEqual(
          vav.thermostat._setpoint_schedule, self._global_setpoint_schedule
      )
      self.assertEqual(vav.hot_water_system, self._global_hot_water_system)
      self.assertEqual(vav.max_air_flow_rate, vav_max_air_flow_rate)
      self.assertEqual(
          vav._reheat_max_water_flow_factor, vav_reheat_max_water_flow_factor
      )
      self.assertEqual(vav._zone_id, coord)

  def test_vav_device_ids(self):
    expected_vav_ids = [
        "vav_room_0",
        "vav_room_1",
        "vav_room_2",
    ]

    vav_ids = []
    for coord in self._zone_identifier:
      vav = self._hvac.vavs[f"vav_{coord}"]
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

    zones = list(self._zone_identifier)
    zone_to_vavs = {z: [f"vav_{z}"] for z in zones}
    test_hvac.initialize_zone_identifier(zones, zone_to_vavs)

    with self.subTest("check_zone_assignment_is_equal"):
      self.assertEqual(test_hvac._vavs.keys(), self._hvac.vavs.keys())


if __name__ == "__main__":
  absltest.main()
