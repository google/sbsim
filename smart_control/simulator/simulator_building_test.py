from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.simulator import air_handler
from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import hot_water_system as hot_water_system_py
from smart_buildings.smart_control.simulator import hvac_floorplan_based
from smart_buildings.smart_control.simulator import setpoint_schedule
from smart_buildings.smart_control.simulator import simulator as simulator_py
from smart_buildings.smart_control.simulator import simulator_building as sb_py
from smart_buildings.smart_control.simulator import simulator_building_test_lib
from smart_buildings.smart_control.simulator import simulator_flexible_floor_plan
from smart_buildings.smart_control.simulator import step_function_occupancy
from smart_buildings.smart_control.simulator import weather_controller


class SimulatorBuildingTest(
    simulator_building_test_lib.SimulatorBuildingTestBase
):

  def get_sim_building(
      self,
      initial_rejection_count: int = 0,
      zones: list[building_pb2.ZoneInfo] | None = None,
      simulator: (
          simulator_py.Simulator
          | simulator_flexible_floor_plan.SimulatorFlexibleGeometries
          | None
      ) = None,
  ) -> sb_py.SimulatorBuilding:
    sim = simulator or self._create_small_simulator()
    return sb_py.SimulatorBuilding(
        simulator=sim,
        occupancy=self.occupancy,
        zones=zones,
    )


class FloorPlanBasedSimulatorBuildingTest(
    simulator_building_test_lib.SimulatorBuildingTestBase
):

  def setUp(self):
    self.zone_ids = ["room_0", "room_1", "room_2"]

    self.wc = weather_controller.WeatherController(
        default_low_temp=280,
        default_high_temp=290,
    )
    self.occ = step_function_occupancy.StepFunctionOccupancy(
        work_start_time=pd.Timedelta("9h"),
        work_end_time=pd.Timedelta("17h"),
        work_occupancy=1.0,
        nonwork_occupancy=0.0,
    )

    ahu = air_handler.AirHandler(
        recirculation=0.3,
        heating_air_temp_setpoint=270,
        cooling_air_temp_setpoint=288,
        fan_static_pressure=20000.0,
        fan_efficiency=0.8,
    )
    hws = hot_water_system_py.construct_hot_water_system(
        supply_water_temperature_setpoint=260,
        water_pump_differential_head=3,
        water_pump_efficiency=0.6,
        device_id="hws_id",
    )
    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour=9,
        evening_start_hour=18,
        comfort_temp_window=(292, 295),
        eco_temp_window=(290, 297),
        holidays={7, 223, 245},
    )

    zone_to_vavs = {z: [f"vav_{z}"] for z in self.zone_ids}
    self.hvac = hvac_floorplan_based.FloorPlanBasedHvac(
        zone_identifier=self.zone_ids,
        air_handler=ahu,
        hot_water_system=hws,
        schedule=schedule,
        vav_max_air_flow_rate=0.2,
        vav_reheat_max_water_flow_factor=0.4,
        zone_to_vavs=zone_to_vavs,
    )

    building_mock = mock.create_autospec(
        building_py.FloorPlanBasedBuilding, instance=True
    )
    building_mock.get_zone_average_temps.return_value = {
        "room_0": 295,
        "room_1": 295,
        "room_2": 295,
    }
    building_mock.floor_plan = np.array([[1]])
    building_mock.room_dict = {"room_0": [], "room_1": [], "room_2": []}
    building_mock.custom_zone_to_vavs = None

    self.sim = simulator_flexible_floor_plan.SimulatorFlexibleGeometries(
        hvac=self.hvac,
        building=building_mock,
        weather_controller=self.wc,
        start_timestamp=pd.Timestamp("2021-01-01 00:00"),
        time_step_sec=300,
        convergence_threshold=0.01,
        iteration_limit=100,
        iteration_warning=50,
    )

    super().setUp()

  def get_sim_building(
      self,
      initial_rejection_count: int = 0,
      zones: list[building_pb2.ZoneInfo] | None = None,
      simulator: (
          simulator_py.Simulator
          | simulator_flexible_floor_plan.SimulatorFlexibleGeometries
          | None
      ) = None,
  ) -> sb_py.SimulatorBuilding:
    sim = simulator or self.sim
    return sb_py.SimulatorBuilding(
        simulator=sim,
        occupancy=self.occ,
        zones=zones,
    )

  def test_devices(self):
    self.assertLen(self.building.devices, 5)

  def test_init_with_zones_uses_provided_zones_overwrites_hvac_zones(self):
    zones = [
        building_pb2.ZoneInfo(zone_id="the_real_zone_1"),
        building_pb2.ZoneInfo(zone_id="the_real_zone_2"),
    ]
    sim_building = self.get_sim_building(zones=zones)

    self.assertEqual(zones, sim_building.zones)
    self.assertEqual(
        zones, list(sim_building.simulator.hvac.zone_infos.values())
    )

  def test_init_with_zones_uses_provided_zones_overwrites_hvac_devices(self):
    zones = [
        building_pb2.ZoneInfo(zone_id="new_room_0", devices=["vav_custom_0"]),
        building_pb2.ZoneInfo(zone_id="new_room_1", devices=["vav_custom_1"]),
    ]
    sim_building = self.get_sim_building(zones=zones)

    with self.subTest("HVAC devices are overwritten"):
      self.assertEqual(
          sim_building.simulator.hvac.get_vav_ids_for_zone("new_room_0"),
          ["vav_custom_0"],
      )
      self.assertEqual(
          sim_building.simulator.hvac.get_zones_for_vav("vav_custom_0"),
          ["new_room_0"],
      )

    # TODO(b/512158835) - Update after cascading zone info overrides into the
    # room dict as well.
    with self.subTest("FloorPlanBasedBuilding room dict is not yet updated"):
      self.assertEqual(
          sim_building.simulator.building.room_dict,
          {"room_0": [], "room_1": [], "room_2": []},
      )
      self.assertIsNone(sim_building.simulator.building.custom_zone_to_vavs)


if __name__ == "__main__":
  absltest.main()
