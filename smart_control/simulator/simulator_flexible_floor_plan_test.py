"""Tests for simulator."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import constants
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import simulator_flexible_floor_plan as simulator_py
from smart_control.simulator import step_function_occupancy
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.utils import conversion_utils


class FlexibleFloorplanSimulatorTest(parameterized.TestCase):

  def _create_deprecated_scenario_building(self, initial_temp):
    """Returns building with specified initial temperature.

    This test building will be used in various heating/cooling scenarios.

    Args:
      initial_temp: Initial temperature of all CVs in building.
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (20, 30)
    building_shape = (3, 3)
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=5.0, heat_capacity=800.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=5.0, heat_capacity=800.0, density=3000.0
    )

    building = building_py.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )
    return building

  def _create_dummy_floor_plan_small(self):
    """Creates a normal dummy floor plan."""
    plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])

    return plan

  def _create_dummy_floor_plan_small_with_fenestrations(self):
    """Creates a normal dummy floor plan with fenestrations."""
    plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 4, 4, 4, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 4, 4, 4, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        [2, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    return plan

  def _create_scenario_floor_plan(self) -> None:
    """Matches the previous deprecated building.

    Creates a dummy floor plan intended to match exactly the former building
    layout tested in the deprecated Simulation. Specifically, this is a
    rectangular building with 3 by 3 20x30 CV rooms for a total of 9 rooms.

    Returns:
      None
    """
    top = bottom = [2] * (98)
    wall = [2] + [1] * 96 + [2]
    inside = [2, 1, 1] + [0] * 30 + [1] + [0] * 30 + [1] + [0] * 30 + [1, 1, 2]

    b = np.array([
        top,
        wall,
        wall,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        wall,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        wall,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        inside,
        wall,
        wall,
        bottom,
    ])
    return b

  def _create_dummy_floor_plan_weird_shape(self):
    """Creates a weirdly shaped dummy floor plan."""

    plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2, 1, 1, 1, 2],
        [2, 1, 0, 1, 1, 1, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 1, 1, 1, 0, 1, 2],
        [2, 1, 1, 1, 2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    return plan

  def _create_small_building(
      self,
      initial_temp,
      match_diffusers=False,
      include_radiative_heat_transfer=False,
      floor_plan=None,
      include_interior_mass=False,
  ):
    """Returns building with specified initial temperature.

    The building returned will have a matrix size of: 21 x 10, this should be
    left as a comment in tests when relevant. Additionally initial_temp is
    added as a parameter for clarity in the tests.

    Args:
      initial_temp: Initial temperature of all CVs in building.
      match_diffusers: borrow the diffuser allocation scheme of the deprecated
        building (for testing purposes)
      include_radiative_heat_transfer: include radiative heat transfer
      floor_plan: floor plan to use
      include_interior_mass: include interior mass
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=1.0, heat_capacity=500.0, density=3000.0
    )

    if floor_plan is None:
      floor_plan = self._create_dummy_floor_plan_small()

    zone_map = copy.deepcopy(floor_plan)

    if include_radiative_heat_transfer or include_interior_mass:
      if include_radiative_heat_transfer:
        inside_air_radiative_properties = building_py.RadiationProperties(
            alpha=0.0, epsilon=0.0, tau=1.0, rho=None
        )
        inside_wall_radiative_properties = building_py.RadiationProperties(
            alpha=0.4, epsilon=0.6, tau=0.0, rho=None
        )
        building_exterior_radiative_properties = (
            building_py.RadiationProperties(
                alpha=0.65, epsilon=0.35, tau=0.0, rho=None
            )
        )
      else:
        inside_air_radiative_properties = None
        inside_wall_radiative_properties = None
        building_exterior_radiative_properties = None

      if include_interior_mass:
        interior_mass_properties = building_py.MaterialProperties(
            conductivity=5.0, heat_capacity=300.0, density=2000.0
        )
      else:
        interior_mass_properties = None

      building = building_py.FloorPlanBasedBuilding(
          cv_size_cm=cv_size_cm,
          floor_height_cm=floor_height_cm,
          initial_temp=initial_temp,
          inside_air_properties=inside_air_properties,
          inside_wall_properties=inside_wall_properties,
          building_exterior_properties=building_exterior_properties,
          floor_plan=floor_plan,
          zone_map=zone_map,
          buffer_from_walls=0,
          inside_air_radiative_properties=inside_air_radiative_properties,
          inside_wall_radiative_properties=inside_wall_radiative_properties,
          building_exterior_radiative_properties=building_exterior_radiative_properties,  # pylint: disable=line-too-long
          include_radiative_heat_transfer=include_radiative_heat_transfer,
          view_factor_method="ScriptF",
          interior_mass_properties=interior_mass_properties,
          include_interior_mass=include_interior_mass,
      )
    else:
      building = building_py.FloorPlanBasedBuilding(
          cv_size_cm=cv_size_cm,
          floor_height_cm=floor_height_cm,
          initial_temp=initial_temp,
          inside_air_properties=inside_air_properties,
          inside_wall_properties=inside_wall_properties,
          building_exterior_properties=building_exterior_properties,
          floor_plan=floor_plan,
          zone_map=zone_map,
          buffer_from_walls=0,
      )

    if match_diffusers:
      deprecated_building = self._create_small_building_deprecated(initial_temp)
      building.diffusers = np.pad(
          deprecated_building.diffusers, 2, "constant", constant_values=0
      )

    return building

  def _create_simulator_and_building(
      self,
      initial_temp=292.0,
      include_interior_mass=True,
      include_radiative_heat_transfer=False,
      convergence_threshold=0.001,
      iteration_limit=100,
  ):
    """Creates a building and simulator instance with shared parameters."""
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    # Building geometry and base properties
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=1.0, density=1.2
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=500.0, density=3000.0
    )
    interior_mass_properties = building_py.MaterialProperties(
        conductivity=0.5, heat_capacity=1000.0, density=2000.0
    )

    floor_plan = self._create_dummy_floor_plan_small()
    zone_map = copy.deepcopy(floor_plan)

    building = building_py.FloorPlanBasedBuilding(
        cv_size_cm=cv_size_cm,
        floor_height_cm=floor_height_cm,
        initial_temp=initial_temp,
        inside_air_properties=inside_air_properties,
        inside_wall_properties=inside_wall_properties,
        building_exterior_properties=building_exterior_properties,
        floor_plan=floor_plan,
        zone_map=zone_map,
        buffer_from_walls=0,
        interior_mass_properties=interior_mass_properties,
        include_interior_mass=include_interior_mass,
        include_radiative_heat_transfer=include_radiative_heat_transfer,
        view_factor_method="ScriptF"
        if include_radiative_heat_transfer
        else None,
    )

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )
    return simulator, building

  def _create_weirdly_shaped_building(self, initial_temp):
    """Returns weird building with specified initial temperature.

    The building returned will have a matrix size of: 21 x 10, this should be
    left as a comment in tests when relevant. Additionally initial_temp is
    added as a parameter for clarity in the tests.

    Args:
      initial_temp: Initial temperature of all CVs in building.
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=500.0, density=3000.0
    )

    floor_plan = self._create_dummy_floor_plan_weird_shape()
    zone_map = copy.deepcopy(floor_plan)

    building = building_py.FloorPlanBasedBuilding(
        cv_size_cm=cv_size_cm,
        floor_height_cm=floor_height_cm,
        initial_temp=initial_temp,
        inside_air_properties=inside_air_properties,
        inside_wall_properties=inside_wall_properties,
        building_exterior_properties=building_exterior_properties,
        floor_plan=floor_plan,
        zone_map=zone_map,
        buffer_from_walls=0,
    )
    return building

  def _create_small_building_deprecated(self, initial_temp):
    """Returns building with specified initial temperature.

    The building returned will have a matrix size of: 21 x 10, this should be
    left as a comment in tests when relevant. Additionally initial_temp is
    added as a parameter for clarity in the tests.

    Args:
      initial_temp: Initial temperature of all CVs in building.
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (8, 6)
    building_shape = (2, 1)
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=500.0, density=3000.0
    )

    building = building_py.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    return building

  def _create_small_hvac(self):
    """Returns hvac matching zones for small test building."""
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    boiler = boiler_py.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        "boiler_id",
    )

    recirculation = 0.3
    heating_air_temp_setpoint = 270
    cooling_air_temp_setpoint = 288
    fan_differential_pressure = 20000.0
    fan_efficiency = 0.8

    air_handler = air_handler_py.AirHandler(
        recirculation,
        heating_air_temp_setpoint,
        cooling_air_temp_setpoint,
        fan_differential_pressure,
        fan_efficiency,
    )

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

    zone_identifier = ["room_1", "room_2"]

    hvac = floorplan_hvac_py.FloorPlanBasedHvac(
        zone_identifier=zone_identifier,
        air_handler=air_handler,
        boiler=boiler,
        schedule=schedule,
        vav_max_air_flow_rate=0.45,
        vav_reheat_max_water_flow_rate=0.02,
    )
    return hvac

  def _create_scenario_hvac(self, zone_identifier):
    """Returns hvac matching zones for small test building."""
    reheat_water_setpoint = 350
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    boiler = boiler_py.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        "boiler_id",
    )

    recirculation = 0.6
    heating_air_temp_setpoint = 291
    cooling_air_temp_setpoint = 295
    fan_differential_pressure = 20000.0
    fan_efficiency = 0.8

    air_handler = air_handler_py.AirHandler(
        recirculation,
        heating_air_temp_setpoint,
        cooling_air_temp_setpoint,
        fan_differential_pressure,
        fan_efficiency,
    )

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

    hvac = floorplan_hvac_py.FloorPlanBasedHvac(
        zone_identifier=zone_identifier,
        air_handler=air_handler,
        boiler=boiler,
        schedule=schedule,
        vav_max_air_flow_rate=0.45,
        vav_reheat_max_water_flow_rate=0.02,
    )
    return hvac

  def _create_scenario_building(self, initial_temp, match_old_diffusers=False):
    """Returns building with specified initial temperature.

    This test building will be used in various heating/cooling scenarios.

    Args:
      initial_temp: Initial temperature of all CVs in building.
      match_old_diffusers: take the old diffusers and match them
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=5.0, heat_capacity=800.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=5.0, heat_capacity=800.0, density=3000.0
    )

    floor_plan = self._create_scenario_floor_plan()
    zone_map = copy.deepcopy(floor_plan)

    building = building_py.FloorPlanBasedBuilding(
        cv_size_cm,
        floor_height_cm,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
        floor_plan=floor_plan,
        zone_map=zone_map,
        buffer_from_walls=0,
    )

    if match_old_diffusers:
      old_building = self._create_deprecated_scenario_building(
          initial_temp=initial_temp
      )
      building.diffusers = np.pad(old_building.diffusers, 1, mode="constant")
    return building

  def test_init(self):
    building = self._create_small_building(initial_temp=293)
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    self.assertEqual(simulator.building, building)
    self.assertEqual(simulator._weather_controller, weather_controller)
    self.assertEqual(simulator._time_step_sec, time_step_sec)
    self.assertEqual(simulator.time_step_sec, time_step_sec)
    self.assertEqual(simulator._convergence_threshold, convergence_threshold)
    self.assertEqual(simulator._iteration_limit, iteration_limit)
    self.assertEqual(simulator._iteration_warning, iteration_warning)
    self.assertEqual(simulator._current_timestamp, start_timestamp)

  def test_reset(self):
    initial_temp = 293
    building = self._create_small_building(initial_temp)
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    simulator.building.temp[2][2] += 10.0
    simulator.building.temp[0][3] += 10.0
    simulator.building.input_q[2][2] = 1000.0
    simulator.building.input_q[0][3] = 1000.0

    simulator.hvac.boiler._return_water_temperature_sensor += 10.0
    simulator.hvac.boiler._water_pump_differential_head += 100.0
    simulator.hvac.boiler._reheat_water_setpoint += 2.0

    simulator.hvac.air_handler._air_flow_rate += 0.1
    simulator.hvac.air_handler._fan_differential_pressure = 0.1

    for coord in simulator.hvac._zone_identifier:
      vav = simulator.hvac.vavs[coord]
      vav.thermostat._setpoint_schedule.morning_start_hour += 1.0
      vav.thermostat._setpoint_schedule.comfort_temp_window = (280, 310)
      vav.max_air_flow_rate += 0.1
      vav._reheat_max_water_flow_rate += 0.1

    simulator._current_timestamp += pd.Timedelta(360.0, unit="seconds")
    simulator.reset()
    self.assertEqual(simulator.building, building)
    expected_hvac = self._create_small_hvac()
    expected_air_handler = expected_hvac.air_handler
    self.assertEqual(
        simulator._hvac.air_handler.recirculation,
        expected_air_handler.recirculation,
    )
    self.assertEqual(
        simulator._hvac.air_handler.heating_air_temp_setpoint,
        expected_air_handler.heating_air_temp_setpoint,
    )
    self.assertEqual(
        simulator._hvac.air_handler.cooling_air_temp_setpoint,
        expected_air_handler.cooling_air_temp_setpoint,
    )
    self.assertEqual(
        simulator._hvac.air_handler.fan_differential_pressure,
        expected_air_handler.fan_differential_pressure,
    )
    self.assertEqual(
        simulator._hvac.air_handler.fan_efficiency,
        expected_air_handler.fan_efficiency,
    )

    expected_boiler = expected_hvac.boiler
    self.assertEqual(
        simulator._hvac.boiler.reheat_water_setpoint,
        expected_boiler._reheat_water_setpoint,
    )
    self.assertEqual(
        simulator._hvac.boiler._water_pump_differential_head,
        expected_boiler._water_pump_differential_head,
    )
    self.assertEqual(
        simulator._hvac.boiler._water_pump_efficiency,
        expected_boiler._water_pump_efficiency,
    )
    self.assertEqual(simulator._hvac.boiler._total_flow_rate, 0)

    self.assertEqual(simulator._current_timestamp, start_timestamp)
    self.assertEqual(simulator.building.temp[2][2], initial_temp)
    self.assertEqual(simulator.building.temp[0][3], initial_temp)
    self.assertEqual(simulator.building.input_q[2][2], 0)
    self.assertEqual(simulator.building.input_q[0][3], 0)

  def test_get_cv_temp_estimate_cell_no_change(self):
    """This tests that temperatures don"t change in stable conditions.

    This test sets up a small building at temperature 292. The ambient
    conditions are also 292.
    """
    # Set up simulation parameters
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    # Control Volume matrix shape: (21, 10)
    building = self._create_small_building(initial_temp=292.0)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 292.0
    convection_coefficient = 12.0

    expected_temp_estimate = 292.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Test every cell.
    for x in range(12):
      for y in range(9):
        # Get estimate for cell x,y
        temp_estimate = simulator._get_cv_temp_estimate(
            (x, y),
            temperature_estimates,
            ambient_temperature,
            convection_coefficient,
        )
        # Due to floating point precision errors.
        self.assertAlmostEqual(
            temp_estimate,
            expected_temp_estimate,
            msg=f"Cell ({x}, {y}) changed unexpectedly.",
            delta=1e-5,
        )

  @parameterized.named_parameters(
      ("Corner", (1, 1)), ("Edge", (1, 2)), ("Interior", (2, 2))
  )
  def test_get_cv_temp_estimate_cell_increase_from_conduction(
      self, cell_coordinates
  ):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    initial_building_temp = 292.0
    initial_cell_temp = 290.0

    building = self._create_small_building(initial_temp=initial_building_temp)
    building.temp[cell_coordinates] = initial_cell_temp

    temperature_estimates = building.temp.copy()

    # Set ambient temperature to cell temp so there is no convection transfer.
    ambient_temperature = initial_cell_temp
    convection_coefficient = 12.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    temp_estimate = simulator._get_cv_temp_estimate(
        cell_coordinates,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    self.assertGreater(temp_estimate, initial_cell_temp)

  @parameterized.named_parameters(
      ("Corner", (1, 1)), ("Edge", (1, 2)), ("Interior", (2, 2))
  )
  def test_get_cv_temp_estimate_cell_decrease_from_conduction(
      self, cell_coordinates
  ):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    initial_building_temp = 292.0
    initial_cell_temp = 294.0

    building = self._create_small_building(initial_temp=initial_building_temp)
    building.temp[cell_coordinates] = initial_cell_temp

    temperature_estimates = building.temp.copy()

    # Set ambient temperature to cell temp so there is no convection transfer.
    ambient_temperature = initial_cell_temp
    convection_coefficient = 12.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    temp_estimate = simulator._get_cv_temp_estimate(
        cell_coordinates,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    self.assertLess(temp_estimate, initial_cell_temp)

  def test_get_cv_temp_estimate_convection_corner_increases_faster_than_edge(
      self,
  ):
    """Tests that the speed at which cell temps increase from convection.

    Corner cells are more exposed to air, so they should transfer/absorb heat
    through convection more rapidly than edges. All cells start at 292, so
    there should be no transfer through conduction. Ambient temperature is set
    higher (300). A corner and edge cell are chosen to get estimates. These
    estimates are compared to each other and to the initial temp.
    """
    # Set up simulation parameters.
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    initial_building_temp = 292.0

    building = self._create_small_building(initial_temp=initial_building_temp)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 300.0
    convection_coefficient = 12.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Get estimates for corner and edge cells.
    corner_temp_estimate = simulator._get_cv_temp_estimate(
        (1, 1),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )
    edge_temp_estimate = simulator._get_cv_temp_estimate(
        (1, 7),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    # Corner should gain heat faster than edge.
    with self.subTest("corner_gains_faster_than_edge"):
      self.assertGreater(corner_temp_estimate, edge_temp_estimate)

    # Both should be increasing.
    with self.subTest("both_increase"):
      self.assertGreater(edge_temp_estimate, initial_building_temp)

  @parameterized.named_parameters(
      ("traditional_CVs", (1, 1), (5, 1)),
      ("CVs_different_for_weird_floorplan", (1, 3), (2, 4)),
      ("bottom_non_traditional", (9, 3), (8, 4)),
  )
  def test_weird_building_get_cv_temp_estimate_convection_corner_increases(
      self, corner_cv, edge_cv
  ):
    """Tests that the speed at which cell temps increase from convection.

      Corner cells are more exposed to air, so they should transfer/absorb heat
      through convection more rapidly than edges. All cells start at 292, so
      there should be no transfer through conduction. Ambient temperature is set
      higher (300). A corner and edge cell are chosen to get estimates. These
      estimates are compared to each other and to the initial temp.

    Args:
      corner_cv: inds for a corner cv
      edge_cv: inds for an edge cv
    """
    # Set up simulation parameters.
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    initial_building_temp = 292.0

    building = self._create_weirdly_shaped_building(
        initial_temp=initial_building_temp
    )

    temperature_estimates = building.temp.copy()

    ambient_temperature = 300.0
    convection_coefficient = 12.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Get estimates for corner and edge cells.
    corner_temp_estimate = simulator._get_cv_temp_estimate(
        corner_cv,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )
    edge_temp_estimate = simulator._get_cv_temp_estimate(
        edge_cv,
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    # Corner should gain heat faster than edge.
    with self.subTest("corner_gains_faster_than_edge"):
      self.assertGreater(corner_temp_estimate, edge_temp_estimate)

    # Both should be increasing.
    with self.subTest("both_increase"):
      self.assertGreater(edge_temp_estimate, initial_building_temp)

  def test_get_cv_temp_estimate_convection_corner_decreases_faster_than_edge(
      self,
  ):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    initial_building_temp = 292.0

    building = self._create_small_building(initial_temp=initial_building_temp)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 285.0
    convection_coefficient = 12.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    corner_temp_estimate = simulator._get_cv_temp_estimate(
        (1, 1),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )
    edge_temp_estimate = simulator._get_cv_temp_estimate(
        (1, 7),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    # Corner should lose heat faster than edge.
    with self.subTest("corner_loses_faster_than_edge"):
      self.assertLess(corner_temp_estimate, edge_temp_estimate)

    # Both should be decreasing.
    with self.subTest("both_decrease"):
      self.assertLess(edge_temp_estimate, initial_building_temp)

  def test_update_temperature_estimates_changes_temperatures(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_small_building(
        initial_temp=292.0, match_diffusers=True
    )
    temperature_estimates = building.temp.copy()

    # Set last temperature to 0.0 to ensure all cells will have new temp
    # estimates.
    building.temp *= 0.0

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    simulator.update_temperature_estimates(
        temperature_estimates,
        ambient_temperature=70.0,
        convection_coefficient=12.0,
    )

    # loop over the floor plan including outside air
    for x in range(1, temperature_estimates.shape[0]):
      for y in range(1, temperature_estimates.shape[1]):
        with self.subTest(f"temp{x}_{y}"):
          self.assertNotEqual(temperature_estimates[x][y], 292.0)

  def test_update_temperature_estimates_return_value(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_small_building(initial_temp=292.0)
    temperature_estimates = building.temp.copy()

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    _, max_delta = simulator.update_temperature_estimates(
        temperature_estimates,
        ambient_temperature=292.0,
        convection_coefficient=12.0,
    )
    self.assertAlmostEqual(max_delta, 0.0, places=3)

  def test_finite_differences_timestep_does_not_converge(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 3000.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.01
    iteration_limit = 5
    iteration_warning = 3
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_small_building(initial_temp=292.0)

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    with self.assertLogs() as logs:
      converged = sim.finite_differences_timestep(
          ambient_temperature=0.0, convection_coefficient=12.0
      )

    self.assertFalse(
        converged, msg="finite_differences_timestep unexpectedly converged."
    )

    # Check logs for warning
    self.assertLen(
        [
            x
            for x in logs.output
            if "Max iteration count reached, max_delta = 0." in x
        ],
        1,
    )

  def test_step_sim_heating_scenario_avg_temps_increase(self):
    """Tests that the average temperature increases.

    Ambient temperatures are set high.
    """
    # Constant temp of 300C
    weather_controller = weather_controller_py.WeatherController(300.0, 300.0)

    time_step_sec = 300.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 292.0
    building = self._create_scenario_building(initial_temp=initial_temperature)

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    for _ in range(5):
      sim.step_sim()

    avg_temperatures = building.get_zone_average_temps()

    for temperature in avg_temperatures.values():
      self.assertGreater(temperature, initial_temperature)

  def test_step_sim_heating_scenario_zone_temperature_speeds(self):
    """Tests that certain zones heat faster than others.

    Ambient temperatures are set high. Corner zones should heat fastest,
    followed by edge zones, lastly the center zone.
    """
    # Constant temp of 300C
    weather_controller = weather_controller_py.WeatherController(300.0, 300.0)

    time_step_sec = 3000.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 292.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    for _ in range(5):
      sim.step_sim()

    avg_temperatures = building.get_zone_average_temps()

    corner_temp = avg_temperatures["room_1"]
    edge_temp = avg_temperatures["room_2"]
    interior_temp = avg_temperatures["room_5"]

    self.assertGreater(corner_temp, edge_temp)
    self.assertGreater(edge_temp, interior_temp)
    self.assertGreater(interior_temp, initial_temperature)

  def test_step_sim_heating_scenario_vavs_cools(self):
    """Tests that the vavs work to keep the building cool.

    Initial temperatures are set high. After a time step, thermostats
    should enter cooling mode and keep the building cool.
    """
    # Constant temp of 300C
    weather_controller = weather_controller_py.WeatherController(310.0, 310.0)

    time_step_sec = 3000.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 310.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    for _ in range(5):
      sim.step_sim()

    # Average building temperature should decrease.
    self.assertLess(building.temp.mean(), initial_temperature)

  def test_step_sim_cooling_scenario_vavs_heat(self):
    """Tests that the vavs work to keep the building warm.

    Initial temperatures are set low. After a time step, thermostats
    should enter heating mode and keep the building warm.
    """
    # Constant temp of 300C
    weather_controller = weather_controller_py.WeatherController(275.0, 275.0)

    time_step_sec = 3000.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 275.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    for _ in range(5):
      sim.step_sim()

    # Average building temperature should increase.
    self.assertGreater(building.temp.mean(), initial_temperature)

  def test_step_sim_increments_current_time(self):
    weather_controller = weather_controller_py.WeatherController(296.0, 296.0)
    time_step_sec = 300.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")
    expected_end_timestamp = pd.Timestamp("12-21-2012") + pd.Timedelta(
        1500, "s"
    )

    initial_temperature = 296.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    for _ in range(5):
      sim.step_sim()

    self.assertEqual(sim._current_timestamp, expected_end_timestamp)

  def test_step_sim_sets_boiler_return_water_temperature_sensor(self):
    weather_controller = weather_controller_py.WeatherController(296.0, 296.0)
    time_step_sec = 300.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 200.0
    expected_return_water_temperature = 301.895482

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    sim.step_sim()

    self.assertAlmostEqual(
        sim._hvac.boiler.return_water_temperature_sensor,
        expected_return_water_temperature,
        delta=1e-5,
    )

  def test_reward_info(self):
    weather_controller = weather_controller_py.WeatherController(296.0, 296.0)
    time_step_sec = 300.0
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("12-21-2012")

    initial_temperature = 200.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(
        initial_temp=initial_temperature, match_old_diffusers=True
    )

    hvac = self._create_scenario_hvac(
        zone_identifier=building._room_dict.keys()
    )

    sim = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    occupancy = step_function_occupancy.StepFunctionOccupancy(
        pd.Timedelta(9, unit="h"), pd.Timedelta(17, unit="h"), 10, 0.1
    )
    reward_info = sim.reward_info(occupancy)

    self.assertEqual(
        conversion_utils.pandas_to_proto_timestamp(sim._current_timestamp),
        reward_info.start_timestamp,
    )
    self.assertEqual(
        conversion_utils.pandas_to_proto_timestamp(
            sim._current_timestamp + pd.Timedelta(sim._time_step_sec, unit="s")
        ),
        reward_info.end_timestamp,
    )

    expected_zone_reward_infos = {}
    zone_identifier = [
        "room_1",
        "room_2",
        "room_3",
        "room_4",
        "room_5",
        "room_6",
        "room_7",
        "room_8",
        "room_9",
    ]
    for coords in zone_identifier:
      zone_id = conversion_utils.floor_plan_based_zone_identifier_to_id(coords)
      occupancy_value = occupancy.average_zone_occupancy(
          zone_id,
          sim._current_timestamp,
          sim._current_timestamp + pd.Timedelta(sim._time_step_sec, unit="s"),
      )
      air_flow_rate = sim._hvac.air_handler.air_flow_rate
      air_flow_rate_setpoint = sim._hvac.vavs[coords].max_air_flow_rate
      heating_setpoint, cooling_setpoint = (
          sim._hvac.vavs[coords]
          .thermostat.get_setpoint_schedule()
          .get_temperature_window(sim._current_timestamp)
      )
      zone_temperature = sim.building.get_zone_average_temps()[coords]

      expected_zone_info = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
          heating_setpoint_temperature=heating_setpoint,
          cooling_setpoint_temperature=cooling_setpoint,
          zone_air_temperature=zone_temperature,
          air_flow_rate_setpoint=air_flow_rate_setpoint,
          air_flow_rate=air_flow_rate,
          average_occupancy=occupancy_value,
      )
      expected_zone_reward_infos[zone_id] = expected_zone_info

    self.assertEqual(reward_info.zone_reward_infos, expected_zone_reward_infos)

    air_handler_reward_info = reward_info.air_handler_reward_infos[
        sim._hvac.air_handler.device_id()
    ]

    blower_electrical_energy_rate = (
        sim._hvac.air_handler.compute_intake_fan_energy_rate()
        + sim._hvac.air_handler.compute_exhaust_fan_energy_rate()
    )

    self.assertEqual(
        blower_electrical_energy_rate,
        air_handler_reward_info.blower_electrical_energy_rate,
    )

    recirculation_temp = sim.building.temp.mean()
    ambient_temp = sim._weather_controller.get_current_temp(
        sim._current_timestamp
    )
    air_conditioning_electrical_energy_rate = (
        sim._hvac.air_handler.compute_thermal_energy_rate(
            recirculation_temp, ambient_temp
        )
    )
    self.assertEqual(
        air_conditioning_electrical_energy_rate,
        air_handler_reward_info.air_conditioning_electrical_energy_rate,
    )

    boiler_reward_info = reward_info.boiler_reward_infos[
        sim._hvac.boiler.device_id()
    ]
    natural_gas_heating_energy_rate = (
        sim._hvac.boiler.compute_thermal_energy_rate(
            sim._hvac.boiler.return_water_temperature_sensor, ambient_temp
        )
    )
    self.assertAlmostEqual(
        natural_gas_heating_energy_rate,
        boiler_reward_info.natural_gas_heating_energy_rate,
        places=3,
    )

    pump_electrical_energy_rate = sim._hvac.boiler.compute_pump_power()
    self.assertEqual(
        pump_electrical_energy_rate,
        boiler_reward_info.pump_electrical_energy_rate,
    )

  def test_update_temperature_estimates_return_value_with_lwx(  # pylint: disable=line-too-long
      self,
  ):
    """Test that the temperature estimates are updated correctly with LWX"""

    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_small_building(
        initial_temp=292.0, include_radiative_heat_transfer=True
    )
    # temperature_estimates = building.temp.copy()

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    converged = simulator.finite_differences_timestep(
        ambient_temperature=292, convection_coefficient=12.0
    )

    self.assertTrue(
        converged,
        msg=(
            "finite_differences_timestep converged with radiative heat"
            " transfer."
        ),
    )

  def test_update_temperature_estimates_return_value_with_lwx_no_interior_walls(  # pylint: disable=line-too-long
      self,
  ):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp("2012-12-21")

    plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])

    building = self._create_small_building(
        initial_temp=292.0,
        include_radiative_heat_transfer=True,
        floor_plan=plan,
    )

    simulator = simulator_py.SimulatorFlexibleGeometries(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    converged = simulator.finite_differences_timestep(
        ambient_temperature=292, convection_coefficient=12.0
    )

    self.assertTrue(
        converged,
        msg=(
            "finite_differences_timestep converged with radiative heat"
            " transfer."
        ),
    )

  def test_interior_mass_temperatures_update(self):
    """Test that interior mass temperatures are updated during simulation."""
    simulator, building = self._create_simulator_and_building(
        convergence_threshold=0.001,
        iteration_limit=100,
        include_interior_mass=True,
    )

    # Store initial interior mass temperatures
    initial_interior_mass_temp = building.interior_mass_temp.copy()

    # Run a timestep with different ambient temperature to cause heat transfer
    converged = simulator.finite_differences_timestep(
        ambient_temperature=300.0, convection_coefficient=12.0
    )

    self.assertTrue(converged)

    # Check that interior mass temperatures have increased
    # Use NumPy boolean indexing with mask for cleaner array comparison
    temps_increased = np.any(
        building.interior_mass_temp[building.interior_mass_mask]
        > initial_interior_mass_temp[building.interior_mass_mask]
    )

    self.assertTrue(
        temps_increased,
        msg="Interior mass temperatures should increase during simulation",
    )

  def test_interior_mass_convergence(self):
    """Test that simulation with interior mass converges."""
    simulator, _ = self._create_simulator_and_building(
        convergence_threshold=0.001, iteration_limit=100
    )

    # Test convergence with same temperature (should converge quickly)
    converged = simulator.finite_differences_timestep(
        ambient_temperature=292.0, convection_coefficient=12.0
    )

    self.assertTrue(
        converged,
        msg=(
            "Simulation with interior mass should converge when ambient temp"
            " equals initial temp"
        ),
    )

  def test_interior_mass_affects_heat_transfer(self):
    """Test that interior mass affects heat transfer in the building."""
    # Building without interior mass
    simulator_no_mass, building_no_mass = self._create_simulator_and_building(
        convergence_threshold=0.001,
        iteration_limit=100,
        include_interior_mass=False,
    )
    # Building with interior mass
    simulator_with_mass, building_with_mass = (
        self._create_simulator_and_building(
            convergence_threshold=0.001,
            iteration_limit=100,
            include_interior_mass=True,
        )
    )

    # Run simulation with higher ambient temperature
    ambient_temp = 300.0
    convection_coeff = 12.0

    simulator_no_mass.finite_differences_timestep(
        ambient_temperature=ambient_temp,
        convection_coefficient=convection_coeff,
    )
    simulator_with_mass.finite_differences_timestep(
        ambient_temperature=ambient_temp,
        convection_coefficient=convection_coeff,
    )

    # Compare average air temperatures
    avg_temp_no_mass = np.mean(
        building_no_mass.temp[
            building_with_mass.floor_plan
            == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
        ]
    )
    avg_temp_with_mass = np.mean(
        building_with_mass.temp[
            building_with_mass.floor_plan
            == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
        ]
    )

    # Building with interior mass should heat up differently due to thermal
    #  inertia
    # The exact relationship depends on material properties, but they should
    # differ
    self.assertGreater(
        avg_temp_no_mass - avg_temp_with_mass,
        0,
        msg=(
            "Average temperature without interior mass should be greater than"
            " with interior mass"
        ),
    )

  def test_interior_mass_convergence_with_lwx(self):
    """Test that simulation with interior mass converges with LWX
    (longwave interior radiative heat transfer)."""
    simulator, _ = self._create_simulator_and_building(
        convergence_threshold=0.001,
        iteration_limit=100,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
    )

    # Test convergence with same temperature (should converge quickly)
    converged = simulator.finite_differences_timestep(
        ambient_temperature=292.0, convection_coefficient=12.0
    )

    self.assertTrue(
        converged,
        msg="converged.",
    )


if __name__ == "__main__":
  absltest.main()
