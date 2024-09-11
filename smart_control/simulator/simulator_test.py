"""Tests for simulator.

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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac as hvac_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import simulator as simulator_py
from smart_control.simulator import step_function_occupancy
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.utils import conversion_utils


class SimulatorTest(parameterized.TestCase):

  def _create_small_building(self, initial_temp):
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
        'boiler_id',
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

    zone_coordinates = [(0, 0), (1, 0)]

    hvac = hvac_py.Hvac(
        zone_coordinates, air_handler, boiler, schedule, 0.45, 0.02
    )
    return hvac

  def _create_scenario_building(self, initial_temp):
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

  def _create_scenario_hvac(self):
    """Returns hvac matching zones for scenario building."""
    reheat_water_setpoint = 350
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    boiler = boiler_py.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        'boiler_id',
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

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
    )

    zone_coordinates = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    vav_max_air_flow_rate = 0.45
    vav_reheat_max_water_flow_rate = 0.02
    hvac = hvac_py.Hvac(
        zone_coordinates,
        air_handler,
        boiler,
        schedule,
        vav_max_air_flow_rate,
        vav_reheat_max_water_flow_rate,
    )
    return hvac

  def test_init(self):
    building = mock.create_autospec(building_py.Building)
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('2012-12-21')

    simulator = simulator_py.Simulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    self.assertEqual(simulator._building, building)
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
    start_timestamp = pd.Timestamp('2012-12-21')

    simulator = simulator_py.Simulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    simulator._building.temp[2][2] += 10.0
    simulator._building.temp[0][3] += 10.0
    simulator._building.input_q[2][2] = 1000.0
    simulator._building.input_q[0][3] = 1000.0

    simulator.hvac.boiler._return_water_temperature_sensor += 10.0
    simulator.hvac.boiler._water_pump_differential_head += 100.0
    simulator.hvac.boiler._reheat_water_setpoint += 2.0

    simulator.hvac.air_handler._air_flow_rate += 0.1
    simulator.hvac.air_handler._fan_differential_pressure = 0.1

    for coord in simulator.hvac._zone_coordinates:
      vav = simulator.hvac.vavs[coord]
      vav.thermostat._setpoint_schedule.morning_start_hour += 1.0
      vav.thermostat._setpoint_schedule.comfort_temp_window = (280, 310)
      vav.max_air_flow_rate += 0.1
      vav._reheat_max_water_flow_rate += 0.1

    simulator._current_timestamp += pd.Timedelta(360.0, unit='seconds')
    simulator.reset()
    self.assertEqual(simulator._building, building)
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
    self.assertEqual(simulator._building.temp[2][2], initial_temp)
    self.assertEqual(simulator._building.temp[0][3], initial_temp)
    self.assertEqual(simulator._building.input_q[2][2], 0)
    self.assertEqual(simulator._building.input_q[0][3], 0)

  def test_get_cv_temp_estimate_cell_no_change(self):
    """This tests that temperatures don't change in stable conditions.

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
    start_timestamp = pd.Timestamp('2012-12-21')

    # Control Volume matrix shape: (21, 10)
    building = self._create_small_building(initial_temp=292.0)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 292.0
    convection_coefficient = 12.0

    expected_temp_estimate = 292.0

    simulator = simulator_py.Simulator(
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
    for x in range(21):
      for y in range(10):
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
            msg=f'Cell ({x}, {y}) changed unexpectedly.',
            delta=1e-5,
        )

  @parameterized.named_parameters(
      ('Corner', (0, 0)), ('Edge', (0, 2)), ('Interior', (1, 1))
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
    start_timestamp = pd.Timestamp('2012-12-21')

    initial_building_temp = 292.0
    initial_cell_temp = 290.0

    building = self._create_small_building(initial_temp=initial_building_temp)
    building.temp[cell_coordinates] = initial_cell_temp

    temperature_estimates = building.temp.copy()

    # Set ambient temperature to cell temp so there is no convection transfer.
    ambient_temperature = initial_cell_temp
    convection_coefficient = 12.0

    simulator = simulator_py.Simulator(
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
      ('Corner', (0, 0)), ('Edge', (0, 2)), ('Interior', (1, 1))
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
    start_timestamp = pd.Timestamp('2012-12-21')

    initial_building_temp = 292.0
    initial_cell_temp = 294.0

    building = self._create_small_building(initial_temp=initial_building_temp)
    building.temp[cell_coordinates] = initial_cell_temp

    temperature_estimates = building.temp.copy()

    # Set ambient temperature to cell temp so there is no convection transfer.
    ambient_temperature = initial_cell_temp
    convection_coefficient = 12.0

    simulator = simulator_py.Simulator(
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
    start_timestamp = pd.Timestamp('2012-12-21')

    initial_building_temp = 292.0

    building = self._create_small_building(initial_temp=initial_building_temp)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 300.0
    convection_coefficient = 12.0

    simulator = simulator_py.Simulator(
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
        (0, 0),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )
    edge_temp_estimate = simulator._get_cv_temp_estimate(
        (0, 2),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    # Corner should gain heat faster than edge.
    self.assertGreater(corner_temp_estimate, edge_temp_estimate)

    # Both should be increasing.
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
    start_timestamp = pd.Timestamp('2012-12-21')

    initial_building_temp = 292.0

    building = self._create_small_building(initial_temp=initial_building_temp)

    temperature_estimates = building.temp.copy()

    ambient_temperature = 285.0
    convection_coefficient = 12.0

    simulator = simulator_py.Simulator(
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
        (0, 0),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )
    edge_temp_estimate = simulator._get_cv_temp_estimate(
        (0, 2),
        temperature_estimates,
        ambient_temperature,
        convection_coefficient,
    )

    # Corner should lose heat faster than edge.
    self.assertLess(corner_temp_estimate, edge_temp_estimate)

    # Both should be decreasing.
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
    start_timestamp = pd.Timestamp('2012-12-21')

    building = self._create_small_building(initial_temp=292.0)
    temperature_estimates = building.temp.copy()

    # Set last temperature to 0.0 to ensure all cells will have new temp
    # estimates.
    building.temp *= 0.0

    simulator = simulator_py.Simulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    temperature_estimates, _ = simulator.update_temperature_estimates(
        temperature_estimates,
        ambient_temperature=210.0,
        convection_coefficient=12.0,
    )

    for x in range(temperature_estimates.shape[0]):
      for y in range(temperature_estimates.shape[1]):
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
    start_timestamp = pd.Timestamp('2012-12-21')

    building = self._create_small_building(initial_temp=292.0)
    temperature_estimates = building.temp.copy()

    simulator = simulator_py.Simulator(
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

  def test_finite_differences_timestep_converges_with_warning(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 3000.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.01
    iteration_limit = 100
    iteration_warning = 5
    start_timestamp = pd.Timestamp('2012-12-21')

    building = self._create_small_building(initial_temp=292.0)

    sim = simulator_py.Simulator(
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

    self.assertTrue(
        converged,
        msg='finite_differences_timestep unexpectedly failed to converge.',
    )

    # Check logs for warning
    self.assertLen(
        [
            x
            for x in logs.output
            if x.endswith('Step 4, not converged in 5 steps, max_delta = 0.029')
        ],
        1,
    )

  def test_finite_differences_timestep_does_not_converge(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 3000.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.01
    iteration_limit = 5
    iteration_warning = 3
    start_timestamp = pd.Timestamp('2012-12-21')

    building = self._create_small_building(initial_temp=292.0)

    sim = simulator_py.Simulator(
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
        converged, msg='finite_differences_timestep unexpectedly converged.'
    )

    # Check logs for warning
    self.assertLen(
        [
            x
            for x in logs.output
            if x.endswith('Max iteration count reached, max_delta = 0.029')
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 292.0
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 292.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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

    corner_temp = avg_temperatures[(0, 0)]
    edge_temp = avg_temperatures[(0, 1)]
    interior_temp = avg_temperatures[(1, 1)]

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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 310.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 275.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')
    expected_end_timestamp = pd.Timestamp('12-21-2012') + pd.Timedelta(
        1500, 's'
    )

    initial_temperature = 296.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 200.0
    expected_return_water_temperature = 301.895482

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
    hvac = self._create_scenario_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 10
    start_timestamp = pd.Timestamp('12-21-2012')

    initial_temperature = 200.0

    # Building is 3x3 zones.
    building = self._create_scenario_building(initial_temp=initial_temperature)

    sim = simulator_py.Simulator(
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
        pd.Timedelta(9, unit='h'), pd.Timedelta(17, unit='h'), 10, 0.1
    )
    reward_info = sim.reward_info(occupancy)

    self.assertEqual(
        conversion_utils.pandas_to_proto_timestamp(sim._current_timestamp),
        reward_info.start_timestamp,
    )
    self.assertEqual(
        conversion_utils.pandas_to_proto_timestamp(
            sim._current_timestamp + pd.Timedelta(sim._time_step_sec, unit='s')
        ),
        reward_info.end_timestamp,
    )

    expected_zone_reward_infos = {}
    zone_coordinates = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for coords in zone_coordinates:
      zone_id = conversion_utils.zone_coordinates_to_id(coords)
      occupancy_value = occupancy.average_zone_occupancy(
          zone_id,
          sim._current_timestamp,
          sim._current_timestamp + pd.Timedelta(sim._time_step_sec, unit='s'),
      )
      air_flow_rate = sim._hvac.air_handler.air_flow_rate
      air_flow_rate_setpoint = sim._hvac.vavs[coords].max_air_flow_rate
      heating_setpoint, cooling_setpoint = (
          sim._hvac.vavs[coords]
          .thermostat.get_setpoint_schedule()
          .get_temperature_window(sim._current_timestamp)
      )
      zone_temperature = sim._building.get_zone_average_temps()[coords]

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

    recirculation_temp = sim._building.temp.mean()
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


if __name__ == '__main__':
  absltest.main()
