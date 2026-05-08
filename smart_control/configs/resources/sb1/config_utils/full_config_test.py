"""Tests for Building 'SB-1' config files used in simulation experiments."""

from unittest import mock
import warnings

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_buildings.smart_control.configs.resources.sb1.config_utils import conftest
from smart_buildings.smart_control.configs.resources.sb1.config_utils import data_files
from smart_buildings.smart_control.configs.resources.sb1.config_utils import full_config
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment  # pylint: disable=unused-import
from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret
from smart_buildings.smart_control.simulator import building
from smart_buildings.smart_control.simulator import hvac_floorplan_based
from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy
from smart_buildings.smart_control.simulator import setpoint_schedule
from smart_buildings.smart_control.simulator import simulator_building
from smart_buildings.smart_control.simulator import stochastic_convection_simulator
from smart_buildings.smart_control.simulator import tf_simulator
from smart_buildings.smart_control.simulator import weather_controller
from smart_buildings.smart_control.utils import observation_normalizer

FloorPlanBasedBuilding = building.FloorPlanBasedBuilding
FloorPlanBasedHvac = hvac_floorplan_based.FloorPlanBasedHvac
HybridActionEnvironment = hybrid_action_environment.HybridActionEnvironment
RandomizedOccupancy = randomized_arrival_departure_occupancy.RandomizedArrivalDepartureOccupancy  # pylint: disable=line-too-long
ReplayWeatherController = weather_controller.ReplayWeatherController
SetpointEnergyCarbonRegretFunction = setpoint_energy_carbon_regret.SetpointEnergyCarbonRegretFunction  # pylint: disable=line-too-long
SimulatorBuilding = simulator_building.SimulatorBuilding
StandardScoreObservationNormalizer = observation_normalizer.StandardScoreObservationNormalizer  # pylint: disable=line-too-long
TFSimulator = tf_simulator.TFSimulator

# environment has lots of info logs, which cause "Test log too large" errors,
# and prevent us from seeing the reasons for test failures,
# so disable info level logging:
logging.set_verbosity(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning)


class EnvironmentConfigTest(parameterized.TestCase):

  @classmethod
  def _create_environment(cls):
    return environment.Environment()

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    full_config.set_gin_config()

    # env takes a long time to load, so we are doing it once at the class level:
    cls.env = cls._create_environment()

    cls.sim_building = cls.env.building
    cls.occupancy = cls.sim_building.occupancy
    cls.sim = cls.sim_building.simulator

    cls.building = cls.sim.building
    cls.hvac = cls.sim.hvac
    cls.wc = cls.sim.weather_controller

    cls.expected_action_names = conftest.ACTION_NAMES
    cls.expected_action_normalizer_names = conftest.ACTION_NORMALIZER_NAMES
    cls.expected_default_action_values = conftest.DEFAULT_ACTION_VALUES

  def test_environment(self):
    self.assertIsInstance(self.env, environment.Environment)

  def test_properties(self):
    self.assertEqual(self.env.time_step_mins, 5)
    self.assertEqual(
        self.env.start_timestamp, pd.Timestamp(full_config.START_TIMESTAMP)
    )
    self.assertEqual(self.env.num_days_in_episode, full_config.N_DAYS)
    self.assertEqual(self.env.discount_factor, 0.9)

  #
  # SIM
  #

  def test_weather_controller(self):
    wc = self.wc
    self.assertIsInstance(wc, ReplayWeatherController)

    self.assertEqual(wc.convection_coefficient, 100.0)
    self.assertEqual(wc.csv_filepath, data_files.get_weather_data_filepath())
    self.assertEqual(wc.min_time, pd.Timestamp("2024-01-01 01:00:00", tz="UTC"))
    self.assertEqual(wc.max_time, pd.Timestamp("2024-12-30 09:00:00", tz="UTC"))

    with self.subTest("weather_data"):
      df = wc.weather_df
      self.assertIsInstance(df, pd.DataFrame)
      self.assertEqual(df.shape, (8529, 15))
      self.assertEqual(df.columns.tolist(), conftest.WEATHER_COLUMNS)

    with self.subTest("time_range"):
      timestamp = pd.Timestamp(full_config.START_TIMESTAMP)
      self.assertEqual(wc.get_current_temp(timestamp), 285.15)

      future_timestamp = timestamp + pd.Timedelta(days=full_config.N_DAYS + 1)
      self.assertEqual(wc.get_current_temp(future_timestamp), 289.15)

  def test_reward_function(self):
    reward_function = self.env.reward_function
    self.assertIsInstance(reward_function, SetpointEnergyCarbonRegretFunction)
    self.assertEqual(reward_function.energy_cost_weight, 0.2)
    self.assertEqual(reward_function.carbon_emission_weight, 0.2)
    self.assertEqual(reward_function.productivity_weight, 0.6)

  def test_building(self):
    bldg = self.building
    self.assertIsInstance(bldg, building.FloorPlanBasedBuilding)
    self.assertEqual(bldg.floor_plan_filepath, data_files.FLOOR_PLAN_FILEPATH)
    self.assertEqual(bldg.zone_map_filepath, data_files.FLOOR_PLAN_FILEPATH)
    self.assertEqual(bldg.cv_size_cm, 10)
    self.assertEqual(bldg.floor_height_cm, 300.0)
    self.assertEqual(bldg.initial_temp, 294.0)

    with self.subTest("material_properties"):
      self.assertEqual(
          bldg.inside_air_properties,
          building.MaterialProperties(
              conductivity=50.0,
              heat_capacity=700.0,
              density=1.0,
          ),
      )
      self.assertEqual(
          bldg.inside_wall_properties,
          building.MaterialProperties(
              conductivity=50.0,
              heat_capacity=700.0,
              density=1.0,
          ),
      )
      self.assertEqual(
          bldg.building_exterior_properties,
          building.MaterialProperties(
              conductivity=0.05,
              heat_capacity=700.0,
              density=1.0,
          ),
      )

  def test_convection_simulator(self):
    simulator = self.building.convection_simulator
    self.assertIsInstance(
        simulator, stochastic_convection_simulator.StochasticConvectionSimulator
    )
    self.assertEqual(simulator.p, 0.5)
    self.assertEqual(simulator.distance, 25)

  def test_building_zones(self):
    df = self.sim_building.zones_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = []
    for i in range(1, 127):
      # FYI: right now, the floorplan-based hvac does not support dynamic floor
      # assignments. however in the future, once supported, we can assign them
      # using logic like: `floor = 1 if i <= 53 else 2`
      floor = 0
      expected_records.append({
          "zone_id": f"room_{i}",
          "building_id": "US-SIM-001",
          "zone_description": "Simulated zone",
          "area": 0.0,
          "devices": [f"VAV_{i}"],
          "zone_type": "ROOM",
          "floor": floor,
      })
    self.assertEqual(df.to_dict("records"), expected_records)

  def test_building_devices(self):
    df = self.sim_building.devices_df
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 128)

    vav_ids = [f"VAV_{i}" for i in range(1, 127)]
    expected_device_ids = ["hws", "ahs"] + vav_ids
    self.assertCountEqual(df["device_id"].tolist(), expected_device_ids)

  def test_hvac(self):
    hvac = self.hvac
    self.assertIsInstance(hvac, hvac_floorplan_based.FloorPlanBasedHvac)

    with self.subTest("properties"):
      self.assertEqual(hvac.vav_max_air_flow_rate, 2.0)
      self.assertEqual(hvac.vav_reheat_max_water_flow_factor, 0.03)

    with self.subTest("setpoint_schedule"):
      schedule = hvac.schedule
      self.assertIsInstance(schedule, setpoint_schedule.SetpointSchedule)
      self.assertEqual(schedule.morning_start_hour, 6)
      self.assertEqual(schedule.evening_start_hour, 19)
      self.assertEqual(schedule.comfort_temp_window, (294, 297))
      self.assertEqual(schedule.eco_temp_window, (289, 298))
      self.assertEqual(schedule.time_zone, "US/Pacific")

  def test_simulator(self):
    sim = self.sim
    self.assertIsInstance(sim, TFSimulator)
    self.assertIsInstance(sim.building, FloorPlanBasedBuilding)
    self.assertIsInstance(sim.hvac, FloorPlanBasedHvac)
    self.assertIsInstance(sim.weather_controller, ReplayWeatherController)
    self.assertEqual(sim.time_step_sec, 300)
    self.assertEqual(sim.convergence_threshold, 0.1)
    self.assertEqual(sim.iteration_limit, 100)
    self.assertEqual(sim.iteration_warning, 20)
    self.assertEqual(
        sim.start_timestamp, pd.Timestamp(full_config.START_TIMESTAMP)
    )

  def test_occupancy(self):
    self.assertIsInstance(self.occupancy, RandomizedOccupancy)

    self.assertEqual(self.occupancy.zone_assignment, 1)
    self.assertEqual(self.occupancy.earliest_expected_arrival_hour, 7)
    self.assertEqual(self.occupancy.latest_expected_arrival_hour, 12)
    self.assertEqual(self.occupancy.earliest_expected_departure_hour, 13)
    self.assertEqual(self.occupancy.latest_expected_departure_hour, 19)
    self.assertEqual(self.occupancy.step_size, pd.Timedelta(300, unit="second"))
    self.assertEqual(self.occupancy.time_zone, "US/Pacific")

  def test_simulator_building(self):
    self.assertIsInstance(self.sim_building, SimulatorBuilding)
    self.assertIsInstance(self.sim_building.simulator, TFSimulator)
    self.assertIsInstance(self.sim_building.occupancy, RandomizedOccupancy)

  #
  # ENV
  #

  def test_observation_config(self):
    normalizer = self.env.observation_normalizer
    self.assertIsInstance(normalizer, StandardScoreObservationNormalizer)

    observation_names = normalizer.normalization_constants.keys()
    self.assertLen(observation_names, 54)
    with self.subTest("outside_air_temperature_sensor"):
      self.assertIn("outside_air_temperature_sensor", observation_names)

  def test_action_names(self):
    self.assertCountEqual(self.env.action_names, self.expected_action_names)

  def test_action_normalizers(self):
    action_normalizers = self.env.action_config.action_normalizers
    self.assertCountEqual(
        action_normalizers.keys(), self.expected_action_normalizer_names
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="supply_water_setpoint",
          action_name="supply_water_setpoint",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=310.0,
          max_native_value=350.0,
      ),
      dict(
          testcase_name="differential_pressure",
          action_name="differential_pressure",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=20.0,
      ),
      dict(
          testcase_name="ahu_1_supply_air_temperature_setpoint",
          action_name="ahu_1_supply_air_temperature_setpoint",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=285.0,
          max_native_value=305.0,
      ),
      dict(
          testcase_name="ahu_1_static_pressure_setpoint",
          action_name="ahu_1_static_pressure_setpoint",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=20000.0,
      ),
      dict(
          testcase_name="ahu_2_supply_air_temperature_setpoint",
          action_name="ahu_2_supply_air_temperature_setpoint",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=285.0,
          max_native_value=305.0,
      ),
      dict(
          testcase_name="ahu_2_static_pressure_setpoint",
          action_name="ahu_2_static_pressure_setpoint",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=20000.0,
      ),
  )
  def test_action_normalizer_values(
      self,
      action_name,
      min_normalized_value,
      max_normalized_value,
      min_native_value,
      max_native_value,
  ):
    normalizer = self.env.action_config.action_normalizers[action_name]
    self.assertEqual(normalizer.min_normalized_value, min_normalized_value)
    self.assertEqual(normalizer.max_normalized_value, max_normalized_value)
    self.assertEqual(normalizer.min_native_value, min_native_value)
    self.assertEqual(normalizer.max_native_value, max_native_value)

  def test_default_actions(self):
    self.assertSequenceAlmostEqual(
        self.env.default_policy_values.numpy().tolist(),
        self.expected_default_action_values,
        places=5,
    )

  def test_action_fields_df(self):
    df = self.env.action_fields_df
    self.assertIsInstance(df, pd.DataFrame)
    self.assertCountEqual(df.to_dict("records"), conftest.ACTION_FIELDS)


class HybridActionEnvironmentConfigTest(EnvironmentConfigTest):

  @classmethod
  def _create_environment(cls):
    return HybridActionEnvironment()  # pylint:disable=no-value-for-parameter these are set by the gin config!

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.expected_action_names = conftest.HYBRID_ACTION_NAMES
    cls.expected_action_normalizer_names = conftest.HYBRID_ACTION_NORMALIZER_NAMES  # pylint: disable=line-too-long
    cls.expected_default_action_values = conftest.HYBRID_DEFAULT_ACTION_VALUES

  def test_environment(self):
    self.assertIsInstance(self.env, HybridActionEnvironment)

  @parameterized.named_parameters(
      dict(
          testcase_name="supervisor_run_command",
          action_name="supervisor_run_command",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=1.0,
      ),
      dict(
          testcase_name="ahu_1_supervisor_run_command",
          action_name="ahu_1_supervisor_run_command",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=1.0,
      ),
      dict(
          testcase_name="ahu_2_supervisor_run_command",
          action_name="ahu_2_supervisor_run_command",
          min_normalized_value=-1.0,
          max_normalized_value=1.0,
          min_native_value=0.0,
          max_native_value=1.0,
      ),
  )
  def test_discrete_action_normalizer_values(
      self,
      action_name,
      min_normalized_value,
      max_normalized_value,
      min_native_value,
      max_native_value,
  ):
    normalizer = self.env.action_config.action_normalizers[action_name]
    self.assertEqual(normalizer.min_normalized_value, min_normalized_value)
    self.assertEqual(normalizer.max_normalized_value, max_normalized_value)
    self.assertEqual(normalizer.min_native_value, min_native_value)
    self.assertEqual(normalizer.max_native_value, max_native_value)

  def test_action_fields_df(self):
    df = self.env.action_fields_df
    self.assertIsInstance(df, pd.DataFrame)
    self.assertCountEqual(df.to_dict("records"), conftest.HYBRID_ACTION_FIELDS)


class OverrideConfigTest(parameterized.TestCase):

  def test_override_config_values(self):
    full_config.set_gin_config(
        productivity_weight=0.1,
        energy_cost_weight=0.2,
        carbon_emission_weight=0.7,
        earliest_expected_arrival_hour=5,
        latest_expected_arrival_hour=10,
        earliest_expected_departure_hour=15,
        latest_expected_departure_hour=20,
    )
    env = environment.Environment()

    with self.subTest("reward_weights"):
      reward_function = env.reward_function
      self.assertEqual(reward_function.productivity_weight, 0.1)
      self.assertEqual(reward_function.energy_cost_weight, 0.2)
      self.assertEqual(reward_function.carbon_emission_weight, 0.7)

    with self.subTest("occupancy_hours"):
      occupancy = env.building.occupancy
      self.assertEqual(occupancy.earliest_expected_arrival_hour, 5)
      self.assertEqual(occupancy.latest_expected_arrival_hour, 10)
      self.assertEqual(occupancy.earliest_expected_departure_hour, 15)
      self.assertEqual(occupancy.latest_expected_departure_hour, 20)

  def test_weather_data_year_corresponds_with_timestamp(self):
    with mock.patch.object(
        data_files, "get_weather_data_filepath", return_value="example.csv"
    ) as mock_get_weather_data_filepath:
      full_config.set_gin_config(
          start_timestamp="2023-05-10 08:00:00",
          weather_data_filepath=None,
      )
      mock_get_weather_data_filepath.assert_called_once_with(2023)


if __name__ == "__main__":
  absltest.main()
