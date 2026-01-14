"""More tests for the environment, to ensure the LLM agent can use it."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
import pandas as pd

from smart_buildings.smart_control.environment import conftest
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.models import base_building
from smart_buildings.smart_control.models import base_reward_function
from smart_buildings.smart_control.utils import building_image_generator
from smart_buildings.smart_control.utils import controller_writer
from smart_buildings.smart_control.utils import observation_normalizer

HybridActionEnvironment = hybrid_action_environment.HybridActionEnvironment


class LLMEnvironmentTest(absltest.TestCase):
  """Ensures the environment has what it needs for an LLM agent use case."""

  def setUp(self):
    super().setUp()
    self.env = conftest.create_environment()

  def test_initialization(self):
    self.assertIsInstance(self.env, environment.Environment)

    with self.subTest(name="building"):
      self.assertIsInstance(self.env.building, base_building.BaseBuilding)

    with self.subTest(name="reward_function"):
      self.assertIsInstance(
          self.env.reward_function, base_reward_function.BaseRewardFunction
      )

    with self.subTest(name="observation_normalizer"):
      self.assertIsInstance(
          self.env.observation_normalizer,
          observation_normalizer.StandardScoreObservationNormalizer,
      )

    with self.subTest(name="action_config"):
      self.assertIsInstance(self.env.action_config, environment.ActionConfig)

    with self.subTest(name="default_actions"):
      self.assertEmpty(self.env.default_policy_values)

  def test_properties(self):
    with self.subTest(name="step_count"):
      self.assertEqual(self.env.step_count, 0)

    with self.subTest(name="time_step_mins"):
      self.assertEqual(self.env.time_step_mins, 5)

    with self.subTest(name="time_zone"):
      self.assertEqual(self.env.time_zone, "US/Pacific")

    with self.subTest(name="current_simulation_timestamp"):
      ts = self.env.current_simulation_timestamp
      self.assertIsNone(ts.tz)
      self.assertEqual(ts, pd.Timestamp("2021-06-07 12:00:01"))

    with self.subTest(name="current_local_timestamp"):
      ts = self.env.current_local_timestamp
      self.assertEqual(ts, pd.Timestamp("2021-06-07 12:00:01", tz="US/Pacific"))

  def test_building_devices(self):
    df = self.env.building.devices_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            "device_id": "air_handler_1",
            "namespace": "",
            "code": "",
            "zone_id": "zone_1",
            "device_type": "AHU",
            "observable_fields": ["measurement_1"],
            "action_fields": [
                "setpoint_1",
                "supervisor_run_command",
                "supply_air_heating_temperature_setpoint",
            ],
            "observable_field_types": {"measurement_1": "VALUE_CONTINUOUS"},
            "action_field_types": {
                "setpoint_1": "VALUE_CONTINUOUS",
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_air_heating_temperature_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "boiler_1",
            "namespace": "",
            "code": "",
            "zone_id": "zone_1",
            "device_type": "BLR",
            "observable_fields": ["measurement_2"],
            "action_fields": [
                "setpoint_2",
                "setpoint_3",
                "setpoint_4",
                "supervisor_run_command",
                "supply_water_setpoint",
            ],
            "observable_field_types": {"measurement_2": "VALUE_CONTINUOUS"},
            "action_field_types": {
                "setpoint_2": "VALUE_CONTINUOUS",
                "setpoint_3": "VALUE_CONTINUOUS",
                "setpoint_4": "VALUE_CONTINUOUS",
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_water_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "air_handler_2",
            "namespace": "",
            "code": "",
            "zone_id": "zone_2",
            "device_type": "AHU",
            "observable_fields": ["measurement_3"],
            "action_fields": [
                "supervisor_run_command",
                "supply_air_heating_temperature_setpoint",
            ],
            "observable_field_types": {"measurement_3": "VALUE_CONTINUOUS"},
            "action_field_types": {
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_air_heating_temperature_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "vav_1",
            "namespace": "",
            "code": "",
            "zone_id": "zone_2",
            "device_type": "VAV",
            "observable_fields": ["measurement_4"],
            "action_fields": ["setpoint_5"],
            "observable_field_types": {"measurement_4": "VALUE_CONTINUOUS"},
            "action_field_types": {"setpoint_5": "VALUE_CONTINUOUS"},
        },
    ]
    self.assertEqual(df.to_dict("records"), expected_records)

  def test_building_zones(self):
    df = self.env.building.zones_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            "zone_id": "zone_1",
            "building_id": "SimpleBuilding",
            "zone_description": "zone_1",
            "area": 0.0,
            "devices": ["air_handler_1", "boiler_1"],
            "zone_type": "UNDEFINED",
            "floor": 0,
        },
        {
            "zone_id": "zone_2",
            "building_id": "SimpleBuilding",
            "zone_description": "zone_2",
            "area": 0.0,
            "devices": ["air_handler_2", "vav_1"],
            "zone_type": "UNDEFINED",
            "floor": 0,
        },
    ]
    self.assertEqual(df.to_dict("records"), expected_records)

  def test_action_fields_df(self):
    self.assertIsInstance(self.env.action_fields_df, pd.DataFrame)
    records = self.env.action_fields_df.to_dict("records")
    expected_records = [
        {
            "field_id": "air_handler_1_supply_air_heating_temperature_setpoint",
            "device_id": "air_handler_1",
            "device_type": "AHU",
            "zone_id": "zone_1",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 295.0,
            "max_normalized_value": 1.0,
            "min_native_value": 285.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "boiler_1_supply_water_setpoint",
            "device_id": "boiler_1",
            "device_type": "BLR",
            "zone_id": "zone_1",
            "setpoint_name": "supply_water_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 350.0,
            "max_normalized_value": 1.0,
            "min_native_value": 310.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "air_handler_2_supply_air_heating_temperature_setpoint",
            "device_id": "air_handler_2",
            "device_type": "AHU",
            "zone_id": "zone_2",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 295.0,
            "max_normalized_value": 1.0,
            "min_native_value": 285.0,
            "min_normalized_value": -1.0,
        },
    ]
    self.assertCountEqual(records, expected_records)

  def test_step(self):
    self.env.reset()
    with self.subTest(name="wants normalized action values"):
      self.assertEqual(self.env.step_count, 0)
      self.env.step([0, 0, 0])  # normalized action values
      self.assertEqual(self.env.step_count, 1)

  def test_observations(self):
    n_device_measurements = 4  # see all "_measurement" in conftest.LAYOUT
    n_auxiliary_measurements = 7
    n_observations = n_device_measurements + n_auxiliary_measurements
    self.assertEqual(
        self.env.observation_spec(),
        conftest.create_observation_spec(n_observations),
    )

  def test_actions(self):
    self.assertEqual(
        self.env.action_spec(), conftest.create_action_spec(n_continuous=3)
    )
    self.assertSequenceEqual(
        self.env.action_names,
        [
            "air_handler_1_supply_air_heating_temperature_setpoint",
            "boiler_1_supply_water_setpoint",
            "air_handler_2_supply_air_heating_temperature_setpoint",
        ],
    )


class LLMHybridActionEnvironmentTest(absltest.TestCase):
  """Ensures the environment has what it needs for an LLM agent use case."""

  def setUp(self):
    super().setUp()
    self.env = conftest.create_hybrid_action_environment(
        layout=conftest.DEMO_LAYOUT
    )

  def test_initialization(self):
    self.assertIsInstance(self.env, HybridActionEnvironment)
    with self.subTest(name="building"):
      self.assertIsInstance(self.env.building, base_building.BaseBuilding)

    with self.subTest(name="reward_function"):
      self.assertIsInstance(
          self.env.reward_function, base_reward_function.BaseRewardFunction
      )

    with self.subTest(name="observation_normalizer"):
      self.assertIsInstance(
          self.env.observation_normalizer,
          observation_normalizer.StandardScoreObservationNormalizer,
      )

    with self.subTest(name="action_config"):
      self.assertIsInstance(self.env.action_config, environment.ActionConfig)

  def test_building_devices(self):
    df = self.env.building.devices_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            "device_id": "air_handler_1",
            "namespace": "",
            "code": "",
            "zone_id": "zone_1",
            "device_type": "AHU",
            "observable_fields": [],
            "action_fields": [
                "supervisor_run_command",
                "supply_air_heating_temperature_setpoint",
            ],
            "observable_field_types": {},
            "action_field_types": {
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_air_heating_temperature_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "boiler_1",
            "namespace": "",
            "code": "",
            "zone_id": "zone_1",
            "device_type": "BLR",
            "observable_fields": [],
            "action_fields": [
                "supervisor_run_command",
                "supply_water_setpoint",
            ],
            "observable_field_types": {},
            "action_field_types": {
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_water_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "air_handler_2",
            "namespace": "",
            "code": "",
            "zone_id": "zone_2",
            "device_type": "AHU",
            "observable_fields": [],
            "action_fields": [
                "supervisor_run_command",
                "supply_air_heating_temperature_setpoint",
            ],
            "observable_field_types": {},
            "action_field_types": {
                "supervisor_run_command": "VALUE_CONTINUOUS",
                "supply_air_heating_temperature_setpoint": "VALUE_CONTINUOUS",
            },
        },
        {
            "device_id": "outside_air_sensor",
            "namespace": "",
            "code": "",
            "zone_id": "zone_2",
            "device_type": "UNDEFINED",
            "observable_fields": ["outside_air_temperature_sensor"],
            "action_fields": [],
            "observable_field_types": {
                "outside_air_temperature_sensor": "VALUE_CONTINUOUS"
            },
            "action_field_types": {},
        },
    ]
    self.assertEqual(df.to_dict("records"), expected_records)

  def test_building_zones(self):
    df = self.env.building.zones_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            "zone_id": "zone_1",
            "building_id": "SimpleBuilding",
            "zone_description": "zone_1",
            "area": 0.0,
            "devices": ["air_handler_1", "boiler_1"],
            "zone_type": "UNDEFINED",
            "floor": 0,
        },
        {
            "zone_id": "zone_2",
            "building_id": "SimpleBuilding",
            "zone_description": "zone_2",
            "area": 0.0,
            "devices": ["air_handler_2", "outside_air_sensor"],
            "zone_type": "UNDEFINED",
            "floor": 0,
        },
    ]
    self.assertEqual(df.to_dict("records"), expected_records)

  def test_properties(self):
    with self.subTest(name="time_zone"):
      self.assertEqual(self.env.time_zone, "US/Pacific")

    with self.subTest(name="current_simulation_timestamp"):
      self.assertEqual(
          self.env.current_simulation_timestamp,
          pd.Timestamp("2021-06-07 12:00:01"),
      )

    with self.subTest(name="step_count"):
      self.assertEqual(self.env.step_count, 0)

  def test_action_fields_df(self):
    df = self.env.action_fields_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            "field_id": "air_handler_1_supervisor_run_command",
            "device_id": "air_handler_1",
            "device_type": "AHU",
            "zone_id": "zone_1",
            "setpoint_name": "supervisor_run_command",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "DISCRETE",
            "max_native_value": 1.0,
            "max_normalized_value": 1.0,
            "min_native_value": 0.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "air_handler_1_supply_air_heating_temperature_setpoint",
            "device_id": "air_handler_1",
            "device_type": "AHU",
            "zone_id": "zone_1",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 295.0,
            "max_normalized_value": 1.0,
            "min_native_value": 285.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "boiler_1_supervisor_run_command",
            "device_id": "boiler_1",
            "device_type": "BLR",
            "zone_id": "zone_1",
            "setpoint_name": "supervisor_run_command",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "DISCRETE",
            "max_native_value": 1.0,
            "max_normalized_value": 1.0,
            "min_native_value": 0.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "boiler_1_supply_water_setpoint",
            "device_id": "boiler_1",
            "device_type": "BLR",
            "zone_id": "zone_1",
            "setpoint_name": "supply_water_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 350.0,
            "max_normalized_value": 1.0,
            "min_native_value": 310.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "air_handler_2_supervisor_run_command",
            "device_id": "air_handler_2",
            "device_type": "AHU",
            "zone_id": "zone_2",
            "setpoint_name": "supervisor_run_command",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "DISCRETE",
            "max_native_value": 1.0,
            "max_normalized_value": 1.0,
            "min_native_value": 0.0,
            "min_normalized_value": -1.0,
        },
        {
            "field_id": "air_handler_2_supply_air_heating_temperature_setpoint",
            "device_id": "air_handler_2",
            "device_type": "AHU",
            "zone_id": "zone_2",
            "setpoint_name": "supply_air_heating_temperature_setpoint",
            "value_type": "VALUE_CONTINUOUS",
            "action_type": "CONTINUOUS",
            "max_native_value": 295.0,
            "max_normalized_value": 1.0,
            "min_native_value": 285.0,
            "min_normalized_value": -1.0,
        },
    ]
    self.assertCountEqual(df.to_dict("records"), expected_records)

  def test_step(self):
    self.env.reset()
    with self.subTest(name="wants normalized action values"):
      self.assertEqual(self.env.step_count, 0)
      self.env.step({
          "discrete_action": [0, 0, 0],
          "continuous_action": [-1.0, 0.0, 1.0],
      })  # normalized action values
      self.assertEqual(self.env.step_count, 1)

  def test_observations(self):
    n_device_measurements = 1  # see all "_measurement" in conftest.DEMO_LAYOUT
    n_auxiliary_measurements = 7
    n_observations = n_device_measurements + n_auxiliary_measurements
    with self.subTest(name="observation_spec"):
      self.assertEqual(
          self.env.observation_spec(),
          conftest.create_observation_spec(n_observations),
      )

  def test_actions(self):
    with self.subTest(name="action_spec"):
      self.assertEqual(
          self.env.action_spec(),
          conftest.create_hybrid_action_spec(n_continuous=3, n_discrete=3),
      )

    with self.subTest(name="action_names"):
      self.assertSequenceEqual(
          self.env.action_names,
          [
              "air_handler_1_supply_air_heating_temperature_setpoint",
              "air_handler_1_supervisor_run_command",
              "boiler_1_supply_water_setpoint",
              "boiler_1_supervisor_run_command",
              "air_handler_2_supply_air_heating_temperature_setpoint",
              "air_handler_2_supervisor_run_command",
          ],
      )


#
# METRICS WRITER TESTS
#


class EnvironmentMetricsWriterTest(parameterized.TestCase):
  """Ensures the environment metrics are written."""

  def setUp(self):
    super().setUp()
    self.metrics_path = self.create_tempdir().full_path
    writer_factory = controller_writer.ProtoWriterFactory()
    self.env = conftest.create_environment(
        metrics_path=self.metrics_path, writer_factory=writer_factory
    )

  def test_metrics_writer(self):
    self.assertIsInstance(
        self.env._metrics_writer, controller_writer.ProtoWriter
    )
    self.assertStartsWith(
        self.env._metrics_writer._output_dir, self.metrics_path
    )

  def test_reset_writes_metrics(self):
    # the setup for this test is a little more complex, since the reset() method
    # creates a new metrics writer...
    # so we are mocking the writer_factory.create method to return a mock writer
    writer = mock.create_autospec(controller_writer.ProtoWriter, instance=True)

    with mock.patch.object(
        self.env._writer_factory, "create", return_value=writer, autospec=True
    ) as mock_create_method:
      self.env.reset()

    mock_create_method.assert_called_once()
    writer.write_device_infos.assert_called_once_with(self.env.building.devices)
    writer.write_zone_infos.assert_called_once_with(self.env.building.zones)

  @parameterized.parameters("get_reward", "get_reward_info")
  def test_reward_methods_write_metrics(self, method_name):
    self.env._metrics_writer = mock.Mock()

    getattr(self.env, method_name)()

    with self.subTest(name="writes reward_info"):
      self.env._metrics_writer.write_reward_info.assert_called_once()

    with self.subTest(name="writes reward_response"):
      self.env._metrics_writer.write_reward_response.assert_called_once()

  @parameterized.parameters("get_observation_response", "_get_observation")
  def test_observation_methods_write_metrics(self, method_name):
    self.env._metrics_writer = mock.Mock()
    self.env._building_image_generator = mock.create_autospec(
        building_image_generator.BuildingImageGenerator, instance=True
    )

    getattr(self.env, method_name)()

    with self.subTest(name="writes observation_response"):
      self.env._metrics_writer.write_observation_response.assert_called_once()

    with self.subTest("writes building image if generator is set"):
      self.env._metrics_writer.write_building_image.assert_called_once()

  def test_step_writes_metrics(self):
    self.env._metrics_writer = mock.Mock()

    self.env.step([0, 0, 0])

    self.env._metrics_writer.write_action_response.assert_called_once()


if __name__ == "__main__":
  absltest.main()
