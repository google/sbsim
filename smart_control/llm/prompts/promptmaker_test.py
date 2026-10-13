from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.prompts import promptmaker
from smart_buildings.smart_control.llm.schema import output_schema
from smart_buildings.smart_control.utils.proto_parsers import observation_response_parser
from smart_buildings.smart_control.utils.proto_parsers import reward_info_parser

WEIGHTS = {
    'energy_cost_weight': 0.3,
    'carbon_emission_weight': 0.2,
    'productivity_weight': 0.5,
}

WEIGHTS_INCLUDED_CONTENT = (
    'We have assigned a weight to designate the importance of each objective.'
)

BUILDING_INFO = {
    'stories': 'two',
    'sqft': 96_000,
    'location': 'Mountain View, California',
    'name': 'SB-1',
}


class PromptmakerTest(absltest.TestCase):
  """Tests for the Promptmaker class, with weights present but not included."""

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.env.reward_function.weights = WEIGHTS
    self.pm = promptmaker.Promptmaker(env=self.env)
    self.expected_promtpmaker_type = 'Promptmaker'

  def test_initialization(self):
    self.assertIsInstance(self.pm, promptmaker.Promptmaker)

  def test_attributes(self):
    with self.subTest(name='required_attributes'):
      self.assertEqual(
          self.pm.output_schema_class,
          output_schema.SetpointsAction,
      )
      self.assertEqual(self.pm.env, self.env)

    with self.subTest(name='configuration_attributes'):
      self.assertFalse(self.pm.include_weights)
      self.assertEqual(self.pm.occupancy_mode_min, 10)
      self.assertEqual(self.pm.temp_display_unit, 'Fahrenheit')

    with self.subTest(name='building_info'):
      building_info = self.pm.building_info
      self.assertIsInstance(building_info, promptmaker.BuildingInfo)
      self.assertEqual(building_info.stories, 'two')
      self.assertEqual(building_info.sqft, 96_000)
      self.assertEqual(building_info.location, 'Mountain View, California')

    with self.subTest(name='proto_parsers'):
      self.assertFalse(self.pm.lazy_init_protos)
      self.assertIsInstance(
          self.pm.observation_response_parser,
          observation_response_parser.ObservationResponseParser,
      )
      self.assertIsInstance(
          self.pm.reward_info_parser,
          reward_info_parser.RewardInfoParser,
      )

  # PROPERTIES

  def test_json_metadata(self):
    json_metadata = self.pm.json_metadata

    with self.subTest(name='type'):
      self.assertEqual(json_metadata['type'], self.expected_promtpmaker_type)

    with self.subTest(name='include_weights'):
      self.assertEqual(json_metadata['include_weights'], False)

    with self.subTest(name='occupancy_mode_min'):
      self.assertEqual(json_metadata['occupancy_mode_min'], 10)

    with self.subTest(name='temp_display_unit'):
      self.assertEqual(json_metadata['temp_display_unit'], 'Fahrenheit')

    with self.subTest(name='building_info'):
      self.assertEqual(json_metadata['building_info'], BUILDING_INFO)

  def test_weights(self):
    self.assertEqual(
        self.pm.weights,
        {
            'energy_cost_weight': 0.3,
            'carbon_emission_weight': 0.2,
            'comfort_weight': 0.5,
        },
    )

  def test_setpoints_df(self):
    df = self.pm.setpoints_df
    self.assertIsInstance(df, pd.DataFrame)

    expected_records = [
        {
            'device_id': 'air_handler_1',
            'setpoint_name': 'supervisor_run_command',
            'setpoint_type': 'DISCRETE',
            'units': 'On/Off',
            'min_native_value': 0.0,
            'max_native_value': 1.0,
        },
        {
            'device_id': 'air_handler_1',
            'setpoint_name': 'supply_air_heating_temperature_setpoint',
            'setpoint_type': 'CONTINUOUS',
            'units': 'Kelvin',
            'min_native_value': 285.0,
            'max_native_value': 295.0,
        },
        {
            'device_id': 'air_handler_2',
            'setpoint_name': 'supervisor_run_command',
            'setpoint_type': 'DISCRETE',
            'units': 'On/Off',
            'min_native_value': 0.0,
            'max_native_value': 1.0,
        },
        {
            'device_id': 'air_handler_2',
            'setpoint_name': 'supply_air_heating_temperature_setpoint',
            'setpoint_type': 'CONTINUOUS',
            'units': 'Kelvin',
            'min_native_value': 285.0,
            'max_native_value': 295.0,
        },
        {
            'device_id': 'boiler_1',
            'setpoint_name': 'supervisor_run_command',
            'setpoint_type': 'DISCRETE',
            'units': 'On/Off',
            'min_native_value': 0.0,
            'max_native_value': 1.0,
        },
        {
            'device_id': 'boiler_1',
            'setpoint_name': 'supply_water_setpoint',
            'setpoint_type': 'CONTINUOUS',
            'units': 'Kelvin',
            'min_native_value': 310.0,
            'max_native_value': 350.0,
        },
    ]
    self.assertListEqual(df.to_dict('records'), expected_records)

  def test_validity_intervals(self):
    self.assertEqual(
        self.pm.validity_intervals,
        [5, 10, 15, 20, 30, 45, 60, 75, 90, 120],
    )

  # PROMPT CONTENT

  def test_prompt(self):
    prompt = self.pm.prompt
    with self.subTest(name='objectives_section'):
      self.assertIn(self.pm.objectives_section, prompt)

    with self.subTest(name='zone_info_section'):
      self.assertIn(self.pm.zone_info_section, prompt)

    with self.subTest(name='occupancy_modes_section'):
      self.assertIn(self.pm.occupancy_modes_section, prompt)

    with self.subTest(name='hvac_system_guidelines_section'):
      self.assertIn(self.pm.hvac_system_guidelines_section, prompt)

    with self.subTest(name='action_guidelines_section'):
      self.assertIn(self.pm.action_guidelines_section, prompt)

    with self.subTest(name='current_conditions_section'):
      self.assertIn(self.pm.current_conditions_section, prompt)

    with self.subTest(name='current_action_section'):
      self.assertIn(self.pm.current_action_section, prompt)

    with self.subTest(name='formatting_instructions_section'):
      self.assertIn(self.pm.formatting_instructions_section, prompt)

  def test_objectives_section(self):
    section = self.pm.objectives_section
    self.assertIn('## Objectives', section)
    self.assertIn('### Role', section)
    self.assertIn('### Overall Goal', section)

    with self.subTest(name='includes_building_info'):
      self.assertIn('**Building Information**', section)
      table = self.pm.building_info_series.to_markdown(index=True)
      self.assertIn(table, section)

    with self.subTest(name='weights_present_but_not_included'):
      self.assertIsNotNone(self.env.reward_function.weights)
      self.assertNotIn(WEIGHTS_INCLUDED_CONTENT, section)

  def test_zone_info_section(self):
    section = self.pm.zone_info_section
    self.assertIn('## Zone Information', section)
    self.assertIn('### Zone Comfort', section)

  def test_occupancy_modes_section(self):
    section = self.pm.occupancy_modes_section
    self.assertIn('## Occupancy Modes', section)
    self.assertIn('### Heating and Cooling Guidelines', section)

    with self.subTest(name='uses_occupancy_mode_min'):
      self.assertIn(
          '**Occupancy mode** is when the building has at least 10 occupants.',
          section,
      )
      self.assertIn(
          '**Efficiency mode** is when the building has fewer than 10'
          ' occupants.',
          section,
      )

  def test_hvac_system_guidelines_section(self):
    section = self.pm.hvac_system_guidelines_section

    with self.subTest(name='contains_section_headers'):
      self.assertIn('## HVAC System Control Guidelines', section)
      self.assertIn('### Devices and Setpoints', section)
      self.assertIn(
          '### Air Conditioner (AC) / Air Handler (AHU) Guidelines',
          section,
      )
      self.assertIn('### Boiler (BLR) Guidelines', section)
      self.assertIn('### Zone Temperature Control Guidelines', section)

    with self.subTest(name='mentions_specific_devices'):
      self.assertIn(
          '**AC-1**: Air Conditioner / Air Handler Unit (for all zones on the'
          ' first floor)',
          section,
      )
      self.assertIn(
          '**AC-2**: Air Conditioner / Air Handler Unit (for all zones on the'
          ' second floor)',
          section,
      )
      self.assertIn('**BLR**: Boiler (for both floors)', section)

    with self.subTest(name='mentions_key_setpoints'):
      self.assertIn("'supervisor_run_command'", section)
      self.assertIn("'static_pressure_setpoint'", section)
      self.assertIn("'supply_air_temperature_setpoint'", section)
      self.assertIn("'differential_pressure_setpoint'", section)
      self.assertIn("'supply_water_setpoint'", section)

  def test_action_guidelines_section(self):
    section = self.pm.action_guidelines_section

    with self.subTest(name='contains_section_header'):
      self.assertIn('## Action Guidelines', section)

    with self.subTest(name='includes_device_setpoints_table'):
      self.assertIn(self.pm.setpoints_df.to_markdown(index=False), section)

    with self.subTest(name='includes_temp_display_unit'):
      self.assertIn(
          'you should communicate temperatures in Fahrenheit instead',
          section,
      )

  def test_current_conditions_section(self):
    section = self.pm.current_conditions_section

    with self.subTest(name='contains_section_headers'):
      self.assertIn('## Current Conditions', section)
      self.assertIn('### Current Zone Temperatures', section)
      self.assertIn('### Current Power Consumption', section)

    with self.subTest(name='includes_current_local_time'):
      self.assertIn(
          'The current local time is: Monday, June 07, 2021 12:00 PM PDT',
          section,
      )

    with self.subTest(name='includes_current_outside_air_temperature'):
      self.assertIn(
          'The current outside air temperature is: 295.0 Kelvin',
          section,
      )

    with self.subTest(name='includes_occupant_counts'):
      self.assertIn('Total number of zones: 2', section)
      self.assertIn(
          'Current number of occupants: 10',
          section,
      )
      self.assertIn(
          'Current number of occupants exposed to unacceptable comfort'
          ' conditions: 0',
          section,
      )

    parser = self.pm.reward_info_parser
    self.assertIsNotNone(parser)

    # pytype: disable=attribute-error
    with self.subTest(name='includes_current_zone_temperatures_table'):
      table = parser.zone_conditions_histogram.to_markdown(index=True)
      self.assertIn(table, section)

    with self.subTest(name='includes_current_power_consumption_table'):
      table = parser.energy_consumption_df_watts.to_markdown(index=False)
      self.assertIn(table, section)
    # pytype: enable=attribute-error

  def test_current_action_section(self):
    section = self.pm.current_action_section

    with self.subTest(name='contains_section_header'):
      self.assertIn('## Current Action', section)

    with self.subTest(name='specifies_discrete_action_commands'):
      self.assertIn(
          'According to your strategy, decide to turn each device ON (1) or OFF'
          " (0), using their discrete 'supervisor_run_command' setpoints.",
          section,
      )

    with self.subTest(name='specifies_validity_interval_options'):
      self.assertIn(
          'Finally, select a validity interval from the following options:'
          ' [5, 10, 15, 20, 30, 45, 60, 75, 90, 120]',
          section,
      )

  def test_formatting_instructions_section(self):
    section = self.pm.formatting_instructions_section
    self.assertIn('## Formatting Instructions', section)


class PromptmakerWeightsUnavailableTest(absltest.TestCase):
  """Tests for the Promptmaker class, with weights not present."""

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )

  def test_weights_not_requested_or_present(self):
    pm = promptmaker.Promptmaker(env=self.env)
    # Weights are not requested:
    self.assertFalse(pm.include_weights)
    # Weights are not present:
    self.assertFalse(hasattr(self.env.reward_function, 'weights'))

    section = pm.objectives_section
    self.assertNotIn(WEIGHTS_INCLUDED_CONTENT, section)

  def test_weights_requested_but_not_present(self):
    pm = promptmaker.Promptmaker(env=self.env, include_weights=True)
    # Weights are requested:
    self.assertTrue(pm.include_weights)
    # Weights are not present:
    self.assertFalse(hasattr(self.env.reward_function, 'weights'))

    section = pm.objectives_section
    self.assertNotIn(WEIGHTS_INCLUDED_CONTENT, section)


class PromptmakerWeightsInclusionTest(absltest.TestCase):
  """Tests for the Promptmaker class, with weights present and included."""

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.env.reward_function.weights = WEIGHTS
    self.pm = promptmaker.Promptmaker(env=self.env, include_weights=True)

  def test_weights(self):
    self.assertEqual(
        self.pm.weights,
        {
            'energy_cost_weight': 0.3,
            'carbon_emission_weight': 0.2,
            'comfort_weight': 0.5,
        },
    )

  def test_weights_included(self):
    weights = self.pm.weights
    self.assertIsInstance(weights, dict)

    section = self.pm.objectives_section
    self.assertIn(WEIGHTS_INCLUDED_CONTENT, section)
    weights_table = pd.Series(weights, name='weight').to_markdown(index=True)
    self.assertIn(weights_table, section)


class PromptmakerLazyInitProtosTest(parameterized.TestCase):

  ATTRIBUTE_NAMES = (
      dict(
          testcase_name='base_prompt',
          attribute_name='base_prompt',
      ),
      dict(
          testcase_name='current_conditions_section',
          attribute_name='current_conditions_section',
      ),
  )

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.pm = promptmaker.Promptmaker(self.env, lazy_init_protos=True)

  @parameterized.named_parameters(*ATTRIBUTE_NAMES)
  def test_lazy_init_protos_raises_when_protos_not_set(self, attribute_name):
    self.assertIsNone(self.pm._observation_response_parser)
    self.assertIsNone(self.pm._reward_info_parser)

    with self.assertRaisesRegex(
        ValueError, 'Observation response parser is None.'
    ):
      _ = getattr(self.pm, attribute_name)

  @parameterized.named_parameters(*ATTRIBUTE_NAMES)
  def test_lazy_init_protos_ok_when_protos_are_set(self, attribute_name):
    self.assertIsNone(self.pm._observation_response_parser)
    self.assertIsNone(self.pm._reward_info_parser)

    self.pm.set_protos(
        observation_response=self.env.get_observation_response(),
        reward_info=self.env.get_reward_info(),
    )
    self.assertIsInstance(
        self.pm.observation_response_parser,
        observation_response_parser.ObservationResponseParser,
    )
    self.assertIsInstance(
        self.pm.reward_info_parser,
        reward_info_parser.RewardInfoParser,
    )
    _ = getattr(self.pm, attribute_name)  # No error thrown.


if __name__ == '__main__':
  absltest.main()
