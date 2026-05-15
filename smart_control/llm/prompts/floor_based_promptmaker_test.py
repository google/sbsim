from absl.testing import absltest

from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.prompts import floor_based_promptmaker
from smart_buildings.smart_control.utils import temperature_conversion as tc


class FloorBasedPromptmakerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.zone_reward_configs = {
        'zone_1': {
            'zone_air_temperature': 292.1,
            'heating_setpoint_temperature': 294.0,
            'cooling_setpoint_temperature': 296.0,
            'average_occupancy': 5.0,
        },
        'zone_2': {
            'zone_air_temperature': 296.2,
            'heating_setpoint_temperature': 294.0,
            'cooling_setpoint_temperature': 296.0,
            'average_occupancy': 10.0,
        },
        'zone_3': {
            'zone_air_temperature': 297.9,
            'heating_setpoint_temperature': 294.0,
            'cooling_setpoint_temperature': 296.0,
            'average_occupancy': 3.0,
        },
    }
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.MULTI_FLOOR_LAYOUT,
        zone_reward_configs=self.zone_reward_configs,
    )
    self.pm = floor_based_promptmaker.FloorBasedPromptmaker(
        env=self.env, temp_display_unit=tc.TempUnit.KELVIN
    )

  def test_zone_conditions_histogram_by_floor(self):
    df = self.pm.zone_conditions_histogram_by_floor
    # The histogram is transposed in FloorBasedPromptmaker.
    # Index should include occupancy_count, setpoint_range, exposed_count,
    # and floor distribution(s).
    self.assertIn('occupancy_count', df.index)
    self.assertIn('setpoint_range', df.index)
    self.assertIn('exposed_count', df.index)

    floor_rows = [i for i in df.index if str(i).startswith('occ@floor')]
    self.assertCountEqual(floor_rows, ['occ@floor1', 'occ@floor2'])

    # Global occupancy: 5 at 292, 10 at 296, 3 at 298.
    self.assertEqual(df.loc['occupancy_count', 292.0], 5)
    self.assertEqual(df.loc['occupancy_count', 296.0], 10)
    self.assertEqual(df.loc['occupancy_count', 298.0], 3)

    # Floor 1 distribution: zone_1 (temp 292) and zone_2 (temp 296).
    # Since they are normalized, each should be 0.5 at their respective bins.
    self.assertEqual(df.loc['occ@floor1', 292.0], 0.5)
    self.assertEqual(df.loc['occ@floor1', 296.0], 0.5)

    # Floor 2 distribution: zone_3 (temp 298).
    self.assertEqual(df.loc['occ@floor2', 298.0], 1.0)

  def test_zone_conditions_histogram_by_floor_is_always_kelvin(self):
    # Setup promptmaker with Fahrenheit as display unit
    pm = floor_based_promptmaker.FloorBasedPromptmaker(
        env=self.env,
        temp_display_unit=tc.TempUnit.FAHRENHEIT,
    )

    df = pm.zone_conditions_histogram_by_floor

    # Even though display unit is F, the table data passed to LLM stays in K.
    # Global occupancy: 5 at 292, 10 at 296, 3 at 298.
    self.assertEqual(df.loc['occupancy_count', 292.0], 5)
    self.assertEqual(df.loc['occupancy_count', 296.0], 10)
    self.assertEqual(df.loc['occupancy_count', 298.0], 3)

    # Verify the prompt text mentions Fahrenheit
    self.assertIn('communicate temperatures in Fahrenheit', pm.base_prompt)

  def test_current_conditions_section(self):
    section = self.pm.current_conditions_section
    self.assertIn('## Current Conditions', section)
    self.assertIn('### Current Zone Temperatures', section)
    self.assertIn('by floor:', section)
    self.assertIn("The rows starting with 'occ@floor'", section)

    table = self.pm.zone_conditions_histogram_by_floor.to_markdown(index=True)
    self.assertIn(table, section)


if __name__ == '__main__':
  absltest.main()
