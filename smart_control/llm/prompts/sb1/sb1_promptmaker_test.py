from absl.testing import absltest

from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.prompts import promptmaker_test
from smart_buildings.smart_control.llm.prompts.sb1 import sb1_promptmaker


class SB1PromptmakerTest(promptmaker_test.PromptmakerTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.env.reward_function.weights = promptmaker_test.WEIGHTS
    self.pm = sb1_promptmaker.SB1Promptmaker(env=self.env)
    self.expected_promtpmaker_type = 'SB1Promptmaker'

  def test_initialization(self):
    self.assertIsInstance(self.pm, sb1_promptmaker.SB1Promptmaker)


class SB1FloorBasedPromptmakerTest(promptmaker_test.PromptmakerTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_hybrid_action_environment(
        layout=env_conftest.DEMO_LAYOUT
    )
    self.env.reward_function.weights = promptmaker_test.WEIGHTS
    self.pm = sb1_promptmaker.SB1FloorBasedPromptmaker(env=self.env)
    self.expected_promtpmaker_type = 'SB1FloorBasedPromptmaker'

  def test_initialization(self):
    self.assertIsInstance(self.pm, sb1_promptmaker.SB1FloorBasedPromptmaker)

  def test_current_conditions_section(self):
    section = self.pm.current_conditions_section
    self.assertIn('## Current Conditions', section)
    self.assertIn('### Current Zone Temperatures', section)
    self.assertIn('by floor:', section)
    self.assertIn("The row 'occupancy_count'", section)
    self.assertIn("The rows starting with 'occ@floor'", section)

    # Check if the table is present
    table = self.pm.zone_conditions_histogram_by_floor.to_markdown(index=True)
    self.assertIn(table, section)


if __name__ == '__main__':
  absltest.main()
