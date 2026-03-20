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


if __name__ == '__main__':
  absltest.main()
