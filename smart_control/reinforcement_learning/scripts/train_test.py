"""Tests for gin config generation script."""

import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from tf_agents.agents.tf_agent import TFAgent

from smart_control.reinforcement_learning.scripts.train import RLAgentTrainer
from smart_control.reinforcement_learning.utils.constants import RL_STARTER_BUFFERS_DIR
from smart_control.utils.constants import SB1_GIN_CONFIG_FILEPATH

TEST_STARTER_BUFFER_DIRPATH = os.path.join(RL_STARTER_BUFFERS_DIR, "default")


class RLAgentTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_dirpath = self.enter_context(self.temp_dir)  # handles teardown

    self.trainer = RLAgentTrainer(
        experiment_name="testing-123",
        config_filepath=SB1_GIN_CONFIG_FILEPATH,
        starter_buffer_path=TEST_STARTER_BUFFER_DIRPATH,
        # minimal param values to decrease training time:
        train_iterations=1,
        collect_steps_per_iteration=1,
        batch_size=256,
        log_interval=1,
        eval_interval=1,
        num_eval_episodes=1,
        checkpoint_interval=1,
        learner_iterations=1,
    )
    # override results dir to use the temporary directory (will get cleaned up)
    self.trainer.results_dirpath = self.temp_dirpath

  def test_save_experiment_params(self):
    self.assertFalse(os.path.isfile(self.trainer.params_json_filepath))
    self.assertFalse(os.path.isfile(self.trainer.params_txt_filepath))

    self.trainer.save_experiment_parameters()

    with self.subTest("saves experiment parameters to file"):
      self.assertTrue(os.path.isfile(self.trainer.params_json_filepath))
      self.assertTrue(os.path.isfile(self.trainer.params_txt_filepath))

    with self.subTest("saved param values are as expected"):
      params = json.loads(self.trainer.params_json_filepath)
      self.assertEqual(params, self.trainer.experiment_params)

  @parameterized.parameters([{"agent_type": "sac"}, {"agent_type": "ddpg"}])
  def test_train_agent(self, agent_type):
    self.trainer.agent_type = agent_type  # overwrite the agent type

    self.assertFalse(os.path.isfile(self.trainer.done_filepath))

    trained_agent = self.trainer.train_agent()
    with self.subTest("it trains an RL agent"):
      self.assertIsInstance(trained_agent, TFAgent)

    with self.subTest("it saves artifacts to the results directory"):
      self.assertTrue(os.path.isdir(self.trainer.metrics_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.collect_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.eval_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.saved_model_dirpath))
      self.assertTrue(os.path.isfile(self.trainer.done_filepath))


if __name__ == "__main__":
  absltest.main()
