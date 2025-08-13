"""Tests for training RL agents.

FYI: training an agent with the minimal config takes around two minutes.
We are skipping training tests by default, to decrease the build time.
However you can enable training tests by setting the `TEST_RL_TRAINING`
environment variable to "true".
"""

import smart_control.reinforcement_learning.tf_import_fix  # isort:skip # pylint:disable=bad-import-order,unused-import

import json
import os
import tempfile
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from tf_agents.agents.tf_agent import TFAgent

from smart_control.reinforcement_learning.scripts.conftest import create_and_setup_test_environment
from smart_control.reinforcement_learning.scripts.train import RLAgentTrainer
from smart_control.utils.constants import SB1_GIN_CONFIG_FILEPATH

TEST_RL_TRAINING = bool(
    os.getenv("TEST_RL_TRAINING", default="false") == "true"
)
SKIP_REASON = "It takes a long time to train the RL agent."

# TEST_STARTER_BUFFER_DIRPATH = os.path.join(RL_STARTER_BUFFERS_DIR, "default")


class RLAgentTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.temp_dirpath = self.enter_context(self.temp_dir)  # handles teardown

    self.trainer = RLAgentTrainer(
        experiment_name="testing-123",
        config_filepath=SB1_GIN_CONFIG_FILEPATH,
        # starter_buffer_path=TEST_STARTER_BUFFER_DIRPATH,
        starter_buffer_name="test",
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
    # override results dir to use the temporary directory (will get cleaned up):
    self.trainer.results_dirpath = self.temp_dirpath
    # override environment config to decrease number of training steps:
    self.trainer.create_and_setup_environment = create_and_setup_test_environment  # pylint:disable=line-too-long

  def test_save_experiment_params(self):
    self.assertFalse(os.path.isfile(self.trainer.params_json_filepath))
    self.assertFalse(os.path.isfile(self.trainer.params_txt_filepath))

    self.trainer.save_experiment_params()

    with self.subTest("saves experiment parameters to file"):
      self.assertTrue(os.path.isfile(self.trainer.params_json_filepath))
      self.assertTrue(os.path.isfile(self.trainer.params_txt_filepath))

    with self.subTest("saved param values are as expected"):
      with open(
          self.trainer.params_json_filepath, "r", encoding="utf-8"
      ) as json_file:
        params = json.load(json_file)

      self.assertIsInstance(params["timestamp"], str)
      del params["timestamp"]
      self.assertEqual(params, self.trainer.experiment_params)

  @unittest.skipUnless(TEST_RL_TRAINING, SKIP_REASON)
  @parameterized.parameters([{"agent_type": "sac"}, {"agent_type": "ddpg"}])
  def test_train_agent(self, agent_type):
    self.trainer.agent_type = agent_type  # overwrite the agent type

    trained_agent = self.trainer.train_agent()

    with self.subTest("it trains an RL agent"):
      self.assertIsInstance(trained_agent, TFAgent)

    with self.subTest("it saves artifacts to the results directory"):
      self.assertTrue(os.path.isdir(self.trainer.metrics_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.collect_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.eval_dirpath))
      self.assertTrue(os.path.isdir(self.trainer.saved_model_dirpath))
      # self.assertTrue(os.path.isfile(self.trainer.done_filepath))


if __name__ == "__main__":
  absltest.main()
