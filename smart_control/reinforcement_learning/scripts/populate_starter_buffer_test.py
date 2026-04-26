"""Test for RL starter buffer population script."""

from datetime import datetime
import os
import shutil
import tempfile

from absl.testing import absltest
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.specs import TensorSpec
from tf_agents.trajectories.trajectory import Trajectory

from smart_control.reinforcement_learning.scripts.populate_starter_buffer import StarterBufferGenerator
from smart_control.reinforcement_learning.utils.constants import ONE_DAY_CONFIG_FILEPATH


class StarterBufferPopulationTest(absltest.TestCase):

  def setUp(self):
    """Sets up a temporary directory for each test."""
    super().setUp()
    self.buffer_dirpath = tempfile.mkdtemp()

  def tearDown(self):
    """Cleans up the temporary directory after each test."""
    super().tearDown()
    if os.path.isdir(self.buffer_dirpath):
      shutil.rmtree(self.buffer_dirpath)

  def test_starter_buffer_population(self):
    # using small arbitrary values for faster completion:
    capacity = 100  # default:50_000
    steps_per_run = 5  # default:100
    buffer_generator = StarterBufferGenerator(
        buffer_name="testing-123",
        config_filepath=ONE_DAY_CONFIG_FILEPATH,
        buffer_capacity=capacity,
        steps_per_run=steps_per_run,
        num_runs=1,  # default:5
        sequence_length=2,  # default:2
    )
    buffer_generator.buffer_dirpath = self.buffer_dirpath  # use temp dir
    replay_buffer = buffer_generator.populate()

    with self.subTest("returns a replay buffer"):
      self.assertIsInstance(
          replay_buffer, (ReverbReplayBuffer, TFUniformReplayBuffer)
      )
      self.assertEqual(replay_buffer.capacity, capacity)
      if isinstance(replay_buffer, ReverbReplayBuffer):
        expected_num_frames = steps_per_run - 1
      else:
        expected_num_frames = steps_per_run
      self.assertEqual(int(replay_buffer.num_frames()), expected_num_frames)

      trajectory = replay_buffer.data_spec
      self.assertIsInstance(trajectory, Trajectory)
      # action:
      self.assertIsInstance(trajectory.action, BoundedTensorSpec)
      self.assertEqual(trajectory.action.shape[0], 2)
      self.assertEqual(trajectory.action.minimum.item(), -1)
      self.assertEqual(trajectory.action.maximum.item(), 1)
      # discount:
      self.assertIsInstance(trajectory.discount, BoundedTensorSpec)
      self.assertEqual(trajectory.discount.minimum.item(), 0)
      self.assertEqual(trajectory.discount.maximum.item(), 1)
      # observations:
      self.assertIsInstance(trajectory.observation, TensorSpec)
      self.assertEqual(trajectory.observation.shape[0], 53)
      # reward:
      self.assertIsInstance(trajectory.reward, TensorSpec)

    with self.subTest("stores artifacts in the specified directory"):
      self.assertTrue(os.path.isdir(self.buffer_dirpath))

      # Reverb writes checkpoint files; TFUniform fallback does not.
      if hasattr(replay_buffer, "py_client"):
        timestamp_dirname = os.listdir(self.buffer_dirpath)[0]
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertTrue(timestamp_dirname.startswith(today))

        filenames = [
            "DONE",
            "chunks.tfrecord",
            "items.tfrecord",
            "tables.tfrecord",
        ]
        timestamp_dirpath = os.path.join(self.buffer_dirpath, timestamp_dirname)
        for filename in filenames:
          filepath = os.path.join(timestamp_dirpath, filename)
          self.assertTrue(os.path.isfile(filepath))


if __name__ == "__main__":
  absltest.main()
