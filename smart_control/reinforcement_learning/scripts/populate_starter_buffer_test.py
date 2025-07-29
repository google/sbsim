"""Test for RL starter buffer population script."""

from datetime import datetime
import os
import shutil

from absl.testing import absltest
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.specs import TensorSpec
from tf_agents.trajectories.trajectory import Trajectory

from smart_control.reinforcement_learning.scripts.populate_starter_buffer import populate_replay_buffer
from smart_control.reinforcement_learning.utils.constants import DEFAULT_CONFIG_FILEPATH
from smart_control.reinforcement_learning.utils.constants import RL_STARTER_BUFFERS_DIR

TEST_BUFFER_DIRPATH = os.path.join(RL_STARTER_BUFFERS_DIR, "test")


class StarterBufferPopulationTest(absltest.TestCase):

  def test_starter_buffer_population(self):
    # setup:
    if os.path.isdir(TEST_BUFFER_DIRPATH):
      shutil.rmtree(TEST_BUFFER_DIRPATH)

    # using small arbitrary values for faster completion:
    capacity = 100  # default:50_000
    steps_per_run = 5  # default:100
    replay_buffer = populate_replay_buffer(
        buffer_dirpath=TEST_BUFFER_DIRPATH,
        config_filepath=DEFAULT_CONFIG_FILEPATH,
        buffer_capacity=capacity,
        steps_per_run=steps_per_run,
        num_runs=1,  # default:5
        sequence_length=2,  # default:2
    )

    with self.subTest("returns a replay buffer"):
      self.assertIsInstance(replay_buffer, ReverbReplayBuffer)
      self.assertEqual(replay_buffer.name, "reverb_replay_buffer")
      self.assertEqual(replay_buffer.capacity, capacity)
      self.assertEqual(replay_buffer.num_frames(), steps_per_run - 1)

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

    with self.subTest("stores checkpoints in the specified directory"):
      self.assertTrue(os.path.isdir(TEST_BUFFER_DIRPATH))

      # creates a timestamped sub-directory:
      timestamp_subdir = os.listdir(TEST_BUFFER_DIRPATH)[0]  # dir name
      today = datetime.now().strftime("%Y-%m-%d")
      self.assertTrue(timestamp_subdir.startswith(today))

      # saves files, including "DONE" when complete:
      filenames = [
          "DONE",
          "chunks.tfrecord",
          "items.tfrecord",
          "tables.tfrecord",
      ]
      for filename in filenames:
        filepath = os.path.join(TEST_BUFFER_DIRPATH, timestamp_subdir, filename)
        self.assertTrue(os.path.isfile(filepath))

    # clean up:
    shutil.rmtree(TEST_BUFFER_DIRPATH)


if __name__ == "__main__":
  absltest.main()
