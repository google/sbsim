"""Replay buffer tests."""

import sys
import tempfile
import unittest

import tensorflow as tf

from smart_control.reinforcement_learning.replay_buffer import replay_buffer as replay_buffer_lib

# we are skipping these tests on Mac for now, until we can resolve the dm-reverb
# package installation on Mac. See: https://github.com/google/sbsim/issues/102
RUNNING_ON_MAC = sys.platform.startswith("darwin")
SKIP_REASON = "Issues installing dm-reverb on Mac."


class ReverbInstallationTest(unittest.TestCase):
  """Testing if we can install the dm-reverb package. Skipping on Mac for now.
  We can remove the skip logic and push to GitHub Actions to test / prove our
  ability to install across all platforms. Then we can remove this test class.
  """

  @unittest.skipIf(RUNNING_ON_MAC, SKIP_REASON)
  def test_reverb_installation(self):
    import reverb  # pylint:disable=import-outside-toplevel

    print("Reverb imported successfully.")
    print(dir(reverb))
    assert True


class TFUniformFallbackCheckpointTest(unittest.TestCase):
  """Tests checkpoint save/restore behavior for TFUniform fallback."""

  def test_save_and_load_checkpoint_in_fallback_mode(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      data_spec = tf.TensorSpec(shape=(), dtype=tf.float32)

      old_has_reverb = replay_buffer_lib._HAS_REVERB
      try:
        replay_buffer_lib._HAS_REVERB = False

        manager = replay_buffer_lib.ReplayBufferManager(
            data_spec=data_spec,
            capacity=100,
            checkpoint_dir=tmp_dir,
            sequence_length=2,
        )
        replay_buffer, observer = manager.create_replay_buffer()
        observer(tf.constant(1.0, dtype=tf.float32))
        self.assertEqual(replay_buffer.num_frames().numpy(), 1)

        saved_checkpoint_path = manager.save_checkpoint()
        self.assertIsNotNone(saved_checkpoint_path)
        self.assertEqual(
            manager._tf_checkpoint_manager.latest_checkpoint,  # pylint:disable=protected-access
            saved_checkpoint_path,
        )
        self.assertTrue(tf.io.gfile.exists(saved_checkpoint_path + ".index"))

        new_manager = replay_buffer_lib.ReplayBufferManager(
            data_spec=data_spec,
            capacity=100,
            checkpoint_dir=tmp_dir,
            sequence_length=2,
        )
        restored_buffer, _ = new_manager.load_replay_buffer()
        self.assertEqual(restored_buffer.num_frames().numpy(), 1)
      finally:
        replay_buffer_lib._HAS_REVERB = old_has_reverb


if __name__ == "__main__":
  unittest.main()
