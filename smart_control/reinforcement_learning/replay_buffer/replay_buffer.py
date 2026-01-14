"""Reinforcement learning replay buffers."""

import logging
from typing import Optional, Tuple

import reverb
import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

logger = logging.getLogger(__name__)


class ReplayBufferManager:
  """Manager for creating and interacting with Reverb replay buffers.

  This class simplifies the setup, interaction, and checkpointing of Reverb
  replay buffers for reinforcement learning agents.
  It provides methods to create a new buffer, add data, sample from the buffer,
  and save/restore buffer state.
  """

  def __init__(self, data_spec, capacity, checkpoint_dir, sequence_length=2):
    self.data_spec = data_spec
    self.capacity = capacity
    self.checkpoint_dir = checkpoint_dir
    self.sequence_length = sequence_length
    self.table_name = "uniform_table"
    self._is_initialized = False
    self.server = None
    self.replay_buffer = None
    self.observer = None

  def create_replay_buffer(self):
    """Create the replay buffer."""
    # Create the table
    table = reverb.Table(
        self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    # Create the checkpointer
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    # Create the server
    reverb_server = reverb.Server(
        [table], port=None, checkpointer=reverb_checkpointer
    )

    # Create the replay buffer
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        self.data_spec,
        sequence_length=self.sequence_length,
        table_name=self.table_name,
        local_server=reverb_server,
    )

    # Create the observer that adds trajectories to the buffer
    observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        self.table_name,
        sequence_length=self.sequence_length,
        stride_length=1,
    )

    # Save as attributes and mark as initialized
    self.server = reverb_server
    self.replay_buffer = replay_buffer
    self.observer = observer
    self._is_initialized = True

    return replay_buffer, observer

  def load_replay_buffer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Load an existing replay buffer from a saved checkpoint.

    This method reconstructs the replay buffer, server, and observer based on
    the
    saved state in the checkpoint directory.

    Returns:
        A tuple of (replay_buffer, observer).
    """
    # Create the table with the same parameters as before
    table = reverb.Table(
        self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    # Create the checkpointer pointing to the checkpoint directory
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    # Create the server with the existing table and checkpointer.
    reverb_server = reverb.Server(
        [table], port=None, checkpointer=reverb_checkpointer
    )

    # Create the replay buffer and observer using the restored server.
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        self.data_spec,
        sequence_length=self.sequence_length,
        table_name=self.table_name,
        local_server=reverb_server,
    )

    observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        self.table_name,
        sequence_length=self.sequence_length,
        stride_length=1,
    )

    # Save as attributes and mark as initialized.
    self.server = reverb_server
    self.replay_buffer = replay_buffer
    self.observer = observer
    self._is_initialized = True

    logging.info("Replay buffer loaded from checkpoint")
    return replay_buffer, observer

  def get_replay_buffer_and_observer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Get the replay buffer and observer.

    Creates them if not already initialized.

    Returns:
        A tuple of (replay_buffer, observer).
    """
    if not self._is_initialized:
      return self.create_replay_buffer()
    return self.replay_buffer, self.observer

  def get_dataset(
      self, batch_size: int = 64, num_steps: Optional[int] = None
  ) -> tf.data.Dataset:
    """Get a TensorFlow dataset for sampling from the replay buffer.

    Args:
        batch_size: Number of sequences to sample in each batch.
        num_steps: Number of steps to sample for each sequence. If None,
          defaults to sequence_length.

    Returns:
        A TensorFlow dataset that samples from the replay buffer.

    Raises:
        RuntimeError: If the replay buffer has not been initialized yet.
    """
    if not self._is_initialized:
      raise RuntimeError(
          "Replay buffer not initialized. Call create_replay_buffer or"
          " load_replay_buffer first."
      )

    if num_steps is None:
      num_steps = self.sequence_length

    return self.replay_buffer.as_dataset(
        sample_batch_size=batch_size, num_steps=num_steps
    )

  def num_frames(self) -> int:
    """Get the current number of frames in the replay buffer.

    Returns:
        The number of frames currently in the buffer.
    """
    if not self._is_initialized:
      return 0
    return self.replay_buffer.num_frames()

  def clear(self) -> None:
    """Clear all data from the replay buffer."""
    if not self._is_initialized:
      return

    # Close the existing server and create a new one
    self.server.stop()

    # Recreate everything
    self.create_replay_buffer()
    logging.info("Replay buffer cleared and recreated")

  def close(self) -> None:
    """Close the replay buffer server and clean up resources."""
    if self._is_initialized and self.server:
      self.server.stop()
      self._is_initialized = False
      logging.info("Replay buffer server stopped")
