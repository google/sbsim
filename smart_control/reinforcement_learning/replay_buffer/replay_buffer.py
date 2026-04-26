"""Reinforcement learning replay buffers."""

import logging
import os
from typing import Any, Optional, Tuple

import tensorflow as tf

# Try Reverb; fall back to TFUniform if unavailable (e.g., macOS/py3.11).
try:
  import reverb  # type: ignore
  from tf_agents.replay_buffers import reverb_replay_buffer
  from tf_agents.replay_buffers import reverb_utils

  _HAS_REVERB = True
except ImportError:  # pragma: no cover
  reverb = None  # type: ignore
  reverb_replay_buffer = None  # type: ignore
  reverb_utils = None  # type: ignore
  _HAS_REVERB = False

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

logger = logging.getLogger(__name__)


class ReplayBufferManager:
  """Manager for creating and interacting with RL replay buffers.

  If `dm-reverb` is available, uses a Reverb-backed buffer.
  Otherwise, falls back to `TFUniformReplayBuffer` (portable, no external deps).
  """

  def __init__(
      self, data_spec, capacity, checkpoint_dir, sequence_length: int = 2
  ):
    self.data_spec = data_spec
    self.capacity = capacity
    self.checkpoint_dir = checkpoint_dir
    self.sequence_length = sequence_length
    self.table_name = "uniform_table"

    self._is_initialized = False
    self._use_reverb = bool(_HAS_REVERB)
    self.server = None
    self.replay_buffer = None
    self.observer = None
    self._tf_checkpoint = None
    self._tf_checkpoint_manager = None

  # ---------------------------
  # Creation / Loading
  # ---------------------------
  def create_replay_buffer(self) -> Tuple[Any, Any]:
    """Create the replay buffer and an observer callable.

    Returns:
      (replay_buffer, observer)
    """
    if self._use_reverb:
      # ------ Reverb-backed path ------
      table = reverb.Table(  # type: ignore[attr-defined]
          self.table_name,
          max_size=self.capacity,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          rate_limiter=reverb.rate_limiters.MinSize(1),
      )

      reverb_checkpointer = (
          reverb.platform.checkpointers_lib.DefaultCheckpointer(
              path=self.checkpoint_dir
          )
      )

      reverb_server = reverb.Server(  # type: ignore[attr-defined]
          [table], port=None, checkpointer=reverb_checkpointer
      )

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

      self.server = reverb_server
      self.replay_buffer = replay_buffer
      self.observer = observer
      self._is_initialized = True
      return replay_buffer, observer

    else:
      # ------ TFUniform fallback path ------
      replay_buffer = TFUniformReplayBuffer(
          data_spec=self.data_spec,
          batch_size=1,  # keep 1 since we’ll add a batch dim
          max_length=self.capacity,
      )
      tf_checkpoint_dir = os.path.join(self.checkpoint_dir, "tf_uniform_ckpt")
      tf.io.gfile.makedirs(tf_checkpoint_dir)
      self._tf_checkpoint = tf.train.Checkpoint(replay_buffer=replay_buffer)
      self._tf_checkpoint_manager = tf.train.CheckpointManager(
          self._tf_checkpoint, directory=tf_checkpoint_dir, max_to_keep=1
      )

      # Wrap observer to add a batch dimension expected by add_batch
      def _uniform_observer(traj):
        batched = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), traj)
        replay_buffer.add_batch(batched)

      observer = _uniform_observer

      self.server = None
      self.replay_buffer = replay_buffer
      self.observer = observer
      self._is_initialized = True
      logger.info(
          "Using TFUniformReplayBuffer fallback (dm-reverb not available)."
      )
      return replay_buffer, observer

  def load_replay_buffer(self) -> Tuple[Any, Any]:
    """Load an existing replay buffer from a saved checkpoint (if Reverb).

    For TFUniform fallback, this recreates a fresh buffer (no external ckpt).
    """
    if self._use_reverb:
      # ------ Reverb-backed path with checkpointer ------
      table = reverb.Table(  # type: ignore[attr-defined]
          self.table_name,
          max_size=self.capacity,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          rate_limiter=reverb.rate_limiters.MinSize(1),
      )

      reverb_checkpointer = (
          reverb.platform.checkpointers_lib.DefaultCheckpointer(
              path=self.checkpoint_dir
          )
      )

      reverb_server = reverb.Server(  # type: ignore[attr-defined]
          [table], port=None, checkpointer=reverb_checkpointer
      )

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

      self.server = reverb_server
      self.replay_buffer = replay_buffer
      self.observer = observer
      self._is_initialized = True
      logger.info("Replay buffer loaded from checkpoint (Reverb).")
      return replay_buffer, observer

    # TFUniform fallback: restore via tf.train.Checkpoint if available.
    replay_buffer, observer = self.create_replay_buffer()
    self.load_checkpoint()
    return replay_buffer, observer

  def get_replay_buffer_and_observer(self) -> Tuple[Any, Any]:
    """Return (replay_buffer, observer), creating if necessary."""
    if not self._is_initialized:
      return self.create_replay_buffer()
    return self.replay_buffer, self.observer

  # ---------------------------
  # Dataset / Introspection
  # ---------------------------
  def get_dataset(
      self, batch_size: int = 64, num_steps: Optional[int] = None
  ) -> tf.data.Dataset:
    """Get a tf.data.Dataset for sampling from the replay buffer."""
    if not self._is_initialized:
      raise RuntimeError(
          "Replay buffer not initialized. Call create_replay_buffer or"
          " load_replay_buffer first."
      )

    if num_steps is None:
      num_steps = self.sequence_length

    return self.replay_buffer.as_dataset(  # type: ignore[union-attr]
        sample_batch_size=batch_size, num_steps=num_steps
    )

  def num_frames(self) -> int:
    """Number of frames currently in the buffer."""
    if not self._is_initialized:
      return 0
    # Both Reverb and TFUniform expose num_frames()
    return int(self.replay_buffer.num_frames())  # type: ignore[union-attr]

  # ---------------------------
  # Lifecycle
  # ---------------------------
  def clear(self) -> None:
    """Clear all data from the replay buffer."""
    if not self._is_initialized:
      return

    if self._use_reverb and self.server is not None:
      # Stop existing server and recreate everything.
      self.server.stop()
      self.create_replay_buffer()
      logger.info("Reverb replay buffer cleared and recreated.")
    else:
      # TFUniform: recreate a fresh buffer (portable, consistent behavior).
      self.create_replay_buffer()
      logger.info("TFUniform replay buffer cleared and recreated.")

  def save_checkpoint(self) -> Optional[str]:
    """Save replay-buffer state when checkpointing is supported."""
    if not self._is_initialized:
      return None

    if self._use_reverb and self.replay_buffer is not None:
      checkpoint_client = getattr(self.replay_buffer, "py_client", None)
      if checkpoint_client is None:
        return None
      checkpoint_client.checkpoint()
      return self.checkpoint_dir

    if self._tf_checkpoint_manager is None:
      return None
    return self._tf_checkpoint_manager.save()

  def load_checkpoint(self) -> Optional[str]:
    """Restore TFUniform replay-buffer state if a checkpoint exists."""
    if self._use_reverb:
      # Reverb restoration is handled by its checkpointer in load_replay_buffer.
      return self.checkpoint_dir

    if self._tf_checkpoint is None or self._tf_checkpoint_manager is None:
      return None

    latest_checkpoint = self._tf_checkpoint_manager.latest_checkpoint
    if latest_checkpoint is None:
      logger.info(
          "No TFUniform checkpoint found; starting with an empty buffer."
      )
      return None

    self._tf_checkpoint.restore(latest_checkpoint).expect_partial()
    logger.info(
        "Loaded TFUniform replay buffer checkpoint: %s", latest_checkpoint
    )
    return latest_checkpoint

  def close(self) -> None:
    """Close the replay buffer server and clean up resources."""
    if not self._is_initialized:
      return

    if self._use_reverb and self.server is not None:
      self.server.stop()
      logger.info("Replay buffer server stopped.")
    self._is_initialized = False
    self.server = None
    self.replay_buffer = None
    self.observer = None
    self._tf_checkpoint = None
    self._tf_checkpoint_manager = None
