"""Reinforcement learning replay buffers."""

import logging
from typing import Optional, Tuple, Any

import tensorflow as tf

# --- Try Reverb; fall back to TFUniform if unavailable (e.g., macOS/py3.11) ---
try:
  import reverb  # type: ignore
  from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
  _HAS_REVERB = True
except Exception:  # pragma: no cover
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

  def __init__(self, data_spec, capacity, checkpoint_dir, sequence_length: int = 2):
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

      reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(  # type: ignore[attr-defined]
          path=self.checkpoint_dir
      )

      reverb_server = reverb.Server(  # type: ignore[attr-defined]
          [table], port=None, checkpointer=reverb_checkpointer
      )

      replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(  # type: ignore[union-attr]
          self.data_spec,
          sequence_length=self.sequence_length,
          table_name=self.table_name,
          local_server=reverb_server,
      )

      observer = reverb_utils.ReverbAddTrajectoryObserver(  # type: ignore[union-attr]
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
      # For unbatched envs, batch_size=1 is standard.
      replay_buffer = TFUniformReplayBuffer(
          data_spec=self.data_spec,
          batch_size=1,
          max_length=self.capacity,
      )

      # In TFUniform, the driver observer is simply `replay_buffer.add_batch`.
      observer = replay_buffer.add_batch

      self.server = None
      self.replay_buffer = replay_buffer
      self.observer = observer
      self._is_initialized = True
      logger.info("Using TFUniformReplayBuffer fallback (dm-reverb not available).")
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

      reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(  # type: ignore[attr-defined]
          path=self.checkpoint_dir
      )

      reverb_server = reverb.Server(  # type: ignore[attr-defined]
          [table], port=None, checkpointer=reverb_checkpointer
      )

      replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(  # type: ignore[union-attr]
          self.data_spec,
          sequence_length=self.sequence_length,
          table_name=self.table_name,
          local_server=reverb_server,
      )

      observer = reverb_utils.ReverbAddTrajectoryObserver(  # type: ignore[union-attr]
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

    # ------ TFUniform fallback: no external checkpoint to load ------
    logger.info("TFUniform fallback: creating a fresh replay buffer (no checkpoint).")
    return self.create_replay_buffer()

  def get_replay_buffer_and_observer(self) -> Tuple[Any, Any]:
    """Return (replay_buffer, observer), creating if necessary."""
    if not self._is_initialized:
      return self.create_replay_buffer()
    return self.replay_buffer, self.observer

  # ---------------------------
  # Dataset / Introspection
  # ---------------------------
  def get_dataset(self, batch_size: int = 64, num_steps: Optional[int] = None) -> tf.data.Dataset:
    """Get a tf.data.Dataset for sampling from the replay buffer."""
    if not self._is_initialized:
      raise RuntimeError(
          "Replay buffer not initialized. Call create_replay_buffer or load_replay_buffer first."
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
