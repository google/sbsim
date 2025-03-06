import os
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from smart_control.refactor.utils.config import OUTPUT_DATA_PATH


class ReplayBufferManager:
    """Manager for creating and interacting with Reverb replay buffers.
    
    This class simplifies the setup, interaction, and checkpointing of Reverb replay
    buffers for reinforcement learning agents. It provides methods to create a new 
    buffer, add data, sample from the buffer, and save/restore buffer state.
    """
    
    def __init__(self, 
                 data_spec: Any,
                 capacity: int = 50000,
                 checkpoint_dir: str = f"{OUTPUT_DATA_PATH}/reverb_checkpoint",
                 table_name: str = 'uniform_table',
                 sequence_length: int = 2,
                 port: Optional[int] = None,
                 min_size_to_sample: int = 1,
                 stride_length: int = 1):
        """Initialize the ReplayBufferManager.
        
        Args:
            data_spec: The data specification for items stored in the buffer.
            capacity: Maximum number of items stored in the buffer.
            checkpoint_dir: Directory path for saving checkpoints.
            table_name: Name of the reverb table.
            sequence_length: Length of sequences sampled from the buffer.
            port: Port for the reverb server. If None, a port is automatically chosen.
            min_size_to_sample: Minimum number of items in buffer before sampling.
            stride_length: Stride length for adding trajectories to buffer.
        """
        self.data_spec = data_spec
        self.capacity = capacity
        self.checkpoint_dir = checkpoint_dir
        self.table_name = table_name
        self.sequence_length = sequence_length
        self.port = port
        self.min_size_to_sample = min_size_to_sample
        self.stride_length = stride_length
        
        # Initialize as None, to be created in create_replay_buffer
        self.server = None
        self.replay_buffer = None
        self.observer = None
        self._is_initialized = False
    
    def create_replay_buffer(self) -> Tuple[reverb_replay_buffer.ReverbReplayBuffer, reverb_utils.ReverbAddTrajectoryObserver]:
        """Create and initialize the replay buffer.
        
        Returns:
            A tuple of (replay_buffer, observer) for interacting with the buffer.
        """
        # Create the table
        table = reverb.Table(
            name=self.table_name,
            max_size=self.capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(self.min_size_to_sample),
        )
        
        # Set up checkpointing
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(path=self.checkpoint_dir)
        
        # Create the server
        self.server = reverb.Server(
            tables=[table], 
            port=self.port, 
            checkpointer=checkpointer
        )
        
        # Create the replay buffer
        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.data_spec,
            sequence_length=self.sequence_length,
            table_name=self.table_name,
            local_server=self.server,
        )
        
        # Create the observer to add data to the buffer
        self.observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client=self.replay_buffer.py_client, 
            table_name=self.table_name, 
            sequence_length=self.sequence_length, 
            stride_length=self.stride_length
        )
        
        self._is_initialized = True
        logging.info(f"Replay buffer created with server running on port {self.server.port}")
        
        return self.replay_buffer, self.observer
    
    def get_replay_buffer_and_observer(self) -> Tuple[reverb_replay_buffer.ReverbReplayBuffer, reverb_utils.ReverbAddTrajectoryObserver]:
        """Get the replay buffer and observer. Creates them if not already initialized.
        
        Returns:
            A tuple of (replay_buffer, observer).
        """
        if not self._is_initialized:
            return self.create_replay_buffer()
        return self.replay_buffer, self.observer
    
    def get_dataset(self, batch_size: int = 64, num_steps: Optional[int] = None) -> tf.data.Dataset:
        """Get a TensorFlow dataset for sampling from the replay buffer.
        
        Args:
            batch_size: Number of sequences to sample in each batch.
            num_steps: Number of steps to sample for each sequence. If None, 
                       defaults to sequence_length.
                       
        Returns:
            A TensorFlow dataset that samples from the replay buffer.
        """
        if not self._is_initialized:
            raise RuntimeError("Replay buffer not initialized. Call create_replay_buffer first.")
        
        if num_steps is None:
            num_steps = self.sequence_length
        
        return self.replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=num_steps
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