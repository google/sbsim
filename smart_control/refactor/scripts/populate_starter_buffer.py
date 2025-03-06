#!/usr/bin/env python3
"""
Script to populate an initial replay buffer for RL training.
This creates a starter buffer with exploration data that can be used
to bootstrap the training process.
"""

import os
import logging

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.train import actor
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import spec_utils

from smart_control.refactor.observers import (PrintStatusObserver, CompositeObserver)
from smart_control.refactor.utils.config import CONFIG_PATH, OUTPUT_DATA_PATH
from smart_control.learning.reinforcement_learning.sac.learning_utils import load_environment
from smart_control.refactor.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.refactor.policies.schedule_policy import create_baseline_schedule_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]'
)
logger = logging.getLogger(__name__)

def populate_replay_buffer(
    buffer_capacity=50000,
    buffer_path=None,
    steps_per_run=100,
    num_runs=100
):
    """
    Populates a replay buffer with initial exploration data.
    
    Args:
        buffer_capacity: Maximum size of the replay buffer
        buffer_path: Directory to save the replay buffer
        steps_per_run: Number of steps per actor run
        num_runs: Number of actor runs to perform
        use_random_policy: Whether to use a random policy for exploration (True) 
                          or create a SAC agent (False)
    """
    # Use the standard config file
    scenario_config_path = os.path.join(CONFIG_PATH, "sim_config_4_day.gin")
    
    # Default buffer path if not provided
    if buffer_path is None:
        buffer_path = os.path.join(OUTPUT_DATA_PATH, "initial_replay_buffer")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
    
    # Load environment
    logger.info("Loading environment from standard config")
    collect_env = load_environment(scenario_config_path)
    collect_env._metrics_path = None  # Collection env doesn't need metrics
    collect_env._occupancy_normalization_constant = 125.0
    
    # Wrap in TF environment
    collect_tf_env = tf_py_environment.TFPyEnvironment(collect_env)
    
    # Create policy for collection
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    _, __, time_step_spec = spec_utils.get_tensor_specs(collect_tf_env)

    collection_policy = create_baseline_schedule_policy(collect_tf_env)
    
    # Initialize replay buffer
    logger.info(f"Creating replay buffer at: {buffer_path}")
    logger.info(f"Buffer capacity: {buffer_capacity}, Sequence length: 2")
    
    # Always use sequence_length of 2
    replay_manager = ReplayBufferManager(
        time_step_spec,
        buffer_capacity,
        buffer_path,
        sequence_length=2
    )
    
    replay_buffer, replay_buffer_observer = replay_manager.create_replay_buffer()
    
    # Create observers
    print_observer = PrintStatusObserver(
        status_interval_steps=1,  # Print status every 100 steps
        environment=collect_tf_env,
        replay_buffer=replay_buffer
    )
    
    # Combine observers
    observers = CompositeObserver([print_observer, replay_buffer_observer])
    
    # Create collect actor
    logger.info("Setting up collect actor")
    collect_actor = actor.Actor(
        collect_tf_env.pyenv.envs[0],  # Use underlying PyEnv
        py_tf_eager_policy.PyTFEagerPolicy(collection_policy),
        steps_per_run=steps_per_run,
        train_step=train_step,
        observers=[observers]
    )
    
    # Run collection
    logger.info(f"Starting collection for {num_runs} runs of {steps_per_run} steps each")
    total_steps = 0
    
    for current_run in range(num_runs):
        # Run collection
        logger.info(f"Run {current_run+1}/{num_runs} (total steps so far: {total_steps})")
        collect_actor.run()
        
        # Update total steps
        total_steps += steps_per_run
        
        # Checkpoint buffer periodically
        logger.info(f"Completed run {current_run+1}/{num_runs}. Checkpointing buffer...")
        replay_buffer.py_client.checkpoint()
    
    # Final checkpoint and stats
    logger.info(f"Completed all runs, total steps: {total_steps}. Checkpointing buffer one last time...")
    replay_buffer.py_client.checkpoint()
    logger.info(f"Final replay buffer size: {replay_buffer.num_frames()} frames")
    
    return replay_buffer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate a replay buffer with initial exploration data')
    parser.add_argument('--capacity', type=int, default=50000, help='Replay buffer capacity')
    parser.add_argument('--buffer-path', type=str, default=None, help='Path to save the replay buffer')
    parser.add_argument('--steps-per-run', type=int, default=10, help='Number of steps per actor run')
    parser.add_argument('--num-runs', type=int, default=2, help='Number of actor runs to perform')
    
    args = parser.parse_args()
    
    populate_replay_buffer(
        buffer_capacity=args.capacity,
        buffer_path=args.buffer_path,
        steps_per_run=args.steps_per_run,
        num_runs=args.num_runs
    )
