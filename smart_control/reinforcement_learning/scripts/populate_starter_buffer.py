"""Script to populate an initial replay buffer for RL training.

This creates a starter buffer with exploration data that can be used to
bootstrap the training process.
"""

import argparse
import logging
import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import trajectory

from smart_buildings.smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_buildings.smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_buildings.smart_control.reinforcement_learning.policies.schedule_policy import create_baseline_schedule_policy
from smart_buildings.smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_buildings.smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_buildings.smart_control.reinforcement_learning.utils.config import OUTPUT_DATA_PATH
from smart_buildings.smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
)
logger = logging.getLogger(__name__)


def populate_replay_buffer(
    buffer_name,
    buffer_capacity,
    steps_per_run,
    num_runs,
    sequence_length,
    env_gin_config_file_path,
):
  """Populates a replay buffer with initial exploration data.

  Args:
    buffer_name: Name with which to save replay buffer. Buffer will be at
      smart_control/reinforcement_learning/data/starter_buffers/{buffer_name}
    buffer_capacity: Maximum size of the replay buffer
    steps_per_run: Number of steps per actor run
    num_runs: Number of actor runs to perform
    sequence_length: Length of sequences to store in the replay buffer
    env_gin_config_file_path: Path to the environment configuration file

  Returns:
    The replay buffer.
  """

  buffer_path = os.path.join(
      OUTPUT_DATA_PATH,
      f'{buffer_name}_seqlen{sequence_length}_exp{num_runs*steps_per_run}',
  )
  logger.info('Buffer path: %s', buffer_path)

  # Create directory if it doesn't exist
  try:
    os.makedirs(
        os.path.dirname(buffer_path + '/anything-here'), exist_ok=False
    )  # added '/anything-here' such that the path is a directory
  except FileExistsError as err:
    logger.exception(
        'This buffer path already exists. This would override the existing'
        ' buffer. Please use another name'
    )
    raise FileExistsError('Buffer name already exists, would be overriden') from err  # pylint: disable=line-too-long

  # Load environment
  logger.info('Loading environment from standard config')
  collect_env = create_and_setup_environment(
      env_gin_config_file_path, metrics_path=None
  )

  # Wrap in TF environment
  collect_tf_env = tf_py_environment.TFPyEnvironment(collect_env)

  # Create policy for collection
  train_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_tf_env)

  collection_policy = create_baseline_schedule_policy(collect_tf_env)

  # Initialize replay buffer
  logger.info('Creating replay buffer at: %s', buffer_path)
  logger.info(
      'Buffer capacity: %d, Sequence length: %d',
      buffer_capacity,
      sequence_length,
  )

  # Get the policy's info spec
  policy_info_spec = collection_policy.info_spec

  # Create a trajectory spec properly
  collect_data_spec = trajectory.Trajectory(
      step_type=time_step_spec.step_type,
      observation=time_step_spec.observation,
      action=action_spec,
      policy_info=policy_info_spec,
      next_step_type=time_step_spec.step_type,
      reward=time_step_spec.reward,
      discount=time_step_spec.discount,
  )

  # Use this data spec when creating the replay buffer
  replay_manager = ReplayBufferManager(
      collect_data_spec,  # Use the complete data spec
      buffer_capacity,
      buffer_path,
      sequence_length=sequence_length,
  )

  replay_buffer, replay_buffer_observer = replay_manager.create_replay_buffer()

  # Create observers
  print_observer = PrintStatusObserver(
      status_interval_steps=1,  # Print status every step
      environment=collect_tf_env,
      replay_buffer=replay_buffer,
  )

  # Combine observers
  observers = CompositeObserver([print_observer, replay_buffer_observer])

  # Create collect actor
  logger.info('Setting up collect actor')
  collect_actor = actor.Actor(
      collect_tf_env.pyenv.envs[0],  # Use underlying PyEnv
      py_tf_eager_policy.PyTFEagerPolicy(collection_policy),
      steps_per_run=steps_per_run,
      train_step=train_step,
      observers=[observers],
  )

  # Run collection
  logger.info(
      'Starting collection for %d runs of %d steps each',
      num_runs,
      steps_per_run,
  )
  total_steps = 0

  for current_run in range(num_runs):
    # Run collection
    logger.info(
        'Run %d/%d (total steps so far: %d)',
        current_run + 1,
        num_runs,
        total_steps,
    )
    collect_actor.run()

    # Update total steps
    total_steps += steps_per_run

    # Checkpoint buffer periodically
    logger.info(
        'Completed run %d/%d. Checkpointing buffer...',
        current_run + 1,
        num_runs,
    )
    replay_buffer.py_client.checkpoint()

  # Final checkpoint and stats
  logger.info(
      'Completed all runs, total steps: %d. '
      'Checkpointing buffer one last time...',
      total_steps,
  )

  replay_buffer.py_client.checkpoint()
  logger.info('Final replay buffer size: %d frames', replay_buffer.num_frames())

  return replay_buffer


if __name__ == '__main__':

  config_filepath = os.path.join(CONFIG_PATH, 'sim_config_1_day.gin')

  # fmt: off
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description='Populate a replay buffer with initial exploration data')
  parser.add_argument('--buffer-name', type=str, required=True, help='Name to identify the saved replay buffer')
  parser.add_argument('--capacity', type=int, default=50000, help='Replay buffer capacity')
  parser.add_argument('--steps-per-run', type=int, default=100, help='Number of steps per actor run')
  parser.add_argument('--num-runs', type=int, default=5, help='Number of actor runs to perform')
  parser.add_argument('--sequence-length', type=int, default=2, help='Sequence length for the replay buffer')
  parser.add_argument('--env-gin-config-file-path', type=str, default=config_filepath, help='Environment config file')
  # pylint: enable=line-too-long
  # fmt: on
  args = parser.parse_args()

  populate_replay_buffer(
      buffer_name=args.buffer_name,
      buffer_capacity=args.capacity,
      steps_per_run=args.steps_per_run,
      num_runs=args.num_runs,
      sequence_length=args.sequence_length,
      env_gin_config_file_path=args.env_gin_config_file_path,
  )
