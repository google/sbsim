"""Script to populate an initial replay buffer for RL training.

This creates a starter buffer with exploration data that can be used to
bootstrap the training process.
"""

import logging
import os
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import trajectory

from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.policies.schedule_policy import create_baseline_schedule_policy
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.constants import ONE_DAY_CONFIG_FILEPATH
from smart_control.reinforcement_learning.utils.constants import RL_STARTER_BUFFERS_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
from smart_control.utils.constants import ROOT_DIR

# this is used by the gin config (see "sim_config_1_day.gin")
# pylint:disable-next=unused-import
from smart_control.reinforcement_learning.utils.config import get_histogram_path  # isort:skip


# LOGGING

logger = logging.getLogger(__name__)

# logging.basicConfig(
#    level=logging.INFO,
#    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
# )

logging.basicConfig(
    level=logging.INFO,
    format='[%(message)s]',
)


# FLAGS

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='buffer_name',
    default='default',
    help=(
        'Name used to identify the replay buffer. Corresponds with directory'
        ' name where files will be saved.'
    ),
)
flags.DEFINE_string(
    name='config_filepath',
    default=ONE_DAY_CONFIG_FILEPATH,
    help='Environment config file',
)
flags.DEFINE_integer(
    name='capacity', default=50000, help='Replay buffer capacity'
)
flags.DEFINE_integer(
    name='steps_per_run', default=100, help='Number of steps per actor run'
)
flags.DEFINE_integer(
    name='num_runs', default=5, help='Number of actor runs to perform'
)
flags.DEFINE_integer(
    name='sequence_length',
    default=2,
    help='Sequence length for the replay buffer',
)


class StarterBufferGenerator:
  """Populates a replay buffer with initial exploration data.

  Args:
    buffer_name: Name of directory where the replay buffer will be saved.
    config_filepath: Path to the environment gin configuration file.
    buffer_capacity: Maximum size of the replay buffer.
    steps_per_run: Number of steps per actor run.
    num_runs: Number of actor runs to perform.
    sequence_length: Length of sequences to store in the replay buffer.
  """

  def __init__(
      self,
      buffer_name: str = 'default',
      config_filepath: str = ONE_DAY_CONFIG_FILEPATH,
      buffer_capacity: int = 50000,
      steps_per_run: int = 100,
      num_runs: int = 5,
      sequence_length: int = 2,
  ):
    self.buffer_name = buffer_name
    self.config_filepath = config_filepath
    self.buffer_capacity = int(buffer_capacity)
    self.steps_per_run = int(steps_per_run)
    self.num_runs = int(num_runs)
    self.sequence_length = int(sequence_length)

    self.buffer_dirpath = os.path.join(RL_STARTER_BUFFERS_DIR, self.buffer_name)

  def populate(self) -> ReverbReplayBuffer:
    """Returns: The replay buffer."""
    logger.info('Buffer dirpath: %s', os.path.abspath(self.buffer_dirpath))

    os.makedirs(self.buffer_dirpath, exist_ok=True)
    done_filepath = os.path.join(self.buffer_dirpath, 'DONE')
    if os.path.isfile(done_filepath):
      raise FileExistsError(
          'Starter buffer already exists, would be overwritten'
      )

    # Load environment
    logger.info(
        'Loading environment from config: %s',
        os.path.abspath(self.config_filepath),
    )
    collect_env = create_and_setup_environment(
        self.config_filepath, metrics_path=None
    )

    # Wrap in TF environment
    collect_tf_env = tf_py_environment.TFPyEnvironment(collect_env)

    # Create policy for collection
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_tf_env)

    collection_policy = create_baseline_schedule_policy(collect_tf_env)

    # Initialize replay buffer
    logger.info(
        'Creating replay buffer at: %s', os.path.abspath(self.buffer_dirpath)
    )
    logger.info(
        'Buffer capacity: %d, Sequence length: %d',
        self.buffer_capacity,
        self.sequence_length,
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
        data_spec=collect_data_spec,  # Use the complete data spec
        capacity=self.buffer_capacity,
        checkpoint_dir=self.buffer_dirpath,
        sequence_length=self.sequence_length,
    )

    replay_buffer, replay_buffer_observer = (
        replay_manager.create_replay_buffer()
    )

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
        env=collect_tf_env.pyenv.envs[0],  # Use underlying PyEnv
        policy=py_tf_eager_policy.PyTFEagerPolicy(collection_policy),
        steps_per_run=self.steps_per_run,
        train_step=train_step,
        observers=[observers],
    )

    # Run collection
    logger.info(
        'Starting collection for %d runs of %d steps each',
        self.num_runs,
        self.steps_per_run,
    )
    total_steps = 0

    for current_run in range(self.num_runs):
      # Run collection
      logger.info(
          'Run %d/%d (total steps so far: %d)',
          current_run + 1,
          self.num_runs,
          total_steps,
      )
      collect_actor.run()

      # Update total steps
      total_steps += self.steps_per_run

      # Checkpoint buffer periodically
      logger.info(
          'Completed run %d/%d. Checkpointing buffer...',
          current_run + 1,
          self.num_runs,
      )
      replay_buffer.py_client.checkpoint()

    # Final checkpoint and stats
    logger.info(
        'Completed all runs, total steps: %d. '
        'Checkpointing buffer one last time...',
        total_steps,
    )

    replay_buffer.py_client.checkpoint()
    logger.info(
        'Final replay buffer size: %d frames', replay_buffer.num_frames()
    )

    return replay_buffer


def main(argv: Sequence[str]):
  """When running absl app, we need the `argv` param, even though it is unused.

  See:

    + https://abseil.io/docs/python/guides/app
    + https://google.github.io/styleguide/pyguide.html#317-main
    + go/python-readability-advice#unused_argv
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config_filepath = FLAGS.config_filepath
  if not os.path.isabs(config_filepath):
    config_filepath = os.path.join(ROOT_DIR, config_filepath)

  buffer_generator = StarterBufferGenerator(
      buffer_name=FLAGS.buffer_name,
      config_filepath=config_filepath,
      buffer_capacity=FLAGS.capacity,
      steps_per_run=FLAGS.steps_per_run,
      num_runs=FLAGS.num_runs,
      sequence_length=FLAGS.sequence_length,
  )
  buffer_generator.populate()


if __name__ == '__main__':

  app.run(main)
