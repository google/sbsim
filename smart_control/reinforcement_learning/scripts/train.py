"""
Script to train a reinforcement learning agent using a pre-populated replay
buffer.

This script sets up the training process with separate collection and evaluation
components.
"""

# OK so we are running into an error
# TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
# https://github.com/tensorflow/tensorflow/issues/59869
# As a workaround, we need to set this env var before loading tensorflow
# https://github.com/GrahamDumpleton/wrapt/issues/231#issuecomment-1455800902
# fmt: off
import os  # isort:skip
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
# fmt: on

# pylint:disable=wrong-import-position
from datetime import datetime
import json
import logging
import shutil
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tqdm import tqdm

from smart_control.reinforcement_learning.agents.ddpg_agent import create_ddpg_agent
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent
from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.constants import ONE_DAY_CONFIG_FILEPATH
from smart_control.reinforcement_learning.utils.constants import RL_EXPERIMENT_RESULTS_DIR
from smart_control.reinforcement_learning.utils.constants import RL_STARTER_BUFFERS_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# from smart_control.utils.constants import ROOT_DIR
# from smart_control.utils.constants import DEFAULT_CONFIG_FILEPATH

# this is used by the gin config (see "sim_config_day1.gin")
# pylint:disable-next=unused-import
from smart_control.reinforcement_learning.utils.config import get_histogram_path  # isort:skip

# pylint:enable=wrong-import-position

DEFAULT_STARTER_BUFFER_DIRPATH = os.path.join(RL_STARTER_BUFFERS_DIR, 'default')

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    # format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
    format='[%(message)s]',
)
logger = logging.getLogger(__name__)

# FLAGS

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='experiment_name',
    default=None,
    help='Name of the experiment. This is used to save TensorBoard summaries',
    required=True,
)
flags.DEFINE_string(
    name='starter_buffer_path',
    default=DEFAULT_STARTER_BUFFER_DIRPATH,
    help='Path to the starter replay buffer (e.g. "/path/to/my_buffer").',
    # required=True,
)
flags.DEFINE_string(
    name='config_filepath',
    default=ONE_DAY_CONFIG_FILEPATH,  # DEFAULT_CONFIG_FILEPATH,
    help='Path to the scenario config file (e.g. "/path/to/sim_config.gin")',
)
flags.DEFINE_enum(
    name='agent_type',
    default='sac',
    enum_values=['sac', 'td3', 'ddpg'],
    help='Type of agent to train (sac, td3, or ddpg)',
)
flags.DEFINE_integer(
    name='train_iterations',
    default=300,
    help='Number of training iterations',
)
flags.DEFINE_integer(
    name='collect_steps_per_training_iteration',
    default=50,
    help='Number of collection steps per iteration',
)
flags.DEFINE_integer(
    name='batch_size',
    default=256,
    help=(
        'Batch size for training (each gradient update uses this many '
        'elements from the replay buffer batched)'
    ),
)
flags.DEFINE_integer(
    name='eval_interval',
    default=10,
    help='Interval for evaluating the agent',
)
flags.DEFINE_integer(
    name='num_eval_episodes',
    default=1,
    help='Number of episodes for evaluation',
)
flags.DEFINE_integer(
    name='log_interval',
    default=1,
    help='Interval for logging training metrics',
)
flags.DEFINE_integer(
    name='checkpoint_interval',
    default=10,
    help='Interval for checkpointing the replay buffer',
)
flags.DEFINE_integer(
    name='learner_iterations',
    default=200,
    help=(
        'Number of iterations (gradient updates) to run the agent '
        'learner per training loop'
    ),
)


class RLAgentTrainer:
  """Trains a reinforcement learning agent using a pre-populated replay buffer.

  Args:
      experiment_name: Name of the experiment. Corresponds with the name of a
        directory where results will be saved.
      starter_buffer_path: Path to the pre-populated replay buffer directory.
      config_filepath: Path to the scenario configuration file.
      agent_type: Type of agent to train ('sac', 'td3', 'ddpg').
      train_iterations: Number of training iterations.
      collect_steps_per_iteration: Number of collection steps per training
        iteration.
      batch_size: Batch size for training.
      log_interval: Interval for logging training metrics.
      eval_interval: Interval for evaluating the agent.
      num_eval_episodes: Number of episodes for evaluation.
      checkpoint_interval: Interval for checkpointing the replay buffer.
      learner_iterations: Number of iterations to run the agent learner per
        training loop.
  """

  def __init__(
      self,
      experiment_name: str,
      starter_buffer_path: str = DEFAULT_STARTER_BUFFER_DIRPATH,
      config_filepath: str = ONE_DAY_CONFIG_FILEPATH,
      agent_type: str = 'sac',
      train_iterations: int = 100000,
      collect_steps_per_iteration: int = 1,
      batch_size: int = 256,
      log_interval: int = 100,
      eval_interval: int = 1000,
      num_eval_episodes: int = 5,
      checkpoint_interval: int = 1000,
      learner_iterations: int = 200,
  ):
    self.experiment_name = experiment_name
    self.starter_buffer_dirpath = starter_buffer_path
    self.config_filepath = config_filepath
    self.agent_type = agent_type
    self.train_iterations = int(train_iterations)
    self.collect_steps_per_iteration = int(collect_steps_per_iteration)
    self.batch_size = int(batch_size)
    self.log_interval = int(log_interval)
    self.eval_interval = int(eval_interval)
    self.num_eval_episodes = int(num_eval_episodes)
    self.checkpoint_interval = int(checkpoint_interval)
    self.learner_iterations = int(learner_iterations)

    if self.agent_type not in ['sac', 'ddpg']:
      raise ValueError(
          'Agent {self.agent_type} has not (yet) been implemented. Please'
          " choose one of: ['sac', 'ddpg']."
      )

    # todo: validate all integers are greater than zero

    self.experiment_dirname = self.experiment_name.replace(' ', '')
    self.results_dirpath = os.path.join(
        RL_EXPERIMENT_RESULTS_DIR, self.experiment_dirname
    )

    # these will be set later during training:
    self.train_env = None
    self.eval_env = None
    self.agent = None

  @property
  def done_filepath(self):
    """The DONE file is a convention for replay buffers. We are borrowing it.
    After the agent is trained we will create this file.
    """
    return os.path.join(self.results_dirpath, 'DONE')

  def mark_as_complete(self):
    """Create the DONE file to indicate the agent has completed its training."""
    with open(self.done_filepath, 'w', encoding='utf-8') as f:
      f.write('Training Complete!')

  def setup_results_dir(self):
    logger.info(
        'Experiment results will be saved to %s',
        os.path.abspath(self.results_dirpath),
    )

    # try:
    #  os.makedirs(self.results_dirpath, exist_ok=False)
    # except FileExistsError as exc:
    #  logger.exception(
    #      'Directory %s already exists. Exiting.', self.results_dirpath
    #  )
    #  raise FileExistsError(
    #      f'Directory {self.results_dirpath} already exists. Exiting.'
    #  ) from exc
    os.makedirs(self.results_dirpath, exist_ok=True)
    # when testing we are creating the dir beforehand, check for results instead
    if os.path.isfile(self.done_filepath):
      raise FileExistsError('Results directory already exists')

  @property
  def experiment_params(self):
    return {
        'experiment_name': self.experiment_name,
        'config_filepath': os.path.abspath(self.config_filepath),
        'starter_buffer_path': os.path.abspath(self.starter_buffer_dirpath),
        'agent_type': self.agent_type,
        'train_iterations': self.train_iterations,
        'collect_steps_per_iteration': self.collect_steps_per_iteration,
        'batch_size': self.batch_size,
        'log_interval': self.log_interval,
        'eval_interval': self.eval_interval,
        'num_eval_episodes': self.num_eval_episodes,
        'checkpoint_interval': self.checkpoint_interval,
        'learner_iterations': self.learner_iterations,
    }

  @property
  def params_json_filepath(self):
    return os.path.join(self.results_dirpath, 'experiment_parameters.json')

  @property
  def params_txt_filepath(self):
    return os.path.join(self.results_dirpath, 'experiment_parameters.txt')

  def save_experiment_params(self, params: dict = None, save_path: str = None):
    """
    Save experiment parameters to a JSON file, as well as to a TXT file.

    Args:
        params: Dictionary containing experiment parameters.
        save_path: Path to save the parameters file.
    """
    params = params or self.experiment_params
    params['timestamp'] = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

    save_path = save_path or self.results_dirpath

    logger.info(
        'Saving experiment parameters to %s',
        os.path.abspath(self.params_json_filepath),
    )
    with open(self.params_json_filepath, 'w', encoding='utf-8') as f:
      json.dump(params, f, indent=4)

    logger.info(
        'Saving experiment parameters to %s',
        os.path.abspath(self.params_txt_filepath),
    )
    with open(self.params_txt_filepath, 'w', encoding='utf-8') as f:
      f.write('Experiment Parameters:\n')
      f.write('=====================\n\n')
      for key, value in params.items():
        f.write(f'{key}: {value}\n')

  def copy_replay_buffer(self):
    # Create a new buffer path in the experiment directory
    new_buffer_path = os.path.join(self.results_dirpath, 'replay_buffer')
    os.makedirs(new_buffer_path, exist_ok=True)

    # Copy the original buffer to the new location
    logger.info(
        'Creating a copy of replay buffer from %s to %s',
        os.path.abspath(self.starter_buffer_dirpath),
        os.path.abspath(new_buffer_path),
    )

    # First check if starter_buffer_path is a file or directory
    if os.path.isfile(self.starter_buffer_dirpath):
      # If it's a file, copy it directly
      shutil.copy2(self.starter_buffer_dirpath, new_buffer_path)
    else:
      # If it's a directory, copy all contents
      for item in os.listdir(self.starter_buffer_path):
        source_item = os.path.join(self.starter_buffer_path, item)
        dest_item = os.path.join(new_buffer_path, item)
        if os.path.isfile(source_item):
          shutil.copy2(source_item, dest_item)
        else:
          shutil.copytree(source_item, dest_item)

    logger.info('Replay buffer copied to %s', new_buffer_path)
    return new_buffer_path

  def create_agent(self, action_spec, time_step_spec):
    logger.info('Creating %s agent', self.agent_type)
    if self.agent_type.lower() == 'sac':
      logger.info('Creating SAC agent')
      agent = create_sac_agent(
          time_step_spec=time_step_spec, action_spec=action_spec
      )
    elif self.agent_type.lower() == 'ddpg':
      logger.info('Creating DDPG agent')
      agent = create_ddpg_agent(
          time_step_spec=time_step_spec, action_spec=action_spec
      )
    else:
      logger.exception('Unsupported agent type: %s', self.agent_type)
      raise ValueError(f'Unsupported agent type: {self.agent_type}')

    return agent

  @property
  def metrics_dirpath(self):
    return os.path.join(self.results_dirpath, 'metrics')

  @property
  def collect_dirpath(self):
    return os.path.join(self.results_dirpath, 'collect')

  @property
  def eval_dirpath(self):
    return os.path.join(self.results_dirpath, 'eval')

  @property
  def saved_model_dirpath(self):
    return os.path.join(self.results_dirpath, 'policies')

  def train_agent(self) -> tf_agent.TFAgent:
    self.setup_results_dir()
    self.save_experiment_parameters()

    # ENVIRONMENTS

    logger.info(
        'Creating train and eval environments with scenario config path: %s',
        self.config_filepath,
    )
    # metrics_dirpath = os.path.join(self.results_dirpath, 'metrics')
    train_env = create_and_setup_environment(
        self.config_filepath, metrics_path=self.metrics_dirpath
    )
    eval_env = create_and_setup_environment(
        self.config_filepath, metrics_path=None
    )

    # Wrap in TF environments
    train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

    # AGENT

    # Create global step for training
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    # Get specs
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_tf_env)

    # Create agent based on type
    self.agent = self.create_agent(action_spec, time_step_spec)

    # Create policies
    collect_policy = self.agent.collect_policy
    eval_policy = greedy_policy.GreedyPolicy(self.agent.policy)

    # Set up metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=self.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=self.num_eval_episodes
        ),
    ]

    # REPLAY BUFFER

    # Create a new buffer path in the experiment directory
    new_buffer_path = self.copy_starter_buffer()

    # Initialize replay buffer manager with the copied buffer path
    logger.info('Instantiating replay buffer manager with copied buffer')
    replay_manager = ReplayBufferManager(
        data_spec=self.agent.collect_data_spec,
        capacity=50000,  # Use default capacity
        checkpoint_dir=new_buffer_path,  # Use the copied buffer path
        sequence_length=2,
        # should we keep these defaults, or use the dynamic parameter values?
    )
    logger.info(
        'Replay buffer size before loading: %d frames',
        replay_manager.num_frames(),
    )

    # Load the copied replay buffer
    logger.info('Loading replay buffer from %s', new_buffer_path)
    replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()
    logger.info(
        'Replay buffer size after loading: %d frames',
        replay_manager.num_frames(),
    )

    # Create dataset for sampling from the buffer
    logger.info('Creating dataset for sampling from replay buffer')
    dataset = replay_buffer.as_dataset(
        sample_batch_size=self.batch_size, num_steps=2, num_parallel_calls=3
    ).prefetch(3)

    # OBSERVERS

    print_observer = PrintStatusObserver(
        status_interval_steps=1,  # Print status every step
        environment=train_tf_env,
        replay_buffer=replay_buffer,
    )

    eval_print_observer = PrintStatusObserver(
        status_interval_steps=1,
        environment=eval_tf_env,
        replay_buffer=replay_buffer,
    )

    collect_observers = CompositeObserver(
        [print_observer, replay_buffer_observer]
    )

    # ACTORS

    # Create collect actor
    logger.info('Creating collect actor...')
    # collect_dirpath = os.path.join(self.results_dirpath, 'collect')
    collect_actor = actor.Actor(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(collect_policy),
        train_step,
        steps_per_run=self.collect_steps_per_iteration,
        metrics=actor.collect_metrics(1),
        observers=[collect_observers],
        summary_dir=self.collect_dirpath,
        summary_interval=1,
    )

    # Create eval actor
    logger.info('Creating eval actor...')
    # eval_dirpath = os.path.join(self.results_dirpath, 'eval')
    eval_actor = actor.Actor(
        env=eval_env,
        policy=py_tf_eager_policy.PyTFEagerPolicy(eval_policy),
        train_step=train_step,
        episodes_per_run=self.num_eval_episodes,
        metrics=actor.eval_metrics(self.num_eval_episodes),
        observers=[eval_print_observer],
        summary_dir=self.eval_dirpath,
        summary_interval=1,
    )

    # LEARNER

    # Create learner
    # https://github.com/tensorflow/tensorflow/issues/59869
    # saved_model_dirpath = os.path.join(self.results_dirpath, 'policies')
    saved_model_trigger = triggers.PolicySavedModelTrigger(
        saved_model_dir=self.saved_model_dirpath,
        agent=self.agent,
        train_step=train_step,
        interval=self.eval_interval,
    )
    log_trigger = triggers.StepPerSecondLogTrigger(
        train_step=train_step, interval=self.log_interval
    )
    logger.info('Creating learner')
    agent_learner = learner.Learner(
        root_dir=self.results_dirpath,
        train_step=train_step,
        agent=self.agent,
        experience_dataset_fn=lambda: dataset,
        summary_interval=1,
        triggers=[saved_model_trigger, log_trigger],
    )

    # Main training loop
    logger.info('Starting training for %d iterations', self.train_iterations)

    # Reset metrics
    for m in train_metrics:
      m.reset()

    # Main training loop
    for i in tqdm(range(self.train_iterations)):
      # Get current training step value before operations
      current_step = train_step.numpy()
      logger.info(
          'Starting training loop iteration %d (step %d)', i, current_step
      )

      # Evaluate periodically
      if i % self.eval_interval == 0:
        logger.info('Evaluating at iteration %d (step %d)', i, current_step)
        eval_actor.run()

        # Write eval summaries with the current global step
        with eval_actor.summary_writer.as_default():
          for m in eval_metrics:
            tf.summary.scalar(m.name, m.result(), step=current_step)
          eval_actor.summary_writer.flush()

      # Collect experience
      logger.info(
          'Starting collection for loop iteration %d (step %d)', i, current_step
      )
      collect_actor.run()

      # Write collect summaries with the current global step
      with collect_actor.summary_writer.as_default():
        for m in train_metrics:
          tf.summary.scalar(m.name, m.result(), step=current_step)
        collect_actor.summary_writer.flush()

      # Train the agent using the specified learner iterations
      # This will internally increment the train_step
      logger.info('Training agent for loop iteration %d', i)
      agent_learner.run(iterations=self.learner_iterations)

      # Checkpoint replay buffer periodically based on the new argument
      if i % self.checkpoint_interval == 0:
        logger.info('Checkpointing replay buffer')
        replay_buffer.py_client.checkpoint()

      train_step.assign_add(1)

    # Final checkpoint and evaluation
    logger.info(
        'Training complete. Performing final evaluation and checkpointing.'
    )
    replay_buffer.py_client.checkpoint()
    eval_actor.run()

    # Write final evaluation metrics with the final step
    with eval_actor.summary_writer.as_default():
      current_step = train_step.numpy()
      for m in eval_metrics:
        tf.summary.scalar(m.name, m.result(), step=current_step)
        logger.info('Final Eval %s: %s', m.name, m.result())
      eval_actor.summary_writer.flush()

    self.mark_as_complete()
    logger.info(
        'Agent training completed. Saved models in %s',
        os.path.abspath(self.results_dirpath),
    )
    return self.agent


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  experiment_name = FLAGS.experiment_name
  experiment_name = experiment_name.replace(' ', '_')

  # STARTER BUFFER DIRPATH:
  buffer_dirpath = FLAGS.starter_buffer_path
  if not buffer_dirpath:
    buffer_names = [d for d in os.listdir(RL_STARTER_BUFFERS_DIR) if 'buffer' in d]  # pylint:disable=line-too-long
    if any(buffer_names):
      buffer_name = buffer_names[-1]
      print('USING MOST RECENTLY GENERATED STARTER BUFFER:', buffer_name)
      buffer_dirpath = os.path.join(RL_STARTER_BUFFERS_DIR, buffer_name)
    else:
      raise ValueError(
          'There are no starter buffer files available. Please generate one'
          ' using the starter buffer generation script.'
      )
  if not os.path.isabs(buffer_dirpath):
    buffer_dirpath = os.path.join(RL_STARTER_BUFFERS_DIR, buffer_dirpath)

  trainer = RLAgentTrainer(
      starter_buffer_path=buffer_dirpath,
      config_filepath=FLAGS.config_filepath,
      experiment_name=experiment_name,
      agent_type=FLAGS.agent_type,
      train_iterations=FLAGS.train_iterations,
      collect_steps_per_iteration=FLAGS.collect_steps_per_training_iteration,
      batch_size=FLAGS.batch_size,
      eval_interval=FLAGS.eval_interval,
      num_eval_episodes=FLAGS.num_eval_episodes,
      log_interval=FLAGS.log_interval,
      checkpoint_interval=FLAGS.checkpoint_interval,
      learner_iterations=FLAGS.learner_iterations,
  )
  trainer.train_agent()


if __name__ == '__main__':

  app.run(main)
