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


# LOGGING

logging.basicConfig(
    level=logging.INFO,
    # format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
    format='[%(message)s]',
)
logger = logging.getLogger(__name__)

# FLAGS

flags.DEFINE_string(
    name='experiment_name',
    default=None,
    help='Name of the experiment. This is used to save TensorBoard summaries',
    required=True,
)
flags.DEFINE_string(
    name='starter_buffer_path',
    default=None,
    help=(
        'Path to the starter replay buffer (e.g. "/path/to/my_buffer"). If not'
        ' supplied, will check the "starter_buffers" dir and use the most'
        ' recently generated buffer. Use the starter buffer generation script'
        ' to generate a starter buffer.'
    ),
    # required=True,
)
flags.DEFINE_string(
    name='scenario_config_path',
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


FLAGS = flags.FLAGS

# SCRIPT


def save_experiment_parameters(params, save_path):
  """
  Save experiment parameters to a JSON file.

  Args:
      params: Dictionary containing experiment parameters
      save_path: Path to save the parameters file
  """
  # Create a parameters file path
  params_file = os.path.join(save_path, 'experiment_parameters.json')

  # Add timestamp to parameters
  params['timestamp'] = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

  # Save parameters to file
  logger.info('Saving experiment parameters to %s', params_file)
  with open(params_file, 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=4)

  # Also save as a readable text file for quick reference
  params_txt = os.path.join(save_path, 'experiment_parameters.txt')
  with open(params_txt, 'w', encoding='utf-8') as f:
    f.write('Experiment Parameters:\n')
    f.write('=====================\n\n')
    for key, value in params.items():
      f.write(f'{key}: {value}\n')

  logger.info(
      'Experiment parameters saved to %s and %s', params_file, params_txt
  )


def train_agent(
    experiment_name: str,
    starter_buffer_path: str,
    scenario_config_path: str = ONE_DAY_CONFIG_FILEPATH,
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
  """
  Trains a reinforcement learning agent using a pre-populated replay buffer.

  Args:
      experiment_name: Name of the experiment
      starter_buffer_path: Path to the pre-populated replay buffer
      scenario_config_path: Path to the scenario configuration file
      agent_type: Type of agent to train ('sac' or 'td3')
      train_iterations: Number of training iterations
      collect_steps_per_iteration: Number of collection steps per training
        iteration
      batch_size: Batch size for training
      log_interval: Interval for logging training metrics
      eval_interval: Interval for evaluating the agent
      num_eval_episodes: Number of episodes for evaluation
      checkpoint_interval: Interval for checkpointing the replay buffer
      learner_iterations: Number of iterations to run the agent learner per
        training loop
  """

  # SETUP

  # Generate timestamp for summary directory
  current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
  experiment_dirname = f'{experiment_name}_{current_time}'
  summary_dir = os.path.join(RL_EXPERIMENT_RESULTS_DIR, experiment_dirname)
  logger.info(
      'Experiment results will be saved to %s', os.path.abspath(summary_dir)
  )

  try:
    os.makedirs(summary_dir, exist_ok=False)
  except FileExistsError as exc:
    logger.exception('Directory %s already exists. Exiting.', summary_dir)
    raise FileExistsError(
        f'Directory {summary_dir} already exists. Exiting.'
    ) from exc

  # Save experiment parameters
  experiment_params = {
      'starter_buffer_path': starter_buffer_path,
      'experiment_name': experiment_name,
      'agent_type': agent_type,
      'train_iterations': train_iterations,
      'collect_steps_per_iteration': collect_steps_per_iteration,
      'batch_size': batch_size,
      'log_interval': log_interval,
      'eval_interval': eval_interval,
      'num_eval_episodes': num_eval_episodes,
      'checkpoint_interval': checkpoint_interval,
      'learner_iterations': learner_iterations,
      'scenario_config_path': scenario_config_path,
  }
  save_experiment_parameters(experiment_params, summary_dir)

  # ENVIRONMENTS

  # Create train and eval environments
  logger.info(
      'Creating train and eval environments with scenario config path: %s',
      scenario_config_path,
  )
  metrics_dirpath = os.path.join(summary_dir, 'metrics')
  train_env = create_and_setup_environment(
      scenario_config_path, metrics_path=metrics_dirpath
  )
  eval_env = create_and_setup_environment(
      scenario_config_path, metrics_path=None
  )

  # Wrap in TF environments
  train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

  # Create global step for training
  train_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  # Get specs
  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_tf_env)

  # AGENT

  # Create agent based on type
  logger.info('Creating %s agent', agent_type)
  if agent_type.lower() == 'sac':
    logger.info('Creating SAC agent')
    agent = create_sac_agent(
        time_step_spec=time_step_spec, action_spec=action_spec
    )
  elif agent_type.lower() == 'ddpg':
    logger.info('Creating DDPG agent')
    agent = create_ddpg_agent(
        time_step_spec=time_step_spec, action_spec=action_spec
    )
  else:
    logger.exception('Unsupported agent type: %s', agent_type)
    raise ValueError(f'Unsupported agent type: {agent_type}')

  # Create policies
  collect_policy = agent.collect_policy
  eval_policy = greedy_policy.GreedyPolicy(agent.policy)

  # Set up metrics
  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(),
      tf_metrics.AverageEpisodeLengthMetric(),
  ]

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  # REPLAY BUFFER

  # Create a new buffer path in the experiment directory
  new_buffer_path = os.path.join(summary_dir, 'replay_buffer')
  os.makedirs(new_buffer_path, exist_ok=True)

  # Copy the original buffer to the new location
  logger.info(
      'Creating a copy of replay buffer from %s to %s',
      os.path.abspath(starter_buffer_path),
      os.path.abspath(new_buffer_path),
  )

  # First check if starter_buffer_path is a file or directory
  if os.path.isfile(starter_buffer_path):
    # If it's a file, copy it directly
    shutil.copy2(starter_buffer_path, new_buffer_path)
  else:
    # If it's a directory, copy all contents
    for item in os.listdir(starter_buffer_path):
      source_item = os.path.join(starter_buffer_path, item)
      dest_item = os.path.join(new_buffer_path, item)
      if os.path.isfile(source_item):
        shutil.copy2(source_item, dest_item)
      else:
        shutil.copytree(source_item, dest_item)

  logger.info('Replay buffer copied to %s', new_buffer_path)

  # Initialize replay buffer manager with the copied buffer path
  logger.info('Instantiating replay buffer manager with copied buffer')
  replay_manager = ReplayBufferManager(
      agent.collect_data_spec,
      50000,  # Use default capacity
      new_buffer_path,  # Use the copied buffer path
      sequence_length=2,
  )
  logger.info(
      'Replay buffer size before loading: %d frames',
      replay_manager.num_frames(),
  )

  # Load the copied replay buffer
  logger.info('Loading replay buffer from %s', new_buffer_path)
  replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()
  logger.info(
      'Replay buffer size after loading: %d frames', replay_manager.num_frames()
  )

  # Create dataset for sampling from the buffer
  logger.info('Creating dataset for sampling from replay buffer')
  dataset = replay_buffer.as_dataset(
      sample_batch_size=batch_size, num_steps=2, num_parallel_calls=3
  ).prefetch(3)

  # OBSERVERS

  # Create print observer for collection
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

  # Combine observers
  collect_observers = CompositeObserver(
      [print_observer, replay_buffer_observer]
  )

  # ACTORS

  # Create collect actor
  logger.info('Creating collect and eval actors')
  collect_actor = actor.Actor(
      train_env,
      py_tf_eager_policy.PyTFEagerPolicy(collect_policy),
      train_step,
      steps_per_run=collect_steps_per_iteration,
      metrics=actor.collect_metrics(1),
      observers=[collect_observers],
      summary_dir=os.path.join(summary_dir, 'collect'),
      summary_interval=1,
  )

  # Create eval actor
  logger.info('Creating eval actor')
  eval_actor = actor.Actor(
      eval_env,
      py_tf_eager_policy.PyTFEagerPolicy(eval_policy),
      train_step,
      episodes_per_run=num_eval_episodes,
      metrics=actor.eval_metrics(num_eval_episodes),
      observers=[eval_print_observer],
      summary_dir=os.path.join(summary_dir, 'eval'),
      summary_interval=1,
  )

  # LEARNER

  # Create learner
  saved_model_dirpath = os.path.join(summary_dir, 'policies')
  saved_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir=saved_model_dirpath,
      agent=agent,
      train_step=train_step,
      interval=eval_interval,
  )
  log_trigger = triggers.StepPerSecondLogTrigger(
      train_step=train_step, interval=log_interval
  )
  logger.info('Creating learner')
  agent_learner = learner.Learner(
      root_dir=summary_dir,
      train_step=train_step,
      agent=agent,
      experience_dataset_fn=lambda: dataset,
      summary_interval=1,
      triggers=[saved_model_trigger, log_trigger],
  )
  # > https://github.com/tensorflow/tensorflow/issues/59869

  # Main training loop
  logger.info('Starting training for %d iterations', train_iterations)

  # Reset metrics
  for m in train_metrics:
    m.reset()

  # Main training loop
  for i in tqdm(range(train_iterations)):
    # Get current training step value before operations
    current_step = train_step.numpy()
    logger.info(
        'Starting training loop iteration %d (step %d)', i, current_step
    )

    # Evaluate periodically
    if i % eval_interval == 0:
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
    agent_learner.run(iterations=learner_iterations)

    # Checkpoint replay buffer periodically based on the new argument
    if i % checkpoint_interval == 0:
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

  logger.info('Agent training completed. Saved models in %s', summary_dir)
  return agent


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

  train_agent(
      starter_buffer_path=buffer_dirpath,
      scenario_config_path=FLAGS.scenario_config_path,
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


if __name__ == '__main__':

  app.run(main)
