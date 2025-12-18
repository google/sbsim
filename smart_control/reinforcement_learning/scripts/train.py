"""
Script to train a reinforcement learning agent using a pre-populated replay
buffer.

This script sets up the training process with separate collection and evaluation
components.
"""

import smart_control.reinforcement_learning.tf_import_fix  # isort:skip # pylint:disable=bad-import-order,unused-import

from datetime import datetime
import json
import logging
import os
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
from tf_agents.drivers import py_driver
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tqdm import tqdm

from smart_control.reinforcement_learning.agents.ddpg_agent import create_ddpg_agent
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.constants import ONE_DAY_CONFIG_FILEPATH
from smart_control.reinforcement_learning.utils.constants import RL_EXPERIMENT_RESULTS_DIR
from smart_control.reinforcement_learning.utils.constants import RL_STARTER_BUFFERS_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
from smart_control.reinforcement_learning.utils.config import get_histogram_path  # isort:skip # pylint:disable=unused-import
from smart_control.reinforcement_learning.utils.seed import set_seed

# ---- GLOBAL SEED FOR REPRODUCIBILITY ----
set_seed(42)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='[%(message)s]',
)
logger = logging.getLogger(__name__)

# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='experiment_name',
    default=None,
    help='Name of the experiment. This is used to save TensorBoard summaries',
)
flags.DEFINE_string(
    name='starter_buffer_name',
    default='default',
    help=(
        'Name used to identify the replay buffer. Corresponds with directory'
        ' name where the files have been saved.'
    ),
)
flags.DEFINE_string(
    name='train_config_filepath',
    default=ONE_DAY_CONFIG_FILEPATH,
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
    help='Batch size for training',
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
    help='Number of gradient updates per training loop',
)


class RLAgentTrainer:
  """Trains a reinforcement learning agent using a pre-populated replay buffer."""

  def __init__(
      self,
      experiment_name: str,
      starter_buffer_name: str = 'default',
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
    self.starter_buffer_name = starter_buffer_name
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

    self.starter_buffer_dirpath = os.path.join(
        RL_STARTER_BUFFERS_DIR, self.starter_buffer_name
    )

    if self.agent_type not in ['sac', 'ddpg']:
      raise ValueError(
          f'Agent {self.agent_type} has not (yet) been implemented. Please'
          " choose one of: ['sac', 'ddpg']."
      )

    experiment_dirname = self.experiment_name.replace(' ', '')
    self.results_dirpath = os.path.join(
        RL_EXPERIMENT_RESULTS_DIR, experiment_dirname
    )
    self.create_and_setup_environment = create_and_setup_environment
    self.agent = None
    self.replay_manager = None

  def setup_results_dir(self):
    logger.info(
        'Experiment results will be saved to %s',
        os.path.abspath(self.results_dirpath),
    )
    os.makedirs(self.results_dirpath, exist_ok=True)
    os.makedirs(self.saved_model_dirpath, exist_ok=True)

  @property
  def params_json_filepath(self):
    return os.path.join(self.results_dirpath, "experiment_parameters.json")

  @property
  def params_txt_filepath(self):
    return os.path.join(self.results_dirpath, "experiment_parameters.txt")

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

  def save_experiment_params(self):
    params = self.experiment_params
    params['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')

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

  def copy_starter_buffer(self):
    new_buffer_path = os.path.join(self.results_dirpath, 'replay_buffer')
    os.makedirs(new_buffer_path, exist_ok=True)

    logger.info(
        'Creating a copy of replay buffer from %s to %s',
        os.path.abspath(self.starter_buffer_dirpath),
        os.path.abspath(new_buffer_path),
    )

    if os.path.isfile(self.starter_buffer_dirpath):
      shutil.copy2(self.starter_buffer_dirpath, new_buffer_path)
    else:
      for item in os.listdir(self.starter_buffer_dirpath):
        source_item = os.path.join(self.starter_buffer_dirpath, item)
        dest_item = os.path.join(new_buffer_path, item)
        if os.path.isfile(source_item):
          shutil.copy2(source_item, dest_item)
        else:
          shutil.copytree(source_item, dest_item, dirs_exist_ok=True)

    logger.info('Replay buffer copied to %s', new_buffer_path)
    return new_buffer_path

  def create_agent(self, action_spec, time_step_spec):
    logger.info('Creating %s agent', self.agent_type)
    if self.agent_type.lower() == 'sac':
      logger.info('Creating SAC agent')
      return create_sac_agent(time_step_spec, action_spec)
    elif self.agent_type.lower() == 'ddpg':
      logger.info('Creating DDPG agent')
      return create_ddpg_agent(time_step_spec, action_spec)
    else:
      raise ValueError(f'Unsupported agent type: {self.agent_type}')

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
    self.save_experiment_params()

    # ENVIRONMENTS
    logger.info(
        'Creating train and eval environments with scenario config path: %s',
        self.config_filepath,
    )
    train_env = self.create_and_setup_environment(self.config_filepath, metrics_path=self.metrics_dirpath)
    eval_env = self.create_and_setup_environment(self.config_filepath, metrics_path=None)

    train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

    # AGENT
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_tf_env)
    self.agent = self.create_agent(action_spec, time_step_spec)

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
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.num_eval_episodes),
    ]

    # REPLAY BUFFER
    new_buffer_path = self.copy_starter_buffer()

    logger.info('Instantiating replay buffer manager with copied buffer')
    replay_manager = ReplayBufferManager(
        data_spec=self.agent.collect_data_spec,
        capacity=50000,
        checkpoint_dir=new_buffer_path,
        sequence_length=2,
    )
    self.replay_manager = replay_manager

    logger.info(
        'Replay buffer size before loading: %d frames',
        replay_manager.num_frames(),
    )

    replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()

    logger.info(
        'Replay buffer size after loading: %d frames',
        replay_manager.num_frames(),
    )

    dataset = replay_manager.get_dataset(
        batch_size=self.batch_size,
        num_steps=replay_manager.sequence_length,
    )

    # OBSERVERS
    print_observer = PrintStatusObserver(1, train_tf_env, replay_buffer)
    eval_print_observer = PrintStatusObserver(1, eval_tf_env, replay_buffer)

    # CRITICAL: replay_buffer_observer must be FIRST
    collect_observers = [
        replay_buffer_observer,  # MUST be first
        print_observer,
    ]

    # COLLECT DRIVER (using PyDriver for proper observer integration)
    logger.info('Creating collect driver...')
    collect_py_policy = py_tf_eager_policy.PyTFEagerPolicy(
        collect_policy, use_tf_function=True
    )

    collect_driver = py_driver.PyDriver(
        env=train_env,
        policy=collect_py_policy,
        observers=collect_observers,
        max_steps=self.collect_steps_per_iteration,
    )

    # EVAL ACTOR
    logger.info('Creating eval actor...')
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
    saved_model_trigger = triggers.PolicySavedModelTrigger(
        saved_model_dir=self.saved_model_dirpath,
        agent=self.agent,
        train_step=train_step,
        interval=self.eval_interval,
    )
    log_trigger = triggers.StepPerSecondLogTrigger(train_step, interval=self.log_interval)

    logger.info('Creating learner')
    agent_learner = learner.Learner(
        root_dir=self.results_dirpath,
        train_step=train_step,
        agent=self.agent,
        experience_dataset_fn=lambda: dataset,
        summary_interval=1,
        triggers=[saved_model_trigger, log_trigger],
    )

    # MAIN TRAINING LOOP
    logger.info("Starting training for %d iterations", self.train_iterations)

    # Reset metrics
    for m in train_metrics:
      m.reset()

    for i in tqdm(range(self.train_iterations)):
      step_val = train_step.numpy()
      logger.info('Starting training loop iteration %d (step %d)', i, step_val)

      # Evaluate periodically
      if i % self.eval_interval == 0:
        logger.info("Evaluating at iteration %d (step %d)", i, step_val)
        eval_actor.run()

        # Write eval summaries
        with eval_actor.summary_writer.as_default():
          for m in eval_metrics:
            tf.summary.scalar(m.name, m.result(), step=step_val)
          eval_actor.summary_writer.flush()

      # Collect experience
      logger.info("Starting collection for loop iteration %d (step %d)", i, step_val)
      logger.info("Replay frames before collect: %d", self.replay_manager.num_frames())

      time_step = train_env.reset()  # Get initial time_step for PyDriver
      collect_driver.run(time_step)  # Pass time_step to driver

      logger.info("Replay frames after collect: %d", self.replay_manager.num_frames())
      logger.info("Replay Buffer Size: %d", self.replay_manager.num_frames())

      # Check if we have enough data to train
      if replay_manager.num_frames() < self.batch_size:
        logger.info(
            "Warming up: frames=%d < batch_size=%d",
            replay_manager.num_frames(),
            self.batch_size
        )
      else:
        # Train the agent
        logger.info("Training agent for loop iteration %d", i)
        agent_learner.run(iterations=self.learner_iterations)

      # Checkpoint replay buffer periodically
      if i % self.checkpoint_interval == 0:
        logger.info("Checkpointing replay buffer")
        replay_buffer.py_client.checkpoint()

      train_step.assign_add(1)

    # Final checkpoint and evaluation
    logger.info('Training complete. Performing final evaluation and checkpointing.')
    replay_buffer.py_client.checkpoint()
    eval_actor.run()

    # Write final evaluation metrics
    with eval_actor.summary_writer.as_default():
      current_step = train_step.numpy()
      for m in eval_metrics:
        tf.summary.scalar(m.name, m.result(), step=current_step)
        logger.info('Final Eval %s: %s', m.name, m.result())
      eval_actor.summary_writer.flush()

    logger.info(
        "Agent training completed. Saved models in %s",
        os.path.abspath(self.results_dirpath)
    )
    return self.agent


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  trainer = RLAgentTrainer(
      starter_buffer_name=FLAGS.starter_buffer_name,
      config_filepath=FLAGS.train_config_filepath,
      experiment_name=FLAGS.experiment_name,
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


if __name__ == "__main__":
  app.run(main)
