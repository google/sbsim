"""
Script to train a reinforcement learning agent using a pre-populated replay buffer.
This script sets up the training process with separate collection and evaluation components.
"""

from datetime import datetime
import json
import os
import shutil

os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import logging

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
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.config import EXPERIMENT_RESULTS_PATH
from smart_control.reinforcement_learning.utils.config import ROOT_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
)
logger = logging.getLogger(__name__)


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
  logger.info(f'Saving experiment parameters to {params_file}')
  with open(params_file, 'w') as f:
    json.dump(params, f, indent=4)

  # Also save as a readable text file for quick reference
  params_txt = os.path.join(save_path, 'experiment_parameters.txt')
  with open(params_txt, 'w') as f:
    f.write('Experiment Parameters:\n')
    f.write('=====================\n\n')
    for key, value in params.items():
      f.write(f'{key}: {value}\n')

  logger.info(f'Experiment parameters saved to {params_file} and {params_txt}')


def train_agent(
    starter_buffer_path,
    experiment_name,
    agent_type='sac',
    train_iterations=100000,
    collect_steps_per_iteration=1,
    batch_size=256,
    log_interval=100,
    eval_interval=1000,
    num_eval_episodes=5,
    checkpoint_interval=1000,
    learner_iterations=200,
    scenario_config_path=None,
):
  """
  Trains a reinforcement learning agent using a pre-populated replay buffer.

  Args:
      starter_buffer_path: Path to the pre-populated replay buffer
      experiment_name: Name of the experiment
      agent_type: Type of agent to train ('sac' or 'td3')
      train_iterations: Number of training iterations
      collect_steps_per_iteration: Number of collection steps per training iteration
      batch_size: Batch size for training
      log_interval: Interval for logging training metrics
      eval_interval: Interval for evaluating the agent
      num_eval_episodes: Number of episodes for evaluation
      checkpoint_interval: Interval for checkpointing the replay buffer
      learner_iterations: Number of iterations to run the agent learner per training loop
      scenario_config_path: Path to the scenario configuration file (optional)
  """
  # Set up scenario config path if not provided
  if scenario_config_path is None:
    scenario_config_path = os.path.join(CONFIG_PATH, 'sim_config_1_day.gin')

  # Generate timestamp for summary directory
  current_time = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
  summary_dir = os.path.join(
      EXPERIMENT_RESULTS_PATH, f'{experiment_name}_{current_time}'
  )
  logger.info(f'Experiment results will be saved to {summary_dir}')

  try:
    os.makedirs(summary_dir, exist_ok=False)
  except FileExistsError:
    logger.exception(f'Directory {summary_dir} already exists. Exiting.')
    raise FileExistsError(f'Directory {summary_dir} already exists. Exiting.')

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

  # Create train and eval environments
  logger.info(
      'Creating train and eval environments with scenatio config path:'
      f' {scenario_config_path}'
  )
  train_env = create_and_setup_environment(
      scenario_config_path, metrics_path=os.path.join(summary_dir, 'metrics')
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

  # Create agent based on type
  logger.info(f'Creating {agent_type} agent')
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
    logger.exception(f'Unsupported agent type: {agent_type}')
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

  # Create a new buffer path in the experiment directory
  new_buffer_path = os.path.join(summary_dir, 'replay_buffer')
  os.makedirs(new_buffer_path, exist_ok=True)

  # Copy the original buffer to the new location
  logger.info(
      f'Creating a copy of replay buffer from {starter_buffer_path} to'
      f' {new_buffer_path}'
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

  logger.info(f'Replay buffer copied to {new_buffer_path}')

  # Initialize replay buffer manager with the copied buffer path
  logger.info('Instantiating replay buffer manager with copied buffer')
  replay_manager = ReplayBufferManager(
      agent.collect_data_spec,
      50000,  # Use default capacity
      new_buffer_path,  # Use the copied buffer path
      sequence_length=2,
  )
  logger.info(
      f'Replay buffer size before loading: {replay_manager.num_frames()} frames'
  )

  # Load the copied replay buffer
  logger.info(f'Loading replay buffer from {new_buffer_path}')
  replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()
  logger.info(
      f'Replay buffer size after loading: {replay_manager.num_frames()} frames'
  )

  # Create dataset for sampling from the buffer
  logger.info('Creating dataset for sampling from replay buffer')
  dataset = replay_buffer.as_dataset(
      sample_batch_size=batch_size, num_steps=2, num_parallel_calls=3
  ).prefetch(3)

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

  # Create learner
  logger.info('Creating learner')
  agent_learner = learner.Learner(
      root_dir=summary_dir,
      train_step=train_step,
      agent=agent,
      experience_dataset_fn=lambda: dataset,
      summary_interval=1,
      triggers=[
          triggers.PolicySavedModelTrigger(
              os.path.join(summary_dir, 'policies'),
              agent,
              train_step,
              interval=eval_interval,
          ),
          triggers.StepPerSecondLogTrigger(train_step, interval=log_interval),
      ],
  )

  # Main training loop
  logger.info(f'Starting training for {train_iterations} iterations')

  # Reset metrics
  for m in train_metrics:
    m.reset()

  # Main training loop
  for i in tqdm(range(train_iterations)):
    # Get current training step value before operations
    current_step = train_step.numpy()
    logger.info(f'Starting training loop iteration {i} (step {current_step})')

    # Evaluate periodically
    if i % eval_interval == 0:
      logger.info(f'Evaluating at iteration {i} (step {current_step})')
      eval_actor.run()

      # Write eval summaries with the current global step
      with eval_actor.summary_writer.as_default():
        for m in eval_metrics:
          tf.summary.scalar(m.name, m.result(), step=current_step)
        eval_actor.summary_writer.flush()

    # Collect experience
    logger.info(
        f'Starting collection for loop iteration {i} (step {current_step})'
    )
    collect_actor.run()

    # Write collect summaries with the current global step
    with collect_actor.summary_writer.as_default():
      for m in train_metrics:
        tf.summary.scalar(m.name, m.result(), step=current_step)
      collect_actor.summary_writer.flush()

    # Train the agent using the specified learner iterations
    # This will internally increment the train_step
    logger.info(f'Training agent for loop iteration {i}')
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
      logger.info(f'Final Eval {m.name}: {m.result()}')
    eval_actor.summary_writer.flush()

  logger.info(f'Agent training completed. Saved models in {summary_dir}')
  return agent


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
      description=(
          'Train a reinforcement learning agent using a pre-populated replay'
          ' buffer'
      )
  )
  parser.add_argument(
      '--starter-buffer-path',
      type=str,
      required=True,
      help='Path to the starter replay buffer',
  )
  parser.add_argument(
      '--agent-type',
      type=str,
      default='sac',
      choices=['sac', 'td3', 'ddpg'],
      help='Type of agent to train (sac or td3)',
  )
  parser.add_argument(
      '--train-iterations',
      type=int,
      default=300,
      help='Number of training iterations',
  )
  parser.add_argument(
      '--collect-steps-per-training-iteration',
      type=int,
      default=50,
      help='Number of collection steps per iteration',
  )
  parser.add_argument(
      '--batch-size',
      type=int,
      default=256,
      help=(
          'Batch size for training (each gradient update uses                  '
          '                                                    this many'
          ' elements from the replay buffer batched)'
      ),
  )

  parser.add_argument(
      '--eval-interval',
      type=int,
      default=10,
      help='Interval for evaluating the agent',
  )
  parser.add_argument(
      '--num-eval-episodes',
      type=int,
      default=1,
      help='Number of episodes for evaluation',
  )
  parser.add_argument(
      '--log-interval',
      type=int,
      default=1,
      help='Interval for logging training metrics',
  )
  parser.add_argument(
      '--experiment-name',
      type=str,
      required=True,
      help=(
          'Name of the experiment. This be used to                             '
          '                                                save TensorBoard'
          ' summaries'
      ),
  )
  parser.add_argument(
      '--checkpoint-interval',
      type=int,
      default=10,
      help='Interval for checkpointing the replay buffer',
  )
  parser.add_argument(
      '--learner-iterations',
      type=int,
      default=200,
      help=(
          'Number of iterations (gradient updates)                             '
          '                                                 to run the agent'
          ' learner per training loop'
      ),
  )
  parser.add_argument(
      '--scenario-config-path',
      type=str,
      default=os.path.join(
          ROOT_DIR,
          'smart_control',
          'configs',
          'resources',
          'sb1',
          'sim_config.gin',
      ),
      help=(
          'Path to the scenario config file.                                   '
          '                                            Default is'
          ' sim_config.gin'
      ),
  )

  args = parser.parse_args()

  # Make it work for both relative and absolute paths
  if not os.path.isabs(args.starter_buffer_path):
    args.starter_buffer_path = os.path.join(ROOT_DIR, args.starter_buffer_path)

  if not os.path.isabs(args.scenario_config_path):
    args.scenario_config_path = os.path.join(
        ROOT_DIR, args.scenario_config_path
    )

  train_agent(
      starter_buffer_path=args.starter_buffer_path,
      experiment_name=args.experiment_name,
      agent_type=args.agent_type,
      train_iterations=args.train_iterations,
      collect_steps_per_iteration=args.collect_steps_per_training_iteration,
      batch_size=args.batch_size,
      eval_interval=args.eval_interval,
      num_eval_episodes=args.num_eval_episodes,
      log_interval=args.log_interval,
      checkpoint_interval=args.checkpoint_interval,
      learner_iterations=args.learner_iterations,
      scenario_config_path=args.scenario_config_path,
  )
