"""
Script to evaluate a trained reinforcement learning policy.
This script loads a saved policy and evaluates it on a configured environment.
"""

from datetime import datetime
import logging
import os
import shutil
import tempfile

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor

from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.observers.trajectory_recorder_observer import TrajectoryRecorderObserver
from smart_control.reinforcement_learning.policies.saved_model_policy import SavedModelPolicy
from smart_control.reinforcement_learning.policies.schedule_policy import create_baseline_schedule_policy
from smart_control.reinforcement_learning.utils.config import EXPERIMENT_RESULTS_PATH
from smart_control.reinforcement_learning.utils.config import ROOT_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]",
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(policy_dir):
  """
  Find the latest policy checkpoint in a directory.

  Args:
      policy_dir: Path to the directory containing checkpoints

  Returns:
      Path to the latest checkpoint or None if no checkpoints found
  """
  # Check if there's a checkpoints directory
  checkpoints_dir = os.path.join(policy_dir, "checkpoints")
  if os.path.exists(checkpoints_dir):
    # Look for checkpoint directories
    checkpoint_dirs = [
        d
        for d in os.listdir(checkpoints_dir)
        if d.startswith("policy_checkpoint_")
    ]

    if checkpoint_dirs:
      # Sort by checkpoint number and get the latest
      latest_checkpoint = sorted(
          checkpoint_dirs, key=lambda x: int(x.split("_")[-1])
      )[-1]

      return os.path.join(checkpoints_dir, latest_checkpoint)

  # If we're here, either there's no checkpoints dir or no checkpoints in it
  return None


def create_merged_saved_model(policy_dir):
  """
  Create a temporary directory with a complete SavedModel by merging:
  1. Model structure from policy_dir
  2. Variables from the latest checkpoint

  Args:
      policy_dir: Base directory containing policies and checkpoints

  Returns:
      Path to temporary directory with complete model
  """
  # First check for greedy_policy (preferred) or policy directories
  model_structure_dir = None
  if os.path.exists(os.path.join(policy_dir, "greedy_policy")):
    model_structure_dir = os.path.join(policy_dir, "greedy_policy")
    logger.info("Using model structure from greedy_policy directory")
  else:
    raise ValueError(f"No policy structure directories found in {policy_dir}")

  # Find latest checkpoint for variables
  latest_checkpoint = find_latest_checkpoint(policy_dir)
  if not latest_checkpoint:
    logger.warning("No checkpoints found, using original model structure only")
    return model_structure_dir

  logger.info("Found latest checkpoint at: %s", latest_checkpoint)

  # Create temporary directory for merged model
  temp_dir = tempfile.mkdtemp(prefix="merged_policy_")
  logger.info("Created temporary directory for merged model: %s", temp_dir)

  # Copy model structure files (everything except 'variables' directory)
  for item in os.listdir(model_structure_dir):
    if item != "variables":
      source = os.path.join(model_structure_dir, item)
      dest = os.path.join(temp_dir, item)
      if os.path.isdir(source):
        shutil.copytree(source, dest)
      else:
        shutil.copy2(source, dest)

  # Create variables directory
  variables_dir = os.path.join(temp_dir, "variables")
  os.makedirs(variables_dir, exist_ok=True)

  # Copy latest checkpoint variables
  checkpoint_vars_dir = os.path.join(latest_checkpoint, "variables")
  for item in os.listdir(checkpoint_vars_dir):
    source = os.path.join(checkpoint_vars_dir, item)
    dest = os.path.join(variables_dir, item)
    shutil.copy2(source, dest)

  logger.info("Successfully created merged model at %s", temp_dir)
  return temp_dir


def evaluate_policy(
    policy_dir,
    gin_config_path,
    experiment_name,
    num_eval_episodes=10,
    save_trajectory=True,
):
  """
  Evaluates a trained policy on a configured environment.

  Args:
      policy_dir: Path to the directory containing the saved policy
      gin_config_path: Path to the .gin config file
      experiment_name: Name of the evaluation experiment
      num_eval_episodes: Number of episodes to evaluate
      save_trajectory: Whether to save detailed trajectory data for each episode
  """
  # Get base directory for evaluation results
  base_dir = os.path.dirname(EXPERIMENT_RESULTS_PATH)
  eval_results_path = os.path.join(base_dir, "eval_results")
  os.makedirs(eval_results_path, exist_ok=True)

  # Generate timestamp for results directory
  current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
  results_dir = os.path.join(
      eval_results_path, f"{experiment_name}_{current_time}"
  )
  logger.info("Evaluation results will be saved to %s", results_dir)

  try:
    os.makedirs(results_dir, exist_ok=False)
  except FileExistsError as exc:
    logger.exception("Directory %s already exists. Exiting.", results_dir)
    raise FileExistsError(
        f"Directory {results_dir} already exists. Exiting."
    ) from exc

  # Create metrics directory
  metrics_dir = os.path.join(results_dir, "metrics")
  os.makedirs(metrics_dir, exist_ok=True)

  # Create eval environment
  logger.info("Creating evaluation environment")
  eval_env = create_and_setup_environment(
      gin_config_path, metrics_path=metrics_dir
  )

  # Wrap in TF environment
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

  # Create global step counter
  eval_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  # Create policy based on the type
  temp_dir = None
  try:
    if policy_dir == "schedule":
      logger.info("Using schedule policy")
      policy = create_baseline_schedule_policy(eval_tf_env)
    else:
      # Create a merged saved model with structure from policy dir and variables
      # from latest checkpoint
      temp_dir = create_merged_saved_model(policy_dir)

      # Use SavedModelPolicy for saved model
      logger.info("Loading saved model from %s", temp_dir)
      policy = SavedModelPolicy(
          temp_dir, eval_tf_env.time_step_spec(), eval_tf_env.action_spec()
      )
      logger.info("Saved model policy created")

    # Set up metrics
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        tf_metrics.MaxReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.MinReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
    ]

    observers_list = []

    print_observer = PrintStatusObserver(
        status_interval_steps=1, environment=eval_tf_env, replay_buffer=None
    )

    observers_list.append(print_observer)

    # Record trajectory observer
    trajectory_dir = None
    if save_trajectory:
      trajectory_dir = os.path.join(results_dir, "trajectories")
      os.makedirs(trajectory_dir, exist_ok=True)

    if save_trajectory and trajectory_dir:
      trajectory_observer = TrajectoryRecorderObserver(
          save_dir=trajectory_dir, environment=eval_tf_env
      )
      observers_list.append(trajectory_observer)

    observers = CompositeObserver(observers_list)

    # Create eval actor with observers
    logger.info("Creating evaluation actor")
    eval_actor = actor.Actor(
        eval_env,
        py_tf_eager_policy.PyTFEagerPolicy(policy),
        eval_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        observers=[observers],
        summary_dir=os.path.join(results_dir, "eval"),
        summary_interval=1,
    )

    # Run evaluation
    logger.info("Starting evaluation for %d episodes", num_eval_episodes)
    eval_actor.run()

    # Write evaluation summaries
    with eval_actor.summary_writer.as_default():
      for m in eval_metrics:
        tf.summary.scalar(m.name, m.result(), step=eval_step.numpy())
        logger.info("Eval %s: %s", m.name, m.result())
      eval_actor.summary_writer.flush()

    logger.info("Evaluation completed. Saved results in %s", results_dir)
    return

  finally:
    # Clean up temporary directory if created
    if temp_dir and os.path.exists(temp_dir):
      logger.info("Cleaning up temporary directory: %s", temp_dir)
      shutil.rmtree(temp_dir)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      description="Evaluate a trained reinforcement learning policy"
  )
  parser.add_argument(
      "--policy-dir",
      type=str,
      required=True,
      help=(
          "Path to the directory containing the saved policy. To               "
          "                                                         use"
          " schedule policy, just type `schedule`"
      ),
  )
  parser.add_argument(
      "--gin-config",
      type=str,
      default=os.path.join(
          ROOT_DIR,
          "smart_control",
          "configs",
          "resources",
          "sb1",
          "sim_config.gin",
      ),
      help="Path to the .gin config file",
  )
  parser.add_argument(
      "--num-eval-episodes",
      type=int,
      default=1,
      help="Number of episodes for evaluation",
  )
  parser.add_argument(
      "--experiment-name",
      type=str,
      required=True,
      help="Name of the evaluation experiment",
  )

  args = parser.parse_args()

  # Make it work for both relative and absolute paths
  gin_config_path_ = args.gin_config
  if not os.path.isabs(args.gin_config):
    gin_config_path_ = os.path.join(ROOT_DIR, args.gin_config)

  if not os.path.isabs(args.policy_dir) and args.policy_dir != "schedule":
    args.policy_dir = os.path.join(ROOT_DIR, args.policy_dir)

  evaluate_policy(
      policy_dir=args.policy_dir,
      gin_config_path=gin_config_path_,
      experiment_name=args.experiment_name,
      num_eval_episodes=args.num_eval_episodes,
  )
