"""
Script to evaluate a trained reinforcement learning policy.
This script loads a saved policy and evaluates it on a configured environment.
"""

# from datetime import datetime
import logging
import os
import shutil
import tempfile
from typing import Sequence

from absl import app
from absl import flags
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
from smart_control.reinforcement_learning.utils.constants import ONE_DAY_CONFIG_FILEPATH
from smart_control.reinforcement_learning.utils.constants import RL_EXPERIMENT_EVAL_DIR
from smart_control.reinforcement_learning.utils.constants import RL_EXPERIMENT_RESULTS_DIR
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
from smart_control.utils.constants import ROOT_DIR

# from smart_control.utils.constants import SB1_GIN_CONFIG_FILEPATH

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]",
)
logger = logging.getLogger(__name__)

# FLAGS

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="eval_experiment_name",
    default=None,
    help="Name of the evaluation experiment",
    # required=True,
)

flags.DEFINE_string(
    name="eval_policy_dirpath",
    default=None,
    help=(
        "Path to the directory containing the saved policy. To use schedule"
        " policy, use: 'schedule'"
    ),
    # required=True,
)

flags.DEFINE_string(
    name="eval_config_filepath",
    default=ONE_DAY_CONFIG_FILEPATH,  # SB1_GIN_CONFIG_FILEPATH,
    help="Path to the .gin config file",
)

flags.DEFINE_integer(
    name="num_eval_episodes",
    default=1,
    help="Number of episodes for evaluation",
)


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
    raise ValueError(
        "No policy structure directories found in"
        f" {os.path.abspath(policy_dir)}"
    )

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
    experiment_name: str,
    config_filepath: str,
    policy_dirpath: str = None,
    num_eval_episodes: int = 10,
    save_trajectory: bool = True,
):
  """
  Evaluates a trained policy on a configured environment.

  Args:
      experiment_name: Name of the experiment to evaluate. Corresponds with an
        existing directory in the "experiment_results" directory.
      policy_dirpath: Path to the directory containing the saved policy or
        "schedule".
      config_filepath: Path to the .gin config file
      num_eval_episodes: Number of episodes to evaluate
      save_trajectory: Whether to save detailed trajectory data for each episode
  """
  # Get base directory for evaluation results
  # base_dir = os.path.dirname(RL_EXPERIMENT_RESULTS_DIR)
  # eval_results_path = os.path.join(base_dir, "experiment_eval")
  # eval_results_path = os.path.join(RL_EXPERIMENT_RESULTS_DIR,
  # "experiment_eval")
  # os.makedirs(eval_results_path, exist_ok=True)
  os.makedirs(RL_EXPERIMENT_EVAL_DIR, exist_ok=True)

  # results directory
  # current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
  # results_dir = os.path.join(
  #    eval_results_path, f"{experiment_name}_{current_time}"
  # )
  experiment_dirname = experiment_name.replace(" ", "")
  results_dir = os.path.join(RL_EXPERIMENT_EVAL_DIR, experiment_dirname)
  logger.info("Evaluation results will be saved to %s", results_dir)
  # try:
  #  os.makedirs(results_dir, exist_ok=False)
  # except FileExistsError as exc:
  #  logger.exception(
  #      "Directory %s already exists. Exiting.", os.path.abspath(results_dir)
  #  )
  #  raise FileExistsError(
  #      f"Directory {os.path.abspath(results_dir)} already exists. Exiting."
  #  ) from exc
  os.makedirs(results_dir, exist_ok=True)

  # ENV

  # Create metrics directory
  metrics_dir = os.path.join(results_dir, "metrics")
  os.makedirs(metrics_dir, exist_ok=True)

  # Create eval environment
  logger.info("Creating evaluation environment")
  eval_env = create_and_setup_environment(
      gin_config_file=config_filepath, metrics_path=metrics_dir
  )

  # Wrap in TF environment
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

  # Create global step counter
  eval_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  # Create policy based on the type
  temp_policy_dirpath = None
  try:
    if policy_dirpath == "schedule":
      logger.info("Using schedule policy")
      policy = create_baseline_schedule_policy(eval_tf_env)
    else:
      experiment_results_dirpath = os.path.join(
          RL_EXPERIMENT_RESULTS_DIR, experiment_dirname
      )
      policy_dirpath = os.path.join(experiment_results_dirpath, "policies")

      # Create a merged saved model with structure from policy dir and variables
      # from latest checkpoint
      temp_policy_dirpath = create_merged_saved_model(policy_dirpath)

      # Use SavedModelPolicy for saved model
      logger.info("Loading saved model from %s", temp_policy_dirpath)
      policy = SavedModelPolicy(
          saved_model_path=temp_policy_dirpath,
          time_step_spec=eval_tf_env.time_step_spec(),
          action_spec=eval_tf_env.action_spec(),
      )
      logger.info("Saved model policy created")

    # OBSERVERS

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

    # ACTOR

    # Create eval actor with observers
    logger.info("Creating evaluation actor")
    eval_dirpath = os.path.join(results_dir, "eval")
    eval_actor = actor.Actor(
        env=eval_env,
        policy=py_tf_eager_policy.PyTFEagerPolicy(policy),
        train_step=eval_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        observers=[observers],
        summary_dir=eval_dirpath,
        summary_interval=1,
    )

    # EVAL

    # Run evaluation
    logger.info("Starting evaluation for %d episodes", num_eval_episodes)
    eval_actor.run()

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        tf_metrics.MaxReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.MinReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
    ]
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
    if temp_policy_dirpath and os.path.exists(temp_policy_dirpath):
      logger.info(
          "Cleaning up temporary directory: %s",
          os.path.abspath(temp_policy_dirpath),
      )
      shutil.rmtree(temp_policy_dirpath)


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # handle relative and absolute filepaths:
  config_filepath = FLAGS.eval_config_filepath
  if not os.path.isabs(config_filepath):
    config_filepath = os.path.join(ROOT_DIR, config_filepath)

  policy_dirpath = FLAGS.eval_policy_dirpath
  if (
      policy_dirpath is not None
      and not os.path.isabs(policy_dirpath)
      and policy_dirpath != "schedule"
  ):
    policy_dirpath = os.path.join(ROOT_DIR, policy_dirpath)

  evaluate_policy(
      experiment_name=FLAGS.eval_experiment_name,
      policy_dirpath=policy_dirpath,
      config_filepath=config_filepath,
      num_eval_episodes=FLAGS.num_eval_episodes,
  )


if __name__ == "__main__":

  app.run(main)
