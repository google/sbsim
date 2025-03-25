"""
Script to evaluate a trained reinforcement learning policy.
This script loads a saved policy and evaluates it on a configured environment.
"""

import json
import logging
import os
from datetime import datetime

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor

from smart_control.reinforcement_learning.observers.composite_observer import \
    CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import \
    PrintStatusObserver
from smart_control.reinforcement_learning.observers.trajectory_recorder_observer import \
    TrajectoryRecorderObserver
from smart_control.reinforcement_learning.policies.saved_model_policy import \
    SavedModelPolicy
from smart_control.reinforcement_learning.policies.schedule_policy import \
    create_baseline_schedule_policy
from smart_control.reinforcement_learning.utils.config import (
    CONFIG_PATH, EXPERIMENT_RESULTS_PATH)
from smart_control.reinforcement_learning.utils.environment import \
    create_and_setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]'
)
logger = logging.getLogger(__name__)

def evaluate_policy(
    policy_dir,
    gin_config_path,
    experiment_name,
    num_eval_episodes=10,
    save_trajectory=True
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
    results_dir = os.path.join(eval_results_path, f"{experiment_name}_{current_time}")
    logger.info(f"Evaluation results will be saved to {results_dir}")
    
    try:
        os.makedirs(results_dir, exist_ok=False)
    except FileExistsError:
        logger.exception(f"Directory {results_dir} already exists. Exiting.")
        raise FileExistsError(f"Directory {results_dir} already exists. Exiting.")
    
    # Create metrics directory
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create eval environment
    logger.info("Creating evaluation environment")
    eval_env = create_and_setup_environment(gin_config_path, metrics_path=metrics_dir)
    
    # Wrap in TF environment
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
    
    # Create global step counter
    eval_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    # Create policy based on the type
    if policy_dir == 'schedule':
        logger.info("Using schedule policy")
        policy = create_baseline_schedule_policy(eval_tf_env)
    else:
        # Use SavedModelPolicy for saved model
        logger.info(f"Loading saved model from {policy_dir}")
        policy_path = os.path.join(policy_dir, "greedy_policy")
        policy = SavedModelPolicy(
            policy_path,
            eval_tf_env.time_step_spec(),
            eval_tf_env.action_spec()
        )
        logger.info("Saved model policy created")
    
    # Set up metrics
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        tf_metrics.MaxReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.MinReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps()
    ]
    
    observers_list = []
    
    print_observer = PrintStatusObserver(
        status_interval_steps=1,
        environment=eval_tf_env,
        replay_buffer=None
    )
    
    observers_list.append(print_observer)
    
    # Record trajectory observer
    trajectory_dir = None
    if save_trajectory:
        trajectory_dir = os.path.join(results_dir, 'trajectories')
        os.makedirs(trajectory_dir, exist_ok=True)
        
    if save_trajectory and trajectory_dir:
        trajectory_observer = TrajectoryRecorderObserver(
            save_dir=trajectory_dir,
            environment=eval_tf_env
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
        summary_dir=os.path.join(results_dir, 'eval'),
        summary_interval=1
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation for {num_eval_episodes} episodes")
    eval_actor.run()
    
    # Write evaluation summaries
    with eval_actor.summary_writer.as_default():
        for m in eval_metrics:
            tf.summary.scalar(m.name, m.result(), step=eval_step.numpy())
            logger.info(f"Eval {m.name}: {m.result()}")
        eval_actor.summary_writer.flush()
    
    logger.info(f"Evaluation completed. Saved results in {results_dir}")
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained reinforcement learning policy')
    parser.add_argument('--policy-dir', type=str, required=True, help='Path to the directory containing the saved policy. To \
                                                                       use schedule policy, just type `schedule`')
    parser.add_argument('--gin-config', type=str, default="/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-7_starttimestamp-2023-07-06.gin", help='Path to the .gin config file')
    parser.add_argument('--num-eval-episodes', type=int, default=1, help='Number of episodes for evaluation')
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the evaluation experiment')
    
    args = parser.parse_args()
    
    # If the gin config is just a filename, prepend the CONFIG_PATH
    gin_config_path = args.gin_config
    if not os.path.exists(gin_config_path) and not os.path.isabs(gin_config_path):
        gin_config_path = os.path.join(CONFIG_PATH, gin_config_path)
    
    evaluate_policy(
        policy_dir=args.policy_dir,
        gin_config_path=gin_config_path,
        experiment_name=args.experiment_name,
        num_eval_episodes=args.num_eval_episodes
    )
