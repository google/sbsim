"""
Script to train a reinforcement learning agent using a pre-populated replay buffer.
This script sets up the training process with separate collection and evaluation components.
"""

import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import logging
import time
from datetime import datetime

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils
from tf_agents.policies import py_tf_eager_policy

from smart_control.refactor.observers import (PrintStatusObserver, CompositeObserver)
from smart_control.refactor.utils.config import CONFIG_PATH, EXPERIMENT_RESULTS_PATH
from smart_control.refactor.utils.environment import create_and_setup_environment
from smart_control.refactor.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.refactor.agents.sac_agent import create_sac_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]'
)
logger = logging.getLogger(__name__)

def train_agent(
    starter_buffer_path,
    experiment_name,
    agent_type='sac',
    train_iterations=100000,
    collect_steps_per_iteration=1,
    batch_size=256,
    log_interval=100,
    eval_interval=1000,
    num_eval_episodes=5
):
    """
    Trains a reinforcement learning agent using a pre-populated replay buffer.
    
    Args:
        buffer_path: Path to the pre-populated replay buffer
        agent_type: Type of agent to train ('sac' or 'td3')
        train_iterations: Number of training iterations
        collect_steps_per_iteration: Number of collection steps per training iteration
        batch_size: Batch size for training
        log_interval: Interval for logging training metrics
        eval_interval: Interval for evaluating the agent
        num_eval_episodes: Number of episodes for evaluation
        summary_dir: Directory to save TensorBoard summaries
    """
    # Set up scenario config path
    scenario_config_path = os.path.join(CONFIG_PATH, "sim_config_1_day.gin")
    
    # Generate timestamp for summary directory
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M")
    summary_dir = os.path.join(EXPERIMENT_RESULTS_PATH, f"{experiment_name}_{current_time}")
    logger.info(f"Experiment results will be saved to {summary_dir}")
    
    try:
        os.makedirs(summary_dir, exist_ok=False)
    except FileExistsError:
        logger.exception(f"Directory {summary_dir} already exists. Exiting.")
        raise FileExistsError(f"Directory {summary_dir} already exists. Exiting.")
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
    logger.info("Created summary writers")
    
    # Create train and eval environments
    logger.info("Creating train and eval environments")
    train_env = create_and_setup_environment(scenario_config_path, metrics_path=os.path.join(summary_dir, 'metrics'))
    eval_env = create_and_setup_environment(scenario_config_path, metrics_path=None)

    
    # Wrap in TF environments
    train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
    
    # Create global step for training
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    # Get specs
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_tf_env)
    
    # Create agent based on type
    logger.info(f"Creating {agent_type} agent")
    if agent_type.lower() == 'sac':
        logger.info("Creating SAC agent")
        agent = create_sac_agent(time_step_spec=time_step_spec, action_spec=action_spec)
    else:
        logger.exception(f"Unsupported agent type: {agent_type}. Choose from 'sac' or 'td3'.")
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose from 'sac' or 'td3'.")
    
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
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    
    # Load replay buffer from existing path
    logger.info(f"Instantiating replay buffer manager")
    # Create replay buffer manager and load existing buffer
    replay_manager = ReplayBufferManager(
        agent.collect_data_spec,
        50000,  # Use default capacity
        starter_buffer_path,
        sequence_length=2
    )
    logger.info(f"Replay buffer size before loading starter buffer: {replay_manager.num_frames()} frames")
    
    logger.info(f"Loading starter replay buffer from {starter_buffer_path}")
    replay_buffer, replay_buffer_observer = replay_manager.load_replay_buffer()
    logger.info(f"Replay buffer size after loading starter buffer: {replay_manager.num_frames()} frames")
    
    
    # Create dataset for sampling from the buffer
    logger.info("Creating dataset for sampling from replay buffer")
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2,
        num_parallel_calls=3
    ).prefetch(3)
    
    
    # Create print observer for collection
    print_observer = PrintStatusObserver(
        status_interval_steps=100,  # Print status every 100 steps
        environment=train_tf_env,
        replay_buffer=replay_buffer
    )
    
    # Combine observers
    collect_observers = CompositeObserver([print_observer, replay_buffer_observer] + train_metrics)
    
    # Create collect actor
    logger.info("Creating collect and eval actors")
    collect_actor = actor.Actor(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(collect_policy),
        train_step,
        steps_per_run=collect_steps_per_iteration,
        observers=[collect_observers]
    )
    
    # Create eval actor
    logger.info("Creating eval actor")
    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        observers=eval_metrics
    )
    
    # Create learner
    logger.info("Creating learner")
    agent_learner = learner.Learner(
        root_dir=summary_dir,
        train_step=train_step,
        agent=agent._agent,
        experience_dataset_fn=lambda: dataset,
        triggers=[
            triggers.PolicySavedModelTrigger(
                os.path.join(summary_dir, 'policies'),
                agent,
                train_step,
                interval=eval_interval
            ),
            triggers.StepPerSecondLogTrigger(train_step, interval=log_interval)
        ]
    )

    
    # Training loop
    logger.info(f"Starting training for {train_iterations} iterations")
    
    # Reset metrics
    for m in train_metrics:
        m.reset()
    
    
    # Initial evaluation
    logger.info("Performing initial evaluation")
    collect_actor.run()
    for m in eval_metrics:
        with eval_summary_writer.as_default():
            tf.summary.scalar(m.name, m.result(), step=train_step.numpy())
        logger.info(f"{m.name}: {m.result()}")
        
        
    logger.info("Done!")
    return
    
    # Main training loop
    for i in range(train_iterations):
        # Collect experience
        collect_actor.run()
        
        # Train the agent
        loss_info = agent_learner.run(iterations=1)
        
        # Log metrics periodically
        if i % log_interval == 0:
            logger.info(f"Iteration {i}/{train_iterations}")
            logger.info(f"Step: {train_step.numpy()}")
            
            with train_summary_writer.as_default():
                for m in train_metrics:
                    tf.summary.scalar(m.name, m.result(), step=train_step.numpy())
                    logger.info(f"{m.name}: {m.result()}")
                
                if loss_info:
                    for name, loss in loss_info.items():
                        tf.summary.scalar(f"losses/{name}", loss, step=train_step.numpy())
                        logger.info(f"Loss/{name}: {loss}")
        
        # Evaluate periodically
        if i % eval_interval == 0:
            logger.info(f"Evaluating at iteration {i}")
            eval_actor.run()
            
            with eval_summary_writer.as_default():
                for m in eval_metrics:
                    tf.summary.scalar(m.name, m.result(), step=train_step.numpy())
                    logger.info(f"Eval {m.name}: {m.result()}")
        
        # Checkpoint replay buffer periodically
        if i % 1000 == 0:
            logger.info("Checkpointing replay buffer")
            replay_buffer.py_client.checkpoint()
    
    # Final checkpoint and evaluation
    logger.info("Training complete. Performing final evaluation and checkpointing.")
    replay_buffer.py_client.checkpoint()
    eval_actor.run()
    
    for m in eval_metrics:
        logger.info(f"Final Eval {m.name}: {m.result()}")
    
    logger.info(f"Agent training completed. Saved models in {summary_dir}")
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a reinforcement learning agent using a pre-populated replay buffer')
    parser.add_argument('--starter-buffer-path', type=str, required=True, help='Path to the starter replay buffer')
    parser.add_argument('--agent-type', type=str, default='sac', choices=['sac', 'td3'],
                        help='Type of agent to train (sac or td3)')
    parser.add_argument('--train-iterations', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--collect-steps-per-training-iteration', type=int, default=1, help='Number of collection steps per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Interval for evaluating the agent')
    parser.add_argument('--num-eval-episodes', type=int, default=1, help='Number of episodes for evaluation')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for logging training metrics')
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment. Will be used to save TensorBoard summaries')
    
    args = parser.parse_args()
    
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
    )