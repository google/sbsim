import os
import logging
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.train import actor


from smart_control.refactor.agents import create_sac_agent
from smart_control.refactor.observers import (RenderingObserver, PrintStatusObserver, CompositeObserver)
from smart_control.refactor.utils.metrics import compute_avg_return
from smart_control.refactor.utils.config import CONFIG_PATH, METRICS_PATH, OUTPUT_DATA_PATH
from smart_control.learning.reinforcement_learning.sac.learning_utils import load_environment
from smart_control.refactor.replay_buffer.replay_buffer import ReplayBufferManager

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]'
)

# Set up logging
logger = logging.getLogger(__name__)  # Uses the module name


# Instantiate the environments
logger.info("Instantiating the collect environment...")

collect_scenario_config = os.path.join(CONFIG_PATH, "sim_config_4_day.gin")
collect_env = load_environment(collect_scenario_config)
collect_env._metrics_path = None # Collect env does not need metrics path
collect_env._occupancy_normalization_constant = 125.0
# the collect_env is of type PyEnvironment. Let's wrap it in a TFPyEnvironment
collect_tf_env = tf_py_environment.TFPyEnvironment(collect_env)


logger.info("Instantiating the eval environment...")

eval_scenario_config = os.path.join(CONFIG_PATH, "sim_config_4_day.gin")
eval_env = load_environment(collect_scenario_config)
eval_env._metrics_path = METRICS_PATH
eval_env._occupancy_normalization_constant = 125.0
# the eval_env is of type PyEnvironment. Let's wrap it in a TFPyEnvironment
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)


# Now, let's create the agent
logger.info("Creating the SAC agent...")

train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
agent = create_sac_agent(
    time_step_spec=collect_tf_env.time_step_spec(),
    action_spec=collect_tf_env.action_spec(),
    actor_fc_layers=(64, 64),
    critic_obs_fc_layers=(64, 32),
    critic_action_fc_layers=(64, 32),
    critic_joint_fc_layers=(64, 32),
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    target_update_tau=0.005,
    target_update_period=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    train_step_counter=train_step
)

agent.initialize() # Initialize the agent

# Get specs from the environment
observation_spec = collect_tf_env.observation_spec()
action_spec = collect_tf_env.action_spec()
time_step_spec = ts.time_step_spec(observation_spec)


# Create the replay buffer
# Initialize the manager with your agent's data spec
replay_manager = ReplayBufferManager(
    agent.collect_data_spec,
    50000,
    f"{OUTPUT_DATA_PATH}/refactor_test_collect_buffer",
    sequence_length=2
)

# Create the replay buffer
replay_buffer, replay_buffer_observer = replay_manager.create_replay_buffer()


# Setup the observers
logger.info("Setting up the observers...")

# Vizualization functions
def render_env(environment):
    pass

def plot_metrics(environment, time_zone):
    pass

# Create individual observers
render_observer = RenderingObserver(
    render_interval_steps=5,
    environment=collect_tf_env,
    render_fn=render_env, plot_fn=plot_metrics, time_zone='US/Pacific'
)
print_observer = PrintStatusObserver(
    status_interval_steps=1,
    environment=collect_tf_env,
    replay_buffer=replay_buffer
)

eval_render_observer = RenderingObserver(
    render_interval_steps=5,
    environment=eval_tf_env,
    render_fn=render_env, plot_fn=plot_metrics, time_zone='US/Pacific'
)
eval_print_observer = PrintStatusObserver(
    status_interval_steps=1,
    environment=eval_tf_env,
    replay_buffer=replay_buffer
)

# Composite observer aggregates the individual observers neatly
observers = CompositeObserver([render_observer, print_observer, replay_buffer_observer])
eval_observers = CompositeObserver([eval_render_observer, eval_print_observer])


from tf_agents.policies import py_tf_eager_policy

# Setup collect actor
logger.info("Setting up the collect actor...")

collect_actor = actor.Actor(
    collect_tf_env.pyenv.envs[0],  # Have to use the underlying pyenv because the actor doesn't support TFEnvironments yet
    py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy),  # Have to wrap it like this because using underlying pyenv instead of tf env
    steps_per_run=10,
    train_step=agent.train_step_counter,
    observers=[observers]
)
collect_actor.run()
logger.info("Collect actor has run, now running checkpoint...")
replay_buffer.py_client.checkpoint()
logger.info("Replay buffer num frames: {}".format(replay_buffer.num_frames()))

# Let's test the training loop
logger.info("Running test training...")

if replay_buffer.num_frames() > 0:    
    
    # Sample a batch
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2
    ).prefetch(3)
    
    iterator = iter(dataset)
    
    # Run a few training steps
    for _ in range(5):  # Train for 5 steps
        experience, _ = next(iterator)
        train_info = agent.train(experience)
        logger.info(f"Training step {agent.train_step_counter.numpy()}, Loss: {train_info['loss']}")


# Now, run an evaluation with the trained agent policy
logger.info("Running evaluation...")
compute_avg_return(
    environment=eval_tf_env,
    policy=agent.policy,
    num_episodes=1,
    num_steps=10,
    trajectory_observers=[eval_observers]
)


logger.info("Done!")
