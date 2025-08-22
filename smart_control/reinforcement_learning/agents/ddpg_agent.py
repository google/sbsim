"""DDPG Agent implementation.

This module provides a function to create a DDPG agent with customizable
parameters.
"""

from typing import Optional, Sequence

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.networks import network
from tf_agents.typing import types

from smart_control.reinforcement_learning.agents.networks.ddpg_networks import create_sequential_actor_network
from smart_control.reinforcement_learning.agents.networks.ddpg_networks import create_sequential_critic_network


def create_ddpg_agent(
    time_step_spec: types.TimeStep,
    action_spec: types.NestedTensorSpec,
    # Actor network parameters
    actor_fc_layers: Sequence[int] = (128, 128),
    actor_network: Optional[network.Network] = None,
    # Critic network parameters
    critic_obs_fc_layers: Sequence[int] = (128, 64),
    critic_action_fc_layers: Sequence[int] = (128, 64),
    critic_joint_fc_layers: Sequence[int] = (128, 64),
    critic_network: Optional[network.Network] = None,
    # Optimizer parameters
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    # Agent parameters
    ou_stddev: float = 1.0,
    ou_damping: float = 1.0,
    gamma: float = 0.99,
    target_update_tau: float = 0.005,
    target_update_period: int = 1,
    reward_scale_factor: float = 1.0,
    # Training parameters
    gradient_clipping: Optional[float] = None,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    train_step_counter: Optional[tf.Variable] = None,
) -> tf_agent.TFAgent:
  """Creates a DDPG Agent.

  Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.

      action_spec: A nest of BoundedTensorSpec representing the actions.

      actor_fc_layers: Iterable of fully connected layer units for the actor
                       network.

      actor_network: Optional custom actor network to use.

      critic_obs_fc_layers: Iterable of fully connected layer units for the
                            critic observation network.

      critic_action_fc_layers: Iterable of fully connected layer units for the
                               critic action network.

      critic_joint_fc_layers: Iterable of fully connected layer units for the
                              joint part of the critic network.

      critic_network: Optional custom critic network to use.

      actor_learning_rate: Actor network learning rate.

      critic_learning_rate: Critic network learning rate.

      ou_stddev: Standard deviation for the Ornstein-Uhlenbeck (OU) noise added
                 for exploration.

      ou_damping: Damping factor for the OU noise.

      gamma: Discount factor for future rewards.

      target_update_tau: Factor for soft update of target networks.

      target_update_period: Period for soft update of target networks.

      reward_scale_factor: Multiplicative scale for the reward.

      gradient_clipping: Norm length to clip gradients.

      debug_summaries: Whether to emit debug summaries.

      summarize_grads_and_vars: Whether to summarize gradients and variables.

      train_step_counter: An optional counter to increment every time the train
                          op is run. Defaults to the global_step.

  Returns:
      A TFAgent instance with the DDPG agent.
  """
  # Create train step counter if not provided
  if train_step_counter is None:
    train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

  # Create networks if not provided
  if actor_network is None:
    actor_network = create_sequential_actor_network(
        actor_fc_layers=actor_fc_layers, action_tensor_spec=action_spec
    )

  if critic_network is None:
    critic_network = create_sequential_critic_network(
        obs_fc_layer_units=critic_obs_fc_layers,
        action_fc_layer_units=critic_action_fc_layers,
        joint_fc_layer_units=critic_joint_fc_layers,
    )

  # Create agent
  ddpg_tf_agent = ddpg_agent.DdpgAgent(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network,
      critic_network=critic_network,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=actor_learning_rate
      ),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=critic_learning_rate
      ),
      ou_stddev=ou_stddev,
      ou_damping=ou_damping,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step_counter,
  )

  # Initialize the agent
  ddpg_tf_agent.initialize()

  return ddpg_tf_agent
