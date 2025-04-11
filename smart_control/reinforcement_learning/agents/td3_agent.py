from typing import Optional, Sequence

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.networks import network
from tf_agents.typing import types


class CustomActorNetwork(network.Network):
    """Custom actor network for TD3 agent."""
    
    def __init__(
        self,
        input_tensor_spec,
        output_tensor_spec,
        fc_layer_params=(256, 256),
        name='CustomActorNetwork'
    ):
        super(CustomActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )
        self._output_tensor_spec = output_tensor_spec
        
        # Define the layers
        self._layers = []
        for num_units in fc_layer_params:
            self._layers.append(
                tf.keras.layers.Dense(
                    num_units,
                    activation=tf.keras.activations.relu,
                    kernel_initializer='glorot_uniform'
                )
            )
        
        # Output layer
        self._layers.append(
            tf.keras.layers.Dense(
                output_tensor_spec.shape.num_elements(),
                activation=tf.keras.activations.tanh,
                kernel_initializer='glorot_uniform'
            )
        )

    def call(self, observations, step_type=None, network_state=(), training=False):
        del step_type  # Unused.
        observations = tf.cast(observations, tf.float32)
        
        # The issue is here - BatchSquash expects a tensor with at least 2 dimensions
        # If observations comes in as (1, 53), we need to ensure it stays 2D
        
        # Option 1: Skip the batch squashing if not needed
        output = observations
        for layer in self._layers:
            output = layer(output, training=training)
        
        # Scale the output actions
        action_means = (self._output_tensor_spec.maximum + self._output_tensor_spec.minimum) / 2.0
        action_magnitudes = (self._output_tensor_spec.maximum - self._output_tensor_spec.minimum) / 2.0
        output = action_means + action_magnitudes * output
        
        return output, network_state


def create_sequential_critic_network(
    observation_spec,
    action_spec,
    obs_fc_layer_units,
    action_fc_layer_units,
    joint_fc_layer_units
):
    """Create a sequential critic network for TD3."""
    
    # Create a proper critic network class that inherits from network.Network
    class CriticNetwork(network.Network):
        def __init__(self, name='CriticNetwork'):
            # The input tensor spec is a tuple of (observation_spec, action_spec)
            input_tensor_spec = (observation_spec, action_spec)
            super(CriticNetwork, self).__init__(
                input_tensor_spec=input_tensor_spec,
                state_spec=(),
                name=name
            )
            
            # Define the network layers
            # Observation layers
            self.obs_layers = []
            for num_units in obs_fc_layer_units:
                self.obs_layers.append(
                    tf.keras.layers.Dense(
                        num_units, 
                        activation=tf.keras.activations.relu,
                        kernel_initializer='glorot_uniform'
                    )
                )
                
            # Action layers
            self.action_layers = []
            for num_units in action_fc_layer_units:
                self.action_layers.append(
                    tf.keras.layers.Dense(
                        num_units, 
                        activation=tf.keras.activations.relu,
                        kernel_initializer='glorot_uniform'
                    )
                )
                
            # Joint layers after concatenation
            self.joint_layers = []
            for num_units in joint_fc_layer_units:
                self.joint_layers.append(
                    tf.keras.layers.Dense(
                        num_units, 
                        activation=tf.keras.activations.relu,
                        kernel_initializer='glorot_uniform'
                    )
                )
                
            # Final value layer
            self.value_layer = tf.keras.layers.Dense(
                1, 
                kernel_initializer='glorot_uniform'
            )
            
        def call(self, inputs, step_type=None, network_state=(), training=False):
            observations, actions = inputs
            
            # Process observations
            obs_output = observations
            for layer in self.obs_layers:
                obs_output = layer(obs_output, training=training)
                
            # Process actions
            action_output = actions
            for layer in self.action_layers:
                action_output = layer(action_output, training=training)
                
            # Concatenate observation and action outputs
            joint_input = tf.concat([obs_output, action_output], axis=-1)
            
            # Process joint input
            joint_output = joint_input
            for layer in self.joint_layers:
                joint_output = layer(joint_output, training=training)
                
            # Get final value
            value = self.value_layer(joint_output)
            value = tf.squeeze(value, axis=-1)
            
            return value, network_state
    
    # Create and return an instance of our critic network
    return CriticNetwork()


def create_td3_agent(
    time_step_spec: types.TimeStep,
    action_spec: types.NestedTensorSpec,
    
    # Actor network parameters
    actor_fc_layers: Sequence[int] = (256, 256),
    actor_network: Optional[network.Network] = None,
    
    # Critic network parameters
    critic_obs_fc_layers: Sequence[int] = (256, 128),
    critic_action_fc_layers: Sequence[int] = (256, 128),
    critic_joint_fc_layers: Sequence[int] = (256, 128),
    critic_network_1: Optional[network.Network] = None,
    critic_network_2: Optional[network.Network] = None,
    
    # Optimizer parameters
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    
    # TD3 specific parameters
    exploration_noise_std: float = 0.1,
    target_policy_noise: float = 0.2,
    target_policy_noise_clip: float = 0.5,
    
    # Agent parameters
    gamma: float = 0.99,
    target_update_tau: float = 0.005,
    target_update_period: int = 2,
    reward_scale_factor: float = 1.0,
    
    # Training parameters
    gradient_clipping: Optional[float] = None,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    train_step_counter: Optional[tf.Variable] = None,
) -> tf_agent.TFAgent:
    """Creates a TD3 Agent.
    
    Args:
        time_step_spec: A `TimeStep` spec of the expected time_steps.
        
        action_spec: A nest of BoundedTensorSpec representing the actions.
        
        actor_fc_layers: Iterable of fully connected layer units for the actor network.
        
        actor_network: Optional custom actor network to use.
        
        critic_obs_fc_layers: Iterable of fully connected layer units for the critic 
                              observation network.
                              
        critic_action_fc_layers: Iterable of fully connected layer units for the critic
                                 action network.
                                 
        critic_joint_fc_layers: Iterable of fully connected layer units for the joint 
                                part of the critic network.
                                
        critic_network_1: Optional custom critic network 1 to use.
        
        critic_network_2: Optional custom critic network 2 to use.
        
        actor_learning_rate: Actor network learning rate.
        
        critic_learning_rate: Critic network learning rate.
        
        exploration_noise_std: Standard deviation of the exploration noise.
        
        target_policy_noise: Standard deviation of the noise added to target actions.
        
        target_policy_noise_clip: Value to clip the target policy noise.
        
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
        A BaseAgent instance with the TD3 agent.
    """
    # Create train step counter if not provided
    if train_step_counter is None:
        train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    # Create actor network if not provided
    if actor_network is None:
        actor_network = CustomActorNetwork(
            input_tensor_spec=time_step_spec.observation,
            output_tensor_spec=action_spec,
            fc_layer_params=actor_fc_layers,
            name='TD3ActorNetwork'
        )
    
    # Create critic networks if not provided
    # Create critic networks if not provided
    if critic_network_1 is None:
        critic_network_1 = create_sequential_critic_network(
            observation_spec=time_step_spec.observation,  # Add these parameters
            action_spec=action_spec,                      # Add these parameters
            obs_fc_layer_units=critic_obs_fc_layers,
            action_fc_layer_units=critic_action_fc_layers,
            joint_fc_layer_units=critic_joint_fc_layers
        )
    
    if critic_network_2 is None:
        critic_network_2 = create_sequential_critic_network(
            observation_spec=time_step_spec.observation,  # Add these parameters
            action_spec=action_spec,                      # Add these parameters
            obs_fc_layer_units=critic_obs_fc_layers,
            action_fc_layer_units=critic_action_fc_layers,
            joint_fc_layer_units=critic_joint_fc_layers
        )
    
    # Create agent
    td3_agent_obj = td3_agent.Td3Agent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network,
        critic_network=critic_network_1,
        critic_network_2=critic_network_2,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
        exploration_noise_std=exploration_noise_std,
        target_policy_noise=target_policy_noise,
        target_policy_noise_clip=target_policy_noise_clip,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter
    )
    
    return td3_agent_obj