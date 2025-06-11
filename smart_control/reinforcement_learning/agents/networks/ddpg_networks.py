"""Network architectures for DDPG agent.

This module provides functions to create actor and critic networks for DDPG agents.
"""

import functools
from typing import Sequence

import tensorflow as tf
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.typing import types
from tf_agents.utils import common

# Utility to create dense layers with consistent initialization and activation
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=1.0 / 3.0, mode='fan_in', distribution='uniform'
    ),
)


def create_identity_layer() -> tf.keras.layers.Layer:
  """Creates an identity layer.

  Returns:
      A Lambda layer that returns its input.
  """
  return tf.keras.layers.Lambda(lambda x: x)


def create_fc_network(layer_units: Sequence[int]) -> tf.keras.Model:
  """Creates a fully connected network.

  Args:
      layer_units: A sequence of layer units.

  Returns:
      A sequential model of dense layers.
  """
  return sequential.Sequential([dense(num_units) for num_units in layer_units])


def create_sequential_actor_network(
    actor_fc_layers: Sequence[int],
    action_tensor_spec: types.NestedTensorSpec,
) -> sequential.Sequential:
  """Create a sequential actor network for DDPG.

  Args:
      actor_fc_layers: Units for actor network fully connected layers.
      action_tensor_spec: The action tensor spec.

  Returns:
      A sequential actor network.
  """
  flat_action_spec = tf.nest.flatten(action_tensor_spec)
  if len(flat_action_spec) > 1:
    raise ValueError('Only a single action tensor is supported by this network')
  flat_action_spec = flat_action_spec[0]

  fc_layers = [dense(num_units) for num_units in actor_fc_layers]
  num_actions = flat_action_spec.shape.num_elements()
  action_fc_layer = tf.keras.layers.Dense(
      num_actions,
      activation=tf.keras.activations.tanh,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003
      ),
  )

  scaling_layer = tf.keras.layers.Lambda(
      lambda x: common.scale_to_spec(x, flat_action_spec)
  )
  return sequential.Sequential(fc_layers + [action_fc_layer, scaling_layer])


def create_sequential_critic_network(
    obs_fc_layer_units: Sequence[int],
    action_fc_layer_units: Sequence[int],
    joint_fc_layer_units: Sequence[int],
) -> sequential.Sequential:
  """Create a sequential critic network for DDPG.

  Args:
      obs_fc_layer_units: Units for observation network layers.
      action_fc_layer_units: Units for action network layers.
      joint_fc_layer_units: Units for joint network layers.

  Returns:
      A sequential critic network.
  """

  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  obs_network = (
      create_fc_network(obs_fc_layer_units)
      if obs_fc_layer_units
      else create_identity_layer()
  )
  action_network = (
      create_fc_network(action_fc_layer_units)
      if action_fc_layer_units
      else create_identity_layer()
  )
  joint_network = (
      create_fc_network(joint_fc_layer_units)
      if joint_fc_layer_units
      else create_identity_layer()
  )
  value_fc_layer = tf.keras.layers.Dense(
      1,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003
      ),
  )

  return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({'observation': obs_network, 'action': action_network}),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_fc_layer,
      inner_reshape.InnerReshape([1], []),
  ])
