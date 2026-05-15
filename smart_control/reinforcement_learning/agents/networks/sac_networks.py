"""Network architectures for SAC agent.

This module provides functions to create actor and critic networks for SAC
agents.
"""

import functools
from typing import Sequence

import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.typing import types

# Utility to create dense layers with consistent initialization and activation
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer='glorot_uniform',
)


def create_fc_network(layer_units: Sequence[int]) -> tf.keras.Model:
  """Creates a fully connected network.

  Args:
      layer_units: A sequence of layer units.

  Returns:
      A sequential model of dense layers.
  """
  return sequential.Sequential([dense(num_units) for num_units in layer_units])


def create_identity_layer() -> tf.keras.layers.Layer:
  """Creates an identity layer.

  Returns:
      A Lambda layer that returns its input.
  """
  return tf.keras.layers.Lambda(lambda x: x)


def create_sequential_critic_network(
    obs_fc_layer_units: Sequence[int],
    action_fc_layer_units: Sequence[int],
    joint_fc_layer_units: Sequence[int],
) -> sequential.Sequential:
  """Create a sequential critic network for SAC.

  Args:
      obs_fc_layer_units: Units for observation network layers.
      action_fc_layer_units: Units for action network layers.
      joint_fc_layer_units: Units for joint network layers.

  Returns:
      A sequential critic network.
  """

  # Split the inputs into observations and actions.
  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  # Create an observation network.
  obs_network = (
      create_fc_network(obs_fc_layer_units)
      if obs_fc_layer_units
      else create_identity_layer()
  )

  # Create an action network.
  action_network = (
      create_fc_network(action_fc_layer_units)
      if action_fc_layer_units
      else create_identity_layer()
  )

  # Create a joint network.
  joint_network = (
      create_fc_network(joint_fc_layer_units)
      if joint_fc_layer_units
      else create_identity_layer()
  )

  # Final layer.
  value_layer = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

  return sequential.Sequential(
      [
          tf.keras.layers.Lambda(split_inputs),
          nest_map.NestMap(
              {'observation': obs_network, 'action': action_network}
          ),
          nest_map.NestFlatten(),
          tf.keras.layers.Concatenate(),
          joint_network,
          value_layer,
          inner_reshape.InnerReshape(current_shape=[1], new_shape=[]),
      ],
      name='sequential_critic',
  )


class _TanhNormalProjectionNetworkWrapper(
    tanh_normal_projection_network.TanhNormalProjectionNetwork
):
  """Wrapper to pass predefined `outer_rank` to underlying projection net."""

  def __init__(self, sample_spec, predefined_outer_rank=1):
    super(_TanhNormalProjectionNetworkWrapper, self).__init__(sample_spec)
    self.predefined_outer_rank = predefined_outer_rank

  def call(self, inputs, **kwargs):
    kwargs['outer_rank'] = self.predefined_outer_rank
    if 'step_type' in kwargs:
      del kwargs['step_type']
    return super(_TanhNormalProjectionNetworkWrapper, self).call(
        inputs, **kwargs
    )


def create_sequential_actor_network(
    actor_fc_layers: Sequence[int],
    action_tensor_spec: types.NestedTensorSpec,
) -> sequential.Sequential:
  """Create a sequential actor network for SAC.

  Args:
      actor_fc_layers: Units for actor network fully connected layers.
      action_tensor_spec: The action tensor spec.

  Returns:
      A sequential actor network.
  """

  def tile_as_nest(non_nested_output):
    return tf.nest.map_structure(
        lambda _: non_nested_output, action_tensor_spec
    )

  return sequential.Sequential(
      [dense(num_units) for num_units in actor_fc_layers]
      + [tf.keras.layers.Lambda(tile_as_nest)]
      + [
          nest_map.NestMap(
              tf.nest.map_structure(
                  _TanhNormalProjectionNetworkWrapper, action_tensor_spec
              )
          )
      ]
  )
