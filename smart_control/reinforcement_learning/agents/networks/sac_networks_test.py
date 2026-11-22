"""Tests for SAC network architectures."""

from absl.testing import absltest
import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.specs import tensor_spec
from smart_control.reinforcement_learning.agents.networks import sac_networks

# True: fixed wrapper, pytest passes.
# False: old call, pytest prints the network_state TypeError and fails.
ACCEPT_NETWORK_STATE = True


def _action_spec():
  return tensor_spec.BoundedTensorSpec(
      shape=(2,), dtype=tf.float32, minimum=-1.0, maximum=1.0
  )


def call_with_nestmap_kwargs(accept_network_state):
  """Pass NestMap's kwargs into the projection layer."""
  action_spec = _action_spec()
  if accept_network_state:
    layer = sac_networks._TanhNormalProjectionNetworkWrapper(action_spec)
  else:
    layer = tanh_normal_projection_network.TanhNormalProjectionNetwork(
        action_spec
    )
  return layer.call(
      tf.zeros((1, 8), dtype=tf.float32),
      network_state=(),
      step_type=None,
      training=None,
  )


class SacNetworksTest(absltest.TestCase):
  def test_network_state(self):
    if ACCEPT_NETWORK_STATE:
      distribution = call_with_nestmap_kwargs(True)
      self.assertIsNotNone(distribution)
    else:
      call_with_nestmap_kwargs(False)

  def test_actor_network_create_variables(self):
    observation_spec = tensor_spec.TensorSpec(shape=(4,), dtype=tf.float32)
    actor_network = sac_networks.create_sequential_actor_network(
        actor_fc_layers=(8,),
        action_tensor_spec=_action_spec(),
    )
    actor_network.create_variables(observation_spec)


if __name__ == '__main__':
  absltest.main()