import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts


class SavedModelPolicy(tf_policy.TFPolicy):
  """Policy that uses a saved TF-Agents policy model."""

  def __init__(self, saved_model_path, time_step_spec, action_spec, name=None):
    """Initialize a SavedModelPolicy.

    Args:
        saved_model_path: Path to the saved model.
        time_step_spec: A `TimeStep` spec of the expected time_steps.
        action_spec: A nest of BoundedTensorSpec representing the actions.
        name: The name of this policy.
    """
    self._saved_model_path = saved_model_path

    # Load the saved policy
    self._loaded_model = tf.saved_model.load(saved_model_path)

    # Use empty tuple as default for policy state
    self._policy_state_spec = ()

    super(SavedModelPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=self._policy_state_spec,
        name=name or 'SavedModelPolicy',
    )

  @tf.function
  def _action(self, time_step, policy_state, seed):
    """Implementation of `action`."""
    # Convert the time_step to tensors
    observation = tf.nest.map_structure(
        tf.convert_to_tensor, time_step.observation
    )
    step_type = tf.convert_to_tensor(time_step.step_type)
    reward = tf.convert_to_tensor(time_step.reward)
    discount = tf.convert_to_tensor(time_step.discount)

    # Recreate the time step with tensors
    time_step_tensors = ts.TimeStep(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=observation,
    )

    # Call the action method of the loaded model
    action_step = self._loaded_model.action(time_step_tensors)
    return action_step

  def _distribution(self, time_step, policy_state):
    """Implementation of `distribution`."""
    # Get deterministic action
    action_step = self._action(time_step, policy_state, seed=None)

    # Create deterministic distribution
    def _to_distribution(action):
      return tfp.distributions.Deterministic(loc=action)

    action_distribution = tf.nest.map_structure(
        _to_distribution, action_step.action
    )

    return policy_step.PolicyStep(
        action=action_distribution,
        state=action_step.state,
        info=action_step.info,
    )
