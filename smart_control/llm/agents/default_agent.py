"""Default policy agent.

This agent employs a fixed strategy that uses the environment's default policy
values for all of its actions.

This strategy is overly simplistic, but provides a decent foundation for
child classes to inherit from, and can be useful for testing and debugging the
agent control loop.
"""

from typing import Final

import numpy as np
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.llm.agents import base_agent
from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.llm.schema import output_schema
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import serialization

DEFAULT_JUSTIFICATION: Final[str] = "Default action."
DEFAULT_SETPOINT_JUSTIFICATION: Final[str] = "Default value."


class DefaultPolicyAgent(base_agent.BaseControlAgent):
  """A control agent that uses the environment's default policy values.

  Attributes:
    env: The environment to be controlled. Should be configured with default
      policy values.
  """

  def __init__(self, env: environment.Environment, clip: bool = True):
    """Initializes the instance.

    Args:
      env: The environment to be controlled. Should be configured with
        default policy values.
      clip: Whether to clip setpoint values to the bounds of the valid range. If
        `False`, raises `GuardrailsExceededError`. Otherwise, clips the
        setpoint values to the valid range, and logs a record of the error.
        Defaults to `True`.
    """
    super().__init__()
    self._clip = clip
    self.env = self._validate_environment(env)

  def _validate_environment(
      self, env: environment.Environment
  ) -> environment.Environment:
    """Ensures the environment has default values."""
    if env.action_names is None:
      raise ValueError("Expecting environment to have action names.")

    if env.default_policy_values is None:
      raise ValueError("Expecting environment to have default policy values.")

    if len(env.action_names) != len(env.default_policy_values):
      raise ValueError(
          "Expecting environment to have the same number of action names and"
          " default policy values."
      )

    return env

  @property
  def json_metadata(self) -> serialization.SerializableData:
    """Info about the agent and its setup, to be written to a JSON file."""
    return super().json_metadata | {
        "default_policy": {
            "action_names": self.env.action_names,
            "default_values": self.env.default_action_values,
        },
        "clip": self._clip,
    }

  @property
  def action_context_class(self) -> type[action_context.ActionContext]:
    """The action context class to be used by this agent."""
    if isinstance(self.env, hybrid_action_environment.HybridActionEnvironment):
      return action_context.HybridActionContext
    return action_context.ActionContext

  def get_default_action_context(self) -> action_context.ActionContext:
    """Compiles an action context using the environment's default values."""

    setpoints = []
    for action_name, normalized_value in zip(
        self.env.action_names, self.env.default_action_values
    ):
      device_id, setpoint_name = self.env.id_map.inv[action_name]
      normalizer = self.env.action_normalizers.get(setpoint_name)
      if normalizer is None:
        raise ValueError(f"No normalizer found for setpoint: {setpoint_name}")

      native_value = normalizer.setpoint_value(np.array(normalized_value))
      setpoints.append(
          output_schema.DeviceSetpoint(
              device_id=device_id,
              setpoint_name=setpoint_name,
              setpoint_value=native_value,
              justification=DEFAULT_SETPOINT_JUSTIFICATION,
          )
      )

    return self.action_context_class(
        env=self.env,
        clip=self._clip,
        timestamp=str(self.env.current_local_timestamp),
        justification=DEFAULT_JUSTIFICATION,
        validity_interval=self.env.time_step_mins,
        setpoints=setpoints,
    )

  def get_action_context(
      self,
      observation_response: building_pb2.ObservationResponse | None = None,
      reward_info: reward_pb2.RewardInfo | None = None,
  ) -> action_context.ActionContext:
    """The action context to be used within the agent control loop."""
    del observation_response, reward_info  # Unused in this implementation.
    return self.get_default_action_context()
