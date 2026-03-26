"""Base class for agents that use the control loop."""

import abc
from typing import Any

from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2


SerializableData = dict[str, Any]


class BaseControlAgent(abc.ABC):
  """An AI agent that chooses actions based on info from the environment."""

  @abc.abstractmethod
  def get_action_context(
      self,
      observation_response: building_pb2.ObservationResponse | None = None,
      reward_info: reward_pb2.RewardInfo | None = None,
  ) -> action_context.ActionContext:
    """Returns an action context based on the agent's strategy / policy.

    Args:
      observation_response: The observation response from the environment.
      reward_info: The reward info from the environment.

    Returns:
      An action context based on the agent's strategy / policy.
    """

  @property
  def json_metadata(self) -> SerializableData:
    """Info about the agent and its setup, to be written to a JSON file."""
    return {"type": self.__class__.__name__}
