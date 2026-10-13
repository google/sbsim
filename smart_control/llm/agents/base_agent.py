"""Base class for agents that use the control loop."""

import abc
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any

from smart_buildings.smart_control.llm.schema import action_context
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2 as reward_pb2
from smart_buildings.smart_control.utils import serialization


@dataclasses.dataclass
class AgentErrorRecord:
  """Record of an error produced by an agent.

  Attributes:
    error_type: The class name of the exception.
    error_message: The string representation of the error.
    details: Structured error details (e.g. from Pydantic's ValidationError).
    metadata: Extra metadata about the error (e.g. raw response text).
  """

  error_type: str
  error_message: str
  details: Sequence[Mapping[str, Any]] | None = None
  metadata: Mapping[str, Any] | None = None

  @property
  def json_metadata(self) -> serialization.SerializableData:
    """A JSON-serializable representation of the error record."""
    return serialization.to_serializable(dataclasses.asdict(self))


class BaseControlAgent(abc.ABC):
  """An agent that chooses actions based on info from the environment.

  Attributes:
    errors: A list of errors recorded by the agent during its operation.
  """

  def __init__(self):
    self.errors: list[AgentErrorRecord] = []

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
  def json_metadata(self) -> serialization.SerializableData:
    """Info about the agent and its setup, to be written to a JSON file."""
    return {"type": self.__class__.__name__}
