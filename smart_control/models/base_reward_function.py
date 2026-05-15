"""Base class for smart buildings reward function."""

import abc
from typing import Any

from smart_buildings.smart_control.proto import smart_control_reward_pb2


class BaseRewardFunction(metaclass=abc.ABCMeta):
  """Base class that converts the building energy information into a reward."""

  @abc.abstractmethod
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Returns the real-valued reward for the current state of the building."""

  @property
  def json_metadata(self) -> dict[str, Any]:
    """Info to write into a JSON file. Needs to be serializable."""
    return {"type": self.__class__.__name__,}
