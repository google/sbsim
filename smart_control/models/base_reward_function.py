"""Base class for smart buildings reward function.

    Copyright 2022 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import abc
from smart_control.proto import smart_control_reward_pb2


class BaseRewardFunction(metaclass=abc.ABCMeta):
  """Base class that converts the building energy information into a reward."""

  @abc.abstractmethod
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Returns the real-valued reward for the current state of the building."""

