"""Base class for a convection simulator.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A convection simulator provides a method for simulating airflow convection in
a building.
"""

import abc
from typing import MutableSequence

import numpy as np


class BaseConvectionSimulator(metaclass=abc.ABCMeta):
  """Represents a method of simulating air convection."""

  @abc.abstractmethod
  def apply_convection(
      self,
      room_dict: dict[str, MutableSequence[tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies convection to the temperature array in place.

    Splits up rooms via room_dict.

    Args:
      room_dict: A dictionary mapping of room coordinates.
      temp: An array of temperatures.
    """
