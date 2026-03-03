"""Base class for a convection simulator.

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
