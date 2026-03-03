"""Defines an occupancy model that returns the average number of occupants.

Occupancy refers to the average number of people in a zone within a specified
period of time. Concrete classes can either simulate the occupancy or
estimate the occupancy from Calendar or motion sensors in the buildings.

The occupancy signal is an input to the agent's reward function.
"""

import abc

import pandas as pd


class BaseOccupancy(metaclass=abc.ABCMeta):
  """Provides the RL agent information about how many people are in a zone."""

  @abc.abstractmethod
  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Returns the occupancy within start_time, end_time for the zone.

    If the zone is not found, implementations should raise a ValueError.

    Args:
      zone_id: specific zone identifier for the building.
      start_time: **local time** w/ TZ for the beginning of the interval.
      end_time: **local time** w/ TZ for the end of the interval.

    Returns:
      average number of people in the zone for the interval.
    """
    pass
