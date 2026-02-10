"""Base class for energy cost and carbon, for use in reward function."""

import abc

import pandas as pd


class BaseEnergyCost(metaclass=abc.ABCMeta):
  """Class that returns a cost for energy consumed over a time interval."""

  @abc.abstractmethod
  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Computes cost in USD for energy consumed from start_time to end_time.

    Fundamentally, energy cost is computed as:
      energy_consumed [J] = energy_rate [W] * (end_time - start_time) [s]
      energy_cost = energy_consumed [J] * energy_price [USD/J]

    Most utilities use different units for their pricing, such as kWh,
    Btu/hr, cubic feet per minute, etc., so implementations will have to perform
    all necessary conversions internally.

    Args:
      start_time: starting **local** time for the energy use.
      end_time: ending **local** time for the energy use.
      energy_rate: constant-rate power in Watts consumed over the interval.

    Returns: cost in USD of the energy consumed over the interval.
    """
    pass

  @abc.abstractmethod
  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Returns the mass of carbon emitted from the energy consumption.

    The energy-to-carbon emission is source specific. Assuming a constant
    rate of energy consumption (W) of the time interval bounded by
    start_time and end_time, we can estimate total energy use (J). The type
    of source will convert energy into carbon mass (kg).

    Args:
      start_time: starting **local** time for the energy use.
      end_time: ending **local** time for the energy use.
      energy_rate: constant-rate power in Watts consumed over the interval.

    Returns: carbon mass (kg) emitted
    """
    pass
