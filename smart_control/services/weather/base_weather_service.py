"""Base classes for a weather service that fetches forecast data."""

import abc

import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast


class BaseWeatherService(abc.ABC):
  """A weather service for fetching forecast data."""

  @abc.abstractmethod
  def get_hourly_forecast(
      self,
      *,
      start_timestamp: pd.Timestamp | None = None,
      n_hours: int = 24,
      normalize_to_start: bool = False,
  ) -> base_forecast.BaseForecast:
    """Fetches upcoming hourly forecast data for a specific location.

    The resulting forecast should be sorted by start_timestamp in ascending
    order, and only include periods that end after the specified
    start_timestamp (excludes periods ending in the past).

    Args:
      start_timestamp: The timestamp to start fetching forecast data from.
        Optional, because some services may default to fetching current
        conditions.
      n_hours: The maximum number of hours to fetch forecast data for. It is
        possible this method can return fewer than n_hours periods, for example,
        depending on the availability of forecast data, or depending on the
        requested normalization conditions (the last normalized period may be
        excluded if its adjusted end time would exceed the forecast range).
      normalize_to_start: If False (default), returns forecasts on the hour
        (e.g. 10:00, 11:00, etc.), or whatever was returned by the service. If
        True and start_timestamp is provided, returns forecasts normalized to
        start_timestamp (e.g. 10:05, 11:05, etc.). Uses linear interpolation to
        calculate forecast values at the normalized times.

    Returns:
      A forecast object containing a list of hourly forecast periods.
    """
