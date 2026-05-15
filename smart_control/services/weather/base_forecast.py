"""Base weather forecast.

A weather **forecast** is a collection of sequential forecast periods.

A forecast is generally represented "on the hour" using times like 10:00, 11:00,
12:00, etc. However, it is capable of creating a copy of itself that is
normalized to a given start time (e.g. 10:05, 11:05, 12:05, etc.), and in this
case uses linearly interpolated temperature values for each normalized period.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import functools
from typing import Self

import numpy as np
import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast_period
from smart_buildings.smart_control.utils import temperature_conversion


@dataclasses.dataclass(frozen=True)
class BaseForecast:
  """A collection of forecast periods.

  The forecast can represent any sequence of forecast periods, depending on the
  source of the forecast data. For example:

    - a seven-day forecast, with one or two periods per day
    - a single-day forecast, with one period per hour
    - etc.

  Attributes:
    periods: A sequence of forecast periods.
    interpolation_interval: The time delta between period start times to use
      when constructing an interpolated / normalized version of the forecast.
  """

  periods: Sequence[base_forecast_period.BaseForecastPeriod]
  interpolation_interval: pd.Timedelta = pd.Timedelta(minutes=1)

  def __post_init__(self):
    self._validate_periods()

  def _validate_periods(self):
    """Validates periods are sorted by start_timestamp in ascending order."""
    if not self.periods:
      raise ValueError("Periods cannot be empty.")

    sorted_periods = sorted(self.periods, key=lambda p: p.start_timestamp)
    if list(self.periods) != sorted_periods:
      raise ValueError(
          "Periods must be sorted by start_timestamp in ascending order."
      )

  @property
  def first_period(self) -> base_forecast_period.BaseForecastPeriod:
    """The forecast period with the earliest start time."""
    return self.periods[0]

  @property
  def last_period(self) -> base_forecast_period.BaseForecastPeriod:
    """The forecast period with the latest start time."""
    return self.periods[-1]

  @property
  def temp_unit(self) -> temperature_conversion.TempUnit:
    """The temperature unit for all temperatures in the forecast."""
    return self.first_period.temp_unit

  @property
  def tzinfo(self) -> datetime.tzinfo:
    """The time zone information for all forecast periods."""
    return self.first_period.start_timestamp.tzinfo

  @property
  def time_zone(self) -> str:
    """The time zone name for all forecast periods."""
    return str(self.tzinfo)

  @property
  def df(self) -> pd.DataFrame:
    """The forecast periods, as a pandas DataFrame."""
    return pd.DataFrame((p.as_dict for p in self.periods))

  def filter_periods(
      self,
      ends_after_timestamp: pd.Timestamp | None = None,
      max_periods: int | None = 24,
  ) -> Self:
    """Returns a copy of the forecast, with periods filtered as applicable.

    Args:
      ends_after_timestamp: Filter periods to include only those that end after
        this timestamp (generally used to exclude periods ending in the past).
        If None, no filtering will be done based on the timestamp.
      max_periods: The maximum number of periods to include in the forecast.
        Must be greater than zero because a forecast without any periods would
        be invalid. This filtering step is performed after the timestamp
        filtering step, as applicable. If None, no filtering will be done based
        on the number of periods.

    Returns:
      A copy of the forecast with periods filtered to the given range.
    """
    if ends_after_timestamp is not None:
      ends_after_timestamp = self._parse_timestamp(ends_after_timestamp)

    new_periods = []
    for p in self.periods:
      if ends_after_timestamp and p.end_timestamp <= ends_after_timestamp:
        continue  # Keep only the periods that end after the filter timestamp.
      new_periods.append(p)

    if max_periods is not None:
      new_periods = new_periods[:max_periods]

    return type(self)(periods=new_periods)

  # INTERPOLATION

  @functools.cached_property
  def seconds(self) -> Sequence[float]:
    """A sequence of period start times, as seconds since the epoch (UTC)."""
    return tuple(p.start_seconds for p in self.periods)

  @functools.cached_property
  def temps(self) -> Sequence[float]:
    """A sequence of period temperatures, in ascending order of start time."""
    return tuple(p.temp for p in self.periods)

  def _parse_timestamp(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Ensures the timestamp is timezone-aware, and in the forecasts's timezone.

    Args:
      timestamp: A timezone-aware timestamp.

    Returns:
      A timestamp in forecast's timezone.

    Raises:
      ValueError: if timestamp is timezone-naive.
    """
    if timestamp.tzinfo is None:
      raise ValueError(f"Timestamp {timestamp} must be timezone-aware.")

    if timestamp.tzinfo == self.tzinfo:
      return timestamp

    return timestamp.tz_convert(self.tzinfo)

  def _validate_timestamp_is_in_range(self, timestamp: pd.Timestamp):
    """Ensures the timestamp is within the forecast range."""
    if (
        timestamp < self.first_period.start_timestamp
        or timestamp > self.last_period.end_timestamp
    ):
      raise ValueError(
          f"Timestamp {timestamp} is outside the forecast range: "
          f"({self.first_period.start_timestamp}, "
          f"{self.last_period.end_timestamp})."
      )

  def resample(
      self,
      start_timestamp: pd.Timestamp,
      interval: pd.Timedelta = pd.Timedelta(hours=1),
  ) -> Self:
    """Returns a copy of the forecast, normalized to the given start time.

    The new periods will start at the given start_timestamp and the duration of
    each period will be set to the given interval. The temperature values for
    each period are adjusted using linear interpolation. Periods are generated
    from the requested start_timestamp, up to the end of the forecast range.

    This is the primary method used by the weather service to obtain normalized
    forecast periods with interpolated temperature values.

    Args:
      start_timestamp: The start timestamp to anchor the new forecast periods.
      interval: The time delta between periods / duration of each period.

    Returns:
      A new BaseForecast with resampled periods.

    Raises:
      ValueError: If the `start_timestamp` is timezone-naive, or if the
        `interval` is not a positive duration.
    """
    start_timestamp = self._parse_timestamp(start_timestamp)

    if interval <= pd.Timedelta(0):
      raise ValueError("Interval must be a positive duration.")

    new_periods = []
    i = 0
    while True:
      timestamp = start_timestamp + interval * i
      # Ensure the entire new period is within the bounds of the forecast.
      if timestamp + interval > self.last_period.end_timestamp:
        break
      try:
        self._validate_timestamp_is_in_range(timestamp)
        new_periods.append(
            base_forecast_period.BaseForecastPeriod(
                start_timestamp=timestamp,
                end_timestamp=timestamp + interval,
                temp=np.interp(timestamp.timestamp(), self.seconds, self.temps),
                temp_unit=self.temp_unit,
            )
        )
      except ValueError:
        break  # There are no more periods available in the forecast time range.
      i += 1
    return type(self)(periods=new_periods)

  def interpolate_period(
      self, timestamp: pd.Timestamp
  ) -> base_forecast_period.BaseForecastPeriod:
    """Returns a single forecast period, interpolated to the given timestamp.

    This is an extra, optional convenience method, not used by the weather
    service.

    Args:
      timestamp: The timestamp to interpolate to.

    Returns:
      A forecast period with the given timestamp and interpolated values.

    Raises:
      ValueError: If the timestamp is timezone-naive, or if the timestamp is
        outside the forecast's time range.
    """
    timestamp = self._parse_timestamp(timestamp)
    self._validate_timestamp_is_in_range(timestamp)

    return base_forecast_period.BaseForecastPeriod(
        start_timestamp=timestamp,
        end_timestamp=timestamp + self.interpolation_interval,
        temp=np.interp(timestamp.timestamp(), self.seconds, self.temps),
        temp_unit=self.temp_unit,
    )

  @functools.cached_property
  def interpolated_forecast(self) -> Self:
    """A forecast with minute-level periods and interpolated temperature values.

    This expanded, more granular version of the original forecast covers the
    entire time range of the original forecast.

    This is an extra, optional convenience method, not used by the weather
    service.
    """
    periods = []

    for p in self.periods:
      start_timestamp = p.start_timestamp
      periods.append(self.interpolate_period(start_timestamp))

      while start_timestamp < p.end_timestamp - self.interpolation_interval:
        start_timestamp = start_timestamp + self.interpolation_interval
        periods.append(self.interpolate_period(start_timestamp))

    return BaseForecast(periods=tuple(periods))

  @property
  def interp_df(self) -> pd.DataFrame:
    """A pandas DataFrame of interpolated forecast periods.

    This is an extra, optional convenience method, not used by the weather
    service. It is useful for charting and display purposes.
    """
    return pd.DataFrame((p.as_dict for p in self.interpolated_forecast.periods))
