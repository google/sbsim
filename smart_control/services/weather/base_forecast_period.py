"""Base weather forecast period.

A weather **forecast period** is a range of time for which a given weather
forecast is applicable.
"""

import dataclasses
from typing import Any

import pandas as pd
from smart_buildings.smart_control.utils import temperature_conversion


@dataclasses.dataclass(frozen=True)
class BaseForecastPeriod:
  """A single forecast period.

  A forecast period is a range of time for which a given weather forecast is
  applicable. This can be an hourly forecast period covering a single hour (the
  most common use case), or a longer forecast period covering multiple hours
  (depending on the source of the forecast data).

  The period is bounded by a start and end timestamp, which are validated to be
  timezone-aware, and sharing the same timezone. At a minimum, the forecast
  period provides information about the given temperature (and corresponding
  temperature unit).

  Attributes:
    start_timestamp: The period start time, as a timezone-aware pandas
      Timestamp.
    end_timestamp: The period end time, as a timezone-aware pandas Timestamp.
    temp: The temperature value (e.g. 70.0).
    temp_unit: The temperature unit (e.g. 'Kelvin', 'Celsius', or 'Fahrenheit').
  """

  start_timestamp: pd.Timestamp
  end_timestamp: pd.Timestamp
  temp: float
  temp_unit: temperature_conversion.TempUnit

  def __post_init__(self):
    """Validates the timestamps and their timezones after initialization."""
    self._validate_time_zones()
    self._validate_timestamps()

  def _validate_time_zones(self):
    """Ensures start and end timestamp time zones are present and matching."""
    if self.start_timestamp.tzinfo is None or self.end_timestamp.tzinfo is None:
      raise ValueError(
          f"start_timestamp ({self.start_timestamp}) and end_timestamp"
          f" ({self.end_timestamp}) must have a time zone."
      )

    if self.start_timestamp.tzinfo != self.end_timestamp.tzinfo:
      raise ValueError(
          f"start_timestamp ({self.start_timestamp}) must be in the same time"
          f" zone as end_timestamp ({self.end_timestamp})."
      )

  def _validate_timestamps(self):
    """Ensures start_timestamp is before end_timestamp."""
    if self.start_timestamp >= self.end_timestamp:
      raise ValueError(
          f"start_timestamp ({self.start_timestamp}) must be before "
          f"end_timestamp ({self.end_timestamp})."
      )

  @property
  def duration(self) -> pd.Timedelta:
    """The period's duration, as a pandas Timedelta."""
    return self.end_timestamp - self.start_timestamp

  @property
  def start_date(self) -> str:
    """The period's start date, as a string."""
    return str(self.start_timestamp.date())

  @property
  def end_date(self) -> str:
    """The period's end date, as a string."""
    return str(self.end_timestamp.date())

  @property
  def start_seconds(self) -> float:
    """The period's start time, as seconds since the epoch (UTC)."""
    return self.start_timestamp.timestamp()

  @property
  def end_seconds(self) -> float:
    """The period's end time, as seconds since the epoch (UTC)."""
    return self.end_timestamp.timestamp()

  @property
  def as_dict(self) -> dict[str, Any]:
    """A dictionary of period attributes."""
    return dataclasses.asdict(self) | dict(
        duration=self.duration,
        start_date=self.start_date,
        end_date=self.end_date,
        start_seconds=self.start_seconds,
        end_seconds=self.end_seconds,
    )
