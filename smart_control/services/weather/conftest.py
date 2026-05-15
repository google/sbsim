"""Test fixtures for base weather service and data classes."""

from typing import Sequence

import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast_period
from smart_buildings.smart_control.utils import temperature_conversion


TIME_ZONE = "UTC"
START_TIMESTAMP = pd.Timestamp("2026-01-01 00:07:00", tz=TIME_ZONE)
END_TIMESTAMP = START_TIMESTAMP + pd.Timedelta(minutes=1)
TEMP = 70.0
TEMP_UNIT = temperature_conversion.TempUnit.FAHRENHEIT


def create_period(
    start_timestamp: pd.Timestamp = START_TIMESTAMP,
    end_timestamp: pd.Timestamp = END_TIMESTAMP,
    temp: float = TEMP,
    temp_unit: temperature_conversion.TempUnit = TEMP_UNIT,
) -> base_forecast_period.BaseForecastPeriod:
  """Creates a forecast period for test purposes."""
  return base_forecast_period.BaseForecastPeriod(
      start_timestamp=start_timestamp,
      end_timestamp=end_timestamp,
      temp=temp,
      temp_unit=temp_unit,
  )


def create_hourly_periods(
    start_timestamp: pd.Timestamp = START_TIMESTAMP,
    n_hours: int = 24,
    temp: float = TEMP,
    temp_unit: temperature_conversion.TempUnit = TEMP_UNIT,
) -> Sequence[base_forecast_period.BaseForecastPeriod]:
  """Creates a list of hourly forecast periods for test purposes."""
  periods = [
      create_period(
          start_timestamp=start_timestamp + pd.Timedelta(hours=i),
          end_timestamp=start_timestamp + pd.Timedelta(hours=i + 1),
          temp=temp,
          temp_unit=temp_unit,
      )
      for i in range(n_hours)
  ]
  return periods
