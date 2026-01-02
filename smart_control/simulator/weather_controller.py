"""Controls ambient temperature in simulator.

Copyright 2023 Google LLC

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
import math
import os
from typing import Final, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.utils import conversion_utils as utils

TemperatureBounds = Tuple[float, float]

_SECONDS_IN_A_DAY: Final[float] = 24 * 3600
_DAYS_IN_A_YEAR: Final[int] = 365
_MIN_RADIANS: Final[float] = -math.pi / 2.0
_MAX_RADIANS: Final[float] = 3.0 * math.pi / 2.0
_EPOCH: Final[pd.Timestamp] = pd.Timestamp('1970-01-01', tz='UTC')

WEATHER_CSV_FILEPATH: Final[str] = os.path.join(
    os.path.dirname(__file__),
    '..',
    'configs',
    'resources',
    'sb1',
    'local_weather_moffett_field_20230701_20231122.csv',
)


@gin.configurable
class BaseWeatherController(metaclass=abc.ABCMeta):
  """Represents the weather on any specific time."""

  @abc.abstractmethod
  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Gets outside temp at specified timestamp."""

  # SHOULD THIS BASE CLASS IMPLEMENT get_air_convection_coefficient AS WELL?


@gin.configurable
class WeatherController(BaseWeatherController):
  """Represents the weather on any given day.

  Weather operates as a sinusoid: low at midnight and high at noon.

  Attributes:
    default_low_temp: Default low temperature in K at midnight.
    default_high_temp: Default high temperature in K at noon.
    special_days: Map of day of year (1-365) to 2-tuple (low_temp, high_temp).
    convection_coefficient: Air convection coefficient (W/m2/K).
  """

  def __init__(
      self,
      default_low_temp: float,
      default_high_temp: float,
      special_days: Optional[Mapping[int, TemperatureBounds]] = None,
      convection_coefficient: float = 12.0,
  ):
    self.default_low_temp = default_low_temp
    self.default_high_temp = default_high_temp
    self.special_days = special_days if special_days else {}
    self.convection_coefficient = convection_coefficient

    if self.default_low_temp > self.default_high_temp:
      raise ValueError(
          'default_low_temp cannot be greater than default_high_temp.'
      )

    for day, temps in self.special_days.items():
      low_temp, high_temp = temps
      if low_temp > high_temp:
        raise ValueError(
            f'Low temp cannot be greater than high temp for special day: {day}.'
        )

  def seconds_to_rads(self, seconds_in_day: int) -> float:
    """Returns radians corresponding to number of second in the day.

    Args:
      seconds_in_day: Seconds that have passed so far in the day.
    """
    return (seconds_in_day / _SECONDS_IN_A_DAY) * (
        _MAX_RADIANS - _MIN_RADIANS
    ) + _MIN_RADIANS

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns current temperature in K.

    Args:
      timestamp: Pandas timestamp to get temperature for.
    """
    today = timestamp.dayofyear
    tomorrow = (today + 1) % _DAYS_IN_A_YEAR

    if today in self.special_days:
      today_low, today_high = self.special_days[today]
    else:
      today_low, today_high = self.default_low_temp, self.default_high_temp

    if tomorrow in self.special_days:
      tomorrow_low, _ = self.special_days[tomorrow]
    else:
      tomorrow_low = self.default_low_temp

    high = today_high
    if timestamp.hour < 12:
      low = today_low
    else:
      low = tomorrow_low

    seconds_in_day = (
        timestamp - pd.Timestamp(timestamp.date())
    ).total_seconds()
    rad = self.seconds_to_rads(seconds_in_day)
    temp = 0.5 * (math.sin(rad) + 1) * (high - low) + low
    return temp

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    """Returns the convection coefficient (W/m2/K) based on the current wind.

    Args:
      timestamp: Pandas timestamp to get convection coefficient for.
    """
    return self.convection_coefficient


def get_replay_temperatures(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
) -> Mapping[str, float]:
  """Returns temperature replays from past observations.

  Args:
    observation_responses: array of observations to extract weather from

  Returns: map from timestamp to temp
  """

  def get_outside_air_temp(observation_response):
    for r in observation_response.single_observation_responses:
      if (
          r.single_observation_request.measurement_name
          == 'outside_air_temperature_sensor'
      ):
        return r.continuous_value
    return -1.0

  temps = {}
  for r in observation_responses:
    temp = get_outside_air_temp(r)
    time = utils.proto_to_pandas_timestamp(r.timestamp)
    temps[str(time)] = temp
  return temps


@gin.configurable
class ReplayWeatherController(BaseWeatherController):
  """Weather controller that interplolates real weather from past observations.

  Attributes:
    local_weather_path: Path to local weather CSV file.
    weather_df: Pandas dataframe of historical weather data.
    convection_coefficient: Air convection coefficient (W/m2/K).
    humidity_column: Column name of the humidity in the weather CSV file.
  """

  def __init__(
      self,
      local_weather_path: str = WEATHER_CSV_FILEPATH,
      convection_coefficient: float = 12.0,
      humidity_column: str = 'Humidity',
  ):
    self.local_weather_path = local_weather_path
    self.weather_df = self.read_weather_csv(self.local_weather_path)
    self.convection_coefficient = convection_coefficient
    self.humidity_column = humidity_column

  @property
  def csv_filepath(self) -> str:
    """Alias for the local weather CSV file path."""
    return self.local_weather_path

  def read_weather_csv(self, csv_filepath: str) -> pd.DataFrame:
    """Loads time series weather data from the specified CSV file.

    The CSV file is expected to have at least the following columns:

      + `Time`: the time, as a string, in the format: `%Y%m%d-%H%M`
            (e.g. `20230701-0000`). Assumed to be in UTC.
      + `TempF`: the temperature in Fahrenheit at the specified time.
      + `Humidity`: the relative humidity in percent at the specified time
            (0 to 100).

    Coerces the times to UTC. Updates the index to be seconds since epoch.

    Args:
      csv_filepath: Path to local weather CSV file.

    Returns:
      Pandas dataframe of weather data.
    """
    df = pd.read_csv(csv_filepath)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    df['Time'] = pd.to_datetime(df['Time'], utc=True)

    df.index = (df['Time'] - _EPOCH).dt.total_seconds()
    df.index.name = 'SecondsSinceEpoch'

    return df

  @property
  def min_time(self) -> pd.Timestamp:
    """Earliest timestamp in the weather data."""
    return min(self.weather_df['Time'])

  @property
  def max_time(self) -> pd.Timestamp:
    """Latest timestamp in the weather data."""
    return max(self.weather_df['Time'])

  @property
  def times_in_seconds(self) -> pd.Index:
    """Returns the timestamps of the weather data, as seconds since epoch."""
    return self.weather_df.index

  @property
  def temps_f(self) -> pd.Series:
    """Returns the temperatures in Fahrenheit of the weather data."""
    return self.weather_df['TempF']

  @property
  def humidities(self) -> pd.Series:
    """Returns the humidities of the weather data."""
    return self.weather_df[self.humidity_column]

  def _get_interpolated_value(
      self, timestamp: pd.Timestamp, values: pd.Series
  ) -> float:
    """Helper to get interpolated value from a given series.

    The timestamp need not exactly appear in the weather data, but should be
    within the range of the data.
    If there is no exact match, linear interpolation is used to estimate the
    temperature between the nearest timestamps.

    Args:
      timestamp: Pandas timestamp to get temperature for interpolation. If the
        timestamp is timezone aware, it will be converted to UTC. If the
        timestamp is timezone naive, it will be localized to UTC. This allows
        for accurate comparisons against the min and max timestamps, as well as
        the epoch, which are always timezone aware (in UTC).
      values: Pandas series to interpolate from.

    Returns:
      The interpolated value from the series at the given timestamp.
    """
    # convert timestamp to UTC to enable proper comparisons:
    if timestamp.tzname() is not None:
      # timestamp is timezone aware, unable to localize, so convert to UTC:
      timestamp = timestamp.tz_convert('UTC')
    else:
      # timestamp is timezone naive, unable to convert, so localize to UTC:
      timestamp = timestamp.tz_localize('UTC')

    if timestamp < self.min_time:
      raise ValueError(
          f'Timestamp not in range. Timestamp {timestamp} is before the'
          f' earliest timestamp {self.min_time}.'
      )
    if timestamp > self.max_time:
      raise ValueError(
          f'Timestamp not in range. Timestamp {timestamp} is after the'
          f' latest timestamp {self.max_time}.'
      )

    time_in_seconds = (timestamp - _EPOCH).total_seconds()
    return np.interp(time_in_seconds, self.times_in_seconds, values)

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """For a given timestamp, returns the current temperature in Kelvin."""
    return utils.fahrenheit_to_kelvin(
        self._get_interpolated_value(timestamp, self.temps_f)
    )

  def get_current_humidity(self, timestamp: pd.Timestamp) -> float:
    """For a given timestamp, returns the current humidity level in percent."""
    return self._get_interpolated_value(timestamp, self.humidities)

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    return self.convection_coefficient
