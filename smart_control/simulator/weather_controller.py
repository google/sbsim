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
from typing import Final, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils as utils

TemperatureBounds = Tuple[float, float]

_SECONDS_IN_A_DAY: Final[float] = 24 * 3600
_DAYS_IN_A_YEAR: Final[int] = 365
_MIN_RADIANS: Final[float] = -math.pi / 2.0
_MAX_RADIANS: Final[float] = 3.0 * math.pi / 2.0
_EPOCH: Final[pd.Timestamp] = pd.Timestamp('1970-01-01', tz='UTC')


@gin.configurable
class BaseWeatherController(metaclass=abc.ABCMeta):
  """Represents the weather on any specific time."""

  @abc.abstractmethod
  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Gets outside temp at specified timestamp."""


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
class ReplayWeatherController:
  """Weather controller that interplolates real weather from past observations.

  Attributes:
    local_weather_path: Path to local weather file.
    convection_coefficient: Air convection coefficient (W/m2/K).
  """

  def __init__(
      self,
      local_weather_path: str,
      convection_coefficient: float = 12.0,
  ):
    self._weather_data = pd.read_csv(local_weather_path)
    self._weather_data['Time'] = [
        pd.Timestamp(t, tz='UTC') for t in self._weather_data['Time']
    ]
    self._weather_data.index = [
        (t - _EPOCH).total_seconds() for t in self._weather_data['Time']
    ]
    self.convection_coefficient = convection_coefficient

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns current temperature in K.

    Args:
      timestamp: Pandas timestamp to get temperature for interpolation.
    """
    timestamp = timestamp.tz_convert('UTC')
    min_time = min(self._weather_data['Time'])
    if timestamp < min_time:

      raise ValueError(
          f'Attempting to get weather data at {timestamp}, before the latest'
          f' timestamp {min_time}.'
      )
    max_time = max(self._weather_data['Time'])
    if timestamp > max_time:

      raise ValueError(
          f'Attempting to get weather data at {timestamp}, after the latest'
          f' timestamp {max_time}.'
      )

    times = np.array(self._weather_data.index)
    target_timestamp = (timestamp - _EPOCH).total_seconds()
    temps = self._weather_data['TempF']
    temp_f = np.interp(target_timestamp, times, temps)
    return utils.fahrenheit_to_kelvin(temp_f)

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    return self.convection_coefficient
