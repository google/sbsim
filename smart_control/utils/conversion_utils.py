"""General-purpose conversion utilities for smart control.

Copyright 2022 Google LLC

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

import collections
import datetime
import enum
import functools
import re
import types
from typing import Mapping, Tuple

import holidays
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_reward_pb2

from google.protobuf import timestamp_pb2

_COUNTRY = 'US'
_SECONDS_IN_DAY = 24 * 3600
_WATT_SECONDS_KWH = 1.0 / 3600.0 / 1000.0
_DAYS_IN_WEEK = 7.0


def pandas_to_proto_timestamp(
    pandas_timestamp: pd.Timestamp,
) -> timestamp_pb2.Timestamp:
  """Converts a Pandas Timestamp to a protobuf.Timestamp."""
  ts = timestamp_pb2.Timestamp()
  ts.seconds = int(pandas_timestamp.timestamp())

  # Use micro/nanosecond from Pandas Timestamp API to preserve the precision of
  # the original timestamp.
  ts.nanos = pandas_timestamp.microsecond * 1000 + pandas_timestamp.nanosecond
  return ts


def proto_to_pandas_timestamp(
    proto_timestamp: timestamp_pb2.Timestamp,
) -> pd.Timestamp:
  """Converts a protobuf.Timestamp to Pandas Timestamp."""

  return pd.Timestamp(
      proto_timestamp.seconds, unit='s', tz='UTC'
  ) + pd.Timedelta(proto_timestamp.nanos, unit='ns')


@functools.cache
def _us_holidays() -> Mapping[datetime.date, str]:
  return types.MappingProxyType(holidays.US())


def is_work_day(timestamp: pd.Timestamp):
  """Returns whether timestamp is on a workday."""

  return timestamp.weekday() < 5 and timestamp.date() not in _us_holidays()


def zone_coordinates_to_id(coordinates: Tuple[int, int]) -> str:
  return 'zone_id_' + str(coordinates)


def floor_plan_based_zone_identifier_to_id(identifier: str) -> str:
  return 'zone_id_' + identifier.replace('room_', '')


def zone_id_to_coordinates(zone_id: str) -> Tuple[int, int]:
  p = r'^zone_id_[(](\d+), (\d+)[)]'
  m = re.match(p, zone_id)
  if m:
    return int(m.group(1)), int(m.group(2))
  raise ValueError('Could not convert zone_id to coordinates!')


def normalize_dow(dow: int) -> float:
  """Returns a normalized day of week, mapping [0, 6] to [-1., 1.]."""
  assert dow <= 6 and dow >= 0
  return (float(dow) - 3.0) / 3.0


def normalize_hod(hod: int) -> float:
  """Returns a normlized hour of day, mapping  [0,23] to [-1., 1.]."""
  assert hod <= 23 and hod >= 0
  return (float(hod) - 11.5) / 11.5


# TODO(sipple): Change HOUR_OF_DAY to TIME_OF_DAY to be more explicit.
class TimeIntervalEnum(enum.Enum):
  DAY_OF_WEEK = 1
  HOUR_OF_DAY = 2


def get_radian_time(
    timestamp: pd.Timestamp, time_interval: TimeIntervalEnum
) -> float:
  """Converts the timestamp into a radian for time interval, ranging 0 - 2pi.

  Args:
    timestamp: Current timestamp, in the local timezone.
    time_interval: the cycle width timestamp to map into range 0 - 2pi.

  Returns:
    The radian value for the timestamp.
  """

  day_local = pd.Timestamp(
      year=timestamp.year,
      month=timestamp.month,
      day=timestamp.day,
      tz=timestamp.tz,
  )
  if time_interval == TimeIntervalEnum.DAY_OF_WEEK:
    week_day = day_local.weekday()
    interval_frac = float(week_day) / _DAYS_IN_WEEK
  elif time_interval == TimeIntervalEnum.HOUR_OF_DAY:
    dt = (timestamp - day_local).total_seconds()
    interval_frac = dt / _SECONDS_IN_DAY
  else:
    raise ValueError(f'No cycle conversion for {time_interval}.')
  return 2.0 * np.pi * interval_frac


def kelvin_to_fahrenheit(kelvin: float) -> float:
  """Converts Kelvin to °F.

  Args:
    kelvin: Temperature in Kelvin, where 273K = 32°F.

  Returns:
    The temperature in °F.

  Raises:
    A ValueError if the input value is negative.
  """
  if kelvin <= 0.0:
    raise ValueError('Temperature must be greater than absolute zero.')
  celsius = kelvin - 273.15
  return celsius * 9.0 / 5.0 + 32.0


def fahrenheit_to_kelvin(fahrenheit: float) -> float:
  """Converts °F to Kelvin.

  Args:
    fahrenheit: Temperature in Kelvin, where 273K = 32°F.

  Returns:
    The temperature in K.

  Raises:
    A ValueError if the input value <= absolute 0, −459.67°F.
  """
  if fahrenheit <= -495.67:
    raise ValueError('Temperature must be greater than absolute zero.')
  celsius = (fahrenheit - 32.0) * 5.0 / 9.0
  return celsius + 273.15


def get_reward_info_energy_use(
    reward_info: smart_control_reward_pb2.RewardInfo,
) -> Mapping[str, float]:
  """Converts to energy use in kWh for ac, blower, pump, and nat gas heating."""
  start_timestamp = proto_to_pandas_timestamp(reward_info.start_timestamp)
  end_timestamp = proto_to_pandas_timestamp(reward_info.end_timestamp)
  dt = (end_timestamp - start_timestamp).total_seconds()

  energy_use = collections.defaultdict(float)

  for air_handler_id in reward_info.air_handler_reward_infos:
    energy_use['air_handler_blower_electricity'] += (
        reward_info.air_handler_reward_infos[
            air_handler_id
        ].blower_electrical_energy_rate
        * dt
        * _WATT_SECONDS_KWH
    )
    energy_use['air_handler_air_conditioning'] += (
        reward_info.air_handler_reward_infos[
            air_handler_id
        ].air_conditioning_electrical_energy_rate
        * dt
        * _WATT_SECONDS_KWH
    )

  for boiler_id in reward_info.boiler_reward_infos:
    energy_use['boiler_natural_gas_heating_energy'] += (
        reward_info.boiler_reward_infos[
            boiler_id
        ].natural_gas_heating_energy_rate
        * dt
        * _WATT_SECONDS_KWH
    )
    energy_use['boiler_pump_electrical_energy'] += (
        reward_info.boiler_reward_infos[boiler_id].pump_electrical_energy_rate
        * dt
        * _WATT_SECONDS_KWH
    )

  return energy_use
