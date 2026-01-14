"""Stores and maintains setpoint schedule of HVAC in simulator.

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

import datetime
from typing import Optional, Set, Tuple

import gin
import pandas as pd
import pytz

TemperatureWindow = Tuple[int, int]


@gin.configurable
class SetpointSchedule:
  """Represents the desired temperature bounds of the building for each day.

  Building setpoint schedules have temperature windows for day (comfort) and
  night (eco).
  Weekends are treated as nights. Additionally specific weekdays can be set
  to the night windows as well to account for holidays.

  Temperature windows are a tuple: (heating_setpoint, cooling_setpoint)
  where heating < cooling.

  Attributes:
    morning_start_hour: The hour (0-23) that the building turns to comfort mode.
    evening_start_hour: The hour (0-23) that the building turns to eco mode.
    comfort_temp_window: 2-Tuple containing heating and cooling setpoints in K
      for comfort mode.
    eco_temp_window: 2-Tuple containing heating and cooling setpoints in K for
      eco mode.
    holidays: Set of days of year (1-365) to set as eco mode.
  """

  # TODO(judahg): make holidays a set of Timestamps

  def __init__(
      self,
      morning_start_hour: int,
      evening_start_hour: int,
      comfort_temp_window: TemperatureWindow,
      eco_temp_window: TemperatureWindow,
      holidays: Optional[Set[int]] = None,
      time_zone: datetime.tzinfo = pytz.UTC,
  ):
    if morning_start_hour > evening_start_hour:
      raise ValueError(
          'morning_start_hour must be less than evening_start_hour'
      )

    if comfort_temp_window[0] > comfort_temp_window[1]:
      raise ValueError(
          'comfort_temp_window[0] must be less than comfort_temp_window[1]'
      )

    if eco_temp_window[0] > eco_temp_window[1]:
      raise ValueError(
          'eco_temp_window[0] must be less than eco_temp_window[1]'
      )

    self.morning_start_hour = morning_start_hour
    self.evening_start_hour = evening_start_hour
    self.comfort_temp_window = comfort_temp_window
    self.eco_temp_window = eco_temp_window
    self._time_zone = time_zone
    if holidays:
      self.holidays = holidays
    else:
      self.holidays = set()

  @property
  def time_zone(self) -> datetime.tzinfo:
    """Returns the schedule's time zone."""
    return self._time_zone

  def is_comfort_mode(self, current_timestamp: pd.Timestamp) -> bool:
    """Returns whether setpoint schedule dictates comfort mode.

    Args:
      current_timestamp: Pandas Timestamp to get mode for.
    """
    local_timestamp = self._localize_or_convert_timezone(current_timestamp)
    return (
        local_timestamp.hour >= self.morning_start_hour
        and local_timestamp.hour < self.evening_start_hour
        and local_timestamp.dayofyear not in self.holidays
        and not self.is_weekend(local_timestamp)
    )

  def _localize_or_convert_timezone(
      self, current_timestamp: pd.Timestamp
  ) -> pd.Timestamp:
    if current_timestamp.tz is not None:
      return current_timestamp.tz_convert(self._time_zone)
    else:
      return current_timestamp.tz_localize(pytz.UTC)

  def is_weekend(self, current_timestamp: pd.Timestamp) -> bool:
    """Returns whether current day is a weekend.

    Args:
      current_timestamp: Pandas Timestamp.
    """
    local_timestamp = self._localize_or_convert_timezone(current_timestamp)
    return local_timestamp.day_name() in ['Saturday', 'Sunday']

  def get_temperature_window(
      self, current_timestamp: pd.Timestamp
  ) -> TemperatureWindow:
    """Returns tuple containing low and high setpoints.

    Args:
      current_timestamp: Pandas timestamp to get window for.
    """
    if self.is_comfort_mode(current_timestamp):
      return self.comfort_temp_window
    else:
      return self.eco_temp_window

  def get_plot_data(
      self, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp
  ) -> pd.DataFrame:
    """Returns DataFrame of all transition events in the time window.

    Can be used for plotting purposes.

    Columns: comfort_mode (True/False), start time, end time, heating,
      cooling setpoints.

    Gets the time windows for drawing day/night setpoint rectangles.

    Enables plotting the schedule on the temp timeline chart. Each entry is
    converted to a rectangle, where the x-coordinates are times, and the
    y-coordinates are temps.

    Args:
      start_timestamp: Pandas Timestamp representing start time of the time
        window to be plotted.
      end_timestamp: Pandas Timestamp representing end time of the time window
        to be plotted.
    """

    comfort_modes = []
    start_times = []
    end_times = []
    heating_setpoints = []
    cooling_setpoints = []

    current_timestamp = start_timestamp

    # Start with the start time and increment until the end time.
    # The increment is determined by the next schedule event.
    while current_timestamp < end_timestamp:
      current_comfort_mode = self.is_comfort_mode(current_timestamp)
      local_timestamp = self._localize_or_convert_timezone(current_timestamp)
      comfort_modes.append(current_comfort_mode)

      if current_comfort_mode:
        heating_setpoints.append(self.comfort_temp_window[0])
        cooling_setpoints.append(self.comfort_temp_window[1])

        te = pd.Timestamp(
            year=current_timestamp.year,
            month=current_timestamp.month,
            day=current_timestamp.day,
            hour=self.evening_start_hour,
        )

      else:  # eco mode
        heating_setpoints.append(self.eco_temp_window[0])
        cooling_setpoints.append(self.eco_temp_window[1])

        if local_timestamp.hour >= self.morning_start_hour:
          te = pd.Timestamp(
              year=current_timestamp.year,
              month=current_timestamp.month,
              day=current_timestamp.day,
              hour=self.morning_start_hour,
          ) + pd.Timedelta(days=1)
        else:
          te = pd.Timestamp(
              year=current_timestamp.year,
              month=current_timestamp.month,
              day=current_timestamp.day,
              hour=self.morning_start_hour,
          )

      if te.tz is not None:
        local_timestamp = te.tz_convert(self._time_zone)
      else:
        local_timestamp = te
      if local_timestamp.dayofyear in self.holidays or self.is_weekend(te):
        while te.dayofyear in self.holidays or self.is_weekend(te):
          te = te + pd.Timedelta(days=1)

      if (te - end_timestamp).total_seconds() > 0:
        te = end_timestamp

      end_times.append(te)
      start_times.append(current_timestamp)
      current_timestamp = te

    return pd.DataFrame({
        'comfort_mode': comfort_modes,
        'start_time': start_times,
        'end_time': end_times,
        'heating_setpoint': heating_setpoints,
        'cooling_setpoint': cooling_setpoints,
    })
