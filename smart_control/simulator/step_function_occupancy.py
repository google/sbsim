"""A very basic occupancy model for smart buildings smart control.

Occupancy is the average number of people in a zone during a specific interval.

Occupancy is an input to the RL agent reward function.

This implementation assumes:
  (a) all zones have the same occupancy patterns.
  (b) the occupancy is constant for work periods, and
  (c) the occupancy is constant for non-work periods (off hours, weekends and
  holidays).

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

from typing import Tuple

import gin
import pandas as pd
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.utils import conversion_utils


@gin.configurable
class StepFunctionOccupancy(BaseOccupancy):
  """An occupancy model with constant level for work and non-work times.

  This model ignores the specific zone and returns the same value for all zones.

  Attributes:
    work_start_time: time-of-day when work period starts
    work_end_time: time-of-day when work period ends
    work_occupancy: avg number of people in zone during work times
    nonwork_occupancy: avg number of people in zone during non-work times
  """

  def __init__(
      self,
      work_start_time: pd.Timedelta,
      work_end_time: pd.Timedelta,
      work_occupancy: float,
      nonwork_occupancy: float,
  ):
    self._check_times(work_start_time)
    self._check_times(work_end_time)
    self._work_start_time = work_start_time
    self._work_end_time = work_end_time
    self._work_occupancy = work_occupancy
    self._nonwork_occupancy = nonwork_occupancy

  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Returns the occupancy within start_time, end_time for the zone.

    This model applies a constant for work time and non-work times, and returns
    a weighted average. Evaluates weekends and holidays as non-working times.

    Ignores the zone - assumes each zone is equally occupied.

    Args:
      zone_id: specific zone identifier for the building.
      start_time: **local time** w/o TZ for the beginning of the interval.
      end_time: **local time** w/o TZ for the end of the interval.

    Returns:
      average number of people in the zone for the interval.
    """

    if start_time >= end_time:
      raise ValueError('End time may not occur before start time.')

    work_seconds = 0.0
    nonwork_seconds = 0.0

    # Get the timestamp for midnight of the first day.
    day = pd.Timestamp(
        year=start_time.year, month=start_time.month, day=start_time.day
    )
    current_time = start_time - day

    # Accumulate working and non-working hours for all days.
    while day <= end_time:
      day_end = min(pd.Timedelta(1, unit='day'), end_time - day)

      if conversion_utils.is_work_day(day):
        before_work, during_work, after_work = self._split_workday(
            current_time, day_end
        )
        work_seconds += during_work
        nonwork_seconds += before_work + after_work

      else:  # Holidays and weekends
        nonwork_seconds += (day_end - current_time).total_seconds()

      day += pd.Timedelta(1.0, unit='day')
      current_time = pd.Timedelta(0.0, unit='sec')

    # Compute the weighted average.
    return (
        work_seconds * self._work_occupancy
        + nonwork_seconds * self._nonwork_occupancy
    ) / (work_seconds + nonwork_seconds)

  def _split_workday(
      self, start_time: pd.Timedelta, end_time: pd.Timedelta
  ) -> Tuple[float, float, float]:
    """Splits the interval into sec before, during, and after working hours.

    Args:
      start_time: start of the interval.
      end_time: end of the interval, may be greater than .

    Returns:
      Tuple (seconds before work, seconds during work, seconds after work)
    """

    if start_time > end_time:
      raise ValueError('Cannot have an end time before start time.')
    self._check_times(start_time)

    before_work = 0.0
    during_work = 0.0
    after_work = 0.0

    # Step through before work, during work, and afterwork phases, and
    # add up the times in seconds. If the current time is before any
    # phase, do not add any time, and truncate.
    # If the end_time precedes any phase, truncate the phase, and do not
    # add any time after the current phase.
    current = start_time
    # Don't go beyond the day or the specified end time, whichever is sooner.
    interval_end = min(end_time, pd.Timedelta(24, unit='hour'))
    # The next step to check is either the work start time or interval end.
    # Before work start:
    next_step = min(interval_end, self._work_start_time)

    # Get the time between the curren time and the next step.
    if current < next_step:
      before_work = (next_step - current).total_seconds()
      current = max(current, next_step)

    # After work start and work end:
    next_step = min(interval_end, self._work_end_time)
    if current < next_step:
      during_work = (next_step - current).total_seconds()
      current = next_step

    # Finally, between work end and modnight.
    next_step = interval_end
    if current < next_step:
      after_work = (next_step - current).total_seconds()

    return (before_work, during_work, after_work)

  def _check_times(self, time_delta: pd.Timedelta) -> None:
    if (
        time_delta > pd.Timedelta(24, unit='hour')
        or time_delta.total_seconds() < 0.0
    ):
      raise ValueError('Time delta must be positive and less than one day.')
