"""A randomized occupancy model for the building simulation.

In this model, N occupants arrive between the earliest and latest arrival
hour and depart between the earliest and latest departure hour. The probability
of departure is specified so that the expected arrival and departure times
occur halfway in the interval. For a standard Bernoulli RV, E[X] = n*p, so
p = E[X] / n / 2, where E[X] is the expected number of arrivals, which equals 1.


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
import enum
from typing import Optional, Union

import gin
import numpy as np
import pandas as pd
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.utils import conversion_utils


class OccupancyStateEnum(enum.Enum):
  AWAY = 1
  WORK = 2


class ZoneOccupant:
  """Represents a single occupant in a zone.

  Attributes:
    earliest_expected_arrival_hour: earliest arrivel, 0 - 22
    latest_expected_arrival_hour: latest arrivel, 1 - 23
    earliest_expected_departure_hour: earliest departure, 0 - 22
    latest_expected_departure_hour: latest departure, 1 - 23
    random_state: random state used to generate events
  """

  def __init__(
      self,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      step_size: pd.Timedelta,
      random_state: np.random.RandomState,
      time_zone: Union[datetime.tzinfo, str] = 'UTC',
  ):
    assert (
        earliest_expected_arrival_hour
        < latest_expected_arrival_hour
        < earliest_expected_departure_hour
        < latest_expected_departure_hour
    )

    self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour = latest_expected_arrival_hour
    self._earliest_expected_departure_hour = earliest_expected_departure_hour
    self._latest_expected_departure_hour = latest_expected_departure_hour
    self._step_size = step_size
    self._occupancy_state = OccupancyStateEnum.AWAY
    self._p_arrival = self._get_event_probability(
        earliest_expected_arrival_hour, latest_expected_arrival_hour
    )
    self._p_departure = self._get_event_probability(
        earliest_expected_departure_hour, latest_expected_departure_hour
    )
    self._random_state = random_state
    self._time_zone = time_zone

  def _to_local_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Converts timestamp to local time."""
    if timestamp.tz is None:
      return timestamp
    else:
      return timestamp.tz_convert(self._time_zone)

  def _get_event_probability(self, start_hour, end_hour):
    """Returns the probability of an event based on the number of time steps."""
    assert start_hour < end_hour
    # The window is the number of Bernoulli trials (i.e. tests for arrival).
    window = pd.Timedelta(end_hour - start_hour, unit='hour')
    # The halfway point is the firts half of the trials.
    n_halfway = window / self._step_size / 2.0
    # We'd like to return the probability of event happening in a single time-
    # step. This follow a geometric distribution, where E[X] = 1/p, where
    # E[x] is the expected number of events before the first success. If
    # E[X] is the halfway point, then p = 1 / n_halfway.
    return 1.0 / n_halfway

  def _occupant_arrived(self, timestamp):
    """Makes a random draw to determine whether occupant arrives."""

    local_timestamp = self._to_local_time(timestamp)
    # TODO(sipple): Consider effects when time crosses DST>
    if (
        local_timestamp.hour < self._earliest_expected_arrival_hour
        or local_timestamp.hour > self._latest_expected_arrival_hour
    ):
      return False
    else:
      return self._random_state.rand() < self._p_arrival

  def _occupant_departed(self, timestamp):
    """Makes a random draw to determine whether the occupant departs."""
    local_timestamp = self._to_local_time(timestamp)
    if local_timestamp.hour < self._earliest_expected_departure_hour:
      return False
    else:
      return self._random_state.rand() < self._p_departure

  def peek(self, current_time: pd.Timestamp) -> OccupancyStateEnum:
    """Returns the state (WORK or AWAY) of the occupant for the current_time."""

    local_timestamp = self._to_local_time(current_time)
    day = pd.Timestamp(
        year=local_timestamp.year,
        month=local_timestamp.month,
        day=local_timestamp.day,
    )

    if not conversion_utils.is_work_day(day):
      self._occupancy_state = OccupancyStateEnum.AWAY

    elif self._occupancy_state == OccupancyStateEnum.AWAY:
      if self._occupant_arrived(current_time):
        self._occupancy_state = OccupancyStateEnum.WORK

    else:
      if self._occupant_departed(current_time):
        self._occupancy_state = OccupancyStateEnum.AWAY

    return self._occupancy_state


@gin.configurable
class RandomizedArrivalDepartureOccupancy(BaseOccupancy):
  """Provides the RL agent information about how many people are in a zone.

  Attributes:
    zone_assignment: number of occupants in a zone
    earliest_expected_arrival_hour: earliest arrivel, 0 - 22
    latest_expected_arrival_hour: latest arrivel, 1 - 23
    earliest_expected_departure_hour: earliest departure, 0 - 22
    latest_expected_departure_hour: latest departure, 1 - 23
    seed: integer used to set the random state for repeatability
  """

  def __init__(
      self,
      zone_assignment: int,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      time_step_sec: int,
      seed: Optional[int] = 17321,
      time_zone: str = 'UTC',
  ):
    self._zone_assignment = zone_assignment
    self._zone_occupants = {}
    self._step_size = pd.Timedelta(time_step_sec, unit='second')
    self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour = latest_expected_arrival_hour
    self._earliest_expected_departure_hour = earliest_expected_departure_hour
    self._latest_expected_departure_hour = latest_expected_departure_hour
    self._random_state = np.random.RandomState(seed)
    self._time_zone = time_zone

  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Returns the occupancy within start_time, end_time for the zone.

    If the zone is not found, implementations should raise a ValueError.

    Args:
      zone_id: specific zone identifier for the building.
      start_time: **local time** w/ TZ for the beginning of the interval.
      end_time: **local time** w/ TZ for the end of the interval.

    Returns:
      average number of people in the zone for the interval.
    """

    if zone_id not in self._zone_occupants:
      self._zone_occupants[zone_id] = []
      for _ in range(self._zone_assignment):
        self._zone_occupants[zone_id].append(
            ZoneOccupant(
                self._earliest_expected_arrival_hour,
                self._latest_expected_arrival_hour,
                self._earliest_expected_departure_hour,
                self._latest_expected_departure_hour,
                self._step_size,
                self._random_state,
                self._time_zone,
            )
        )

    num_occupants = 0.0
    for occupant in self._zone_occupants[zone_id]:
      if occupant.peek(start_time) == OccupancyStateEnum.WORK:
        num_occupants += 1.0
    return num_occupants
