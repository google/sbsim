"""A enhanced stochastic occupancy model for building simulation
with minute-level control and different worker types.This model is based on the
LIGHTSWITCHOccupancy model from stochastic_occupancy.py to include minute-level
control instead of hour-level control. It has the same arrival/departure/lunch
logic but it provides more fine-grained control and realistic occupant
behaviour, such as optional weekend work and daily changing work hours and lunch
break times (instead of a constant set of parameters for each occupant leading
to a static occupancy profile that repreats itself every weekday of the year).
The model samples new work and lunch time parameters for each occupant every
day, caches them for the day and clears the cache on the next day to ensure
consistency. It is possible to model low occupancy levels on the weekends by
using different worker types.
"""

import datetime
import enum
from typing import Dict, Union

import gin
import numpy as np
import pandas as pd

from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.utils import conversion_utils

debug_print = False  # Set to False to disable debugging


class OccupancyStateEnum(enum.Enum):
  AWAY = 1
  WORK = 2


class WorkerType(enum.Enum):
  WEEKDAY_ONLY = 1
  WEEKEND_REGULAR = 2  # worker types that works on weekends and weekdays
  WEEKEND_OCCASIONAL = (
      3  # regular weekday workers who need to work on weekends occasionally
  )


class EnhancedZoneOccupant:
  """EnhancedZone Occupant with minute-level contril and day-specific
  parameters."""

  def __init__(
      self,
      earliest_expected_arrival_min: int,
      latest_expected_arrival_min: int,
      earliest_expected_departure_min: int,
      latest_expected_departure_min: int,
      lunch_start_min: int,
      lunch_end_min: int,
      step_size: pd.Timedelta,
      random_state: np.random.RandomState,
      time_zone: Union[datetime.tzinfo, str] = "UTC",
      worker_type: WorkerType = WorkerType.WEEKDAY_ONLY,
      weekend_work_prob: float = 0.10,
      occupant_id: int = 0,
  ):
    assert (
        earliest_expected_arrival_min
        < latest_expected_arrival_min
        < earliest_expected_departure_min
        < latest_expected_departure_min
    )
    assert lunch_start_min < lunch_end_min

    self._earliest_expected_arrival_min = earliest_expected_arrival_min
    self._latest_expected_arrival_min = latest_expected_arrival_min
    self._earliest_expected_departure_min = earliest_expected_departure_min
    self._latest_expected_departure_min = latest_expected_departure_min
    self._lunch_start_min = lunch_start_min
    self._lunch_end_min = lunch_end_min
    self._step_size = step_size
    self._random_state = random_state
    self._time_zone = time_zone
    self.state = OccupancyStateEnum.AWAY
    self.daily_cache = {}
    self.worker_type = worker_type
    self.weekend_work_prob = weekend_work_prob
    self.occupant_id = occupant_id
    self.id = occupant_id
    self.daily_work_cache = {}

  def _generate_cpf(self, start, end, random_state=None):
    if random_state is None:
      random_state = self._random_state
    values = np.arange(start, end + 1)
    probabilities = random_state.rand(len(values))
    cumulative_probabilities = np.cumsum(probabilities / probabilities.sum())
    return values, cumulative_probabilities

  def _sample_event_time(self, start, end, random_state=None):
    if random_state is None:
      random_state = self._random_state
    values, cumulative_probabilities = self._generate_cpf(
        start, end, random_state
    )
    random_value = random_state.rand()
    index = np.searchsorted(cumulative_probabilities, random_value)
    if debug_print:
      print(
          f"Sampled event time: start={start}, end={end}, value={values[index]}"
      )
    return values[index]

  def _sample_lunch_duration(self, random_state=None):
    if random_state is None:
      random_state = self._random_state
    values, cumulative_probabilities = self._generate_cpf(30, 90, random_state)
    random_value = random_state.rand()
    index = np.searchsorted(cumulative_probabilities, random_value)
    if debug_print:
      print(f"Sampled lunch duration: {values[index]} minutes")
    return values[index]

  def _to_local_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.tz is None:
      if self._time_zone is None:
        return timestamp
      return timestamp.tz_localize(self._time_zone)
    return timestamp.tz_convert(self._time_zone)

  def _get_daily_params(
      self, timestamp: pd.Timestamp, local_timestamp: pd.Timestamp = None
  ) -> Dict[str, int]:
    if local_timestamp is None:
      local_timestamp = self._to_local_time(timestamp)

    date_key = local_timestamp.date()

    if self.daily_cache and list(self.daily_cache.keys())[0] != date_key:
      self.daily_cache.clear()

    if date_key in self.daily_cache:
      return self.daily_cache[date_key]

    day_seed = hash(str(date_key) + str(self.occupant_id) + "daily_params") % (
        2**32
    )
    day_random_state = np.random.RandomState(day_seed)

    arrival = self._sample_event_time(
        self._earliest_expected_arrival_min,
        self._latest_expected_arrival_min,
        day_random_state,
    )
    departure = self._sample_event_time(
        self._earliest_expected_departure_min,
        self._latest_expected_departure_min,
        day_random_state,
    )
    lunch_start = self._sample_event_time(
        self._lunch_start_min, self._lunch_end_min, day_random_state
    )
    lunch_duration = self._sample_lunch_duration(day_random_state)

    self.daily_cache[date_key] = {
        "arrival_time": arrival,
        "departure_time": departure,
        "lunch_start_time": lunch_start,
        "lunch_duration": lunch_duration,
    }
    return self.daily_cache[date_key]

  def _minutes_since_midnight(self, local_timestamp: pd.Timestamp) -> int:
    return local_timestamp.hour * 60 + local_timestamp.minute

  def _should_work_today(
      self, timestamp: pd.Timestamp, local_timestamp: pd.Timestamp = None
  ) -> bool:
    if local_timestamp is None:
      local_timestamp = self._to_local_time(timestamp)

    day = pd.Timestamp(
        year=local_timestamp.year,
        month=local_timestamp.month,
        day=local_timestamp.day,
    )
    date_key = day.date()

    if date_key in self.daily_work_cache:
      return self.daily_work_cache[date_key]

    if conversion_utils.is_work_day(day):
      self.daily_work_cache[date_key] = True
      return True

    if self.worker_type == WorkerType.WEEKDAY_ONLY:
      self.daily_work_cache[date_key] = False
      return False

    if self.worker_type == WorkerType.WEEKEND_REGULAR:
      self.daily_work_cache[date_key] = True
      return True

    elif self.worker_type == WorkerType.WEEKEND_OCCASIONAL:
      seed = hash(str(date_key) + str(self.id)) % 2**32
      random_state = np.random.RandomState(seed)
      work_today = random_state.rand() < self.weekend_work_prob
      self.daily_work_cache[date_key] = work_today
      return work_today

    self.daily_work_cache[date_key] = False
    return False

  def _occupant_arrived(
      self, timestamp: pd.Timestamp, local_timestamp: pd.Timestamp = None
  ) -> bool:
    if local_timestamp is None:
      local_timestamp = self._to_local_time(timestamp)

    current_min = self._minutes_since_midnight(local_timestamp)
    params = self._get_daily_params(timestamp, local_timestamp)

    arrived = current_min >= params["arrival_time"]
    if debug_print:
      print(
          f"Check arrival: local_time_hour={local_timestamp.hour},"
          f" arrival_time={params['arrival_time']}, arrived={arrived}"
      )
    return arrived

  def _occupant_departed(
      self, timestamp: pd.Timestamp, local_timestamp: pd.Timestamp = None
  ) -> bool:
    if local_timestamp is None:
      local_timestamp = self._to_local_time(timestamp)

    current_min = self._minutes_since_midnight(local_timestamp)
    params = self._get_daily_params(timestamp, local_timestamp)

    departed = current_min >= params["departure_time"]
    if debug_print:
      print(
          f"Check departure: local_time_hour={local_timestamp.hour},"
          f" departure_time={params['departure_time']}, departed={departed}"
      )
    return departed

  def peek(self, current_time: pd.Timestamp) -> OccupancyStateEnum:
    """Checks the current occupancy state based on the provided timestamp.

    This method determines the occupancy state (AWAY or WORK) based on
    the current time, considering workdays, weekends, arrival/departure times,
    and a lunch break.

    Args:
        current_time: The current timestamp to evaluate.

    Returns:
        The current `OccupancyStateEnum` (AWAY or WORK).
    """
    local_timestamp = self._to_local_time(current_time)

    if debug_print:
      print(
          f"Peek called: current_time={current_time},"
          f" local_time={local_timestamp}, state={self.state}"
      )

    if not self._should_work_today(current_time, local_timestamp):
      self.state = OccupancyStateEnum.AWAY
      return self.state

    # Check arrival and departure
    if self._occupant_arrived(
        current_time, local_timestamp
    ) and not self._occupant_departed(current_time, local_timestamp):
      self.state = OccupancyStateEnum.WORK
    else:
      self.state = OccupancyStateEnum.AWAY

    # Handle lunch break
    if self.state == OccupancyStateEnum.WORK:
      current_min = self._minutes_since_midnight(local_timestamp)
      params = self._get_daily_params(current_time, local_timestamp)
      lunch_start = params["lunch_start_time"]
      lunch_end = lunch_start + params["lunch_duration"]
      if lunch_start <= current_min < lunch_end:
        self.state = OccupancyStateEnum.AWAY
        return OccupancyStateEnum.AWAY

    if debug_print:
      print(f"Occupancy state: {self.state}")

    return self.state


@gin.configurable
class EnhancedOccupancy(BaseOccupancy):
  """Enhanced occupancy model with minute-level control and different
  worker types."""

  def __init__(
      self,
      zone_assignment: int,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      lunch_start_hour: int = 12,
      lunch_end_hour: int = 14,
      time_step: pd.Timedelta = pd.Timedelta(minutes=5),
      time_zone: str = "UTC",
      # 5% of the workforce are regular weekend workers
      weekend_regular_pct: float = 0.05,
      # 5% of the workforce are occasional weekend worker
      weekend_occasional_pct: float = 0.05,
      # 10% chance per weekend day that an occasional worker will work
      occasional_daily_prob: float = 0.10,
  ):
    self._zone_assignment = zone_assignment
    self._zone_occupants = {}
    self._step_size = time_step
    self._earliest_expected_arrival = earliest_expected_arrival_hour * 60
    self._latest_expected_arrival = latest_expected_arrival_hour * 60
    self._earliest_expected_departure = earliest_expected_departure_hour * 60
    self._latest_expected_departure = latest_expected_departure_hour * 60
    self._lunch_start = lunch_start_hour * 60
    self._lunch_end = lunch_end_hour * 60
    self._time_zone = time_zone
    self._weekend_regular_pct = weekend_regular_pct
    self._weekend_occasional_pct = weekend_occasional_pct
    self._occasional_prob = occasional_daily_prob

    total_pct = weekend_regular_pct + weekend_occasional_pct
    if total_pct > 1.0:
      raise ValueError(
          "Total percentage of weekend workers must be less than or equal"
          " to 1.0"
      )

  def _initialize_zone(self, zone_id: str):
    if zone_id not in self._zone_occupants:
      self._zone_occupants[zone_id] = []
      for i in range(self._zone_assignment):
        worker_random_state = np.random.RandomState(
            hash(f"{zone_id}_{i}") % 2**32
        )
        u = worker_random_state.rand()
        if u < self._weekend_regular_pct:
          worker_type = WorkerType.WEEKEND_REGULAR
          weekend_prob = 1.0
        elif u < self._weekend_regular_pct + self._weekend_occasional_pct:
          worker_type = WorkerType.WEEKEND_OCCASIONAL
          weekend_prob = self._occasional_prob
        else:
          worker_type = WorkerType.WEEKDAY_ONLY
          weekend_prob = 0.0

        occupant_random_state = np.random.RandomState(
            (hash(f"{zone_id}_{i}_behavior") % 2**32)
        )

        self._zone_occupants[zone_id].append(
            EnhancedZoneOccupant(
                self._earliest_expected_arrival,
                self._latest_expected_arrival,
                self._earliest_expected_departure,
                self._latest_expected_departure,
                self._lunch_start,
                self._lunch_end,
                self._step_size,
                occupant_random_state,
                self._time_zone,
                worker_type=worker_type,
                weekend_work_prob=weekend_prob,
                occupant_id=i,
            )
        )

  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Calculates the average occupancy within a time interval for a zone.

    Args:
        zone_id: specific zone identifier for the building.
        start_time: **local time** with TZ for the beginning of the interval.
        end_time: **local time** with TZ for the end of the interval.

    Returns:
        Average number of people in the zone for the interval.
    """
    self._initialize_zone(zone_id)

    current_time = start_time
    total_occupancy = 0
    steps = 0

    while current_time < end_time:
      num_occupants = 0
      for occupant in self._zone_occupants[zone_id]:
        state = occupant.peek(current_time)
        if state == OccupancyStateEnum.WORK:
          num_occupants += 1

      total_occupancy += num_occupants
      steps += 1
      current_time += self._step_size

    return total_occupancy / steps if steps > 0 else 0.0

  def get_worker_distribution(self, zone_id: str) -> Dict[str, int]:
    self._initialize_zone(zone_id)
    counts = {"weekday_only": 0, "weekend_regular": 0, "weekend_occasional": 0}
    for occupant in self._zone_occupants[zone_id]:
      if occupant.worker_type == WorkerType.WEEKDAY_ONLY:
        counts["weekday_only"] += 1
      elif occupant.worker_type == WorkerType.WEEKEND_REGULAR:
        counts["weekend_regular"] += 1
      elif occupant.worker_type == WorkerType.WEEKEND_OCCASIONAL:
        counts["weekend_occasional"] += 1
    return counts
