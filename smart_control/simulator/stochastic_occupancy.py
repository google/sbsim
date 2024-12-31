"""
A stochastic occupancy model for building simulation.

This model simulates the behavior of occupants in a building by defining 
arrival, departure, and lunch break times based on random sampling. Each zone 
is assigned a specified number of occupants, and their schedules are generated 
using cumulative probability functions (CPFs) to ensure realistic variability.

For each occupant, arrival and departure times are sampled within defined 
earliest and latest bounds. Lunch break times and durations are also generated 
stochastically. The model determines whether an occupant is present in the 
work zone or away at any given time, accounting for work hours, lunch breaks, 
and holidays.

The `LIGHTSWITCHOccupancy` class calculates the average occupancy for a zone 
over a specified time interval, enabling integration with larger building 
simulation frameworks.

Debugging features are included to provide insights into sampling and state 
transition processes when `debug_print` is enabled.

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

# Modify the code to include debug prints
import datetime
import enum
from typing import Optional, Union

import gin
import numpy as np
import pandas as pd
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.utils import conversion_utils

debug_print = False  # Set to False to disable debugging


class OccupancyStateEnum(enum.Enum):
    AWAY = 1
    WORK = 2

class ZoneOccupant:
    def __init__(
        self,
        earliest_expected_arrival_hour: int,
        latest_expected_arrival_hour: int,
        earliest_expected_departure_hour: int,
        latest_expected_departure_hour: int,
        lunch_start_hour: int,
        lunch_end_hour: int,
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
        assert lunch_start_hour < lunch_end_hour

        self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
        self._latest_expected_arrival_hour = latest_expected_arrival_hour
        self._earliest_expected_departure_hour = earliest_expected_departure_hour
        self._latest_expected_departure_hour = latest_expected_departure_hour
        self._lunch_start_hour = lunch_start_hour
        self._lunch_end_hour = lunch_end_hour
        self._step_size = step_size
        self._random_state = random_state
        self._time_zone = time_zone
        self._occupancy_state = OccupancyStateEnum.AWAY

        # Sample times using CPF-based sampling
        self._arrival_time = self._sample_event_time(
            self._earliest_expected_arrival_hour, self._latest_expected_arrival_hour
        )
                
        self._departure_time = self._sample_event_time(
            self._earliest_expected_departure_hour, self._latest_expected_departure_hour
        )
        self._lunch_start_time = self._sample_event_time(
            self._lunch_start_hour, self._lunch_end_hour
        )
        self._lunch_duration = self._sample_lunch_duration()

        if debug_print:
            print(f"ZoneOccupant initialized with: arrival_time={self._arrival_time}, "
                  f"departure_time={self._departure_time}, lunch_start_time={self._lunch_start_time}, "
                  f"lunch_duration={self._lunch_duration}")

    def _generate_cpf(self, start, end):
        values = np.arange(start, end + 1)
        probabilities = self._random_state.rand(len(values))
        cumulative_probabilities = np.cumsum(probabilities / probabilities.sum())
        return values, cumulative_probabilities

    def _sample_event_time(self, start, end):
        values, cumulative_probabilities = self._generate_cpf(start, end)
        random_value = self._random_state.rand()
        index = np.searchsorted(cumulative_probabilities, random_value)
        if debug_print:
            print(f"Sampled event time: start={start}, end={end}, value={values[index]}")
        return values[index]

    def _sample_lunch_duration(self):
        duration_minutes = np.arange(30, 91, 5)
        values, cumulative_probabilities = self._generate_cpf(30, 90)
        random_value = self._random_state.rand()
        index = np.searchsorted(cumulative_probabilities, random_value)
        if debug_print:
            print(f"Sampled lunch duration: {values[index]} minutes")
        return values[index]

    def _to_local_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        if timestamp.tz is None:
            return timestamp
        return timestamp.tz_convert(self._time_zone)

    def _occupant_arrived(self, timestamp: pd.Timestamp) -> bool:
        local_timestamp = self._to_local_time(timestamp)
        arrived = local_timestamp.hour >= self._arrival_time
        if debug_print:
            print(f"Check arrival: local_time_hour={local_timestamp.hour}, arrival_time={self._arrival_time}, arrived={arrived}")
        return arrived

    def _occupant_departed(self, timestamp: pd.Timestamp) -> bool:
        local_timestamp = self._to_local_time(timestamp)
        departed = local_timestamp.hour >= self._departure_time
        if debug_print:
            print(f"Check departure: local_time_hour={local_timestamp.hour}, departure_time={self._departure_time}, departed={departed}")
        return departed

    def peek(self, current_time: pd.Timestamp) -> OccupancyStateEnum:
        local_timestamp = self._to_local_time(current_time)
        #print(f"Inside peek: current_time={current_time}")
        local_time = local_timestamp.time()  # Extracts time as a datetime.time object
        if debug_print:
            print(f"Peek called: current_time={current_time}, local_time={local_timestamp}, state={self._occupancy_state}")

        day = pd.Timestamp(
            year=local_timestamp.year,
            month=local_timestamp.month,
            day=local_timestamp.day,
        )
    
        # Check if it's a workday
        if not conversion_utils.is_work_day(day):
            self._occupancy_state = OccupancyStateEnum.AWAY
            return self._occupancy_state
    
        # Check arrival and departure
        if self._occupant_arrived(current_time) and not self._occupant_departed(current_time):
            self._occupancy_state = OccupancyStateEnum.WORK
        else:
            self._occupancy_state = OccupancyStateEnum.AWAY
    
        # Handle lunch break
        if self._occupancy_state == OccupancyStateEnum.WORK:
            lunch_start_time = datetime.time(hour=self._lunch_start_time, minute=0)
            lunch_end_time = (datetime.datetime.combine(datetime.date.today(), lunch_start_time) +
                              pd.Timedelta(minutes=self._lunch_duration)).time()
            if lunch_start_time <= local_time < lunch_end_time:
                self._occupancy_state = OccupancyStateEnum.AWAY
                return OccupancyStateEnum.AWAY


        if debug_print:
            print(f"Occupancy state: {self._occupancy_state}")
    
        return self._occupancy_state


@gin.configurable
class LIGHTSWITCHOccupancy(BaseOccupancy):
    def __init__(
        self,
        zone_assignment: int,
        earliest_expected_arrival_hour: int,
        latest_expected_arrival_hour: int,
        earliest_expected_departure_hour: int,
        latest_expected_departure_hour: int,
        lunch_start_hour: int = 12,
        lunch_end_hour: int = 14,
        time_step_sec: int = 3600,
        seed: Optional[int] = 511211,
        time_zone: str = 'UTC',
    ):
        self._zone_assignment = zone_assignment
        self._zone_occupants = {}
        self._step_size = pd.Timedelta(seconds=time_step_sec)
        self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
        self._latest_expected_arrival_hour = latest_expected_arrival_hour
        self._earliest_expected_departure_hour = earliest_expected_departure_hour
        self._latest_expected_departure_hour = latest_expected_departure_hour
        self._lunch_start_hour = lunch_start_hour
        self._lunch_end_hour = lunch_end_hour
        self._random_state = np.random.RandomState(seed)
        self._time_zone = time_zone

    def _initialize_zone(self, zone_id: str):
        if zone_id not in self._zone_occupants:
            self._zone_occupants[zone_id] = []
            for _ in range(self._zone_assignment):
                self._zone_occupants[zone_id].append(
                    ZoneOccupant(
                        self._earliest_expected_arrival_hour,
                        self._latest_expected_arrival_hour,
                        self._earliest_expected_departure_hour,
                        self._latest_expected_departure_hour,
                        self._lunch_start_hour,
                        self._lunch_end_hour,
                        self._step_size,
                        self._random_state,
                        self._time_zone,
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
    
            #print(f"Current time: {current_time}, Occupancy count: {num_occupants}")
            total_occupancy += num_occupants
            steps += 1
            current_time += self._step_size
    
        # Avoid division by zero
        return total_occupancy / steps if steps > 0 else 0.0

