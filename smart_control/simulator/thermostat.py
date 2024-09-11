"""Models a thermostat in the simulation.

The theromstat is given a SetpointSchedule, which defines for any given time
the deadband. The SetpointSchedule also determines when the thermostat should
operate in Comfort mode or Eco mode.

In Comfort mode, the thermostat can be in one of 3 states.  If the temperature
goes beneath the heating setpoint, Heat mode is activated until the temperature
reaches midway between the 2 setpoints. Similarly, if the temperature is higher
than the cooling setpoint, the thermostat enters Cool mode until the mid-point.
Otherways, it enters Off mode.

In Eco mode, there is an additional state, Passive Cool mode. Upon entering
Eco mode, the thermostate is initially placed in this state, and remains that
way until the temperature cools beyond the eco heating setpoint, upon which the
thermostat operates as it did in Comfort mode

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

import enum

import pandas as pd
from smart_control.simulator import setpoint_schedule


class Thermostat:
  """Local thermostat control for each VAV/zone.

  Is constructed by passing in a SetpointSchedule, which, for any timestamp,
  provides heating and cooling setpoints, as well as whether the thermostat
  should operate in Eco mode/.

  Attributes:
    _setpoint_schedule: SetpointSchedule to determine temperature windows.
    _previous_timestamp: Last timestamp the thermostat was called with.
    _current_mode: Current mode thermostat is in.
  """

  class Mode(enum.Enum):
    """Modes of the thermostat.

    Values:
      OFF: Temperature is within windows and does not need active adjustments.
      HEAT: VAV is actively heating zone.
      COOL: VAV is actively cooling zone.
      PASSIVE_COOL: Building is allowed to cool naturally until within eco mode
      window.
    """

    OFF = 0
    HEAT = 1
    COOL = 2
    PASSIVE_COOL = 3

  def __init__(self, schedule: setpoint_schedule.SetpointSchedule):
    self._setpoint_schedule = schedule
    self._previous_timestamp = None
    self._current_mode = self.Mode.OFF

  def get_setpoint_schedule(self) -> setpoint_schedule.SetpointSchedule:
    return self._setpoint_schedule

  def _default_control(
      self,
      zone_temp: float,
      temperature_window: setpoint_schedule.TemperatureWindow,
  ) -> 'Thermostat.Mode':
    """Returns mode based on current mode and current zone temperature.

    Does not consider Passive Cool mode.

    Default control works as follows: if the temperature sinks below the heating
    setpoint, Cool mode is entered until the midpoint temperature is reached.
    Similarly, if the temperature rises above the cooling setpoint, Heat mode is
    entered until the midpoint is reached. In all other cases, the thermostat is
    in Off mode.

    Args:
      zone_temp: Temperature in k of zone.
      temperature_window: 2-Tuple containing temperature bounds.
    """
    heating_setpoint, cooling_setpoint = temperature_window
    mid = 0.5 * (cooling_setpoint - heating_setpoint) + heating_setpoint
    # Case 1: temperature is below the heating set point, then always heat.
    if zone_temp < heating_setpoint:
      self._current_mode = self.Mode.HEAT
    # Case 2: temperature is above the cooling set point, then always cool.
    elif zone_temp > cooling_setpoint:
      self._current_mode = self.Mode.COOL
    # Case 3: in dead band, below midpoint, and heating, then continue heating.
    elif zone_temp < mid and self._current_mode == self.Mode.HEAT:
      self._current_mode = self.Mode.HEAT
    # Case 4: in dead band, above midpoint, and cooling, then continue cooling.
    elif zone_temp > mid and self._current_mode == self.Mode.COOL:
      self._current_mode = self.Mode.COOL
    # Case 5: in dead band, and no heating/cooling is needed.
    else:
      self._current_mode = self.Mode.OFF
    return self._current_mode

  def update(
      self, zone_temp: float, current_timestamp: pd.Timestamp
  ) -> 'Thermostat.Mode':
    """Returns updated mode, allowing passive cool if shifting into eco.

    Should be invoked once per iteration of the simulation, after all
    control volume temperatures have been updated.

    Args:
        zone_temp: Temperature in k of zone.
        current_timestamp: Pandas timestamp.
    """
    temperature_window = self._setpoint_schedule.get_temperature_window(
        current_timestamp
    )
    # In comfort mode, default control.
    if self._setpoint_schedule.is_comfort_mode(current_timestamp):
      self._default_control(zone_temp, temperature_window)
    # Just entered eco mode, allow passive cool.
    elif (
        self._previous_timestamp is not None
        and self._setpoint_schedule.is_comfort_mode(self._previous_timestamp)
    ):
      self._current_mode = self.Mode.PASSIVE_COOL
    # Been in eco mod
    else:
      if (
          self._current_mode == self.Mode.PASSIVE_COOL
          and zone_temp > temperature_window[0]
      ):
        self._current_mode = self.Mode.PASSIVE_COOL
      else:
        self._default_control(zone_temp, temperature_window)
    self._previous_timestamp = current_timestamp
    return self._current_mode
