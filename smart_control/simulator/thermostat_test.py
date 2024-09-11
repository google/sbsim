"""Tests for thermostat.

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

from absl.testing import absltest
import pandas as pd
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import thermostat


def get_mock_schedule():
  morning_start_hour = 9
  evening_start_hour = 18
  comfort_temp_window = (292, 295)
  eco_temp_window = (290, 297)

  return setpoint_schedule.SetpointSchedule(
      morning_start_hour,
      evening_start_hour,
      comfort_temp_window,
      eco_temp_window,
  )


class ThermostatTest(absltest.TestCase):

  def test_init(self):
    schedule = get_mock_schedule()
    t = thermostat.Thermostat(schedule)
    self.assertEqual(t._setpoint_schedule, schedule)
    self.assertIsNone(t._previous_timestamp)
    self.assertEqual(t._current_mode, thermostat.Thermostat.Mode.OFF)

  def test_default_control(self):
    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)
    schedule = get_mock_schedule()
    t = thermostat.Thermostat(schedule)
    window = schedule.get_temperature_window(time)
    low, high = window
    mid = (low + high) / 2
    self.assertEqual(
        t._default_control(low - 1, window), thermostat.Thermostat.Mode.HEAT
    )
    self.assertEqual(
        t._default_control(mid - 1, window), thermostat.Thermostat.Mode.HEAT
    )

    self.assertEqual(
        t._default_control(high + 1, window), thermostat.Thermostat.Mode.COOL
    )
    self.assertEqual(
        t._default_control(mid + 1, window), thermostat.Thermostat.Mode.COOL
    )

    self.assertEqual(
        t._default_control(mid - 1, window), thermostat.Thermostat.Mode.OFF
    )
    self.assertEqual(
        t._default_control(mid + 1, window), thermostat.Thermostat.Mode.OFF
    )

  def test_update_comfort_mode(self):
    time = pd.Timestamp(year=2021, month=5, day=5, hour=11)
    schedule = get_mock_schedule()
    t = thermostat.Thermostat(schedule)
    window = schedule.get_temperature_window(time)
    low, high = window
    mid = (low + high) / 2

    self.assertEqual(t.update(low - 1, time), thermostat.Thermostat.Mode.HEAT)
    self.assertEqual(t.update(mid - 1, time), thermostat.Thermostat.Mode.HEAT)

    self.assertEqual(t.update(high + 1, time), thermostat.Thermostat.Mode.COOL)
    self.assertEqual(t.update(mid + 1, time), thermostat.Thermostat.Mode.COOL)

    self.assertEqual(t.update(mid - 1, time), thermostat.Thermostat.Mode.OFF)
    self.assertEqual(t.update(mid + 1, time), thermostat.Thermostat.Mode.OFF)

  def test_eco_transition(self):
    weekday = pd.Timestamp(year=2021, month=5, day=5, hour=11)
    schedule = get_mock_schedule()
    t = thermostat.Thermostat(schedule)
    weekend = pd.Timestamp(year=2021, month=5, day=8, hour=11)  # a Saturday
    t.update(0, weekday)
    self.assertEqual(
        t.update(0, weekend), thermostat.Thermostat.Mode.PASSIVE_COOL
    )

  def test_eco_mode(self):
    weekday = pd.Timestamp(year=2021, month=5, day=5, hour=11)
    schedule = get_mock_schedule()
    t = thermostat.Thermostat(schedule)
    window = schedule.get_temperature_window(weekday)
    low, high = window
    mid = (low + high) / 2
    weekend = pd.Timestamp(year=2021, month=5, day=8, hour=11)  # a Saturday
    t.update(0, weekday)
    t.update(0, weekday)

    self.assertEqual(
        t.update(mid, weekend), thermostat.Thermostat.Mode.PASSIVE_COOL
    )
    self.assertEqual(t.update(0, weekend), thermostat.Thermostat.Mode.HEAT)


if __name__ == '__main__':
  absltest.main()
