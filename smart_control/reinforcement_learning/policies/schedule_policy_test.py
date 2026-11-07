"""Tests for the schedule policy time handling."""

import unittest

import pandas as pd

from smart_control.reinforcement_learning.policies import schedule_policy

DAY = 'day'
NIGHT = 'night'
PDT = pd.Timedelta(hours=-7)
IST = pd.Timedelta(hours=5, minutes=30)


def _schedule():
  """A schedule that switches to day at 6:00 and to night at 19:00."""
  return [
      schedule_policy.ScheduleEvent(
          pd.Timedelta(6, unit='hour'),
          schedule_policy.DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          DAY,
      ),
      schedule_policy.ScheduleEvent(
          pd.Timedelta(19, unit='hour'),
          schedule_policy.DeviceType.AC,
          'supply_air_heating_temperature_setpoint',
          NIGHT,
      ),
  ]


def _setpoint_at_utc_hour(hod, utc_offset):
  return schedule_policy.get_active_setpoint(
      _schedule(),
      schedule_policy.DeviceType.AC,
      'supply_air_heating_temperature_setpoint',
      schedule_policy.local_schedule_time(hod, utc_offset),
  )


class LocalScheduleTimeTest(unittest.TestCase):

  def test_negative_offset_wraps_to_previous_evening(self):
    # 00:00 UTC is 17:00 the previous day at UTC-7.
    self.assertEqual(
        schedule_policy.local_schedule_time(0, PDT),
        pd.Timedelta(17, unit='hour'),
    )

  def test_offset_without_wrapping_is_unchanged(self):
    # 13:00 UTC is 06:00 at UTC-7.
    self.assertEqual(
        schedule_policy.local_schedule_time(13, PDT),
        pd.Timedelta(6, unit='hour'),
    )

  def test_local_midnight_is_zero(self):
    self.assertEqual(
        schedule_policy.local_schedule_time(7, PDT), pd.Timedelta(0)
    )

  def test_positive_offset_wraps_past_midnight(self):
    # 20:00 UTC is 01:30 the next day at UTC+5:30.
    self.assertEqual(
        schedule_policy.local_schedule_time(20, IST),
        pd.Timedelta(hours=1, minutes=30),
    )


class BaselineScheduleSwitchTest(unittest.TestCase):

  def test_day_setpoint_holds_until_local_evening_start(self):
    # 17:00 and 18:00 local are 00:00 and 01:00 UTC. They are still day.
    self.assertEqual(_setpoint_at_utc_hour(0, PDT), DAY)
    self.assertEqual(_setpoint_at_utc_hour(1, PDT), DAY)

  def test_night_setpoint_starts_at_local_evening_start(self):
    # 19:00 local is 02:00 UTC.
    self.assertEqual(_setpoint_at_utc_hour(2, PDT), NIGHT)

  def test_night_setpoint_holds_overnight(self):
    # 23:00 local is 06:00 UTC, and local midnight is 07:00 UTC.
    self.assertEqual(_setpoint_at_utc_hour(6, PDT), NIGHT)
    self.assertEqual(_setpoint_at_utc_hour(7, PDT), NIGHT)

  def test_day_setpoint_starts_at_local_morning_start(self):
    # 06:00 local is 13:00 UTC, and 16:00 local is 23:00 UTC.
    self.assertEqual(_setpoint_at_utc_hour(13, PDT), DAY)
    self.assertEqual(_setpoint_at_utc_hour(23, PDT), DAY)


if __name__ == '__main__':
  unittest.main()
