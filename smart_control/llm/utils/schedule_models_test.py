import datetime
import zoneinfo

from absl.testing import absltest
from absl.testing import parameterized
from smart_buildings.smart_control.llm.utils import schedule_models

UTC = "UTC"
EST = "America/New_York"
PST = "America/Los_Angeles"

UTC_INFO = zoneinfo.ZoneInfo(UTC)
EST_INFO = zoneinfo.ZoneInfo(EST)
PST_INFO = zoneinfo.ZoneInfo(PST)

TIME = datetime.time(8, 0)  # timezone naive
TIME_UTC = datetime.time(8, 0, tzinfo=UTC_INFO)
TIME_PST = datetime.time(8, 0, tzinfo=PST_INFO)
TIME_EST = datetime.time(8, 0, tzinfo=EST_INFO)

OFF_TIME = datetime.time(18, 0)  # timezone naive
OFF_TIME_UTC = datetime.time(18, 0, tzinfo=UTC_INFO)
OFF_TIME_PST = datetime.time(18, 0, tzinfo=PST_INFO)
OFF_TIME_EST = datetime.time(18, 0, tzinfo=EST_INFO)


class TimeConversionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="naive_time_str_to_utc",
          time_str="08:00",
          time_zone=UTC,
          expected=TIME_UTC,
      ),
      dict(
          testcase_name="naive_time_str_to_eastern",
          time_str="08:00",
          time_zone=EST,
          expected=TIME_EST,
      ),
      dict(
          testcase_name="naive_time_str_to_pacific",
          time_str="08:00",
          time_zone=PST,
          expected=TIME_PST,
      ),
      dict(
          testcase_name="tz_eastern",
          time_str="08:00",
          time_zone=EST,
          expected=TIME_EST,
      ),
  )
  def test_str_to_time_with_zone(self, time_str, time_zone, expected):
    self.assertEqual(
        schedule_models.str_to_time_with_zone(time_str, time_zone), expected
    )


#
# DAILY SCHEDULE TESTS
#


class OperationalDailyScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule = schedule_models.DailySchedule.from_times(
        day_name="Monday", on_time="08:00", off_time="18:00"
    )

  def test_init(self):
    self.assertIsInstance(self.schedule, schedule_models.DailySchedule)

  def test_attributes(self):
    self.assertEqual(self.schedule.day_name, "Monday")
    self.assertEqual(self.schedule.on_time, TIME_UTC)
    self.assertEqual(self.schedule.off_time, OFF_TIME_UTC)
    self.assertEqual(self.schedule.time_zone, UTC)

  def test_is_operational_day(self):
    self.assertTrue(self.schedule.is_operational_day)

  @parameterized.named_parameters(
      dict(testcase_name="during_hours", hour=12, minute=0, expected=True),
      dict(testcase_name="before_hours", hour=7, minute=0, expected=False),
      dict(testcase_name="after_hours", hour=19, minute=0, expected=False),
      dict(testcase_name="start_of_hours", hour=8, minute=0, expected=True),
      dict(testcase_name="end_of_hours", hour=18, minute=0, expected=False),
  )
  def test_is_during_operational_hours(self, hour, minute, expected):
    self.assertEqual(
        self.schedule.is_during_operational_hours(
            datetime.time(hour, minute, tzinfo=UTC_INFO)
        ),
        expected,
    )

  def test_is_during_operational_hours_with_wrong_time_zone_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The comparison time must have the same time zone as the schedule.",
    ):
      self.schedule.is_during_operational_hours(
          datetime.time(12, 0, tzinfo=PST_INFO)
      )

  def test_is_during_operational_hours_with_naive_time_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The comparison time must have a time zone.",
    ):
      self.schedule.is_during_operational_hours(datetime.time(12, 0))


class NonOperationalDailyScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule = schedule_models.DailySchedule.from_times(
        day_name="Monday", on_time=None, off_time=None
    )

  def test_init(self):
    self.assertIsInstance(self.schedule, schedule_models.DailySchedule)

  def test_attributes(self):
    self.assertEqual(self.schedule.day_name, "Monday")
    self.assertIsNone(self.schedule.on_time)
    self.assertIsNone(self.schedule.off_time)

  def test_is_operational_day(self):
    self.assertFalse(self.schedule.is_operational_day)

  @parameterized.parameters(
      datetime.time(12, 0, tzinfo=UTC_INFO),
      datetime.time(7, 0, tzinfo=UTC_INFO),
      datetime.time(19, 0, tzinfo=UTC_INFO),
  )
  def test_is_during_operational_hours(self, time):
    self.assertFalse(self.schedule.is_during_operational_hours(time))


class DailyScheduleValidationsTest(absltest.TestCase):

  def test_invalid_day_name_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown day name: Funday"):
      schedule_models.DailySchedule.from_times(
          day_name="Funday", on_time="08:00", off_time="18:00"
      )

  def test_missing_on_time_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The on_time and off_time must both be specified, or both be None.",
    ):
      schedule_models.DailySchedule.from_times(
          day_name="Monday", on_time=None, off_time="18:00"
      )

  def test_missing_off_time_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The on_time and off_time must both be specified, or both be None.",
    ):
      schedule_models.DailySchedule.from_times(
          day_name="Monday", on_time="08:00", off_time=None
      )

  def test_on_after_off_raises(self):
    with self.assertRaisesRegex(
        ValueError, "The on_time must be before the off_time."
    ):
      schedule_models.DailySchedule.from_times(
          day_name="Monday", on_time="18:00", off_time="08:00"
      )

  def test_same_on_and_off_raises(self):
    with self.assertRaisesRegex(
        ValueError, "The on_time must be before the off_time."
    ):
      schedule_models.DailySchedule.from_times(
          day_name="Monday", on_time="08:00", off_time="08:00"
      )

  def test_invalid_time_zone_raises(self):
    with self.assertRaisesRegex(ValueError, "Invalid time zone: OOPS"):
      schedule_models.DailySchedule.from_times(
          day_name="Monday",
          on_time=None,
          off_time=None,
          time_zone="OOPS",
      )

  def test_naive_on_time_raises(self):
    with self.assertRaisesRegex(
        ValueError, "The on_time needs to have a time zone."
    ):
      schedule_models.DailySchedule(
          day_name="Monday",
          on_time=TIME,
          off_time=OFF_TIME_UTC,
          time_zone=UTC,
      )

  def test_naive_off_time_raises(self):
    with self.assertRaisesRegex(
        ValueError, "The off_time needs to have a time zone."
    ):
      schedule_models.DailySchedule(
          day_name="Monday",
          on_time=TIME_UTC,
          off_time=OFF_TIME,
          time_zone=UTC,
      )

  def test_mismatched_on_time_tz_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The on_time and off_time must have the same time zone",
    ):
      schedule_models.DailySchedule(
          day_name="Monday",
          on_time=TIME_PST,
          off_time=OFF_TIME_UTC,
          time_zone=UTC,
      )

  def test_mismatched_off_time_tz_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "The on_time and off_time must have the same time zone",
    ):
      schedule_models.DailySchedule(
          day_name="Monday",
          on_time=TIME_UTC,
          off_time=OFF_TIME_PST,
          time_zone=UTC,
      )


#
# WEEKLY SCHEDULE TESTS
#


class WeeklyScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.schedule_dict = {
        "Monday": ("06:00", "19:00"),
        "Tuesday": ("06:00", "19:00"),
        "Wednesday": ("06:00", "19:00"),
        "Thursday": ("06:00", "19:00"),
        "Friday": ("09:00", "17:00"),
        "Saturday": (None, None),
        "Sunday": (None, None),
    }
    self.weekly_schedule = schedule_models.WeeklySchedule.from_dict(
        schedule_dict=self.schedule_dict, time_zone=UTC
    )

  def test_init(self):
    self.assertIsInstance(self.weekly_schedule, schedule_models.WeeklySchedule)

  def test_day_names(self):
    self.assertLen(self.weekly_schedule.daily_schedules, 7)

    day_names = [
        schedule.day_name for schedule in self.weekly_schedule.daily_schedules
    ]
    self.assertCountEqual(day_names, list(schedule_models.DAY_NAMES))

  def test_time_zone(self):
    self.assertEqual(self.weekly_schedule.time_zone, UTC)

  def test_get_daily_schedule(self):
    monday_schedule = self.weekly_schedule.get_daily_schedule("Monday")
    self.assertEqual(monday_schedule.day_name, "Monday")
    self.assertEqual(
        monday_schedule.on_time, datetime.time(6, 0, tzinfo=UTC_INFO)
    )
    self.assertEqual(
        monday_schedule.off_time, datetime.time(19, 0, tzinfo=UTC_INFO)
    )

  def test_get_daily_schedule_with_invalid_day_name_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown day name: Funday"):
      self.weekly_schedule.get_daily_schedule("Funday")

  def test_json_metadata(self):
    self.assertEqual(
        self.weekly_schedule.json_metadata,
        {
            "time_zone": UTC,
            "daily_schedules": {
                "Monday": {"on_time": "06:00", "off_time": "19:00"},
                "Tuesday": {"on_time": "06:00", "off_time": "19:00"},
                "Wednesday": {"on_time": "06:00", "off_time": "19:00"},
                "Thursday": {"on_time": "06:00", "off_time": "19:00"},
                "Friday": {"on_time": "09:00", "off_time": "17:00"},
                "Saturday": {"on_time": None, "off_time": None},
                "Sunday": {"on_time": None, "off_time": None},
            },
        },
    )


class WeeklyScheduleValidationsTest(absltest.TestCase):

  def test_missing_day_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "Weekly schedule must have a schedule for each day of the week.",
    ):
      schedule_models.WeeklySchedule.from_dict(
          {"Monday": ("08:00", "18:00")}, time_zone=PST
      )

  def test_extra_day_raises(self):
    # FYI because dictionaries don't allow duplicate keys, we can't use the
    # WeeklySchedule.from_dict constructor to test this validation.
    with self.assertRaisesRegex(
        ValueError,
        "Weekly schedule must have a schedule for each day of the week.",
    ):
      from_times = schedule_models.DailySchedule.from_times
      on_time = "08:00"
      off_time = "18:00"
      schedule_models.WeeklySchedule([
          from_times(day_name="Monday", on_time=on_time, off_time=off_time),
          from_times(day_name="Tuesday", on_time=on_time, off_time=off_time),
          from_times(day_name="Wednesday", on_time=on_time, off_time=off_time),
          from_times(day_name="Thursday", on_time=on_time, off_time=off_time),
          from_times(day_name="Friday", on_time=on_time, off_time=off_time),
          from_times(day_name="Saturday", on_time=None, off_time=None),
          from_times(day_name="Sunday", on_time=None, off_time=None),
          from_times(day_name="Sunday", on_time=None, off_time=None),  # Extra
      ])


if __name__ == "__main__":
  absltest.main()
