import datetime
from unittest import mock
import zoneinfo

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from pandas.tseries import holiday
from smart_buildings.smart_control.environment import conftest as env_conftest
from smart_buildings.smart_control.llm.utils import schedule_models
from smart_buildings.smart_control.llm.utils import schedule_tool

BuildingOperationalMode = schedule_tool.BuildingOperationalMode

TIME_ZONE = "US/Pacific"
CURRENT_LOCAL_TIMESTAMP = pd.Timestamp("2021-06-01 12:00:00", tz=TIME_ZONE)

UPCOMING_HOLIDAYS = (
    {
        "date": pd.Timestamp("2021-06-18 00:00:00"),
        "holiday": "Juneteenth National Independence Day",
        "day_of_year": 169,
        "year": 2021,
        "day_name": "Friday",
    },
    {
        "date": pd.Timestamp("2021-07-05 00:00:00"),
        "holiday": "Independence Day",
        "day_of_year": 186,
        "year": 2021,
        "day_name": "Monday",
    },
    {
        "date": pd.Timestamp("2021-09-06 00:00:00"),
        "holiday": "Labor Day",
        "day_of_year": 249,
        "year": 2021,
        "day_name": "Monday",
    },
    {
        "date": pd.Timestamp("2021-10-11 00:00:00"),
        "holiday": "Columbus Day",
        "day_of_year": 284,
        "year": 2021,
        "day_name": "Monday",
    },
    {
        "date": pd.Timestamp("2021-11-11 00:00:00"),
        "holiday": "Veterans Day",
        "day_of_year": 315,
        "year": 2021,
        "day_name": "Thursday",
    },
)

SCHEDULE_SCENARIOS = (
    {
        "testcase_name": "weekday_morning",
        "timestamp": pd.Timestamp("2025-12-12 08:00:00", tz=TIME_ZONE),
        "weekday_name": "Friday",
        "is_workday": True,
        "is_holiday": False,
        "is_operational_day": True,
        "is_during_operational_hours": True,
        "is_operational": True,
        "operational_mode": schedule_tool.BuildingOperationalMode.ON,
    },
    {
        "testcase_name": "weekday_afternoon",
        "timestamp": pd.Timestamp("2025-12-12 15:30:00", tz=TIME_ZONE),
        "weekday_name": "Friday",
        "is_workday": True,
        "is_holiday": False,
        "is_operational_day": True,
        "is_during_operational_hours": True,
        "is_operational": True,
        "operational_mode": schedule_tool.BuildingOperationalMode.ON,
    },
    {
        "testcase_name": "weekday_nighttime",
        "timestamp": pd.Timestamp("2025-12-12 02:00:00", tz=TIME_ZONE),
        "weekday_name": "Friday",
        "is_workday": True,
        "is_holiday": False,
        "is_operational_day": True,
        "is_during_operational_hours": False,
        "is_operational": False,
        "operational_mode": schedule_tool.BuildingOperationalMode.OFF,
    },
    {
        "testcase_name": "holiday_daytime",  # Christmas, a Thursday
        "timestamp": pd.Timestamp("2025-12-25 11:00:00", tz=TIME_ZONE),
        "weekday_name": "Thursday",
        "is_workday": True,
        "is_holiday": True,
        "is_operational_day": False,
        "is_during_operational_hours": True,
        "is_operational": False,
        "operational_mode": schedule_tool.BuildingOperationalMode.OFF,
    },
    {
        "testcase_name": "weekend_nighttime",
        "timestamp": pd.Timestamp("2025-12-13 02:00:00", tz=TIME_ZONE),
        "weekday_name": "Saturday",
        "is_workday": False,
        "is_holiday": False,
        "is_operational_day": False,
        "is_during_operational_hours": False,
        "is_operational": False,
        "operational_mode": schedule_tool.BuildingOperationalMode.OFF,
    },
)

SCHEDULE_METADATA = {
    "weekly_schedule": {
        "time_zone": "US/Pacific",
        "daily_schedules": {
            "Monday": {"on_time": "07:00", "off_time": "19:00"},
            "Tuesday": {"on_time": "07:00", "off_time": "19:00"},
            "Wednesday": {"on_time": "07:00", "off_time": "19:00"},
            "Thursday": {"on_time": "07:00", "off_time": "19:00"},
            "Friday": {"on_time": "07:00", "off_time": "19:00"},
            "Saturday": {"on_time": None, "off_time": None},
            "Sunday": {"on_time": None, "off_time": None},
        },
    },
    "start_date": None,
    "end_date": None,
    "upcoming_holidays": [
        {
            "date": "2021-06-18",
            "name": "Juneteenth National Independence Day",
            "day_name": "Friday",
        },
        {
            "date": "2021-07-05",
            "name": "Independence Day",
            "day_name": "Monday",
        },
        {"date": "2021-09-06", "name": "Labor Day", "day_name": "Monday"},
        {"date": "2021-10-11", "name": "Columbus Day", "day_name": "Monday"},
        {"date": "2021-11-11", "name": "Veterans Day", "day_name": "Thursday"},
    ],
}


class ScheduleToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_timestamp_now = self.enter_context(
        mock.patch.object(pd.Timestamp, "now", autospec=True)
    )
    self.mock_timestamp_now.return_value = CURRENT_LOCAL_TIMESTAMP
    self.schedule = schedule_tool.ScheduleTool(time_zone=TIME_ZONE)
    self.expected_class = schedule_tool.ScheduleTool

  def test_initialization(self):
    self.assertIsInstance(self.schedule, self.expected_class)

  def test_weekly_schedule(self):
    self.assertIsInstance(
        self.schedule.weekly_schedule, schedule_models.WeeklySchedule
    )

  def test_time_zone(self):
    self.assertEqual(self.schedule.time_zone, TIME_ZONE)

  def test_holiday_calendar(self):
    self.assertIsNone(self.schedule.start_date)
    self.assertIsNone(self.schedule.end_date)
    self.assertIsInstance(self.schedule.cal, holiday.USFederalHolidayCalendar)

  # CURRENT DATE AND TIME

  def test_date_time_properties(self):
    with self.subTest(name="current_local_timestamp"):
      self.assertEqual(
          self.schedule.current_local_timestamp, CURRENT_LOCAL_TIMESTAMP
      )

    with self.subTest(name="current_year"):
      self.assertEqual(self.schedule.current_year, 2021)

    with self.subTest(name="current_date"):
      self.assertEqual(self.schedule.current_date, datetime.date(2021, 6, 1))
      self.assertEqual(self.schedule.current_date_str, "2021-06-01")

    with self.subTest(name="current_time"):
      self.assertEqual(
          self.schedule.current_time,
          datetime.time(12, 0, tzinfo=zoneinfo.ZoneInfo(TIME_ZONE)),
      )
      self.assertEqual(self.schedule.current_time_str, "12:00")

  # HOLIDAY CALENDAR

  def test_get_holidays(self):
    with self.subTest(name="as_index"):
      holidays = self.schedule._get_holidays(return_name=False)
      self.assertIsInstance(holidays, pd.DatetimeIndex)

    with self.subTest(name="as_series"):
      holidays = self.schedule._get_holidays(return_name=True)
      self.assertIsInstance(holidays, pd.Series)

  def test_holidays(self):
    holidays = self.schedule.holidays
    self.assertIsInstance(holidays, set)
    self.assertGreaterEqual(len(holidays), 2474)
    self.assertIn("1970-01-01", holidays)
    self.assertIn("2200-12-25", holidays)

  def test_holidays_df(self):
    df = self.schedule.holidays_df
    self.assertIsInstance(df, pd.DataFrame)
    self.assertGreaterEqual(len(df), 2474)
    self.assertListEqual(
        df.columns.tolist(),
        ["date", "holiday", "day_of_year", "year", "day_name"],
    )

    holidays = df["date"].dt.strftime("%Y-%m-%d").tolist()
    self.assertIn("1970-01-01", holidays)
    self.assertIn("2200-12-25", holidays)

  def test_upcoming_holidays_df(self):
    self.assertEqual(
        self.schedule.upcoming_holidays_df.to_dict("records"),
        list(UPCOMING_HOLIDAYS),
    )

  def test_upcoming_holidays(self):
    self.assertEqual(
        self.schedule.upcoming_holidays,
        [h["date"].strftime("%Y-%m-%d") for h in UPCOMING_HOLIDAYS],
    )

  def test_is_holiday(self):
    self.assertFalse(self.schedule.is_holiday)

  def test_json_metadata(self):
    self.assertEqual(self.schedule.json_metadata, SCHEDULE_METADATA)

  # DAY OF WEEK

  def test_current_weekday_name(self):
    self.assertEqual(self.schedule.current_weekday_name, "Tuesday")

  def test_is_workday(self):
    self.assertTrue(self.schedule.is_workday)

  # CURRENT OPERATIONAL STATUS

  def test_is_operational_day(self):
    self.assertTrue(self.schedule.is_operational_day)

  def test_is_during_operational_hours(self):
    self.assertTrue(self.schedule.is_during_operational_hours)

  def test_building_is_operational(self):
    self.assertTrue(self.schedule.building_is_operational)

  def test_building_operational_mode(self):
    self.assertEqual(
        self.schedule.building_operational_mode,
        schedule_tool.BuildingOperationalMode.ON,
    )


class BuildingScheduleToolTest(ScheduleToolTest):

  def setUp(self):
    super().setUp()
    self.env = env_conftest.create_environment(
        start_timestamp=CURRENT_LOCAL_TIMESTAMP
    )
    self.schedule = schedule_tool.BuildingScheduleTool(env=self.env)
    self.expected_class = schedule_tool.BuildingScheduleTool


#
# SCENARIO TESTS
#


class ScheduleScenariosTest(parameterized.TestCase):
  """Performs scenario testing for different operational modes."""

  @parameterized.named_parameters(*SCHEDULE_SCENARIOS)
  def test_building_operation_schedule(
      self,
      timestamp,
      weekday_name,
      is_workday,
      is_holiday,
      is_operational_day,
      is_during_operational_hours,
      is_operational,
      operational_mode,
  ):
    env = env_conftest.create_environment(start_timestamp=timestamp)
    schedule = schedule_tool.BuildingScheduleTool(env=env)
    with self.subTest(name="current_date_and_time"):
      self.assertEqual(schedule.current_local_timestamp, timestamp)
      self.assertEqual(schedule.current_weekday_name, weekday_name)
      self.assertEqual(schedule.is_workday, is_workday)
      self.assertEqual(
          schedule.is_during_operational_hours, is_during_operational_hours
      )

    with self.subTest(name="holiday_calendar"):
      self.assertEqual(schedule.is_holiday, is_holiday)

    with self.subTest(name="operational_status"):
      self.assertEqual(schedule.is_operational_day, is_operational_day)
      self.assertEqual(schedule.building_is_operational, is_operational)
      self.assertEqual(schedule.building_operational_mode, operational_mode)


#
# CUSTOM HOLIDAY CALENDAR TESTS
#


class MyCustomHolidayCalendar(holiday.AbstractHolidayCalendar):
  """Custom holiday calendar for testing."""

  rules = [
      holiday.Holiday("Founder's Day", month=7, day=1),
      holiday.Holiday("My Birthday", month=9, day=1),
  ]


class CustomHolidayScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_timestamp_now = self.enter_context(
        mock.patch.object(pd.Timestamp, "now", autospec=True)
    )
    self.custom_calendar = MyCustomHolidayCalendar()

  @parameterized.named_parameters(
      dict(
          testcase_name="founders_day",
          timestamp="2024-07-01 10:00:00",
          is_holiday=True,
      ),
      dict(
          testcase_name="my_birthday",
          timestamp="2024-09-01 10:00:00",
          is_holiday=True,
      ),
      dict(
          testcase_name="christmas_day",
          timestamp="2024-12-25 10:00:00",
          is_holiday=False,
      ),
      dict(
          testcase_name="new_years_day",
          timestamp="2025-01-01 10:00:00",
          is_holiday=False,
      ),
  )
  def test_custom_holidays(self, timestamp, is_holiday):
    self.mock_timestamp_now.return_value = pd.Timestamp(timestamp, tz=TIME_ZONE)
    schedule = schedule_tool.ScheduleTool(
        time_zone=TIME_ZONE,
        cal=self.custom_calendar,
    )
    self.assertEqual(schedule.is_holiday, is_holiday)


if __name__ == "__main__":
  absltest.main()
