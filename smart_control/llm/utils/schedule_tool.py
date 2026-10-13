"""Schedule tool.

This tool provides information about the building's operational schedule, by
accessing information such as the current date and time from the environment.

**Operational Modes**

This tool can be used by an agent to determine if the building's devices should
be ON or OFF, based on the time of day, day of week, and holiday calendar.

**Weekly Schedule**

By default, this tool assumes that workdays are Mondays through Fridays, and
that operational hours are from 7:00 AM to 7:00 PM, but these values can be
customized. This includes the ability to specify different operational hours for
different days of the week. See the `schedule_models.WeeklySchedule` class for
more details.

**Holiday Calendar**

We anticipate the need to customize the holiday calendar, because we will be
supporting buildings across different countries. And because even within a given
country, different localities, companies, and building operators may observe
slightly different holiday schedules.

By default, this tool uses the `holiday.USFederalHolidayCalendar` to determine
the holidays, which provides a good baseline for US-based buildings. However,
you can specify a different holiday calendar, as long as it implements the
`holiday.AbstractHolidayCalendar` interface from pandas (as illustrated by the
example below).

```python
from pandas.tseries import holiday

class MyCustomHolidayCalendar(holiday.AbstractHolidayCalendar):
  rules = [
      holiday.Holiday("Founder's Day", month=7, day=1),
      holiday.Holiday("My Birthday", month=9, day=1),
  ]
```
"""

import abc
import datetime
import enum
from typing import Any, Final, TypeAlias

import pandas as pd
from pandas.tseries import holiday
from smart_buildings.smart_control.environment import environment
from smart_buildings.smart_control.llm.utils import schedule_models

SerializableData: TypeAlias = dict[str, Any]


class BuildingOperationalMode(enum.Enum):
  """The operational mode of the building (and its devices)."""

  ON = "ON"
  OFF = "OFF"


OPERATIONAL_MODES = tuple(mode.value for mode in BuildingOperationalMode)


DEFAULT_WEEKLY_SCHEDULE: Final[schedule_models.WeeklySchedule] = (
    schedule_models.WeeklySchedule.from_dict(
        schedule_dict={
            "Monday": ("07:00", "19:00"),
            "Tuesday": ("07:00", "19:00"),
            "Wednesday": ("07:00", "19:00"),
            "Thursday": ("07:00", "19:00"),
            "Friday": ("07:00", "19:00"),
            "Saturday": (None, None),
            "Sunday": (None, None),
        },
        time_zone="US/Pacific",
    )
)


class BaseSchedule(abc.ABC):
  """Abstract interface providing info about a building's operational schedule.

  Requires a child class to implement the `time_zone` and
  `current_local_timestamp` properties, using the building's local time zone.

  Determines if the building's devices should be ON or OFF, based on the time of
  day, day of week, and holiday calendar.

  For the holiday calendar, the US federal holiday calendar will be used by
  default, however you can customize this by passing in your own implementation
  of the `holiday.AbstractHolidayCalendar` interface from pandas.

  The start and end dates are optionally used to filter the range of holidays
  included. If not specified, holidays from all available years will be
  included.

  Attributes:
    time_zone: The time zone to use for all date and time calculations.
    current_local_timestamp: The current date and time in the local time zone.
    weekly_schedule: The operational hours for each day of the week, using the
      building's local time zone.
    cal: The holiday calendar to use for determining holidays. Defaults to the
      US federal holiday calendar.
    start_date: The start date for the holiday calendar (optional).
    end_date: The end date for the holiday calendar (optional).
    n_upcoming_holidays: The number of upcoming holidays to return.
  """

  def __init__(
      self,
      weekly_schedule: schedule_models.WeeklySchedule | None = None,
      cal: holiday.AbstractHolidayCalendar | None = None,
      start_date: str | None = None,
      end_date: str | None = None,
      n_upcoming_holidays: int = 5,
  ):
    """Initializes the instance.

    Args:
      weekly_schedule: The operational schedule for the week. Defaults to
        `DEFAULT_WEEKLY_SCHEDULE`.
      cal: The holiday calendar to use for determining holidays. The calendar
        must implement the `holiday.AbstractHolidayCalendar` interface.
        By default, the US federal holiday calendar is used.
      start_date: The start date used to optionally filter the list of
        holidays. Defaults to None.
      end_date: The end date used to optionally filter the list of
        holidays. Defaults to None.
      n_upcoming_holidays: The number of upcoming holidays to return.
    """
    self.weekly_schedule = weekly_schedule or DEFAULT_WEEKLY_SCHEDULE
    self.start_date = start_date
    self.end_date = end_date
    self.cal = cal or holiday.USFederalHolidayCalendar()
    self.n_upcoming_holidays = n_upcoming_holidays

  #
  # BASE CONTRACT
  #

  @property
  @abc.abstractmethod
  def time_zone(self) -> str:
    """The time zone used for all timestamps and comparisons."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def current_local_timestamp(self) -> pd.Timestamp:
    """The current (timezone-aware) date and time in the local timezone."""
    raise NotImplementedError

  #
  # IMPLEMENTATION METHODS
  #

  @property
  def json_metadata(self) -> SerializableData:
    """Info to write into a JSON file. Needs to be serializable."""
    holidays_df = self.upcoming_holidays_df.copy()
    holidays_df["date"] = holidays_df["date"].dt.strftime("%Y-%m-%d")
    holidays_df = holidays_df.rename(columns={"holiday": "name"})
    holidays = holidays_df[["date", "name", "day_name"]].to_dict("records")

    return {
        "weekly_schedule": self.weekly_schedule.json_metadata,
        "start_date": self.start_date,
        "end_date": self.end_date,
        "upcoming_holidays": holidays,
    }

  # CURRENT DATE AND TIME

  @property
  def current_year(self) -> int:
    """The current year, in the building's local timezone."""
    return self.current_local_timestamp.year

  @property
  def current_date(self) -> datetime.date:
    """The current date, in the building's local timezone."""
    return self.current_local_timestamp.date()

  @property
  def current_date_str(self) -> str:
    """The current date as a string, in the building's local timezone."""
    return self.current_local_timestamp.strftime("%Y-%m-%d")

  @property
  def current_time(self) -> datetime.time:
    """The current (timezone-aware) time, in the building's local timezone."""
    return self.current_local_timestamp.timetz()

  @property
  def current_time_str(self) -> str:
    """The current time as a string, in the building's local timezone."""
    return self.current_local_timestamp.strftime("%H:%M")

  @property
  def current_weekday_name(self) -> str:
    """The current day of the week, in the building's local timezone."""
    return self.current_local_timestamp.strftime("%A")  # > "Monday"

  # HOLIDAY CALENDAR

  def _get_holidays(
      self, return_name: bool = False
  ) -> pd.DatetimeIndex | pd.Series:
    """Returns the holidays as a DatetimeIndex or a Series.

    Args:
      return_name: Whether to return the holidays as a Series.

    Returns:
      A DatetimeIndex or a Series of the holidays.
    """
    return self.cal.holidays(
        start=self.start_date, end=self.end_date, return_name=return_name
    )

  @property
  def holidays(self) -> set[str]:
    """The holiday calendar, as a set of string dates (like '2025-01-01')."""
    return {
        d.strftime("%Y-%m-%d") for d in self._get_holidays(return_name=False)
    }

  @property
  def holidays_df(self) -> pd.DataFrame:
    """The holiday calendar, as a DataFrame."""
    df = self._get_holidays(return_name=True).reset_index()
    df.columns = ["date", "holiday"]
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["day_name"] = df["date"].dt.day_name()
    return df

  @property
  def upcoming_holidays_df(self) -> pd.DataFrame:
    """The next few upcoming holidays, as a DataFrame.

    Use the `n_upcoming_holidays` initialization argument to customize the
    number of holidays to be included.

    Note: It is possible for this dataframe to contain fewer than the requested
    number of holidays, depending on the current date and the end date.

    Returns:
      A DataFrame of the next few upcoming holidays, sorted by date ascending.
    """
    df = self.holidays_df[self.holidays_df["date"].dt.date >= self.current_date]
    df.sort_values(by="date", inplace=True)
    return df.head(self.n_upcoming_holidays)

  @property
  def upcoming_holidays(self) -> list[str]:
    """The next few upcoming holidays.

    Use the `n_upcoming_holidays` initialization argument to customize the
    number of holidays to return.

    Note: It is possible for this list to contain fewer than the requested
    number of holidays, depending on the current date and the end date.

    Returns:
      A list of strings, like '2025-01-01', sorted by date ascending.
    """
    return self.upcoming_holidays_df["date"].dt.strftime("%Y-%m-%d").tolist()

  @property
  def is_holiday(self) -> bool:
    """Whether the current date is a holiday."""
    return self.current_date_str in self.holidays

  # WEEKLY SCHEDULE

  @property
  def current_daily_schedule(self) -> schedule_models.DailySchedule:
    """The daily schedule for the current day of week."""
    return self.weekly_schedule.get_daily_schedule(self.current_weekday_name)

  @property
  def is_workday(self) -> bool:
    """Whether the current date is a workday (not considering holidays)."""
    return self.current_daily_schedule.is_operational_day

  # CURRENT OPERATIONAL STATUS

  @property
  def is_operational_day(self) -> bool:
    """Whether the current date is an operational day."""
    return self.is_workday and not self.is_holiday

  @property
  def is_during_operational_hours(self) -> bool:
    """Whether the current time is during operational hours."""
    return self.current_daily_schedule.is_during_operational_hours(
        self.current_time
    )

  @property
  def building_is_operational(self) -> bool:
    """Whether the building is operational."""
    return self.is_operational_day and self.is_during_operational_hours

  @property
  def building_operational_mode(self) -> BuildingOperationalMode:
    """The building's operational mode."""
    if self.building_is_operational:
      return BuildingOperationalMode.ON
    else:
      return BuildingOperationalMode.OFF


class ScheduleTool(BaseSchedule):
  """Schedule tool using the current date and time in a specified time zone."""

  def __init__(self, time_zone: str = "UTC", **kwargs):
    """Initializes the instance.

    Args:
      time_zone: The time zone to use for all date and time calculations.
        Defaults to UTC.
      **kwargs: Keyword arguments to pass to the base class.
    """
    super().__init__(**kwargs)
    self._time_zone = time_zone

  @property
  def time_zone(self) -> str:
    """Returns the time zone used for all date and time calculations."""
    return self._time_zone

  @property
  def current_local_timestamp(self) -> pd.Timestamp:
    """The current date and time in the local timezone."""
    return pd.Timestamp.now(tz=self.time_zone)


class BuildingScheduleTool(BaseSchedule):
  """A tool for accessing information about the building's operational schedule.

  Uses the time zone and current local timestamp from the environment to
  determine if the building's devices should be ON or OFF, based on the time of
  day, day of week, and holiday calendar.

  Attributes:
    env: The environment to use for getting the time zone and current timestamp.
    **kwargs: Keyword arguments to pass to the base class.
  """

  def __init__(self, env: environment.Environment, **kwargs):
    """Initializes the instance.

    Args:
      env: The environment to use for getting the time zone and current
        timestamp.
      **kwargs: Keyword arguments to pass to the base class.
    """
    super().__init__(**kwargs)
    self.env = env

  @property
  def time_zone(self) -> str:
    """The building's local time zone, from the environment."""
    return self.env.time_zone

  @property
  def current_local_timestamp(self) -> pd.Timestamp:
    """The current date and time, in the building's local timezone."""
    return self.env.current_local_timestamp
