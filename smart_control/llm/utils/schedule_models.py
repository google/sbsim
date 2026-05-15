"""Schedule models.

These models represent the daily and weekly operational schedules for a given
building, and are used by the schedule tool to determine the operational mode
of the building, based on the current day and time.

These daily and weekly schedules are meant to provide general templates that are
applied to all weeks, and are not tied to specific dates. As such, they also do
not account for holidays, which should be incorporated separately.

The models use timezone-aware on and off times, to ensure accurate comparisons.
"""

import calendar
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
from typing import Any, Self
import zoneinfo

# FYI: The calendar module may use different day names depending on the locale.
# We assume the calendar is set to the English locale.
DAY_NAMES = tuple(calendar.day_name)  # ("Monday", "Tuesday", etc.)


def str_to_time_with_zone(time_str: str, time_zone: str) -> datetime.time:
  """Returns a datetime.time object that is timezone aware."""
  tzinfo = zoneinfo.ZoneInfo(time_zone)
  return datetime.time.fromisoformat(time_str).replace(tzinfo=tzinfo)


def display_time(time: datetime.time | None) -> str | None:
  """Displays a given time as a string.

  This method is used for display and JSON serialization purposes only, not for
  comparisons.

  It needs to handle None values because the daily schedule's on and off times
  can be None (which designates a non-operational day).

  Args:
    time: The time to convert, or None.

  Returns:
    The time as a string, like "07:00", or None.
  """
  return time.strftime("%H:%M") if time is not None else None


@dataclasses.dataclass(frozen=True)
class DailySchedule:
  """The planned operational schedule for a given day of week.

  This model assumes that there is a single operational period for the day, and
  that the building and its devices should be "ON" during the hours between the
  `on_time` and `off_time`, and "OFF" otherwise.

  A given day should have both an `on_time` and `off_time` (to designate an
  operational day), or neither (using None values to designate a non-operational
  day). If present, both times must be timezone aware, and share the same time
  zone. When comparing a time to the on_time and off_time, the comparison time
  must also be timezone aware, and have the same time zone (see the
  `is_during_operational_hours` method for more details).

  Attributes:
    day_name: The name of the day of the week (e.g. "Monday").
    on_time: The time of day when devices should be turned on, or None.
    off_time: The time of day when devices should be turned off, or None.
    time_zone: The time zone used for the on_time and off_time. Required, even
      if the on_time and off_time are None.
  """
  day_name: str
  time_zone: str = "UTC"
  on_time: datetime.time | None = None
  off_time: datetime.time | None = None

  # VALIDATIONS

  def __post_init__(self) -> None:
    self._validate_day_name()
    self._validate_time_zone()
    self._validate_times()
    self._validate_times_zones()
    self._validate_times_start_after_end()

  def _validate_day_name(self) -> None:
    """Ensures the day name is valid."""
    if self.day_name not in DAY_NAMES:
      raise ValueError(
          f"Unknown day name: {self.day_name}. Expecting one of: {DAY_NAMES}."
      )

  def _validate_time_zone(self) -> None:
    """Ensures the time zone is present and valid."""
    if self.time_zone is None:
      raise ValueError("The time zone must be specified.")

    try:
      self.tzinfo  # pylint: disable=pointless-statement
    except zoneinfo.ZoneInfoNotFoundError as err:
      raise ValueError(f"Invalid time zone: {self.time_zone}.") from err

  def _validate_times(self) -> None:
    """Ensures both times are present, or neither are."""
    if (self.on_time is None and self.off_time is not None) or (
        self.on_time is not None and self.off_time is None
    ):
      raise ValueError(
          "The on_time and off_time must both be specified, or both be None."
      )

  def _validate_times_zones(self) -> None:
    """Ensures both times have a time zone, and they match the schedule."""
    if self.on_time is None or self.off_time is None:
      return

    if self.on_time.tzinfo is None:
      raise ValueError("The on_time needs to have a time zone.")

    if self.off_time.tzinfo is None:
      raise ValueError("The off_time needs to have a time zone.")

    if (
        self.on_time.tzinfo != self.tzinfo
        or self.off_time.tzinfo != self.tzinfo
    ):
      raise ValueError(
          "The on_time and off_time must have the same time zone, and it must "
          f"match the schedule's time zone: {self.time_zone}."
      )

  def _validate_times_start_after_end(self) -> None:
    """Ensures the on_time is before the off_time."""
    if self.on_time is not None and self.off_time is not None:
      if self.on_time >= self.off_time:
        raise ValueError("The on_time must be before the off_time.")

  # CONSTRUCTOR

  @classmethod
  def from_times(
      cls,
      *,
      day_name: str,
      on_time: str | None,
      off_time: str | None,
      time_zone: str | None = "UTC",
  ) -> Self:
    """Creates a DailySchedule from 24-hr time strings.

    This method allows you to pass timezone-naive strings for convenience. It
    will apply the specified time zone to each of the times to ensure they are
    both timezone aware.

    Args:
      day_name: The name of the day of the week (e.g. "Monday").
      on_time: The time of day when devices should be turned on, as a string
        like "07:00", or None if the day is not operational.
      off_time: The time of day when devices should be turned off, as a string
        like "19:00", or None if the day is not operational.
      time_zone: The time zone to use for the on_time and off_time. Defaults to
        "UTC".

    Returns:
      A DailySchedule instance.
    """
    if on_time is not None:
      on_time = str_to_time_with_zone(on_time, time_zone=time_zone)

    if off_time is not None:
      off_time = str_to_time_with_zone(off_time, time_zone=time_zone)

    return cls(
        day_name=day_name,
        on_time=on_time,
        off_time=off_time,
        time_zone=time_zone,
    )

  # METHODS AND PROPERTIES

  @property
  def tzinfo(self) -> zoneinfo.ZoneInfo:
    """Information about the given time zone, as a zoneinfo.ZoneInfo object."""
    return zoneinfo.ZoneInfo(self.time_zone)

  @property
  def is_operational_day(self) -> bool:
    """Whether this day is scheduled to be an operational day."""
    return self.on_time is not None and self.off_time is not None

  def is_during_operational_hours(self, time: datetime.time) -> bool:
    """Determines if the given time is within the scheduled hours.

    The comparison time needs to be timezone-aware, and have the same time zone
    as the on_time and off_time, which have both already been validated to have
    the same time zone.

    Note about edge cases: The start time is considered operational (inclusive),
    but the end time is considered non-operational (exclusive).

    Args:
      time: The time to check. Must be timezone aware, and have the same time
        zone as the schedule.

    Returns:
      A boolean indicating whether the given time falls within the scheduled
      hours.
    """
    if time.tzinfo is None:
      raise ValueError("The comparison time must have a time zone.")

    if str(time.tzinfo) != str(self.tzinfo):
      raise ValueError(
          "The comparison time must have the same time zone as the schedule."
      )

    if not self.is_operational_day:
      return False

    return self.on_time <= time < self.off_time


@dataclasses.dataclass(frozen=True)
class WeeklySchedule:
  """The operational schedule for a given week.

  The weekly schedule contains a daily schedule for each day of the week.

  Attributes:
    daily_schedules: A sequence of DailySchedules for each day of the week.
  """

  daily_schedules: Sequence[DailySchedule]

  # VALIDATIONS

  def __post_init__(self) -> None:
    self._validate_all_days()

  def _validate_all_days(self) -> None:
    """Ensures all expected day names are present."""
    day_names = [schedule.day_name for schedule in self.daily_schedules]
    if sorted(day_names) != sorted(DAY_NAMES):
      raise ValueError(
          f"Weekly schedule must have a schedule for each day of the week."
          f" Expected: {DAY_NAMES}, got: {day_names}."
      )

  # CONSTRUCTOR

  @classmethod
  def from_dict(
      cls,
      schedule_dict: Mapping[str, Sequence[str | None]],
      time_zone: str | None = "UTC",
  ) -> Self:
    """Creates a WeeklySchedule from a dictionary of DailySchedules."""
    return cls([
        DailySchedule.from_times(
            day_name=day_name,
            on_time=on_time,
            off_time=off_time,
            time_zone=time_zone,
        )
        for day_name, (on_time, off_time) in schedule_dict.items()
    ])

  # PROPERTIES AND METHODS

  @property
  def time_zone(self) -> str:
    """The time zone used for all the daily schedules."""
    return self.daily_schedules[0].time_zone

  def get_daily_schedule(self, day_name: str) -> DailySchedule:
    """Returns the daily schedule for the given day of week.

    Args:
      day_name: The name of the day of the week (e.g. "Monday").

    Raises:
      ValueError: If the day name is not in the weekly schedule.

    Returns:
      The DailySchedule instance for the given day of week.
    """
    for schedule in self.daily_schedules:
      if schedule.day_name == day_name:
        return schedule
    raise ValueError(f"Unknown day name: {day_name}")

  @property
  def json_metadata(self) -> dict[str, Any]:
    """Info about the weekly schedule, in a JSON serializable format."""
    daily_schedules_dict = {
        schedule.day_name: {
            "on_time": display_time(schedule.on_time),
            "off_time": display_time(schedule.off_time),
        }
        for schedule in self.daily_schedules
    }
    return {
        "time_zone": self.time_zone,
        "daily_schedules": daily_schedules_dict,
    }
