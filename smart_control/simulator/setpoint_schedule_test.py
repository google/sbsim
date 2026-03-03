from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from pandas.tseries import holiday
import pytz

from smart_buildings.smart_control.simulator import setpoint_schedule


class SetpointScheduleTest(parameterized.TestCase):

  def test_init(self):
    # make sure all the state in the constructor is stored properly
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 223, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    self.assertEqual(schedule.morning_start_hour, morning_start_hour)
    self.assertEqual(schedule.evening_start_hour, evening_start_hour)
    self.assertEqual(schedule.comfort_temp_window, comfort_temp_window)
    self.assertEqual(schedule.eco_temp_window, eco_temp_window)
    self.assertEqual(schedule.holidays, holidays)

  def test_init_default(self):
    # make sure all the state in the constructor is stored properly
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)

    # test with default value for holidays
    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
    )

    self.assertEqual(schedule.holidays, set())

  def test_init_raises_error_evening_before_morning(self):
    morning_start_hour = 22
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)

    with self.assertRaises(ValueError):
      setpoint_schedule.SetpointSchedule(
          morning_start_hour,
          evening_start_hour,
          comfort_temp_window,
          eco_temp_window,
      )

  def test_init_raises_error_invalid_comfort_window(self):
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (296, 295)
    eco_temp_window = (290, 297)

    with self.assertRaises(ValueError):
      setpoint_schedule.SetpointSchedule(
          morning_start_hour,
          evening_start_hour,
          comfort_temp_window,
          eco_temp_window,
      )

  def test_init_raises_error_invalid_eco_window(self):
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (298, 297)

    with self.assertRaises(ValueError):
      setpoint_schedule.SetpointSchedule(
          morning_start_hour,
          evening_start_hour,
          comfort_temp_window,
          eco_temp_window,
      )

  @parameterized.named_parameters(
      ('first_hour_of_morning', 9, None, True),
      ('last_hour_of_morning', 17, None, True),
      ('first_hour_of_evening', 18, None, False),
      ('last_hour_of_evening', 8, None, False),
      ('first_hour_of_morning_tz', 9, pytz.UTC, True),
      ('last_hour_of_morning_tz', 17, pytz.timezone('US/Pacific'), True),
      ('first_hour_of_evening_tz', 18, pytz.UTC, False),
      ('last_hour_of_evening_tz', 8, pytz.timezone('US/Eastern'), False),
  )
  def test_is_comfort_mode(self, hour, time_zone, expected):
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 32, 245])

    if time_zone is None:
      schedule = setpoint_schedule.SetpointSchedule(
          morning_start_hour,
          evening_start_hour,
          comfort_temp_window,
          eco_temp_window,
          holidays,
      )
    else:
      schedule = setpoint_schedule.SetpointSchedule(
          morning_start_hour,
          evening_start_hour,
          comfort_temp_window,
          eco_temp_window,
          holidays,
          time_zone,
      )
    self.assertEqual(
        schedule.is_comfort_mode(
            pd.Timestamp(year=2021, month=4, day=30, hour=hour, tz=time_zone)
        ),
        expected,
    )

    # on holiday always return False
    self.assertEqual(
        schedule.is_comfort_mode(
            pd.Timestamp(year=2021, month=2, day=1, hour=hour, tz=time_zone)
        ),
        False,
    )

  @parameterized.named_parameters(
      ('Saturday', 2),
      ('Sunday', 3),
      ('Monday', 4),
      ('Tuesday', 5),
      ('Wednesday', 6),
      ('Thursday', 7),
      ('Friday', 8),
  )
  def test_is_weekend(self, day):
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)

    date = pd.Timestamp(year=2021, month=1, day=day, hour=7)
    expected = False
    if date.day_name() == 'Saturday' or date.day_name() == 'Sunday':
      expected = True

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
    )

    self.assertEqual(schedule.is_weekend(date), expected)

  @parameterized.named_parameters(
      ('first_hour_of_morning', 9, (292, 295)),
      ('last_hour_of_morning', 17, (292, 295)),
      ('first_hour_of_evening', 18, (290, 297)),
      ('last_hour_of_evening', 8, (290, 297)),
  )
  def test_get_temperature_window(self, hour, expected):
    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 32, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    self.assertEqual(
        schedule.get_temperature_window(
            pd.Timestamp(year=2021, month=4, day=30, hour=hour)
        ),
        expected,
    )

    # on holiday always return (290, 297)
    self.assertEqual(
        schedule.get_temperature_window(
            pd.Timestamp(year=2021, month=2, day=1, hour=hour)
        ),
        (290, 297),
    )

  def test_get_plot_data(self):
    start_time = pd.Timestamp(year=2021, month=1, day=4, hour=7)
    end_time = pd.Timestamp(year=2021, month=1, day=11, hour=20)
    comfort_modes = [False, True, False, True, False, True, False, True, False]
    start_times = [
        start_time,
        pd.Timestamp(year=2021, month=1, day=4, hour=9),
        pd.Timestamp(year=2021, month=1, day=4, hour=18),
        pd.Timestamp(year=2021, month=1, day=5, hour=9),
        pd.Timestamp(year=2021, month=1, day=5, hour=18),
        pd.Timestamp(year=2021, month=1, day=6, hour=9),
        pd.Timestamp(year=2021, month=1, day=6, hour=18),
        pd.Timestamp(year=2021, month=1, day=11, hour=9),
        pd.Timestamp(year=2021, month=1, day=11, hour=18),
    ]

    end_times = [
        pd.Timestamp(year=2021, month=1, day=4, hour=9),
        pd.Timestamp(year=2021, month=1, day=4, hour=18),
        pd.Timestamp(year=2021, month=1, day=5, hour=9),
        pd.Timestamp(year=2021, month=1, day=5, hour=18),
        pd.Timestamp(year=2021, month=1, day=6, hour=9),
        pd.Timestamp(year=2021, month=1, day=6, hour=18),
        pd.Timestamp(year=2021, month=1, day=11, hour=9),
        pd.Timestamp(year=2021, month=1, day=11, hour=18),
        end_time,
    ]

    heating_setpoints = [290, 292, 290, 292, 290, 292, 290, 292, 290]
    cooling_setpoints = [297, 295, 297, 295, 297, 295, 297, 295, 297]

    df = pd.DataFrame({
        'comfort_mode': comfort_modes,
        'start_time': start_times,
        'end_time': end_times,
        'heating_setpoint': heating_setpoints,
        'cooling_setpoint': cooling_setpoints,
    })

    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 8, 223, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    pd.testing.assert_frame_equal(
        schedule.get_plot_data(start_time, end_time), df
    )


class HolidayScheduleTest(parameterized.TestCase):
  """NOTE: consider updating the SetpointScheduleTest and inheriting from it."""

  def setUp(self):
    super().setUp()
    self.schedule = setpoint_schedule.HolidaySchedule(year=2024)

  def test_calendar(self):
    self.assertIsInstance(self.schedule.cal, holiday.USFederalHolidayCalendar)
    self.assertEqual(self.schedule.cal.name, 'USFederalHolidayCalendar')

  def test_holidays(self):
    holiday_nums_this_year = {1, 15, 50, 148, 171, 186, 246, 288, 316, 333, 360}
    self.assertEqual(self.schedule.holidays, holiday_nums_this_year)

  def test_holidays_df(self):
    records = self.schedule.holidays_df.to_dict('records')
    expected_records = [
        {
            'date': pd.Timestamp('2024-01-01 00:00:00'),
            'holiday': "New Year's Day",
            'day_of_year': 1,
        },
        {
            'date': pd.Timestamp('2024-01-15 00:00:00'),
            'holiday': 'Birthday of Martin Luther King, Jr.',
            'day_of_year': 15,
        },
        {
            'date': pd.Timestamp('2024-02-19 00:00:00'),
            'holiday': "Washington's Birthday",
            'day_of_year': 50,
        },
        {
            'date': pd.Timestamp('2024-05-27 00:00:00'),
            'holiday': 'Memorial Day',
            'day_of_year': 148,
        },
        {
            'date': pd.Timestamp('2024-06-19 00:00:00'),
            'holiday': 'Juneteenth National Independence Day',
            'day_of_year': 171,
        },
        {
            'date': pd.Timestamp('2024-07-04 00:00:00'),
            'holiday': 'Independence Day',
            'day_of_year': 186,
        },
        {
            'date': pd.Timestamp('2024-09-02 00:00:00'),
            'holiday': 'Labor Day',
            'day_of_year': 246,
        },
        {
            'date': pd.Timestamp('2024-10-14 00:00:00'),
            'holiday': 'Columbus Day',
            'day_of_year': 288,
        },
        {
            'date': pd.Timestamp('2024-11-11 00:00:00'),
            'holiday': 'Veterans Day',
            'day_of_year': 316,
        },
        {
            'date': pd.Timestamp('2024-11-28 00:00:00'),
            'holiday': 'Thanksgiving Day',
            'day_of_year': 333,
        },
        {
            'date': pd.Timestamp('2024-12-25 00:00:00'),
            'holiday': 'Christmas Day',
            'day_of_year': 360,
        },
    ]
    self.assertEqual(records, expected_records)

    with self.subTest('correct holiday numbers'):
      holiday_nums = self.schedule.holidays_df['day_of_year'].unique().tolist()
      self.assertEqual(self.schedule.holidays, set(holiday_nums))


if __name__ == '__main__':
  absltest.main()
