"""Tests for setpoint_schedule.

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
from absl.testing import parameterized
import pandas as pd
import pytz
from smart_control.simulator import setpoint_schedule


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


if __name__ == '__main__':
  absltest.main()
