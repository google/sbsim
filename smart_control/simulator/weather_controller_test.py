"""Tests for weather_controller.

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

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_buildings.smart_control.simulator import weather_controller


# pylint: disable=g-long-lambda, unnecessary-lambda-assignment # TODO: consider using named functions instead
class WeatherControllerTest(parameterized.TestCase):

  def test_init_attributes(self):
    low_temp = 40.5
    high_temp = 62.5
    special_days = {110: (30, 70)}
    convection_coefficient = 1.0

    weather = weather_controller.WeatherController(
        low_temp, high_temp, special_days, convection_coefficient
    )

    self.assertEqual(weather.default_low_temp, low_temp)
    self.assertEqual(weather.default_high_temp, high_temp)
    self.assertEqual(weather.special_days, special_days)
    self.assertEqual(weather.convection_coefficient, convection_coefficient)

  def test_default_attributes(self):
    low_temp = 40.5
    high_temp = 62.5

    default_convection_coefficient = 12.0

    weather = weather_controller.WeatherController(low_temp, high_temp)

    self.assertEqual(weather.special_days, {})
    self.assertEqual(
        weather.convection_coefficient, default_convection_coefficient
    )

  def test_init_raises_error_default_temp(self):
    low_temp = 40.5
    high_temp = 15.0

    create_weather_fn = lambda: weather_controller.WeatherController(
        low_temp, high_temp
    )

    self.assertRaises(ValueError, create_weather_fn)

  def test_init_raises_error_special_day_temp(self):
    low_temp = 40.5
    high_temp = 62.5
    special_days = {329: (60.0, 40.0)}

    create_weather_fn = lambda: weather_controller.WeatherController(
        low_temp, high_temp, special_days
    )

    self.assertRaises(ValueError, create_weather_fn)

  @parameterized.named_parameters(
      ('min_rad', 0.0, -math.pi / 2),
      ('max_rad', 3600.0 * 24, 3 * math.pi / 2),
      ('mid_rad', 3600 * 12, math.pi / 2),
  )
  def test_seconds_to_rad(self, seconds, expected):
    low_temp = 40.5
    high_temp = 62.5
    special_days = {110: (30, 70)}

    weather = weather_controller.WeatherController(
        low_temp, high_temp, special_days
    )

    rads = weather.seconds_to_rads(seconds)

    self.assertEqual(rads, expected)

  @parameterized.named_parameters(
      ('min_temp_default', 4, 0, 40.5),
      ('max_temp_default', 4, 12 * 3600, 62.5),
      ('mid_temp_default', 4, 6 * 3600, 51.5),
      ('min_temp_special', 110, 0, 30.0),
      ('max_temp_special', 110, 12 * 3600, 70.0),
      ('mid_temp_special', 110, 6 * 3600, 50.0),
      ('transition_to_special_day', 109, 18 * 3600, 46.25),
      ('transition_from_special_day', 110, 18 * 3600, 55.25),
  )
  def test_get_current_temp(self, day_of_year, seconds_in_day, expected):
    low_temp = 40.5
    high_temp = 62.5
    special_days = {110: (30, 70)}
    weather = weather_controller.WeatherController(
        low_temp, high_temp, special_days
    )
    beginning_of_year = pd.Timestamp('2021-01-01')
    specified_day = beginning_of_year + pd.Timedelta(
        day_of_year - 1, unit='day'
    )  # Jan 1st is day 1
    timestamp = specified_day + pd.Timedelta(seconds_in_day, unit='seconds')

    temp = weather.get_current_temp(timestamp)

    self.assertEqual(temp, expected)

  def test_get_air_convection_coefficient(self):
    low_temp = 40.5
    high_temp = 62.5

    expected_convection_coefficient = 12.0

    weather = weather_controller.WeatherController(low_temp, high_temp)

    convection_coefficient = weather.get_air_convection_coefficient(
        pd.Timestamp('2012-12-21')
    )

    self.assertEqual(convection_coefficient, expected_convection_coefficient)


class ReplayWeatherControllerTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    self.controller = weather_controller.ReplayWeatherController(
        local_weather_path=data_path,
        convection_coefficient=10.0
    )

  def test_replay_weather_controller(self):
    temp = self.controller.get_current_temp(
        pd.Timestamp('2023-07-01 03:00:01+00:00')
    )
    self.assertAlmostEqual(temp, 298.1500, places=5)

  def test_replay_weather_controller_raises_error_before_range(self):
    weather_fn = lambda: self.controller.get_current_temp(
        pd.Timestamp('2023-05-01 03:00:01+00:00')
    )
    self.assertRaises(ValueError, weather_fn)

  def test_replay_weather_controller_raises_error_after_range(self):
    weather_fn = lambda: self.controller.get_current_temp(
        pd.Timestamp('2023-12-01 03:00:01+00:00')
    )
    self.assertRaises(ValueError, weather_fn)


class MoffettReplayWeatherControllerTest(parameterized.TestCase):
  """Tests for ReplayWeatherController using real weather data."""

  def setUp(self):
    super().setUp()
    self.controller = weather_controller.ReplayWeatherController()

  def test_weather_df(self):
    self.assertIsInstance(self.controller.weather_df, pd.DataFrame)
    self.assertEqual(self.controller.weather_df.shape, (3462, 15))

    expected_columns = [
        'Time', 'StationName', 'StationId', 'Location', 'TempC', 'DewPointC',
        'BarometerMbar', 'Rain', 'RainTotal', 'WindspeedKmph',
        'WindDirection', 'SkyCoverage', 'VisibilityKm', 'Humidity', 'TempF'
    ]
    self.assertCountEqual(
        self.controller.weather_df.columns.tolist(),
        expected_columns,
    )

  def test_time_range(self):
    min_time = pd.Timestamp('2023-06-30 17:00:00+00:00')
    max_time = pd.Timestamp('2023-11-22 16:00:00+00:00')

    self.assertEqual(self.controller.min_time, min_time)
    self.assertEqual(self.controller.max_time, max_time)

  def test_times_in_seconds(self):
    self.assertIsInstance(self.controller.times_in_seconds, pd.Index)
    self.assertEqual(self.controller.times_in_seconds.shape, (3462,))

    self.assertEqual(min(self.controller.times_in_seconds), 1688144400.0)
    self.assertEqual(max(self.controller.times_in_seconds), 1700668800.0)

  def test_get_temp_timezones(self):
    with self.subTest('when timestamp is timezone aware'):
      timestamp = pd.Timestamp('2023-07-01 10:00:00+00:00')
      self.assertEqual(timestamp.tzname(), 'UTC')

      temp = self.controller.get_current_temp(timestamp)
      self.assertEqual(temp, 289.15)

    with self.subTest('when timestamp is timezone naive'):
      timestamp = pd.Timestamp('2023-07-01 10:00:00')
      self.assertIsNone(timestamp.tzname())

      temp = self.controller.get_current_temp(timestamp)
      self.assertEqual(temp, 289.15)

  def test_interpolation(self):
    timestamp = pd.Timestamp('2023-07-01 03:00:01+00:00')

    with self.subTest('current_temp'):
      temp_k = self.controller.get_current_temp(timestamp)
      self.assertAlmostEqual(temp_k, 294.1497, places=4)

    with self.subTest('current_humidity'):
      humidity = self.controller.get_current_humidity(timestamp)
      self.assertAlmostEqual(humidity, 65.0, places=5)


if __name__ == '__main__':
  absltest.main()
