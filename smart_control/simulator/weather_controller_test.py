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
from smart_control.simulator import weather_controller


# pylint: disable=g-long-lambda
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

  def test_replay_weather_controller(self):

    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    controller = weather_controller.ReplayWeatherController(data_path, 10.0)

    temp = controller.get_current_temp(
        pd.Timestamp('2023-07-01 03:00:01+00:00')
    )

    self.assertAlmostEqual(temp, 298.1500, places=5)

  def test_replay_weather_controller_raises_error_before_range(self):
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    controller = weather_controller.ReplayWeatherController(data_path, 10.0)

    weather_fn = lambda: controller.get_current_temp(
        pd.Timestamp('2023-05-01 03:00:01+00:00')
    )

    self.assertRaises(ValueError, weather_fn)

  def test_replay_weather_controller_raises_error_after_range(self):
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    controller = weather_controller.ReplayWeatherController(data_path, 10.0)

    weather_fn = lambda: controller.get_current_temp(
        pd.Timestamp('2023-12-01 03:00:01+00:00')
    )

    self.assertRaises(ValueError, weather_fn)


if __name__ == '__main__':
  absltest.main()
