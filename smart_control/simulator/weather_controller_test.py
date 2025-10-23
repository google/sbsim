"""Tests for weather_controller."""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_control.simulator import weather_controller


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

  def test_get_current_irradiance_weather_controller(self):
    """Test clearsky irradiance calculation for WeatherController."""
    low_temp = 273.15
    high_temp = 298.15
    # Mountain View, CA coordinates
    latitude = 37.4
    longitude = -122.1

    weather = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='America/Los_Angeles',
    )

    # Test at noon on a summer day
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='America/Los_Angeles')
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are present and positive
    self.assertIn('ghi', irrad)
    self.assertIn('dni', irrad)
    self.assertIn('dhi', irrad)
    self.assertGreater(irrad['ghi'], 0)
    self.assertGreater(irrad['dni'], 0)
    self.assertGreater(irrad['dhi'], 0)
    # At noon in summer, GHI should be substantial (> 500 W/m2)
    self.assertGreater(irrad['ghi'], 500)
    self.assertEqual(round(irrad['ghi']), 934)
    self.assertEqual(round(irrad['dni']), 872)
    self.assertEqual(round(irrad['dhi']), 121)

  def test_get_current_irradiance_no_location(self):
    """Test that irradiance calculation raises error without location."""
    low_temp = 273.15
    high_temp = 298.15

    weather = weather_controller.WeatherController(low_temp, high_temp)

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')

    with self.assertRaises(ValueError):
      weather.get_current_irradiance(timestamp)

  def test_get_irradiance_poa_weather_controller(self):
    """Test POA irradiance calculation for WeatherController."""
    low_temp = 273.15
    high_temp = 298.15
    latitude = 37.4
    longitude = -122.1

    weather = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    surface_tilt = 30.0  # 30 degrees tilt
    surface_azimuth = 180.0  # South-facing

    poa = weather.get_irradiance_poa(timestamp, surface_tilt, surface_azimuth)

    # POA should be positive at noon
    self.assertGreater(poa, 0)
    # POA should be reasonable (between 0 and ~1200 W/m2)
    self.assertLess(poa, 1200)

  def test_get_current_cloud_cover(self):
    """Test cloud cover interpolation from weather data."""
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    latitude = 37.4
    longitude = -122.1
    controller = weather_controller.ReplayWeatherController(
        data_path, 10.0, latitude=latitude, longitude=longitude, tz='UTC'
    )

    # Test at a time with known cloud cover (0% at midnight)
    timestamp = pd.Timestamp('2023-07-01 00:00:00+00:00')
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    self.assertEqual(cloud_cover, 0.0)
    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    self.assertEqual(cloud_cover, 100.0)

  def test_get_current_irradiance_replay_controller(self):
    """Test irradiance calculation with cloud cover for
    ReplayWeatherController.

    """
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    latitude = 37.4
    longitude = -122.1
    controller = weather_controller.ReplayWeatherController(
        data_path,
        10.0,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
        irradiance_method='campbell_norman',
    )

    # Test at noon
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    irrad = controller.get_current_irradiance(timestamp)

    # Check that all components are present and non-negative
    self.assertIn('ghi', irrad)
    self.assertIn('dni', irrad)
    self.assertIn('dhi', irrad)
    self.assertGreaterEqual(irrad['ghi'], 0)
    self.assertGreaterEqual(irrad['dni'], 0)
    self.assertGreaterEqual(irrad['dhi'], 0)
    self.assertEqual(round(irrad['ghi']), 523.0)
    self.assertEqual(round(irrad['dni']), 235.0)
    self.assertEqual(round(irrad['dhi']), 304.0)

  def test_get_irradiance_poa_replay_controller(self):
    """Test POA irradiance calculation for ReplayWeatherController."""
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    latitude = 37.4
    longitude = -122.1
    controller = weather_controller.ReplayWeatherController(
        data_path, 10.0, latitude=latitude, longitude=longitude, tz='UTC'
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    surface_tilt = 30.0
    surface_azimuth = 180.0

    poa = controller.get_irradiance_poa(
        timestamp, surface_tilt, surface_azimuth
    )

    # POA should be non-negative
    self.assertGreaterEqual(poa, 0)

  def test_get_sky_temperature_weather_controller(self):
    """Test sky temperature calculation for WeatherController."""
    low_temp = 273.15
    high_temp = 298.15

    weather = weather_controller.WeatherController(
        low_temp, high_temp, dewpoint_depression=5.0
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    temp_sky_k = weather.get_current_sky_temperature(timestamp)

    # Sky temperature should be in Kelvin and reasonable
    self.assertGreater(temp_sky_k, 200)  # Above absolute zero
    self.assertLess(temp_sky_k, 350)  # Below very hot temps
    # Sky temperature should be less than or equal to dry bulb temp
    temp_k = weather.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)

  def test_get_sky_temperature_replay_controller(self):
    """Test sky temperature calculation for ReplayWeatherController."""
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    controller = weather_controller.ReplayWeatherController(
        data_path, 10.0, tz='UTC'
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    temp_sky_k = controller.get_current_sky_temperature(timestamp)

    # Sky temperature should be in Kelvin and reasonable
    self.assertGreater(temp_sky_k, 200)
    self.assertLess(temp_sky_k, 350)
    # Sky temperature should be less than or equal to dry bulb temp
    temp_k = controller.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)


if __name__ == '__main__':
  absltest.main()
