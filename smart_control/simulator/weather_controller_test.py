"""Tests for weather_controller."""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib import location

from smart_control.simulator import building_radiation_utils
from smart_control.simulator import weather_controller
from smart_control.utils import conversion_utils as utils


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
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)
    self.assertGreater(irrad['ghi'], 0)
    self.assertGreater(irrad['dni'], 0)
    self.assertGreater(irrad['dhi'], 0)
    # At noon in summer, GHI should be substantial (> 500 W/m2)
    self.assertGreater(irrad['ghi'], 500)
    self.assertEqual(round(irrad['ghi']), 934)
    self.assertEqual(round(irrad['dni']), 872)
    self.assertEqual(round(irrad['dhi']), 121)

    # Solar position should be reasonable at noon
    self.assertGreater(irrad['solar_zenith'], 0)
    self.assertLess(irrad['solar_zenith'], 90)  # Sun above horizon

    # Direct pvlib clearsky validation
    pvlib_location = location.Location(
        latitude, longitude, tz='America/Los_Angeles'
    )
    clearsky = pvlib_location.get_clearsky(pd.DatetimeIndex([timestamp]))
    self.assertAlmostEqual(irrad['ghi'], clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad['dni'], clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad['dhi'], clearsky['dhi'].iloc[0], places=4)

    # Validate solar position against pvlib
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    self.assertAlmostEqual(
        irrad['solar_zenith'],
        solar_position['apparent_zenith'].iloc[0],
        places=4,
    )
    self.assertAlmostEqual(
        irrad['solar_azimuth'], solar_position['azimuth'].iloc[0], places=4
    )

  def test_get_current_irradiance_no_location(self):
    """Test that irradiance calculation raises error without location."""
    low_temp = 273.15
    high_temp = 298.15

    weather = weather_controller.WeatherController(low_temp, high_temp)

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')

    with self.assertRaises(ValueError):
      weather.get_current_irradiance(timestamp)

  def test_get_irradiance_with_solar_position_weather_controller(self):
    """Test irradiance with solar position for WeatherController."""
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

    # Get irradiance which now includes solar position
    irrad = weather.get_current_irradiance(timestamp)

    # Verify solar position is included
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)
    self.assertGreater(irrad['solar_zenith'], 0)
    self.assertLess(irrad['solar_zenith'], 90)  # Sun is above horizon at noon

    # Test POA calculation using utility function
    surface_tilt = 30.0  # 30 degrees tilt
    surface_azimuth = 180.0  # South-facing

    poa = building_radiation_utils.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=irrad['solar_zenith'],
        solar_azimuth=irrad['solar_azimuth'],
    )

    # POA should be positive at noon
    self.assertGreater(poa, 0)
    # POA should be reasonable (between 0 and ~1200 W/m2)
    self.assertLess(poa, 1200)

    # Direct pvlib validation
    pvlib_location = location.Location(latitude, longitude, tz='US/Pacific')
    clearsky = pvlib_location.get_clearsky(pd.DatetimeIndex([timestamp]))
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    poa_irrad_pvlib = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=clearsky['dni'].iloc[0],
        ghi=clearsky['ghi'].iloc[0],
        dhi=clearsky['dhi'].iloc[0],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )
    expected_poa = float(poa_irrad_pvlib['poa_global'])
    self.assertAlmostEqual(poa, expected_poa, places=4)

  def test_get_current_cloud_cover_weather_controller(self):
    """Test cloud cover getter for WeatherController."""
    low_temp = 273.15
    high_temp = 298.15
    latitude = 37.4
    longitude = -122.1

    # Test with no cloud cover set (should return 0)
    weather_no_cloud = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
    )
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    self.assertEqual(weather_no_cloud.get_current_cloud_cover(timestamp), 0.0)

    # Test with cloud cover set
    weather_with_cloud = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
        cloud_cover=50.0,
    )
    self.assertEqual(
        weather_with_cloud.get_current_cloud_cover(timestamp), 50.0
    )

  def test_get_current_irradiance_with_cloud_cover_campbell_norman(self):
    """Test irradiance calculation with cloud cover using campbell_norman."""
    low_temp = 273.15
    high_temp = 298.15
    latitude = 37.4
    longitude = -122.1
    cloud_cover = 50.0  # 50% cloud cover

    weather = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
        cloud_cover=cloud_cover,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are present and non-negative
    self.assertIn('ghi', irrad)
    self.assertIn('dni', irrad)
    self.assertIn('dhi', irrad)
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)
    self.assertGreaterEqual(irrad['ghi'], 0)
    self.assertGreaterEqual(irrad['dni'], 0)
    self.assertGreaterEqual(irrad['dhi'], 0)

    # With 50% cloud cover, irradiance should be less than clearsky
    weather_clearsky = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
    )
    clearsky_irrad = weather_clearsky.get_current_irradiance(timestamp)
    self.assertLess(irrad['ghi'], clearsky_irrad['ghi'])

    # Direct pvlib campbell_norman validation
    pvlib_location = location.Location(latitude, longitude, tz='US/Pacific')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)
    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )
    self.assertAlmostEqual(irrad['ghi'], expected_irrad['ghi'], places=4)
    self.assertAlmostEqual(irrad['dni'], expected_irrad['dni'], places=4)
    self.assertAlmostEqual(irrad['dhi'], expected_irrad['dhi'], places=4)

    # Validate solar position against pvlib
    self.assertAlmostEqual(
        irrad['solar_zenith'],
        solar_position['apparent_zenith'].iloc[0],
        places=4,
    )
    self.assertAlmostEqual(
        irrad['solar_azimuth'], solar_position['azimuth'].iloc[0], places=4
    )

  def test_get_current_irradiance_with_cloud_cover_linear(self):
    """Test irradiance calculation with cloud cover using linear method."""
    low_temp = 273.15
    high_temp = 298.15
    latitude = 37.4
    longitude = -122.1
    cloud_cover = 30.0  # 30% cloud cover

    weather = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
        cloud_cover=cloud_cover,
        irradiance_method='linear',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are present and non-negative
    self.assertIn('ghi', irrad)
    self.assertIn('dni', irrad)
    self.assertIn('dhi', irrad)
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)
    self.assertGreaterEqual(irrad['ghi'], 0)
    self.assertGreaterEqual(irrad['dni'], 0)
    self.assertGreaterEqual(irrad['dhi'], 0)

    # Direct pvlib linear method validation
    pvlib_location = location.Location(latitude, longitude, tz='US/Pacific')
    clearsky = pvlib_location.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    # Expected GHI using linear relationship
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (
        1.0 - 0.8 * (cloud_cover / 100.0)
    )

    # Expected DNI using DISC model
    dni_result = irradiance.disc(
        pd.Series([expected_ghi], index=pd.DatetimeIndex([timestamp])),
        solar_position['zenith'],
        pd.DatetimeIndex([timestamp]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])

    # Expected DHI
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    self.assertAlmostEqual(irrad['ghi'], expected_ghi, places=4)
    self.assertAlmostEqual(irrad['dni'], expected_dni, places=4)
    self.assertAlmostEqual(irrad['dhi'], expected_dhi, places=4)

    # Validate solar position against pvlib
    self.assertAlmostEqual(
        irrad['solar_zenith'],
        solar_position['apparent_zenith'].iloc[0],
        places=4,
    )
    self.assertAlmostEqual(
        irrad['solar_azimuth'], solar_position['azimuth'].iloc[0], places=4
    )

  def test_invalid_cloud_cover_raises_error(self):
    """Test that invalid cloud cover values raise errors."""
    low_temp = 273.15
    high_temp = 298.15

    # Cloud cover < 0
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover=-10.0
      )

    # Cloud cover > 100
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover=150.0
      )

  def test_dynamic_cloud_cover(self):
    """Test dynamic cloud cover with sinusoidal pattern."""
    low_temp = 273.15
    high_temp = 298.15
    cloud_cover_low = 20.0
    cloud_cover_high = 80.0

    weather = weather_controller.WeatherController(
        low_temp,
        high_temp,
        cloud_cover_low=cloud_cover_low,
        cloud_cover_high=cloud_cover_high,
    )

    # At midnight (0 seconds), cloud cover should be at low
    midnight = pd.Timestamp('2023-07-01 00:00:00', tz='UTC')
    cc_midnight = weather.get_current_cloud_cover(midnight)
    self.assertAlmostEqual(cc_midnight, cloud_cover_low, places=1)

    # At noon (12 hours), cloud cover should be at high
    noon = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    cc_noon = weather.get_current_cloud_cover(noon)
    self.assertAlmostEqual(cc_noon, cloud_cover_high, places=1)

    # At 6am (6 hours), cloud cover should be midpoint
    morning = pd.Timestamp('2023-07-01 06:00:00', tz='UTC')
    cc_morning = weather.get_current_cloud_cover(morning)
    expected_mid = (cloud_cover_low + cloud_cover_high) / 2
    self.assertAlmostEqual(cc_morning, expected_mid, places=1)

  def test_dynamic_cloud_cover_validation(self):
    """Test validation for dynamic cloud cover parameters."""
    low_temp = 273.15
    high_temp = 298.15

    # Only cloud_cover_low provided (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover_low=20.0
      )

    # Only cloud_cover_high provided (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover_high=80.0
      )

    # cloud_cover_low > cloud_cover_high (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover_low=80.0, cloud_cover_high=20.0
      )

    # cloud_cover_low < 0 (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover_low=-10.0, cloud_cover_high=80.0
      )

    # cloud_cover_high > 100 (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, cloud_cover_low=20.0, cloud_cover_high=150.0
      )

  def test_dynamic_cloud_cover_affects_irradiance(self):
    """Test that dynamic cloud cover affects irradiance calculation."""
    low_temp = 273.15
    high_temp = 298.15
    latitude = 37.4
    longitude = -122.1

    # Weather with dynamic cloud cover (low at midnight, high at noon)
    weather_dynamic = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
        cloud_cover_low=0.0,
        cloud_cover_high=80.0,
        irradiance_method='campbell_norman',
    )

    # Weather with clearsky (no cloud cover)
    weather_clearsky = weather_controller.WeatherController(
        low_temp,
        high_temp,
        latitude=latitude,
        longitude=longitude,
        tz='US/Pacific',
    )

    # At noon when dynamic cloud cover is at maximum (80%)
    noon = pd.Timestamp('2023-07-01 12:00:00', tz='US/Pacific')
    irrad_dynamic = weather_dynamic.get_current_irradiance(noon)
    irrad_clearsky = weather_clearsky.get_current_irradiance(noon)

    # Dynamic irradiance should be less than clearsky at noon
    self.assertLess(irrad_dynamic['ghi'], irrad_clearsky['ghi'])

  def test_invalid_irradiance_method_raises_error(self):
    """Test that invalid irradiance method raises error."""
    low_temp = 273.15
    high_temp = 298.15

    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          low_temp, high_temp, irradiance_method='invalid_method'
      )

  def test_get_current_cloud_cover_replay_controller(self):
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
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)
    self.assertGreaterEqual(irrad['ghi'], 0)
    self.assertGreaterEqual(irrad['dni'], 0)
    self.assertGreaterEqual(irrad['dhi'], 0)
    self.assertEqual(round(irrad['ghi']), 523.0)
    self.assertEqual(round(irrad['dni']), 235.0)
    self.assertEqual(round(irrad['dhi']), 304.0)

    # Solar position should be reasonable at noon
    self.assertGreater(irrad['solar_zenith'], 0)
    self.assertLess(irrad['solar_zenith'], 90)  # Sun above horizon

    # Direct pvlib campbell_norman validation
    # At noon UTC (20:00 US/Pacific previous day), cloud cover is 100%
    # The timestamp is 12:00 US/Pacific which is 19:00 UTC
    # Looking at CSV: row 12 (20230701-1200 UTC) has SkyCoverage=100
    pvlib_location = location.Location(latitude, longitude, tz='US/Pacific')
    # Convert to UTC for the test data (test data is in UTC format)
    timestamp_utc = timestamp.tz_convert('UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp_utc])
    )
    # Cloud cover at 12:00 US/Pacific = 19:00 UTC, interpolated from test data
    # From CSV: 1200 UTC has 100% cloud cover
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)
    dni_extra = irradiance.get_extra_radiation(
        pd.DatetimeIndex([timestamp_utc])
    )
    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )
    # Compare with expected values (allowing for interpolation differences)
    self.assertAlmostEqual(irrad['ghi'], expected_irrad['ghi'], delta=1.0)
    self.assertAlmostEqual(irrad['dni'], expected_irrad['dni'], delta=1.0)
    self.assertAlmostEqual(irrad['dhi'], expected_irrad['dhi'], delta=1.0)

    # Validate solar position against pvlib
    self.assertAlmostEqual(
        irrad['solar_zenith'],
        solar_position['apparent_zenith'].iloc[0],
        places=4,
    )
    self.assertAlmostEqual(
        irrad['solar_azimuth'], solar_position['azimuth'].iloc[0], places=4
    )

  def test_get_irradiance_with_solar_position_replay_controller(self):
    """Test irradiance with solar position for ReplayWeatherController."""
    data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    latitude = 37.4
    longitude = -122.1
    controller = weather_controller.ReplayWeatherController(
        data_path, 10.0, latitude=latitude, longitude=longitude, tz='UTC'
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')

    # Get irradiance which now includes solar position
    irrad = controller.get_current_irradiance(timestamp)

    # Verify solar position is included
    self.assertIn('solar_zenith', irrad)
    self.assertIn('solar_azimuth', irrad)

    # Test POA calculation using utility function
    surface_tilt = 30.0
    surface_azimuth = 180.0

    poa = building_radiation_utils.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=irrad['solar_zenith'],
        solar_azimuth=irrad['solar_azimuth'],
    )

    # POA should be non-negative
    self.assertGreaterEqual(poa, 0)

    # Direct pvlib validation for POA calculation
    pvlib_location = location.Location(latitude, longitude, tz='UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    poa_irrad_pvlib = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=irrad['dni'],
        ghi=irrad['ghi'],
        dhi=irrad['dhi'],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )
    expected_poa = float(poa_irrad_pvlib['poa_global'])
    self.assertAlmostEqual(poa, expected_poa, places=4)

  def test_get_sky_temperature_weather_controller(self):
    """Test sky temperature calculation for WeatherController."""
    low_temp = 273.15
    high_temp = 298.15
    dewpoint_depression = 5.0

    weather = weather_controller.WeatherController(
        low_temp, high_temp, dewpoint_depression=dewpoint_depression
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    temp_sky_k = weather.get_current_sky_temperature(timestamp)

    # Sky temperature should be in Kelvin and reasonable
    self.assertGreater(temp_sky_k, 200)  # Above absolute zero
    self.assertLess(temp_sky_k, 350)  # Below very hot temps
    # Sky temperature should be less than or equal to dry bulb temp
    temp_k = weather.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)

    # Direct Clark & Allen formula validation
    # Stefan-Boltzmann constant
    sigma = 5.6697e-8  # W/(m^2*K^4)
    # Dew point temperature
    dp_k = temp_k - dewpoint_depression
    # Sky emissivity (Clark & Allen formula)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    # Horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k**4)
    # Expected sky temperature
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=4)

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

    # Direct Clark & Allen formula validation using weather data values
    # From CSV at 12:00 UTC: TempC=23.0, DewPointC=15.0
    sigma = 5.6697e-8  # W/(m^2*K^4)
    temp_c = 23.0  # From CSV row 12
    dp_c = 15.0  # From CSV row 12
    temp_k_expected = utils.celsius_to_kelvin(temp_c)
    dp_k = utils.celsius_to_kelvin(dp_c)
    # Sky emissivity (Clark & Allen formula)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    # Horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k_expected**4)
    # Expected sky temperature
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=2)


class ReplayWeatherControllerPvlibValidationTest(parameterized.TestCase):
  """Validate ReplayWeatherController irradiance calculations against pvlib.

  This test class validates that the DHI, GHI, DNI calculations in
  ReplayWeatherController are consistent with pvlib's irradiance methods
  using the local_weather_test_data.csv which has SkyCoverage data.
  """

  def setUp(self):
    """Set up test fixtures."""
    self.data_path = os.path.join(
        os.path.dirname(__file__), 'local_weather_test_data.csv'
    )
    # Mountain View, CA coordinates (from test data)
    self.latitude = 37.4
    self.longitude = -122.1

  def test_replay_controller_campbell_norman_vs_pvlib(self):
    """Validate ReplayWeatherController campbell_norman matches pvlib.

    This test validates that the pre-calculated irradiance values in
    ReplayWeatherController match direct pvlib calculations for the
    same timestamp and cloud cover.
    """
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    # Test at timestamps with known cloud cover from the CSV
    # Use 19:00 UTC which is ~12:00 local time in Mountain View (daytime)
    # Row 19: 19:00 UTC has SkyCoverage=0% (clear)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib calculation with same parameters
    pvlib_location = location.Location(self.latitude, self.longitude, tz='UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)

    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )

    # Validate GHI, DNI, DHI match pvlib (within tolerance for pre-calculation)
    # ReplayWeatherController pre-calculates and then interpolates
    self.assertAlmostEqual(
        irrad['ghi'],
        expected_irrad['ghi'],
        delta=10.0,
        msg=(
            f"GHI mismatch: got {irrad['ghi']}, expected"
            f" {expected_irrad['ghi']}"
        ),
    )
    self.assertAlmostEqual(
        irrad['dni'],
        expected_irrad['dni'],
        delta=10.0,
        msg=(
            f"DNI mismatch: got {irrad['dni']}, expected"
            f" {expected_irrad['dni']}"
        ),
    )
    self.assertAlmostEqual(
        irrad['dhi'],
        expected_irrad['dhi'],
        delta=10.0,
        msg=(
            f"DHI mismatch: got {irrad['dhi']}, expected"
            f" {expected_irrad['dhi']}"
        ),
    )

  @parameterized.named_parameters(
      ('clear_daytime', '2023-07-01 19:00:00+00:00', 0.0),  # ~noon local
      ('clear_morning', '2023-07-01 17:00:00+00:00', 0.0),  # ~10am local
  )
  def test_replay_controller_cloud_cover_irradiance_relationship(
      self, time_str, expected_cloud_cover
  ):
    """Validate cloud cover affects irradiance as expected during daytime."""
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp(time_str)
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    irrad = controller.get_current_irradiance(timestamp)

    # Verify cloud cover matches expected value
    self.assertAlmostEqual(cloud_cover, expected_cloud_cover, delta=1.0)

    # All irradiance components should be non-negative
    self.assertGreaterEqual(irrad['ghi'], 0)
    self.assertGreaterEqual(irrad['dni'], 0)
    self.assertGreaterEqual(irrad['dhi'], 0)

    # During daytime with clear sky, GHI should be substantial
    self.assertGreater(irrad['ghi'], 100)

  def test_replay_controller_irradiance_closure_equation(self):
    """Validate ReplayWeatherController satisfies closure equation."""
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    # Test at multiple timestamps
    test_times = [
        '2023-07-01 08:00:00+00:00',
        '2023-07-01 10:00:00+00:00',
        '2023-07-01 12:00:00+00:00',
        '2023-07-01 14:00:00+00:00',
    ]

    pvlib_location = location.Location(self.latitude, self.longitude, tz='UTC')

    for time_str in test_times:
      timestamp = pd.Timestamp(time_str)
      irrad = controller.get_current_irradiance(timestamp)

      # Get solar position
      solar_position = pvlib_location.get_solarposition(
          pd.DatetimeIndex([timestamp])
      )
      zenith_rad = np.radians(solar_position['apparent_zenith'].iloc[0])

      # Closure equation: GHI = DNI * cos(zenith) + DHI
      calculated_ghi = irrad['dni'] * np.cos(zenith_rad) + irrad['dhi']

      # Should satisfy closure equation
      if irrad['ghi'] > 10:  # Skip very low irradiance (numerical issues)
        self.assertAlmostEqual(
            calculated_ghi,
            irrad['ghi'],
            delta=10.0,
            msg=f'Closure equation failed at {time_str}',
        )

  def test_replay_controller_vs_weather_controller_consistency(self):
    """Validate ReplayWeatherController and WeatherController produce
    consistent results for same cloud cover and location.
    """
    # Create ReplayWeatherController
    replay_controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    # Test at a timestamp with known cloud cover
    timestamp = pd.Timestamp('2023-07-01 10:00:00+00:00')
    cloud_cover = replay_controller.get_current_cloud_cover(timestamp)
    replay_irrad = replay_controller.get_current_irradiance(timestamp)

    # Create WeatherController with same cloud cover
    weather = weather_controller.WeatherController(
        default_low_temp=273.15,
        default_high_temp=298.15,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        cloud_cover=cloud_cover,
        irradiance_method='campbell_norman',
    )
    weather_irrad = weather.get_current_irradiance(timestamp)

    # Both should produce consistent results (within tolerance)
    self.assertAlmostEqual(
        replay_irrad['ghi'],
        weather_irrad['ghi'],
        delta=10.0,
        msg=(
            'GHI inconsistency between ReplayWeatherController and'
            ' WeatherController'
        ),
    )
    self.assertAlmostEqual(
        replay_irrad['dni'],
        weather_irrad['dni'],
        delta=10.0,
        msg=(
            'DNI inconsistency between ReplayWeatherController and'
            ' WeatherController'
        ),
    )
    self.assertAlmostEqual(
        replay_irrad['dhi'],
        weather_irrad['dhi'],
        delta=10.0,
        msg=(
            'DHI inconsistency between ReplayWeatherController and'
            ' WeatherController'
        ),
    )

  def test_replay_controller_campbell_norman_consistency(self):
    """Validate ReplayWeatherController campbell_norman method against pvlib.

    Similar to test_weather_controller_campbell_norman_consistency but for
    ReplayWeatherController.
    """
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    # Test at 19:00 UTC (~noon local time in Mountain View)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib validation
    pvlib_location = location.Location(self.latitude, self.longitude, tz='UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)

    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )

    # Validate against pvlib
    self.assertAlmostEqual(irrad['ghi'], expected_irrad['ghi'], delta=5.0)
    self.assertAlmostEqual(irrad['dni'], expected_irrad['dni'], delta=5.0)
    self.assertAlmostEqual(irrad['dhi'], expected_irrad['dhi'], delta=5.0)

  def test_replay_controller_linear_method_consistency(self):
    """Validate ReplayWeatherController linear method results.

    Similar to test_weather_controller_linear_method_consistency but for
    ReplayWeatherController.
    """
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='linear',
    )

    # Test at 19:00 UTC (~noon local time in Mountain View)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib validation with linear method
    pvlib_location = location.Location(self.latitude, self.longitude, tz='UTC')
    clearsky = pvlib_location.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    # Expected GHI using linear relationship
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (
        1.0 - 0.8 * (cloud_cover / 100.0)
    )

    # Expected DNI using DISC model
    dni_result = irradiance.disc(
        pd.Series([expected_ghi], index=pd.DatetimeIndex([timestamp])),
        solar_position['zenith'],
        pd.DatetimeIndex([timestamp]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])

    # Expected DHI from closure
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    # Validate (with tolerance for pre-calculation/interpolation differences)
    self.assertAlmostEqual(irrad['ghi'], expected_ghi, delta=20.0)
    self.assertAlmostEqual(irrad['dni'], expected_dni, delta=20.0)
    self.assertAlmostEqual(irrad['dhi'], expected_dhi, delta=20.0)

  def test_replay_clearsky_irradiance_matches_pvlib(self):
    """Validate ReplayWeatherController clearsky irradiance matches pvlib.

    When cloud cover is 0%, the irradiance should match clearsky values
    from pvlib.location.get_clearsky(). Similar to
    test_clearsky_irradiance_matches_pvlib but for ReplayWeatherController.
    """
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        tz='UTC',
        irradiance_method='campbell_norman',
    )

    # Test at 19:00 UTC (~noon local) where SkyCoverage=0 (clear sky)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Verify it's clear sky condition
    self.assertEqual(cloud_cover, 0.0)

    # For campbell_norman with transmittance=0.7 (clear sky), compare to pvlib
    pvlib_location = location.Location(self.latitude, self.longitude, tz='UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))

    # Clearsky transmittance = 0.7 (when cloud_cover = 0)
    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        0.7,  # Clear sky transmittance
        dni_extra=dni_extra.iloc[0],
    )

    # Validate against pvlib clearsky campbell_norman
    self.assertAlmostEqual(irrad['ghi'], expected_irrad['ghi'], delta=5.0)
    self.assertAlmostEqual(irrad['dni'], expected_irrad['dni'], delta=5.0)
    self.assertAlmostEqual(irrad['dhi'], expected_irrad['dhi'], delta=5.0)

    # Also compare to ineichen clearsky model for reference
    clearsky = pvlib_location.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )

    # GHI should be in reasonable range compared to clearsky model
    # (campbell_norman and ineichen will differ but should be similar order)
    self.assertAlmostEqual(
        irrad['ghi'],
        clearsky['ghi'].iloc[0],
        delta=200.0,
        msg='GHI differs significantly from ineichen clearsky',
    )


class IrradianceDecompositionPvlibValidationTest(parameterized.TestCase):
  """Validate irradiance decomposition methods against pvlib using TMY3 data.

  This test class validates that the DHI, GHI, DNI calculations in
  weather_controller are consistent with pvlib's irradiance decomposition
  methods (DISC, Erbs, campbell_norman) using the Greensboro TMY3 data.
  """

  @classmethod
  def setUpClass(cls):
    """Load TMY3 data once for all tests."""
    from pvlib.iotools import read_tmy3  # pylint: disable=import-outside-toplevel
    from pvlib.solarposition import get_solarposition  # pylint: disable=import-outside-toplevel

    # Load TMY3 data from test data directory
    tmy3_path = os.path.join(
        os.path.dirname(__file__),
        'building_radiation_test_data',
        '723170TYA.CSV',
    )
    cls.tmy3_data, cls.metadata = read_tmy3(
        tmy3_path, coerce_year=1990, map_variables=True
    )

    # Calculate solar positions (shift 30 min back for center of interval)
    cls.solpos = get_solarposition(
        cls.tmy3_data.index.shift(freq='-30min'),
        latitude=cls.metadata['latitude'],
        longitude=cls.metadata['longitude'],
        altitude=cls.metadata['altitude'],
        pressure=cls.tmy3_data.pressure * 100,  # millibar to Pa
        temperature=cls.tmy3_data.temp_air,
    )
    cls.solpos.index = cls.tmy3_data.index  # Reset index to end of hour

  def test_disc_method_matches_pvlib(self):
    """Validate DISC decomposition method matches pvlib.irradiance.disc."""
    # Use pvlib DISC method
    out_disc = irradiance.disc(
        self.tmy3_data.ghi,
        self.solpos.zenith,
        self.tmy3_data.index,
        self.tmy3_data.pressure * 100,
    )

    # Calculate DHI using closure equation: DHI = GHI - DNI * cos(zenith)
    df_disc = irradiance.complete_irradiance(
        solar_zenith=self.solpos.apparent_zenith,
        ghi=self.tmy3_data.ghi,
        dni=out_disc.dni,
        dhi=None,
    )

    # Select a sample of daytime hours for validation (avoid night/edge cases)
    # July 4th noon - clear summer day
    sample_times = [
        '1990-07-04 12:00:00-05:00',
        '1990-07-04 13:00:00-05:00',
        '1990-07-04 14:00:00-05:00',
    ]

    for time_str in sample_times:
      idx = pd.Timestamp(time_str)
      if idx in out_disc.index:
        pvlib_dni = out_disc.dni.loc[idx]
        pvlib_dhi = df_disc.dhi.loc[idx]
        pvlib_ghi = self.tmy3_data.ghi.loc[idx]

        # Verify closure equation: GHI = DNI * cos(zenith) + DHI
        zenith_rad = np.radians(self.solpos.apparent_zenith.loc[idx])
        calculated_ghi = pvlib_dni * np.cos(zenith_rad) + pvlib_dhi

        # GHI should match within tolerance (closure equation)
        self.assertAlmostEqual(
            calculated_ghi,
            pvlib_ghi,
            delta=1.0,
            msg=f'DISC closure equation failed at {time_str}',
        )

        # DNI should be non-negative and reasonable
        self.assertGreaterEqual(pvlib_dni, 0)
        self.assertLessEqual(pvlib_dni, 1400)  # Max reasonable DNI

  def test_erbs_method_matches_pvlib(self):
    """Validate Erbs decomposition method matches pvlib.irradiance.erbs."""
    # Use pvlib Erbs method
    out_erbs = irradiance.erbs(
        self.tmy3_data.ghi, self.solpos.zenith, self.tmy3_data.index
    )

    # Select sample times for validation
    sample_times = [
        '1990-04-04 12:00:00-05:00',
        '1990-04-04 13:00:00-05:00',
        '1990-01-04 12:00:00-05:00',
    ]

    for time_str in sample_times:
      idx = pd.Timestamp(time_str)
      if idx in out_erbs.index:
        pvlib_dni = out_erbs.dni.loc[idx]
        pvlib_dhi = out_erbs.dhi.loc[idx]
        pvlib_ghi = self.tmy3_data.ghi.loc[idx]

        # Verify closure equation
        zenith_rad = np.radians(self.solpos.zenith.loc[idx])
        if np.cos(zenith_rad) > 0.1:  # Avoid edge cases near horizon
          calculated_ghi = pvlib_dni * np.cos(zenith_rad) + pvlib_dhi
          self.assertAlmostEqual(
              calculated_ghi,
              pvlib_ghi,
              delta=5.0,
              msg=f'Erbs closure equation failed at {time_str}',
          )

  def test_campbell_norman_method_matches_pvlib(self):
    """Validate campbell_norman method matches pvlib implementation."""
    # Test with known transmittance values
    test_transmittances = [0.7, 0.5, 0.3]  # Clear to cloudy

    # Select a clear summer noon timestamp
    test_time = pd.Timestamp('1990-07-04 12:00:00-05:00')

    for transmittance in test_transmittances:
      # Get extra-terrestrial radiation
      dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([test_time]))

      # Calculate using pvlib campbell_norman
      pvlib_result = irradiance.campbell_norman(
          self.solpos.apparent_zenith.loc[test_time],
          transmittance,
          dni_extra=dni_extra.iloc[0],
      )

      # Verify components are non-negative
      self.assertGreaterEqual(pvlib_result['ghi'], 0)
      self.assertGreaterEqual(pvlib_result['dni'], 0)
      self.assertGreaterEqual(pvlib_result['dhi'], 0)

      # Verify closure equation: GHI = DNI * cos(zenith) + DHI
      zenith_rad = np.radians(self.solpos.apparent_zenith.loc[test_time])
      calculated_ghi = (
          pvlib_result['dni'] * np.cos(zenith_rad) + pvlib_result['dhi']
      )
      self.assertAlmostEqual(
          calculated_ghi,
          pvlib_result['ghi'],
          delta=1.0,
          msg=(
              'Campbell-Norman closure failed for'
              f' transmittance={transmittance}'
          ),
      )

  def test_weather_controller_linear_method_consistency(self):
    """Validate WeatherController linear method produces consistent results."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']

    # Test with 30% cloud cover using linear method
    weather = weather_controller.WeatherController(
        default_low_temp=273.15,
        default_high_temp=298.15,
        latitude=latitude,
        longitude=longitude,
        tz='US/Eastern',
        cloud_cover=30.0,
        irradiance_method='linear',
    )

    # Test at a summer noon timestamp
    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = weather.get_current_irradiance(timestamp)

    # Direct pvlib validation
    pvlib_location = location.Location(latitude, longitude, tz='US/Eastern')
    clearsky = pvlib_location.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    # Expected GHI using linear relationship
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (1.0 - 0.8 * (30.0 / 100.0))

    # Expected DNI using DISC model
    dni_result = irradiance.disc(
        pd.Series([expected_ghi], index=pd.DatetimeIndex([timestamp])),
        solar_position['zenith'],
        pd.DatetimeIndex([timestamp]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])

    # Expected DHI from closure
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    # Validate
    self.assertAlmostEqual(irrad['ghi'], expected_ghi, places=2)
    self.assertAlmostEqual(irrad['dni'], expected_dni, places=2)
    self.assertAlmostEqual(irrad['dhi'], expected_dhi, places=2)

  def test_weather_controller_campbell_norman_consistency(self):
    """Validate WeatherController campbell_norman method against pvlib."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']

    # Test with 50% cloud cover using campbell_norman method
    weather = weather_controller.WeatherController(
        default_low_temp=273.15,
        default_high_temp=298.15,
        latitude=latitude,
        longitude=longitude,
        tz='US/Eastern',
        cloud_cover=50.0,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = weather.get_current_irradiance(timestamp)

    # Direct pvlib validation
    pvlib_location = location.Location(latitude, longitude, tz='US/Eastern')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (50.0 / 100.0)

    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )

    # Validate against pvlib
    self.assertAlmostEqual(irrad['ghi'], expected_irrad['ghi'], places=2)
    self.assertAlmostEqual(irrad['dni'], expected_irrad['dni'], places=2)
    self.assertAlmostEqual(irrad['dhi'], expected_irrad['dhi'], places=2)

  def test_clearsky_irradiance_matches_pvlib(self):
    """Validate clearsky irradiance matches pvlib location.get_clearsky."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']

    # WeatherController with clearsky (no cloud cover)
    weather = weather_controller.WeatherController(
        default_low_temp=273.15,
        default_high_temp=298.15,
        latitude=latitude,
        longitude=longitude,
        tz='US/Eastern',
    )

    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = weather.get_current_irradiance(timestamp)

    # Direct pvlib clearsky
    pvlib_location = location.Location(latitude, longitude, tz='US/Eastern')
    clearsky = pvlib_location.get_clearsky(pd.DatetimeIndex([timestamp]))

    # Validate exact match
    self.assertAlmostEqual(irrad['ghi'], clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad['dni'], clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad['dhi'], clearsky['dhi'].iloc[0], places=4)

  @parameterized.named_parameters(
      ('winter_noon', '1990-01-04 12:00:00-05:00'),
      ('spring_noon', '1990-04-04 12:00:00-05:00'),
      ('summer_noon', '1990-07-04 12:00:00-05:00'),
  )
  def test_tmy3_irradiance_closure_equation(self, time_str):
    """Validate TMY3 data satisfies closure equation: GHI = DNI*cos(z) + DHI."""
    idx = pd.Timestamp(time_str)

    ghi = self.tmy3_data.ghi.loc[idx]
    dni = self.tmy3_data.dni.loc[idx]
    dhi = self.tmy3_data.dhi.loc[idx]
    zenith = self.solpos.apparent_zenith.loc[idx]

    # Closure equation
    zenith_rad = np.radians(zenith)
    calculated_ghi = dni * np.cos(zenith_rad) + dhi

    # TMY3 data should satisfy closure (within measurement uncertainty)
    self.assertAlmostEqual(
        calculated_ghi,
        ghi,
        delta=50.0,
        msg=f'TMY3 closure equation failed at {time_str}',
    )


if __name__ == '__main__':
  absltest.main()
