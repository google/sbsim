"""Tests for weather_controller."""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib import location

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import building_radiation_utils
from smart_control.simulator import constants as sim_constants
from smart_control.simulator import weather_controller
from smart_control.utils import conversion_utils as utils

# Paths to shared weather data and station configuration files.
_WEATHER_DATA_DIR = os.path.join(
    os.path.dirname(__file__), 'building_radiation_test_data'
)


_LOCAL_WEATHER_TEST_DATA_PATH = os.path.join(
    _WEATHER_DATA_DIR, 'local_weather_test_data.csv'
)

_MOFFETT_WEATHER_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'configs',
    'resources',
    'sb1',
    'local_weather_moffett_field_20230701_20231122.csv',
)

_STATION_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'configs',
    'resources',
    'sb1',
    'weather_data',
    'station.json',
)

# Test location constants (Mountain View, CA)
_TEST_LATITUDE = 37.4
_TEST_LONGITUDE = -122.1
_TEST_TIMEZONE_PACIFIC = 'US/Pacific'
_TEST_TIMEZONE_UTC = 'UTC'

# Default temperature bounds for tests (0°C and 25°C expressed in Kelvin)
_DEFAULT_LOW_TEMP_K = utils.celsius_to_kelvin(0.0)
_DEFAULT_HIGH_TEMP_K = utils.celsius_to_kelvin(25.0)


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


class IrradianceTestBase(parameterized.TestCase):
  """Base class for irradiance-related tests with shared setup."""

  def setUp(self):
    super().setUp()
    self.latitude = _TEST_LATITUDE
    self.longitude = _TEST_LONGITUDE
    self.low_temp = _DEFAULT_LOW_TEMP_K
    self.high_temp = _DEFAULT_HIGH_TEMP_K

  def _make_weather_controller(
      self,
      timezone=_TEST_TIMEZONE_PACIFIC,
      cloud_cover=None,
      irradiance_method='clearsky',
      **kwargs,
  ):
    """Factory for WeatherController with test defaults."""
    return weather_controller.WeatherController(
        self.low_temp,
        self.high_temp,
        latitude=self.latitude,
        longitude=self.longitude,
        timezone=timezone,
        cloud_cover=cloud_cover,
        irradiance_method=irradiance_method,
        **kwargs,
    )

  def _make_pvlib_location(self, timezone=_TEST_TIMEZONE_UTC):
    """Factory for pvlib Location with test defaults."""
    return location.Location(self.latitude, self.longitude, tz=timezone)

  def _validate_irradiance_components(self, irrad):
    """Assert all irradiance components are present and non-negative."""
    self.assertGreaterEqual(irrad.ghi, 0)
    self.assertGreaterEqual(irrad.dni, 0)
    self.assertGreaterEqual(irrad.dhi, 0)

  def _validate_solar_position_against_pvlib(
      self, irrad, pvlib_location, timestamp
  ):
    """Assert solar position matches pvlib calculation."""
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    self.assertAlmostEqual(
        irrad.solar_zenith,
        solar_position['apparent_zenith'].iloc[0],
        places=4,
    )
    self.assertAlmostEqual(
        irrad.solar_azimuth,
        solar_position['azimuth'].iloc[0],
        places=4,
    )


class WeatherControllerIrradianceTest(IrradianceTestBase):
  """Tests for WeatherController irradiance methods."""

  def test_get_current_irradiance(self):
    """Test clearsky irradiance calculation for WeatherController."""
    weather = self._make_weather_controller(
        timezone='America/Los_Angeles',
    )

    # Test at noon on a summer day
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='America/Los_Angeles')
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are present and positive
    self.assertGreater(irrad.ghi, 0)
    self.assertGreater(irrad.dni, 0)
    self.assertGreater(irrad.dhi, 0)
    # At noon in summer, GHI should be substantial (> 500 W/m2)
    self.assertGreater(irrad.ghi, 500)
    self.assertEqual(round(irrad.ghi), 934)
    self.assertEqual(round(irrad.dni), 872)
    self.assertEqual(round(irrad.dhi), 121)

    # Solar position should be reasonable at noon
    self.assertGreater(irrad.solar_zenith, 0)
    self.assertLess(irrad.solar_zenith, 90)  # Sun above horizon

    # Direct pvlib clearsky validation
    pvlib_location = location.Location(
        self.latitude, self.longitude, tz='America/Los_Angeles'
    )
    clearsky = pvlib_location.get_clearsky(pd.DatetimeIndex([timestamp]))
    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

    # Validate solar position against pvlib
    self._validate_solar_position_against_pvlib(
        irrad, pvlib_location, timestamp
    )

  def test_get_current_irradiance_no_location(self):
    """Test that irradiance calculation raises error without location."""
    weather = weather_controller.WeatherController(
        self.low_temp, self.high_temp
    )
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    with self.assertRaises(ValueError):
      weather.get_current_irradiance(timestamp)

  def test_get_irradiance_with_solar_position(self):
    """Test irradiance with solar position for WeatherController."""
    weather = self._make_weather_controller()

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad = weather.get_current_irradiance(timestamp)

    # Verify solar position is reasonable at noon
    self.assertGreater(irrad.solar_zenith, 0)
    self.assertLess(irrad.solar_zenith, 90)  # Sun is above horizon at noon

    # Test POA calculation using utility function
    surface_tilt = 30.0  # 30 degrees tilt
    surface_azimuth = 180.0  # South-facing

    poa = building_radiation_utils.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=irrad.solar_zenith,
        solar_azimuth=irrad.solar_azimuth,
    )

    # POA should be positive at noon
    self.assertGreater(poa, 0)
    # POA should be reasonable (between 0 and ~1200 W/m2)
    self.assertLess(poa, 1200)

    # Direct pvlib validation
    pvlib_location = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
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

  def test_get_current_cloud_cover(self):
    """Test cloud cover getter for WeatherController."""
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)

    # Test with no cloud cover set (should return 0)
    weather_no_cloud = self._make_weather_controller()
    self.assertEqual(weather_no_cloud.get_current_cloud_cover(timestamp), 0.0)

    # Test with cloud cover set
    weather_with_cloud = self._make_weather_controller(cloud_cover=50.0)
    self.assertEqual(
        weather_with_cloud.get_current_cloud_cover(timestamp), 50.0
    )

  def test_get_current_irradiance_with_cloud_cover_campbell_norman(self):
    """Test irradiance calculation with cloud cover using campbell_norman."""
    cloud_cover = 50.0  # 50% cloud cover
    weather = self._make_weather_controller(
        cloud_cover=cloud_cover,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are non-negative
    self._validate_irradiance_components(irrad)

    # With 50% cloud cover, irradiance should be less than clearsky
    clearsky_irrad = self._make_weather_controller().get_current_irradiance(
        timestamp
    )
    self.assertLess(irrad.ghi, clearsky_irrad.ghi)

    # Direct pvlib campbell_norman validation
    pvlib_location = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
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
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], places=4)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], places=4)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], places=4)

    # Validate solar position against pvlib
    self._validate_solar_position_against_pvlib(
        irrad, pvlib_location, timestamp
    )

  def test_get_current_irradiance_with_cloud_cover_linear(self):
    """Test irradiance calculation with cloud cover using linear method."""
    cloud_cover = 30.0  # 30% cloud cover
    weather = self._make_weather_controller(
        cloud_cover=cloud_cover,
        irradiance_method='linear',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad = weather.get_current_irradiance(timestamp)

    # Check that all components are non-negative
    self._validate_irradiance_components(irrad)

    # Direct pvlib linear method validation
    pvlib_location = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
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

    self.assertAlmostEqual(irrad.ghi, expected_ghi, places=4)
    self.assertAlmostEqual(irrad.dni, expected_dni, places=4)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, places=4)

    # Validate solar position against pvlib
    self._validate_solar_position_against_pvlib(
        irrad, pvlib_location, timestamp
    )

  def test_invalid_cloud_cover_raises_error(self):
    """Test that invalid cloud cover values raise errors."""
    # Cloud cover < 0
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp, self.high_temp, cloud_cover=-10.0
      )

    # Cloud cover > 100
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp, self.high_temp, cloud_cover=150.0
      )

  def test_dynamic_cloud_cover(self):
    """Test dynamic cloud cover with sinusoidal pattern."""
    cloud_cover_low = 20.0
    cloud_cover_high = 80.0

    weather = weather_controller.WeatherController(
        self.low_temp,
        self.high_temp,
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
    # Only cloud_cover_low provided (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp, self.high_temp, cloud_cover_low=20.0
      )

    # Only cloud_cover_high provided (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp, self.high_temp, cloud_cover_high=80.0
      )

    # cloud_cover_low > cloud_cover_high (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp,
          self.high_temp,
          cloud_cover_low=80.0,
          cloud_cover_high=20.0,
      )

    # cloud_cover_low < 0 (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp,
          self.high_temp,
          cloud_cover_low=-10.0,
          cloud_cover_high=80.0,
      )

    # cloud_cover_high > 100 (should raise error)
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp,
          self.high_temp,
          cloud_cover_low=20.0,
          cloud_cover_high=150.0,
      )

  def test_dynamic_cloud_cover_affects_irradiance(self):
    """Test that dynamic cloud cover affects irradiance calculation."""
    # Weather with dynamic cloud cover (low at midnight, high at noon)
    weather_dynamic = self._make_weather_controller(
        cloud_cover_low=0.0,
        cloud_cover_high=80.0,
        irradiance_method='campbell_norman',
    )

    # Weather with clearsky (no cloud cover)
    weather_clearsky = self._make_weather_controller()

    # At noon when dynamic cloud cover is at maximum (80%)
    noon = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad_dynamic = weather_dynamic.get_current_irradiance(noon)
    irrad_clearsky = weather_clearsky.get_current_irradiance(noon)

    # Dynamic irradiance should be less than clearsky at noon
    self.assertLess(irrad_dynamic.ghi, irrad_clearsky.ghi)

  def test_invalid_irradiance_method_raises_error(self):
    """Test that invalid irradiance method raises error."""
    with self.assertRaises(ValueError):
      weather_controller.WeatherController(
          self.low_temp, self.high_temp, irradiance_method='invalid_method'
      )

  def test_get_sky_temperature_weather_controller(self):
    """Test sky temperature calculation for WeatherController."""
    dewpoint_depression = 5.0

    weather = weather_controller.WeatherController(
        self.low_temp,
        self.high_temp,
        dewpoint_depression=dewpoint_depression,
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    temp_sky_k = weather.get_current_sky_temperature(timestamp)

    # Sky temperature should be in Kelvin and reasonable
    self.assertGreater(temp_sky_k, 200)  # Above absolute zero
    self.assertLess(temp_sky_k, 350)  # Below very hot temps
    # Sky temperature should be less than or equal to dry bulb temp
    temp_k = weather.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)

    # Direct Clark & Allen formula validation using STEFAN_BOLTZMANN_CONSTANT
    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    dp_k = temp_k - dewpoint_depression
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=4)


class ReplayWeatherControllerTest(IrradianceTestBase):
  """Tests for ReplayWeatherController and module-level get_replay_* functions.

  Also covers get_replay_temperatures, get_replay_cloud_cover,
  get_replay_sky_temperature, and get_replay_irradiance, which extract weather
  data from ObservationResponse protos.
  """

  def setUp(self):
    super().setUp()
    self.controller = weather_controller.ReplayWeatherController(
        local_weather_path=_LOCAL_WEATHER_TEST_DATA_PATH,
        convection_coefficient=10.0,
    )

  def _make_observation_response(
      self, measurements, timestamp_seconds=1688212800
  ):
    """Build an ObservationResponse with the given measurement name→value pairs.

    Args:
      measurements: dict mapping measurement_name (str) to continuous_value
        (float).
      timestamp_seconds: Unix timestamp in seconds for the response. Defaults
        to 1688212800 (2023-07-01 12:00:00 UTC).

    Returns:
      A smart_control_building_pb2.ObservationResponse proto.
    """
    single_responses = []
    for name, value in measurements.items():
      single_request = smart_control_building_pb2.SingleObservationRequest(
          device_id='test_device', measurement_name=name
      )
      single_response = smart_control_building_pb2.SingleObservationResponse(
          single_observation_request=single_request,
          continuous_value=value,
      )
      single_responses.append(single_response)
    request = smart_control_building_pb2.ObservationRequest()
    ts_proto = smart_control_building_pb2.ObservationResponse()
    ts_proto.timestamp.seconds = timestamp_seconds
    return smart_control_building_pb2.ObservationResponse(
        timestamp=ts_proto.timestamp,
        request=request,
        single_observation_responses=single_responses,
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

  def test_get_current_cloud_cover_replay_controller(self):
    """Test cloud cover interpolation from weather data."""
    controller = weather_controller.ReplayWeatherController(
        _LOCAL_WEATHER_TEST_DATA_PATH,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        timezone=_TEST_TIMEZONE_UTC,
    )

    # Test at a time with known cloud cover (0% at midnight)
    timestamp = pd.Timestamp('2023-07-01 00:00:00+00:00')
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    self.assertEqual(cloud_cover, 0.0)

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    self.assertEqual(cloud_cover, 100.0)

  def test_get_current_irradiance_replay_controller(self):
    """Test irradiance calculation with cloud cover
    for ReplayWeatherController."""
    controller = weather_controller.ReplayWeatherController(
        _LOCAL_WEATHER_TEST_DATA_PATH,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        timezone=_TEST_TIMEZONE_PACIFIC,
        irradiance_method='campbell_norman',
    )

    # Test at noon
    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad = controller.get_current_irradiance(timestamp)

    # Check that all components are non-negative
    self._validate_irradiance_components(irrad)
    self.assertEqual(round(irrad.ghi), 523.0)
    self.assertEqual(round(irrad.dni), 235.0)
    self.assertEqual(round(irrad.dhi), 304.0)

    # Solar position should be reasonable at noon
    self.assertGreater(irrad.solar_zenith, 0)
    self.assertLess(irrad.solar_zenith, 90)  # Sun above horizon

    # Direct pvlib campbell_norman validation
    pvlib_location = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    timestamp_utc = timestamp.tz_convert('UTC')
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp_utc])
    )
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
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], delta=1.0)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], delta=1.0)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], delta=1.0)

    # Validate solar position against pvlib
    self._validate_solar_position_against_pvlib(
        irrad, pvlib_location, timestamp_utc
    )

  def test_get_irradiance_with_solar_position_replay_controller(self):
    """Test irradiance with solar position for ReplayWeatherController."""
    controller = weather_controller.ReplayWeatherController(
        _LOCAL_WEATHER_TEST_DATA_PATH,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        timezone=_TEST_TIMEZONE_UTC,
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)

    # Test POA calculation using utility function
    surface_tilt = 30.0
    surface_azimuth = 180.0

    poa = building_radiation_utils.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=irrad.solar_zenith,
        solar_azimuth=irrad.solar_azimuth,
    )

    # POA should be non-negative
    self.assertGreaterEqual(poa, 0)

    # Direct pvlib validation for POA calculation
    pvlib_location = self._make_pvlib_location()
    solar_position = pvlib_location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    poa_irrad_pvlib = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=irrad.dni,
        ghi=irrad.ghi,
        dhi=irrad.dhi,
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )
    expected_poa = float(poa_irrad_pvlib['poa_global'])
    self.assertAlmostEqual(poa, expected_poa, places=4)

  def test_get_sky_temperature_replay_controller(self):
    """Test sky temperature calculation for ReplayWeatherController."""
    controller = weather_controller.ReplayWeatherController(
        _LOCAL_WEATHER_TEST_DATA_PATH, 10.0, timezone=_TEST_TIMEZONE_UTC
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00+00:00')
    temp_sky_k = controller.get_current_sky_temperature(timestamp)

    # Sky temperature should be in Kelvin and reasonable
    self.assertGreater(temp_sky_k, 200)
    self.assertLess(temp_sky_k, 350)
    # Sky temperature should be less than or equal to dry bulb temp
    temp_k = controller.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)

    # Direct Clark & Allen formula validation using STEFAN_BOLTZMANN_CONSTANT
    # From CSV at 12:00 UTC: TempC=23.0, DewPointC=15.0
    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    temp_c = 23.0  # From CSV row 12
    dp_c = 15.0  # From CSV row 12
    temp_k_expected = utils.celsius_to_kelvin(temp_c)
    dp_k = utils.celsius_to_kelvin(dp_c)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k_expected**4)
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=2)

  def test_get_replay_temperatures_sensor_present(self):
    """get_replay_temperatures returns the sensor value when present."""
    temp_k = utils.celsius_to_kelvin(25.0)
    obs = self._make_observation_response(
        {'outside_air_temperature_sensor': temp_k}
    )
    result = weather_controller.get_replay_temperatures([obs])
    self.assertEqual(len(result), 1)
    self.assertAlmostEqual(list(result.values())[0], temp_k, places=4)

  def test_get_replay_temperatures_sensor_absent_returns_default(self):
    """get_replay_temperatures returns -1.0 when sensor is absent."""
    obs = self._make_observation_response({'some_other_sensor': 300.0})
    result = weather_controller.get_replay_temperatures([obs])
    self.assertEqual(len(result), 1)
    self.assertEqual(list(result.values())[0], -1.0)

  def test_get_replay_temperatures_multiple_observations(self):
    """get_replay_temperatures handles multiple observations at
    different times."""
    obs1 = self._make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(20.0)},
        timestamp_seconds=1688212800,  # 2023-07-01 12:00:00 UTC
    )
    obs2 = self._make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(25.0)},
        timestamp_seconds=1688216400,  # 2023-07-01 13:00:00 UTC
    )
    result = weather_controller.get_replay_temperatures([obs1, obs2])
    self.assertEqual(len(result), 2)
    values = list(result.values())
    self.assertAlmostEqual(values[0], utils.celsius_to_kelvin(20.0), places=4)
    self.assertAlmostEqual(values[1], utils.celsius_to_kelvin(25.0), places=4)

  def test_get_replay_cloud_cover_sensor_present(self):
    """get_replay_cloud_cover returns the sensor value when present."""
    obs = self._make_observation_response({'cloud_cover_sensor': 50.0})
    result = weather_controller.get_replay_cloud_cover([obs])
    self.assertEqual(len(result), 1)
    self.assertAlmostEqual(list(result.values())[0], 50.0, places=4)

  def test_get_replay_cloud_cover_sensor_absent_returns_default(self):
    """get_replay_cloud_cover returns 0.0 (clear sky) when sensor is absent."""
    obs = self._make_observation_response({'some_other_sensor': 100.0})
    result = weather_controller.get_replay_cloud_cover([obs])
    self.assertEqual(len(result), 1)
    self.assertEqual(list(result.values())[0], 0.0)

  def test_get_replay_sky_temperature_with_both_sensors(self):
    """get_replay_sky_temperature uses dew point sensor when present."""
    temp_k = utils.celsius_to_kelvin(23.0)
    dp_k = utils.celsius_to_kelvin(15.0)
    obs = self._make_observation_response({
        'outside_air_temperature_sensor': temp_k,
        'dew_point_temperature_sensor': dp_k,
    })
    result = weather_controller.get_replay_sky_temperature([obs])
    self.assertEqual(len(result), 1)
    temp_sky_k = list(result.values())[0]

    # Validate against Clark & Allen formula
    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=4)
    # Sky temperature should be less than dry bulb temperature
    self.assertLess(temp_sky_k, temp_k)

  def test_get_replay_sky_temperature_missing_dewpoint_uses_depression(self):
    """get_replay_sky_temperature falls back to dewpoint_depression."""
    temp_k = utils.celsius_to_kelvin(23.0)
    dewpoint_depression = 8.0
    obs = self._make_observation_response(
        {'outside_air_temperature_sensor': temp_k}
    )
    result = weather_controller.get_replay_sky_temperature(
        [obs], dewpoint_depression=dewpoint_depression
    )
    self.assertEqual(len(result), 1)
    temp_sky_k = list(result.values())[0]

    # Validate: dp_k estimated from depression
    dp_k = temp_k - dewpoint_depression
    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected_temp_sky_k = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected_temp_sky_k, places=4)

  def test_get_replay_sky_temperature_missing_temp_skips_entry(self):
    """get_replay_sky_temperature skips entries without temp sensor."""
    obs_no_temp = self._make_observation_response(
        {'dew_point_temperature_sensor': utils.celsius_to_kelvin(15.0)}
    )
    obs_with_temp = self._make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(23.0)}
    )
    result = weather_controller.get_replay_sky_temperature(
        [obs_no_temp, obs_with_temp]
    )
    # Only the observation with temp sensor should produce an entry
    self.assertEqual(len(result), 1)

  def test_get_replay_irradiance_sensors_present(self):
    """get_replay_irradiance returns correct ghi/dni/dhi when sensors
    present."""
    obs = self._make_observation_response({
        'ghi_sensor': 800.0,
        'dni_sensor': 700.0,
        'dhi_sensor': 100.0,
    })
    result = weather_controller.get_replay_irradiance([obs])
    self.assertEqual(len(result), 1)
    irrad = list(result.values())[0]
    self.assertAlmostEqual(irrad['ghi'], 800.0, places=4)
    self.assertAlmostEqual(irrad['dni'], 700.0, places=4)
    self.assertAlmostEqual(irrad['dhi'], 100.0, places=4)

  def test_get_replay_irradiance_sensors_absent_returns_defaults(self):
    """get_replay_irradiance returns 0.0 defaults when sensors are absent."""
    obs = self._make_observation_response({'some_other_sensor': 999.0})
    result = weather_controller.get_replay_irradiance([obs])
    self.assertEqual(len(result), 1)
    irrad = list(result.values())[0]
    self.assertEqual(irrad['ghi'], 0.0)
    self.assertEqual(irrad['dni'], 0.0)
    self.assertEqual(irrad['dhi'], 0.0)

  def test_get_replay_irradiance_multiple_observations(self):
    """get_replay_irradiance handles multiple observations at different
    times."""
    obs1 = self._make_observation_response(
        {'ghi_sensor': 500.0, 'dni_sensor': 400.0, 'dhi_sensor': 100.0},
        timestamp_seconds=1688212800,  # 2023-07-01 12:00:00 UTC
    )
    obs2 = self._make_observation_response(
        {'ghi_sensor': 0.0, 'dni_sensor': 0.0, 'dhi_sensor': 0.0},
        timestamp_seconds=1688216400,  # 2023-07-01 13:00:00 UTC
    )
    result = weather_controller.get_replay_irradiance([obs1, obs2])
    self.assertEqual(len(result), 2)
    values = list(result.values())
    self.assertAlmostEqual(values[0]['ghi'], 500.0, places=4)
    self.assertAlmostEqual(values[1]['ghi'], 0.0, places=4)


class ReplayWeatherControllerPvlibValidationTest(IrradianceTestBase):
  """Validate ReplayWeatherController irradiance calculations against pvlib.

  This test class validates that the DHI, GHI, DNI calculations in
  ReplayWeatherController are consistent with pvlib's irradiance methods
  using the local_weather_test_data.csv which has SkyCoverage data.
  """

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.data_path = _LOCAL_WEATHER_TEST_DATA_PATH

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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    # Test at timestamps with known cloud cover from the CSV
    # Use 19:00 UTC which is ~12:00 local time in Mountain View (daytime)
    # Row 19: 19:00 UTC has SkyCoverage=0% (clear)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib calculation with same parameters
    pvlib_location = self._make_pvlib_location()
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
        irrad.ghi,
        expected_irrad['ghi'],
        delta=10.0,
        msg=f'GHI mismatch: got {irrad.ghi}, expected {expected_irrad["ghi"]}',
    )
    self.assertAlmostEqual(
        irrad.dni,
        expected_irrad['dni'],
        delta=10.0,
        msg=f'DNI mismatch: got {irrad.dni}, expected {expected_irrad["dni"]}',
    )
    self.assertAlmostEqual(
        irrad.dhi,
        expected_irrad['dhi'],
        delta=10.0,
        msg=f'DHI mismatch: got {irrad.dhi}, expected {expected_irrad["dhi"]}',
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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp(time_str)
    cloud_cover = controller.get_current_cloud_cover(timestamp)
    irrad = controller.get_current_irradiance(timestamp)

    # Verify cloud cover matches expected value
    self.assertAlmostEqual(cloud_cover, expected_cloud_cover, delta=1.0)

    # All irradiance components should be non-negative
    self._validate_irradiance_components(irrad)

    # During daytime with clear sky, GHI should be substantial
    self.assertGreater(irrad.ghi, 100)

  def test_replay_controller_irradiance_closure_equation(self):
    """Validate ReplayWeatherController satisfies closure equation."""
    controller = weather_controller.ReplayWeatherController(
        self.data_path,
        10.0,
        latitude=self.latitude,
        longitude=self.longitude,
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    # Test at multiple timestamps
    test_times = [
        '2023-07-01 08:00:00+00:00',
        '2023-07-01 10:00:00+00:00',
        '2023-07-01 12:00:00+00:00',
        '2023-07-01 14:00:00+00:00',
    ]

    pvlib_location = self._make_pvlib_location()

    for time_str in test_times:
      timestamp = pd.Timestamp(time_str)
      irrad = controller.get_current_irradiance(timestamp)

      # Get solar position
      solar_position = pvlib_location.get_solarposition(
          pd.DatetimeIndex([timestamp])
      )
      zenith_rad = np.radians(solar_position['apparent_zenith'].iloc[0])

      # Closure equation: GHI = DNI * cos(zenith) + DHI
      calculated_ghi = irrad.dni * np.cos(zenith_rad) + irrad.dhi

      # Should satisfy closure equation
      if irrad.ghi > 10:  # Skip very low irradiance (numerical issues)
        self.assertAlmostEqual(
            calculated_ghi,
            irrad.ghi,
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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    # Test at a timestamp with known cloud cover
    timestamp = pd.Timestamp('2023-07-01 10:00:00+00:00')
    cloud_cover = replay_controller.get_current_cloud_cover(timestamp)
    replay_irrad = replay_controller.get_current_irradiance(timestamp)

    # Create WeatherController with same cloud cover
    weather = self._make_weather_controller(
        timezone=_TEST_TIMEZONE_UTC,
        cloud_cover=cloud_cover,
        irradiance_method='campbell_norman',
    )
    weather_irrad = weather.get_current_irradiance(timestamp)

    # Both should produce consistent results (within tolerance)
    self.assertAlmostEqual(
        replay_irrad.ghi,
        weather_irrad.ghi,
        delta=10.0,
        msg=(
            'GHI inconsistency between ReplayWeatherController and'
            ' WeatherController'
        ),
    )
    self.assertAlmostEqual(
        replay_irrad.dni,
        weather_irrad.dni,
        delta=10.0,
        msg=(
            'DNI inconsistency between ReplayWeatherController and'
            ' WeatherController'
        ),
    )
    self.assertAlmostEqual(
        replay_irrad.dhi,
        weather_irrad.dhi,
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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    # Test at 19:00 UTC (~noon local time in Mountain View)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib validation
    pvlib_location = self._make_pvlib_location()
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
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], delta=5.0)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], delta=5.0)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], delta=5.0)

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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='linear',
    )

    # Test at 19:00 UTC (~noon local time in Mountain View)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Direct pvlib validation with linear method
    pvlib_location = self._make_pvlib_location()
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
    self.assertAlmostEqual(irrad.ghi, expected_ghi, delta=20.0)
    self.assertAlmostEqual(irrad.dni, expected_dni, delta=20.0)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, delta=20.0)

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
        timezone=_TEST_TIMEZONE_UTC,
        irradiance_method='campbell_norman',
    )

    # Test at 19:00 UTC (~noon local) where SkyCoverage=0 (clear sky)
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = controller.get_current_irradiance(timestamp)
    cloud_cover = controller.get_current_cloud_cover(timestamp)

    # Verify it's clear sky condition
    self.assertEqual(cloud_cover, 0.0)

    # For campbell_norman with transmittance=0.7 (clear sky), compare to pvlib
    pvlib_location = self._make_pvlib_location()
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
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], delta=5.0)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], delta=5.0)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], delta=5.0)

    # Also compare to ineichen clearsky model for reference
    clearsky = pvlib_location.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )

    # GHI should be in reasonable range compared to clearsky model
    # (campbell_norman and ineichen will differ but should be similar order)
    self.assertAlmostEqual(
        irrad.ghi,
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
    """Load TMY3 data once for all tests.

    The file ``building_radiation_test_data/723170TYA.CSV`` is the Typical
    Meteorological Year 3 (TMY3) dataset for Greensboro, NC (USAF station
    723170).  It was obtained directly from the pvlib-python repository:

      https://github.com/pvlib/pvlib-python/blob/main/pvlib/data/723170TYA.CSV

    To reproduce the file locally, run::

      import urllib.request
      url = (
          'https://raw.githubusercontent.com/pvlib/pvlib-python/'
          'main/pvlib/data/723170TYA.CSV'
      )
      urllib.request.urlretrieve(
          url,
          'smart_control/simulator/building_radiation_test_data/723170TYA.CSV',
      )

    The TMY3 format is documented by NLR:
      https://doi.org/10.2172/928611
    """
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
        default_low_temp=_DEFAULT_LOW_TEMP_K,
        default_high_temp=_DEFAULT_HIGH_TEMP_K,
        latitude=latitude,
        longitude=longitude,
        timezone='US/Eastern',
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
    self.assertAlmostEqual(irrad.ghi, expected_ghi, places=2)
    self.assertAlmostEqual(irrad.dni, expected_dni, places=2)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, places=2)

  def test_weather_controller_campbell_norman_consistency(self):
    """Validate WeatherController campbell_norman method against pvlib."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']

    # Test with 50% cloud cover using campbell_norman method
    weather = weather_controller.WeatherController(
        default_low_temp=_DEFAULT_LOW_TEMP_K,
        default_high_temp=_DEFAULT_HIGH_TEMP_K,
        latitude=latitude,
        longitude=longitude,
        timezone='US/Eastern',
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
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], places=2)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], places=2)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], places=2)

  def test_clearsky_irradiance_matches_pvlib(self):
    """Validate clearsky irradiance matches pvlib location.get_clearsky."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']

    # WeatherController with clearsky (no cloud cover)
    weather = weather_controller.WeatherController(
        default_low_temp=_DEFAULT_LOW_TEMP_K,
        default_high_temp=_DEFAULT_HIGH_TEMP_K,
        latitude=latitude,
        longitude=longitude,
        timezone='US/Eastern',
    )

    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = weather.get_current_irradiance(timestamp)

    # Direct pvlib clearsky
    pvlib_location = location.Location(latitude, longitude, tz='US/Eastern')
    clearsky = pvlib_location.get_clearsky(pd.DatetimeIndex([timestamp]))

    # Validate exact match
    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

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


class MoffettReplayWeatherControllerTest(parameterized.TestCase):
  """Tests for ReplayWeatherController using real weather data."""

  def setUp(self):
    super().setUp()
    self.controller = weather_controller.ReplayWeatherController(
        local_weather_path=_MOFFETT_WEATHER_CSV_PATH,
        station_json_path=_STATION_JSON_PATH,
    )

  def test_weather_df(self):
    self.assertIsInstance(self.controller._weather_data, pd.DataFrame)
    self.assertEqual(self.controller._weather_data.shape, (3462, 19))

    expected_columns = [
        'Time',
        'StationName',
        'StationId',
        'Location',
        'TempC',
        'DewPointC',
        'BarometerMbar',
        'Rain',
        'RainTotal',
        'WindspeedKmph',
        'WindDirection',
        'SkyCoverage',
        'VisibilityKm',
        'Humidity',
        'TempF',
        'ghi',
        'dni',
        'dhi',
        'TempSkyC',
    ]
    self.assertCountEqual(
        self.controller._weather_data.columns.tolist(),
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
