"""Tests for solar_radiation module.

Tests are structured using inheritance so that every test that applies to
:class:`SolarRadiation` with a plain :class:`WeatherController` is
automatically re-run with a :class:`ReplayWeatherController`.

Class hierarchy:
  SolarRadiationTest  (uses WeatherController)
  ReplaySolarRadiationTest(SolarRadiationTest)  (uses ReplayWeatherController)
"""

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
from smart_control.simulator import solar_radiation
from smart_control.simulator import weather_controller
from smart_control.utils import conversion_utils as utils

# Paths to shared weather data files.
_LOCAL_WEATHER_TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'local_weather_test_data.csv',
)

# Test location constants (Mountain View, CA)
_TEST_LATITUDE = 37.4
_TEST_LONGITUDE = -122.1
_TEST_TIMEZONE_PACIFIC = 'US/Pacific'
_TEST_TIMEZONE_UTC = 'UTC'

# Default temperature bounds (0 °C and 25 °C in Kelvin)
_DEFAULT_LOW_TEMP_K = utils.celsius_to_kelvin(0.0)
_DEFAULT_HIGH_TEMP_K = utils.celsius_to_kelvin(25.0)

# A daytime timestamp valid for both controller types.
# 2023-07-01 19:00 UTC ≈ noon local time in Mountain View.
_DAYTIME_TIMESTAMP_UTC = pd.Timestamp('2023-07-01 19:00:00', tz='UTC')
_DAYTIME_TIMESTAMP_PACIFIC = pd.Timestamp(
    '2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC
)


# ---------------------------------------------------------------------------
# Stand-alone dataclass tests
# ---------------------------------------------------------------------------
class BuildingInfoTest(absltest.TestCase):
  """Tests for BuildingInfo dataclass."""

  def test_default_values(self):
    info = solar_radiation.BuildingInfo()
    self.assertEqual(info.floor_plan_filepath, '')
    self.assertEqual(info.floor_plan_orientation, 0.0)
    self.assertAlmostEqual(info.lat, 37.4263)
    self.assertAlmostEqual(info.lon, -122.0349)
    self.assertEqual(info.time_zone, 'US/Pacific')
    self.assertIsNone(info.altitude)

  def test_custom_values(self):
    info = solar_radiation.BuildingInfo(
        floor_plan_filepath='/path/to/plan.npy',
        floor_plan_orientation=90.0,
        lat=40.0,
        lon=-74.0,
        time_zone='US/Eastern',
        altitude=100.0,
    )
    self.assertEqual(info.floor_plan_filepath, '/path/to/plan.npy')
    self.assertEqual(info.floor_plan_orientation, 90.0)
    self.assertAlmostEqual(info.lat, 40.0)
    self.assertAlmostEqual(info.lon, -74.0)
    self.assertEqual(info.time_zone, 'US/Eastern')
    self.assertEqual(info.altitude, 100.0)

  def test_valid_orientations(self):
    for angle in [0.0, 90.0, 180.0, 270.0, 360.0]:
      info = solar_radiation.BuildingInfo(floor_plan_orientation=angle)
      self.assertEqual(info.floor_plan_orientation, angle)

  def test_invalid_orientation_below_zero(self):
    with self.assertRaises(ValueError):
      solar_radiation.BuildingInfo(floor_plan_orientation=-1.0)

  def test_invalid_orientation_above_360(self):
    with self.assertRaises(ValueError):
      solar_radiation.BuildingInfo(floor_plan_orientation=361.0)


class IrradianceComponentsTest(absltest.TestCase):
  """Tests for IrradianceComponents dataclass."""

  def test_basic_construction(self):
    irrad = solar_radiation.IrradianceComponents(
        ghi=800.0,
        dni=700.0,
        dhi=100.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    self.assertEqual(irrad.ghi, 800.0)
    self.assertEqual(irrad.dni, 700.0)
    self.assertEqual(irrad.dhi, 100.0)
    self.assertEqual(irrad.solar_zenith, 30.0)
    self.assertEqual(irrad.solar_azimuth, 180.0)
    self.assertIsNone(irrad.timestamp)

  def test_construction_with_timestamp(self):
    ts = pd.Timestamp('2023-07-01 12:00:00', tz='UTC')
    irrad = solar_radiation.IrradianceComponents(
        ghi=800.0,
        dni=700.0,
        dhi=100.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
        timestamp=ts,
    )
    self.assertEqual(irrad.timestamp, ts)


# ---------------------------------------------------------------------------
# Base test helper for irradiance tests
# ---------------------------------------------------------------------------
class IrradianceTestBase(parameterized.TestCase):
  """Base class providing shared helpers for irradiance tests."""

  def setUp(self):
    super().setUp()
    self.latitude = _TEST_LATITUDE
    self.longitude = _TEST_LONGITUDE
    self.low_temp = _DEFAULT_LOW_TEMP_K
    self.high_temp = _DEFAULT_HIGH_TEMP_K

  def _make_pvlib_location(self, timezone=_TEST_TIMEZONE_UTC):
    return location.Location(self.latitude, self.longitude, tz=timezone)

  def _validate_irradiance_components(self, irrad):
    self.assertGreaterEqual(irrad.ghi, 0)
    self.assertGreaterEqual(irrad.dni, 0)
    self.assertGreaterEqual(irrad.dhi, 0)

  def _validate_solar_position_against_pvlib(self, irrad, pvlib_loc, timestamp):
    solar_position = pvlib_loc.get_solarposition(pd.DatetimeIndex([timestamp]))
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


# ---------------------------------------------------------------------------
# SolarRadiation – base test class (uses WeatherController)
# ---------------------------------------------------------------------------
class SolarRadiationTest(IrradianceTestBase):
  """Tests for SolarRadiation using a plain WeatherController.

  Sub-classes only need to override setUp() to swap in a different
  weather controller; all test methods are inherited and re-run.
  """

  def setUp(self):
    super().setUp()
    self.weather_controller = weather_controller.WeatherController(
        self.low_temp, self.high_temp
    )
    self.building_info = solar_radiation.BuildingInfo(
        lat=self.latitude,
        lon=self.longitude,
        time_zone=_TEST_TIMEZONE_PACIFIC,
    )
    self.solar_radiation = solar_radiation.SolarRadiation(
        building_info=self.building_info,
        weather_controller=self.weather_controller,
    )

  # ---- helpers -----------------------------------------------------------

  def _make_solar_radiation(self, **kwargs):
    defaults = dict(
        building_info=self.building_info,
        weather_controller=self.weather_controller,
    )
    defaults.update(kwargs)
    return solar_radiation.SolarRadiation(**defaults)

  # ---- validation tests --------------------------------------------------

  def test_invalid_irradiance_method_raises(self):
    with self.assertRaises(ValueError):
      self._make_solar_radiation(irradiance_method='invalid_method')

  def test_invalid_cloud_cover_raises_error(self):
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover=-10.0)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover=150.0)

  def test_dynamic_cloud_cover_validation(self):
    # Only low provided (should raise)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover_low=20.0)
    # Only high provided (should raise)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover_high=80.0)
    # low > high (should raise)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover_low=80.0, cloud_cover_high=20.0)
    # low < 0 (should raise)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover_low=-10.0, cloud_cover_high=80.0)
    # high > 100 (should raise)
    with self.assertRaises(ValueError):
      self._make_solar_radiation(cloud_cover_low=20.0, cloud_cover_high=150.0)

  # ---- cloud cover tests -------------------------------------------------

  def test_get_current_cloud_cover_static(self):
    sr = self._make_solar_radiation(cloud_cover=50.0)
    self.assertEqual(
        sr.get_current_cloud_cover(_DAYTIME_TIMESTAMP_PACIFIC), 50.0
    )

  def test_get_current_cloud_cover_clearsky(self):
    sr = self._make_solar_radiation()
    self.assertEqual(
        sr.get_current_cloud_cover(_DAYTIME_TIMESTAMP_PACIFIC), 0.0
    )

  def test_dynamic_cloud_cover(self):
    """Sinusoidal pattern: low at midnight, high at noon (US/Pacific)."""
    sr = self._make_solar_radiation(cloud_cover_low=20.0, cloud_cover_high=80.0)
    midnight = pd.Timestamp('2023-07-01 00:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    noon = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    morning = pd.Timestamp('2023-07-01 06:00:00', tz=_TEST_TIMEZONE_PACIFIC)

    self.assertAlmostEqual(sr.get_current_cloud_cover(midnight), 20.0, places=1)
    self.assertAlmostEqual(sr.get_current_cloud_cover(noon), 80.0, places=1)
    self.assertAlmostEqual(sr.get_current_cloud_cover(morning), 50.0, places=1)

  def test_dynamic_cloud_cover_affects_irradiance(self):
    """Dynamic cloud cover at noon should reduce irradiance vs clearsky."""
    sr_dynamic = self._make_solar_radiation(
        cloud_cover_low=0.0,
        cloud_cover_high=80.0,
        irradiance_method='campbell_norman',
    )
    sr_clearsky = self._make_solar_radiation()

    irrad_dynamic = sr_dynamic.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )
    irrad_clearsky = sr_clearsky.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )
    self.assertLess(irrad_dynamic.ghi, irrad_clearsky.ghi)

  # ---- irradiance tests --------------------------------------------------

  def test_get_current_irradiance_clearsky(self):
    """Clearsky irradiance at noon in summer should be substantial."""
    irrad = self.solar_radiation.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )

    self.assertGreater(irrad.ghi, 500)
    self.assertGreater(irrad.dni, 0)
    self.assertGreater(irrad.dhi, 0)
    self.assertGreater(irrad.solar_zenith, 0)
    self.assertLess(irrad.solar_zenith, 90)
    self.assertIsNotNone(irrad.timestamp)

    # Direct pvlib clearsky validation
    pvlib_loc = location.Location(
        self.latitude, self.longitude, tz=_TEST_TIMEZONE_PACIFIC
    )
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

    # Validate solar position against pvlib
    self._validate_solar_position_against_pvlib(
        irrad,
        location.Location(
            self.latitude, self.longitude, tz=_TEST_TIMEZONE_PACIFIC
        ),
        _DAYTIME_TIMESTAMP_PACIFIC,
    )

  def test_get_irradiance_with_solar_position(self):
    """Solar position is reasonable at noon; POA calculation is consistent."""
    irrad = self.solar_radiation.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )
    self.assertGreater(irrad.solar_zenith, 0)
    self.assertLess(irrad.solar_zenith, 90)

    surface_tilt = 30.0
    surface_azimuth = 180.0

    poa = solar_radiation.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=irrad.solar_zenith,
        solar_azimuth=irrad.solar_azimuth,
    )
    self.assertGreater(poa, 0)
    self.assertLess(poa, 1200)

    # Direct pvlib validation
    pvlib_loc = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    solar_position = pvlib_loc.get_solarposition(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    poa_pvlib = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=clearsky['dni'].iloc[0],
        ghi=clearsky['ghi'].iloc[0],
        dhi=clearsky['dhi'].iloc[0],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )
    self.assertAlmostEqual(poa, float(poa_pvlib['poa_global']), places=4)

  def test_get_current_irradiance_campbell_norman(self):
    """Campbell-Norman with 50% cloud cover should be < clearsky."""
    cloud_cover = 50.0
    sr = self._make_solar_radiation(
        cloud_cover=cloud_cover,
        irradiance_method='campbell_norman',
    )
    irrad = sr.get_current_irradiance(_DAYTIME_TIMESTAMP_PACIFIC)

    self._validate_irradiance_components(irrad)
    clearsky_irrad = self.solar_radiation.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )
    self.assertLess(irrad.ghi, clearsky_irrad.ghi)

    # Direct pvlib campbell_norman validation
    pvlib_loc = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    solar_position = pvlib_loc.get_solarposition(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    dni_extra = irradiance.get_extra_radiation(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)
    expected = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )
    self.assertAlmostEqual(irrad.ghi, expected['ghi'], places=4)
    self.assertAlmostEqual(irrad.dni, expected['dni'], places=4)
    self.assertAlmostEqual(irrad.dhi, expected['dhi'], places=4)

    self._validate_solar_position_against_pvlib(
        irrad, pvlib_loc, _DAYTIME_TIMESTAMP_PACIFIC
    )

  def test_get_current_irradiance_linear(self):
    """Linear method with 30% cloud cover."""
    cloud_cover = 30.0
    sr = self._make_solar_radiation(
        cloud_cover=cloud_cover,
        irradiance_method='linear',
    )
    irrad = sr.get_current_irradiance(_DAYTIME_TIMESTAMP_PACIFIC)
    self._validate_irradiance_components(irrad)

    pvlib_loc = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC]), model='ineichen'
    )
    solar_position = pvlib_loc.get_solarposition(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (
        1.0 - 0.8 * (cloud_cover / 100.0)
    )
    dni_result = irradiance.disc(
        pd.Series(
            [expected_ghi],
            index=pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC]),
        ),
        solar_position['zenith'],
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    self.assertAlmostEqual(irrad.ghi, expected_ghi, places=4)
    self.assertAlmostEqual(irrad.dni, expected_dni, places=4)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, places=4)

    self._validate_solar_position_against_pvlib(
        irrad, pvlib_loc, _DAYTIME_TIMESTAMP_PACIFIC
    )

  # ---- sky temperature tests ---------------------------------------------

  def test_get_current_sky_temperature(self):
    """Sky temperature via Clark & Allen formula."""
    dewpoint_depression = 5.0
    sr = self._make_solar_radiation(dewpoint_depression=dewpoint_depression)

    # Use a naive timestamp so WeatherController (which cannot handle
    # tz-aware timestamps) and ReplayWeatherController both work.
    timestamp = pd.Timestamp('2023-07-01 12:00:00')
    temp_sky_k = sr.get_current_sky_temperature(timestamp)

    self.assertGreater(temp_sky_k, 200)
    self.assertLess(temp_sky_k, 350)

    temp_k = self.weather_controller.get_current_temp(timestamp)
    self.assertLessEqual(temp_sky_k, temp_k)

    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    dp_k = temp_k - dewpoint_depression
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected, places=4)

  def test_get_current_sky_temperature_no_weather_controller_raises(self):
    sr = solar_radiation.SolarRadiation(
        building_info=self.building_info,
        weather_controller=None,
    )
    with self.assertRaises(ValueError):
      sr.get_current_sky_temperature(_DAYTIME_TIMESTAMP_UTC)

  # ---- POA irradiance tests ----------------------------------------------

  def test_calculate_poa_irradiance(self):
    """POA calculation matches pvlib reference."""
    irrad_components = solar_radiation.IrradianceComponents(
        ghi=800.0,
        dni=700.0,
        dhi=100.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    poa = solar_radiation.calculate_poa_irradiance(
        irradiance_components=irrad_components,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    self.assertGreater(poa, 0)
    self.assertLess(poa, 1500)

    poa_pvlib = irradiance.get_total_irradiance(
        surface_tilt=30.0,
        surface_azimuth=180.0,
        dni=700.0,
        ghi=800.0,
        dhi=100.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    self.assertAlmostEqual(poa, float(poa_pvlib['poa_global']), places=4)

  def test_calculate_poa_irradiance_building_radiation_utils(self):
    """Backward-compat: building_radiation_utils.calculate_poa_irradiance."""
    irrad_components = solar_radiation.IrradianceComponents(
        ghi=800.0,
        dni=700.0,
        dhi=100.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    poa_sr = solar_radiation.calculate_poa_irradiance(
        irradiance_components=irrad_components,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    poa_utils = building_radiation_utils.calculate_poa_irradiance(
        irradiance_components=irrad_components,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    self.assertAlmostEqual(poa_sr, poa_utils, places=4)

  def test_poa_with_clearsky_irradiance(self):
    """POA from clearsky SolarRadiation output matches pvlib."""
    irrad = self.solar_radiation.get_current_irradiance(
        _DAYTIME_TIMESTAMP_PACIFIC
    )
    poa = solar_radiation.calculate_poa_irradiance(
        irradiance_components=irrad,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=irrad.solar_zenith,
        solar_azimuth=irrad.solar_azimuth,
    )
    self.assertGreater(poa, 0)
    self.assertLess(poa, 1200)

    pvlib_loc = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    solar_position = pvlib_loc.get_solarposition(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    poa_pvlib = irradiance.get_total_irradiance(
        surface_tilt=30.0,
        surface_azimuth=180.0,
        dni=clearsky['dni'].iloc[0],
        ghi=clearsky['ghi'].iloc[0],
        dhi=clearsky['dhi'].iloc[0],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )
    self.assertAlmostEqual(poa, float(poa_pvlib['poa_global']), places=4)

  # ---- get_exterior_radiation tests ----------------------------------------

  def test_get_exterior_radiation_returns_correct_types(self):
    """get_exterior_radiation returns ExteriorRadiationData."""
    timestamp = pd.Timestamp('2023-07-01 12:00:00')
    ext_rad = self.solar_radiation.get_exterior_radiation(timestamp)

    self.assertIsInstance(ext_rad, solar_radiation.ExteriorRadiationData)
    self.assertGreater(ext_rad.ambient_temp_k, 200)
    self.assertLess(ext_rad.ambient_temp_k, 400)
    self.assertGreater(ext_rad.sky_temp_k, 200)
    self.assertLess(ext_rad.sky_temp_k, 400)
    # Sky temp should be ≤ ambient dry-bulb
    self.assertLessEqual(ext_rad.sky_temp_k, ext_rad.ambient_temp_k)
    # Irradiance object should be consistent
    self.assertIsInstance(
        ext_rad.irradiance, solar_radiation.IrradianceComponents
    )
    self.assertGreaterEqual(ext_rad.irradiance.ghi, 0)
    self.assertGreaterEqual(ext_rad.irradiance.dni, 0)
    self.assertGreaterEqual(ext_rad.irradiance.dhi, 0)

  def test_get_exterior_radiation_no_weather_controller_raises(self):
    """get_exterior_radiation raises without a weather controller."""
    sr = solar_radiation.SolarRadiation(
        building_info=self.building_info,
        weather_controller=None,
    )
    with self.assertRaises(ValueError):
      sr.get_exterior_radiation(_DAYTIME_TIMESTAMP_PACIFIC)

  def test_get_exterior_radiation_consistent_with_individual_getters(self):
    """get_exterior_radiation returns same values as individual getter calls."""
    # Use a naive timestamp (works with both WeatherController variants)
    timestamp = pd.Timestamp('2023-07-01 12:00:00')

    ext_rad = self.solar_radiation.get_exterior_radiation(timestamp)

    sky_temp_direct = self.solar_radiation.get_current_sky_temperature(
        timestamp
    )
    irrad_direct = self.solar_radiation.get_current_irradiance(timestamp)

    self.assertAlmostEqual(ext_rad.sky_temp_k, sky_temp_direct, places=6)
    self.assertAlmostEqual(ext_rad.irradiance.ghi, irrad_direct.ghi, places=6)
    self.assertAlmostEqual(ext_rad.irradiance.dni, irrad_direct.dni, places=6)
    self.assertAlmostEqual(ext_rad.irradiance.dhi, irrad_direct.dhi, places=6)


# ---------------------------------------------------------------------------
# ReplaySolarRadiationTest – inherits all tests, swaps the controller
# ---------------------------------------------------------------------------
class ReplaySolarRadiationTest(SolarRadiationTest):
  """Re-runs all SolarRadiationTest tests with ReplayWeatherController."""

  def setUp(self):
    # Skip SolarRadiationTest.setUp; call IrradianceTestBase.setUp directly.
    super(SolarRadiationTest, self).setUp()  # pylint: disable=bad-super-call

    self.weather_controller = weather_controller.ReplayWeatherController(
        local_weather_path=_LOCAL_WEATHER_TEST_DATA_PATH,
        convection_coefficient=10.0,
    )
    self.building_info = solar_radiation.BuildingInfo(
        lat=_TEST_LATITUDE,
        lon=_TEST_LONGITUDE,
        time_zone=_TEST_TIMEZONE_PACIFIC,
    )
    self.solar_radiation = solar_radiation.SolarRadiation(
        building_info=self.building_info,
        weather_controller=self.weather_controller,
    )

  def test_get_current_irradiance_clearsky(self):
    """For Replay, 19:00 UTC ≈ noon local, cloud cover varies by CSV."""
    # ReplayWeatherController doesn't have a location for irradiance
    # unless we specify it.  Use a SolarRadiation with replay controller.
    sr = solar_radiation.SolarRadiation(
        building_info=self.building_info,
        weather_controller=self.weather_controller,
    )
    irrad = sr.get_current_irradiance(_DAYTIME_TIMESTAMP_PACIFIC)

    # Basic sanity
    self._validate_irradiance_components(irrad)
    self.assertGreater(irrad.ghi, 100)
    self.assertIsNotNone(irrad.timestamp)

    # Direct pvlib clearsky validation
    pvlib_loc = location.Location(
        self.latitude, self.longitude, tz=_TEST_TIMEZONE_PACIFIC
    )
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([_DAYTIME_TIMESTAMP_PACIFIC])
    )
    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

  def test_get_current_irradiance_campbell_norman_replay(self):
    """Campbell-Norman with cloud cover from CSV for ReplayWeatherController."""
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=self.latitude,
            lon=self.longitude,
            time_zone=_TEST_TIMEZONE_PACIFIC,
        ),
        weather_controller=self.weather_controller,
        cloud_cover=50.0,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('2023-07-01 12:00:00', tz=_TEST_TIMEZONE_PACIFIC)
    irrad = sr.get_current_irradiance(timestamp)

    self._validate_irradiance_components(irrad)

    pvlib_loc = self._make_pvlib_location(timezone=_TEST_TIMEZONE_PACIFIC)
    timestamp_utc = timestamp.tz_convert('UTC')
    solar_position = pvlib_loc.get_solarposition(
        pd.DatetimeIndex([timestamp_utc])
    )
    cloud_cover = 50.0
    transmittance = 0.7 - 0.5 * (cloud_cover / 100.0)
    dni_extra = irradiance.get_extra_radiation(
        pd.DatetimeIndex([timestamp_utc])
    )
    expected_irrad = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )
    self.assertAlmostEqual(irrad.ghi, expected_irrad['ghi'], delta=1.0)
    self.assertAlmostEqual(irrad.dni, expected_irrad['dni'], delta=1.0)
    self.assertAlmostEqual(irrad.dhi, expected_irrad['dhi'], delta=1.0)


# ---------------------------------------------------------------------------
# pvlib validation tests (clearsky, linear, campbell_norman consistency)
# ---------------------------------------------------------------------------
class SolarRadiationPvlibValidationTest(IrradianceTestBase):
  """Validate SolarRadiation against direct pvlib calculations."""

  def setUp(self):
    super().setUp()
    self.data_path = _LOCAL_WEATHER_TEST_DATA_PATH

  def _make_replay_sr(self, irradiance_method='campbell_norman', **kwargs):
    """Factory for SolarRadiation backed by the replay CSV."""
    wc = weather_controller.ReplayWeatherController(
        local_weather_path=self.data_path,
        convection_coefficient=10.0,
    )
    bi = solar_radiation.BuildingInfo(
        lat=self.latitude,
        lon=self.longitude,
        time_zone=_TEST_TIMEZONE_UTC,
    )
    return solar_radiation.SolarRadiation(
        building_info=bi,
        weather_controller=wc,
        irradiance_method=irradiance_method,
        **kwargs,
    )

  def test_clearsky_irradiance_matches_pvlib(self):
    """Clearsky SolarRadiation matches pvlib location.get_clearsky."""
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=self.latitude, lon=self.longitude, time_zone=_TEST_TIMEZONE_UTC
        )
    )
    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = self._make_pvlib_location()
    clearsky = pvlib_loc.get_clearsky(pd.DatetimeIndex([timestamp]))

    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

  def test_campbell_norman_consistency(self):
    """Campbell-Norman results match pvlib for given cloud cover."""
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=self.latitude,
            lon=self.longitude,
            time_zone=_TEST_TIMEZONE_UTC,
        ),
        cloud_cover=50.0,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = self._make_pvlib_location()
    solar_position = pvlib_loc.get_solarposition(pd.DatetimeIndex([timestamp]))
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (50.0 / 100.0)
    expected = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )

    self.assertAlmostEqual(irrad.ghi, expected['ghi'], places=2)
    self.assertAlmostEqual(irrad.dni, expected['dni'], places=2)
    self.assertAlmostEqual(irrad.dhi, expected['dhi'], places=2)

  def test_linear_method_consistency(self):
    """Linear method results match pvlib for given cloud cover."""
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=self.latitude,
            lon=self.longitude,
            time_zone=_TEST_TIMEZONE_UTC,
        ),
        cloud_cover=30.0,
        irradiance_method='linear',
    )

    timestamp = pd.Timestamp('2023-07-01 19:00:00+00:00')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = self._make_pvlib_location()
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )
    solar_position = pvlib_loc.get_solarposition(pd.DatetimeIndex([timestamp]))
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (1.0 - 0.8 * (30.0 / 100.0))
    dni_result = irradiance.disc(
        pd.Series([expected_ghi], index=pd.DatetimeIndex([timestamp])),
        solar_position['zenith'],
        pd.DatetimeIndex([timestamp]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    self.assertAlmostEqual(irrad.ghi, expected_ghi, places=2)
    self.assertAlmostEqual(irrad.dni, expected_dni, places=2)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, places=2)

  def test_irradiance_closure_equation(self):
    """GHI ≈ DNI * cos(zenith) + DHI for campbell_norman at multiple times."""
    sr = self._make_replay_sr(
        irradiance_method='campbell_norman', cloud_cover=0.0
    )
    pvlib_loc = self._make_pvlib_location()

    for time_str in [
        '2023-07-01 08:00:00+00:00',
        '2023-07-01 12:00:00+00:00',
        '2023-07-01 19:00:00+00:00',
    ]:
      timestamp = pd.Timestamp(time_str)
      irrad = sr.get_current_irradiance(timestamp)

      solar_position = pvlib_loc.get_solarposition(
          pd.DatetimeIndex([timestamp])
      )
      zenith_rad = np.radians(solar_position['apparent_zenith'].iloc[0])
      calculated_ghi = irrad.dni * np.cos(zenith_rad) + irrad.dhi

      if irrad.ghi > 10:
        self.assertAlmostEqual(
            calculated_ghi,
            irrad.ghi,
            delta=10.0,
            msg=f'Closure equation failed at {time_str}',
        )


# ---------------------------------------------------------------------------
# IrradianceDecompositionPvlibValidationTest (TMY3 data)
# ---------------------------------------------------------------------------
class IrradianceDecompositionPvlibValidationTest(parameterized.TestCase):
  """Validate pvlib irradiance decomposition methods using TMY3 data.

  Uses sample TMY3 data for Greensboro Piedmont Triad International Airport
  (USAF 723170) obtained from the pvlib-python repository:
  https://github.com/pvlib/pvlib-python/blob/main/pvlib/data/723170TYA.CSV

  Station metadata::

      USAF:      723170
      Name:      "GREENSBORO PIEDMONT TRIAD INT"
      State:     NC
      TZ:        -5.0 (US/Eastern)
      Latitude:  36.1
      Longitude: -79.95
      Altitude:  273.0 m

  For details on the TMY3 file format, see:

  > Wilcox, S and Marion, W. "Users Manual for TMY3 Data Sets (Revised).",
  > May. 2008. https://doi.org/10.2172/928611

  The test methods below compare our irradiance calculations against pvlib
  reference implementations, following the transposition-gain validation
  example in the pvlib documentation:
  https://github.com/pvlib/pvlib-python/blob/main/docs/examples/irradiance-transposition/plot_transposition_gain.py

  Attributes:
    tmy3_data: A ``pandas.DataFrame`` of TMY3 weather observations read by
      ``pvlib.iotools.read_tmy3`` with timestamps coerced to 1990.
    metadata: A ``dict`` of station metadata returned by ``read_tmy3``
      (latitude, longitude, altitude, time zone, etc.).
    solpos: A ``pandas.DataFrame`` of solar-position angles computed by
      ``pvlib.solarposition.get_solarposition`` at the midpoint of each
      hourly TMY interval.
  """

  @classmethod
  def setUpClass(cls):
    """Load TMY3 data and pre-compute solar positions for all tests.

    Reads the ``723170TYA.CSV`` file (Greensboro, NC) with pvlib's
    ``read_tmy3``, coercing all timestamps to 1990.  Solar positions are
    computed at the midpoint of each hourly interval (shifted by −30 min)
    following pvlib conventions for TMY data.
    """
    from pvlib.iotools import read_tmy3  # pylint: disable=import-outside-toplevel
    from pvlib.solarposition import get_solarposition  # pylint: disable=import-outside-toplevel

    tmy3_path = os.path.join(
        os.path.dirname(__file__),
        'solar_radiation_test_data',
        '723170TYA.CSV',
    )
    cls.tmy3_data, cls.metadata = read_tmy3(
        tmy3_path, coerce_year=1990, map_variables=True
    )
    cls.solpos = get_solarposition(
        cls.tmy3_data.index.shift(freq='-30min'),
        latitude=cls.metadata['latitude'],
        longitude=cls.metadata['longitude'],
        altitude=cls.metadata['altitude'],
        pressure=cls.tmy3_data.pressure * 100,
        temperature=cls.tmy3_data.temp_air,
    )
    cls.solpos.index = cls.tmy3_data.index

  def test_disc_method_matches_pvlib(self):
    out_disc = irradiance.disc(
        self.tmy3_data.ghi,
        self.solpos.zenith,
        self.tmy3_data.index,
        self.tmy3_data.pressure * 100,
    )
    df_disc = irradiance.complete_irradiance(
        solar_zenith=self.solpos.apparent_zenith,
        ghi=self.tmy3_data.ghi,
        dni=out_disc.dni,
        dhi=None,
    )
    for time_str in [
        '1990-07-04 12:00:00-05:00',
        '1990-07-04 13:00:00-05:00',
    ]:
      idx = pd.Timestamp(time_str)
      if idx in out_disc.index:
        zenith_rad = np.radians(self.solpos.apparent_zenith.loc[idx])
        calculated_ghi = (
            out_disc.dni.loc[idx] * np.cos(zenith_rad) + df_disc.dhi.loc[idx]
        )
        self.assertAlmostEqual(
            calculated_ghi,
            self.tmy3_data.ghi.loc[idx],
            delta=1.0,
            msg=f'DISC closure failed at {time_str}',
        )
        self.assertGreaterEqual(out_disc.dni.loc[idx], 0)

  def test_erbs_method_matches_pvlib(self):
    out_erbs = irradiance.erbs(
        self.tmy3_data.ghi, self.solpos.zenith, self.tmy3_data.index
    )
    for time_str in [
        '1990-04-04 12:00:00-05:00',
        '1990-01-04 12:00:00-05:00',
    ]:
      idx = pd.Timestamp(time_str)
      if idx in out_erbs.index:
        zenith_rad = np.radians(self.solpos.zenith.loc[idx])
        if np.cos(zenith_rad) > 0.1:
          calculated_ghi = (
              out_erbs.dni.loc[idx] * np.cos(zenith_rad) + out_erbs.dhi.loc[idx]
          )
          self.assertAlmostEqual(
              calculated_ghi,
              self.tmy3_data.ghi.loc[idx],
              delta=5.0,
              msg=f'Erbs closure failed at {time_str}',
          )

  def test_campbell_norman_method_matches_pvlib(self):
    test_time = pd.Timestamp('1990-07-04 12:00:00-05:00')
    for transmittance in [0.7, 0.5, 0.3]:
      dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([test_time]))
      result = irradiance.campbell_norman(
          self.solpos.apparent_zenith.loc[test_time],
          transmittance,
          dni_extra=dni_extra.iloc[0],
      )
      self.assertGreaterEqual(result['ghi'], 0)
      self.assertGreaterEqual(result['dni'], 0)
      self.assertGreaterEqual(result['dhi'], 0)

      zenith_rad = np.radians(self.solpos.apparent_zenith.loc[test_time])
      calculated_ghi = result['dni'] * np.cos(zenith_rad) + result['dhi']
      self.assertAlmostEqual(
          calculated_ghi,
          result['ghi'],
          delta=1.0,
          msg=f'Campbell-Norman closure failed for t={transmittance}',
      )

  def test_solar_radiation_campbell_norman_consistency(self):
    """SolarRadiation campbell_norman matches pvlib."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=latitude, lon=longitude, time_zone='US/Eastern'
        ),
        cloud_cover=50.0,
        irradiance_method='campbell_norman',
    )

    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = location.Location(latitude, longitude, tz='US/Eastern')
    solar_position = pvlib_loc.get_solarposition(pd.DatetimeIndex([timestamp]))
    dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
    transmittance = 0.7 - 0.5 * (50.0 / 100.0)
    expected = irradiance.campbell_norman(
        solar_position['apparent_zenith'].iloc[0],
        transmittance,
        dni_extra=dni_extra.iloc[0],
    )
    self.assertAlmostEqual(irrad.ghi, expected['ghi'], places=2)
    self.assertAlmostEqual(irrad.dni, expected['dni'], places=2)
    self.assertAlmostEqual(irrad.dhi, expected['dhi'], places=2)

  def test_solar_radiation_linear_consistency(self):
    """SolarRadiation linear method matches pvlib."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=latitude, lon=longitude, time_zone='US/Eastern'
        ),
        cloud_cover=30.0,
        irradiance_method='linear',
    )

    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = location.Location(latitude, longitude, tz='US/Eastern')
    clearsky = pvlib_loc.get_clearsky(
        pd.DatetimeIndex([timestamp]), model='ineichen'
    )
    solar_position = pvlib_loc.get_solarposition(pd.DatetimeIndex([timestamp]))
    expected_ghi = float(clearsky['ghi'].iloc[0]) * (1.0 - 0.8 * (30.0 / 100.0))
    dni_result = irradiance.disc(
        pd.Series([expected_ghi], index=pd.DatetimeIndex([timestamp])),
        solar_position['zenith'],
        pd.DatetimeIndex([timestamp]),
    )
    expected_dni = float(dni_result['dni'].iloc[0])
    zenith_rad = np.radians(solar_position['zenith'].iloc[0])
    expected_dhi = max(0, expected_ghi - expected_dni * np.cos(zenith_rad))

    self.assertAlmostEqual(irrad.ghi, expected_ghi, places=2)
    self.assertAlmostEqual(irrad.dni, expected_dni, places=2)
    self.assertAlmostEqual(irrad.dhi, expected_dhi, places=2)

  def test_solar_radiation_clearsky_matches_pvlib(self):
    """SolarRadiation clearsky matches pvlib location.get_clearsky."""
    latitude = self.metadata['latitude']
    longitude = self.metadata['longitude']
    sr = solar_radiation.SolarRadiation(
        building_info=solar_radiation.BuildingInfo(
            lat=latitude, lon=longitude, time_zone='US/Eastern'
        )
    )
    timestamp = pd.Timestamp('1990-07-04 12:00:00', tz='US/Eastern')
    irrad = sr.get_current_irradiance(timestamp)

    pvlib_loc = location.Location(latitude, longitude, tz='US/Eastern')
    clearsky = pvlib_loc.get_clearsky(pd.DatetimeIndex([timestamp]))

    self.assertAlmostEqual(irrad.ghi, clearsky['ghi'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dni, clearsky['dni'].iloc[0], places=4)
    self.assertAlmostEqual(irrad.dhi, clearsky['dhi'].iloc[0], places=4)

  @parameterized.named_parameters(
      ('winter_noon', '1990-01-04 12:00:00-05:00'),
      ('spring_noon', '1990-04-04 12:00:00-05:00'),
      ('summer_noon', '1990-07-04 12:00:00-05:00'),
  )
  def test_tmy3_irradiance_closure_equation(self, time_str):
    """TMY3 data satisfies closure: GHI = DNI*cos(z) + DHI."""
    idx = pd.Timestamp(time_str)
    ghi = self.tmy3_data.ghi.loc[idx]
    dni = self.tmy3_data.dni.loc[idx]
    dhi = self.tmy3_data.dhi.loc[idx]
    zenith = self.solpos.apparent_zenith.loc[idx]

    zenith_rad = np.radians(zenith)
    calculated_ghi = dni * np.cos(zenith_rad) + dhi

    self.assertAlmostEqual(
        calculated_ghi,
        ghi,
        delta=50.0,
        msg=f'TMY3 closure failed at {time_str}',
    )


# ---------------------------------------------------------------------------
# Replay helper function tests
# ---------------------------------------------------------------------------
def _make_observation_response(measurements, timestamp_seconds=1688212800):
  """Build a fake ObservationResponse proto for testing."""
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


class GetReplayTemperaturesTest(absltest.TestCase):
  """Tests for get_replay_temperatures."""

  def test_sensor_present(self):
    temp_k = utils.celsius_to_kelvin(25.0)
    obs = _make_observation_response({'outside_air_temperature_sensor': temp_k})
    result = solar_radiation.get_replay_temperatures([obs])
    self.assertLen(result, 1)
    self.assertAlmostEqual(list(result.values())[0], temp_k, places=4)

  def test_sensor_absent_returns_default(self):
    obs = _make_observation_response({'some_other_sensor': 300.0})
    result = solar_radiation.get_replay_temperatures([obs])
    self.assertLen(result, 1)
    self.assertEqual(list(result.values())[0], -1.0)

  def test_multiple_observations(self):
    obs1 = _make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(20.0)},
        timestamp_seconds=1688212800,
    )
    obs2 = _make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(25.0)},
        timestamp_seconds=1688216400,
    )
    result = solar_radiation.get_replay_temperatures([obs1, obs2])
    self.assertLen(result, 2)
    values = list(result.values())
    self.assertAlmostEqual(values[0], utils.celsius_to_kelvin(20.0), places=4)
    self.assertAlmostEqual(values[1], utils.celsius_to_kelvin(25.0), places=4)


class GetReplayCloudCoverTest(absltest.TestCase):
  """Tests for get_replay_cloud_cover."""

  def test_sensor_present(self):
    obs = _make_observation_response({'cloud_cover_sensor': 50.0})
    result = solar_radiation.get_replay_cloud_cover([obs])
    self.assertLen(result, 1)
    self.assertAlmostEqual(list(result.values())[0], 50.0, places=4)

  def test_sensor_absent_returns_default(self):
    obs = _make_observation_response({'some_other_sensor': 100.0})
    result = solar_radiation.get_replay_cloud_cover([obs])
    self.assertLen(result, 1)
    self.assertEqual(list(result.values())[0], 0.0)


class GetReplaySkyTemperatureTest(absltest.TestCase):
  """Tests for get_replay_sky_temperature."""

  def test_with_both_sensors(self):
    temp_k = utils.celsius_to_kelvin(23.0)
    dp_k = utils.celsius_to_kelvin(15.0)
    obs = _make_observation_response({
        'outside_air_temperature_sensor': temp_k,
        'dew_point_temperature_sensor': dp_k,
    })
    result = solar_radiation.get_replay_sky_temperature([obs])
    self.assertLen(result, 1)
    temp_sky_k = list(result.values())[0]

    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected, places=4)
    self.assertLess(temp_sky_k, temp_k)

  def test_missing_dewpoint_uses_depression(self):
    temp_k = utils.celsius_to_kelvin(23.0)
    depression = 8.0
    obs = _make_observation_response({'outside_air_temperature_sensor': temp_k})
    result = solar_radiation.get_replay_sky_temperature(
        [obs], dewpoint_depression=depression
    )
    self.assertLen(result, 1)
    temp_sky_k = list(result.values())[0]

    dp_k = temp_k - depression
    sigma = sim_constants.STEFAN_BOLTZMANN_CONSTANT
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    expected = (ir_h / sigma) ** 0.25
    self.assertAlmostEqual(temp_sky_k, expected, places=4)

  def test_missing_temp_skips_entry(self):
    obs_no_temp = _make_observation_response(
        {'dew_point_temperature_sensor': utils.celsius_to_kelvin(15.0)}
    )
    obs_with_temp = _make_observation_response(
        {'outside_air_temperature_sensor': utils.celsius_to_kelvin(23.0)}
    )
    result = solar_radiation.get_replay_sky_temperature(
        [obs_no_temp, obs_with_temp]
    )
    self.assertLen(result, 1)


class GetReplayIrradianceTest(absltest.TestCase):
  """Tests for get_replay_irradiance."""

  def test_sensors_present(self):
    obs = _make_observation_response({
        'ghi_sensor': 800.0,
        'dni_sensor': 700.0,
        'dhi_sensor': 100.0,
    })
    result = solar_radiation.get_replay_irradiance([obs])
    self.assertLen(result, 1)
    irrad = result[0]
    self.assertAlmostEqual(irrad.ghi, 800.0, places=4)
    self.assertAlmostEqual(irrad.dni, 700.0, places=4)
    self.assertAlmostEqual(irrad.dhi, 100.0, places=4)
    self.assertIsNotNone(irrad.timestamp)

  def test_sensors_absent_returns_defaults(self):
    obs = _make_observation_response({'some_other_sensor': 999.0})
    result = solar_radiation.get_replay_irradiance([obs])
    self.assertLen(result, 1)
    irrad = result[0]
    self.assertEqual(irrad.ghi, 0.0)
    self.assertEqual(irrad.dni, 0.0)
    self.assertEqual(irrad.dhi, 0.0)

  def test_multiple_observations(self):
    obs1 = _make_observation_response(
        {'ghi_sensor': 500.0, 'dni_sensor': 400.0, 'dhi_sensor': 100.0},
        timestamp_seconds=1688212800,
    )
    obs2 = _make_observation_response(
        {'ghi_sensor': 0.0, 'dni_sensor': 0.0, 'dhi_sensor': 0.0},
        timestamp_seconds=1688216400,
    )
    result = solar_radiation.get_replay_irradiance([obs1, obs2])
    self.assertLen(result, 2)
    self.assertAlmostEqual(result[0].ghi, 500.0, places=4)
    self.assertAlmostEqual(result[1].ghi, 0.0, places=4)


class GetObservationValueTest(absltest.TestCase):
  """Tests for _get_observation_value helper."""

  def test_value_found(self):
    obs = _make_observation_response({'ghi_sensor': 800.0})
    result = solar_radiation._get_observation_value(obs, 'ghi_sensor')  # pylint: disable=protected-access
    self.assertEqual(result, 800.0)

  def test_value_not_found_returns_default(self):
    obs = _make_observation_response({'ghi_sensor': 800.0})
    result = solar_radiation._get_observation_value(  # pylint: disable=protected-access
        obs, 'nonexistent_sensor', default=42.0
    )
    self.assertEqual(result, 42.0)

  def test_value_not_found_returns_none(self):
    obs = _make_observation_response({'ghi_sensor': 800.0})
    result = solar_radiation._get_observation_value(obs, 'nonexistent_sensor')  # pylint: disable=protected-access
    self.assertIsNone(result)


if __name__ == '__main__':
  absltest.main()
