"""Controls ambient temperature in simulator."""

import abc
import json
import math
import os
from typing import Final, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib import location
import pytz

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils as utils

TemperatureBounds = Tuple[float, float]

_SECONDS_IN_A_DAY: Final[float] = 24 * 3600
_DAYS_IN_A_YEAR: Final[int] = 365
_MIN_RADIANS: Final[float] = -math.pi / 2.0
_MAX_RADIANS: Final[float] = 3.0 * math.pi / 2.0
_EPOCH: Final[pd.Timestamp] = pd.Timestamp('1970-01-01', tz='UTC')

WEATHER_CSV_FILEPATH: Final[str] = os.path.join(
    os.path.dirname(__file__),
    '..',
    'configs',
    'resources',
    'sb1',
    'local_weather_moffett_field_20230701_20231122.csv',
)


def _parse_dms(dms_str: str) -> float:
  """Converts a DMS (degrees minutes seconds) string to decimal degrees.

  Args:
    dms_str: DMS string in the format "DD MM SS.SSS" (e.g. "37 24 35.000").
      Negative values are represented with a leading minus on the degrees part.

  Returns:
    Decimal degrees as a float.
  """
  parts = dms_str.strip().split()
  degrees = float(parts[0])
  minutes = float(parts[1]) if len(parts) > 1 else 0.0
  seconds = float(parts[2]) if len(parts) > 2 else 0.0
  sign = -1 if degrees < 0 else 1
  return degrees + sign * (minutes / 60.0 + seconds / 3600.0)


def load_station_info(station_json_path: str) -> tuple[float, float, str]:
  """Loads station location and timezone from a station JSON file.

  The JSON file is expected to have the following keys:
    - ``lat``: latitude in DMS format (e.g. "37 24 35.000")
    - ``lng``: longitude in DMS format (e.g. "-122 02 56.000")
    - ``timezone``: IANA timezone string (e.g. "America/Los_Angeles")

  Args:
    station_json_path: Path to the station JSON file.

  Returns:
    Tuple of (latitude, longitude, timezone) where lat/lon are decimal degrees
    and timezone is an IANA timezone string.
  """
  with open(station_json_path, encoding='utf-8') as f:
    station = json.load(f)
  latitude = _parse_dms(station['lat'])
  longitude = _parse_dms(station['lng'])
  timezone = station.get('timezone', 'UTC')
  return latitude, longitude, timezone


@gin.configurable
class BaseWeatherController(metaclass=abc.ABCMeta):
  """Represents the weather on any specific time."""

  @abc.abstractmethod
  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Gets outside temp at specified timestamp."""

  # SHOULD THIS BASE CLASS IMPLEMENT get_air_convection_coefficient AS WELL?


@gin.configurable
class WeatherController(BaseWeatherController):
  """Represents the weather on any given day.

  Weather operates as a sinusoid: low at midnight and high at noon.

  Attributes:
    default_low_temp: Default low temperature in K at midnight.
    default_high_temp: Default high temperature in K at noon.
    special_days: Map of day of year (1-365) to 2-tuple (low_temp, high_temp).
    convection_coefficient: Air convection coefficient (W/m2/K).
    timezone: Time zone for the weather data.
    latitude: Latitude for the weather data.
    longitude: Longitude for the weather data.
    dewpoint_depression: Difference between dry bulb and dew point temperatures
      in K. Default is 5.0 K.
    cloud_cover: Static cloud cover in percent (0-100). If None and
      cloud_cover_low/high not set, uses clearsky model.
    cloud_cover_low: Low cloud cover in percent at midnight for dynamic mode.
    cloud_cover_high: High cloud cover in percent at noon for dynamic mode.
    irradiance_method: Method for converting cloud cover to irradiance
      ('clearsky', 'linear', or 'campbell_norman'). Defaults to 'clearsky'.
  """

  def __init__(
      self,
      default_low_temp: float,
      default_high_temp: float,
      special_days: Optional[Mapping[int, TemperatureBounds]] = None,
      convection_coefficient: float = 12.0,
      timezone: str = 'UTC',
      latitude: float | None = None,
      longitude: float | None = None,
      dewpoint_depression: float = 5.0,
      cloud_cover: float | None = None,
      cloud_cover_low: float | None = None,
      cloud_cover_high: float | None = None,
      irradiance_method: str = 'clearsky',
  ):
    self.default_low_temp = default_low_temp
    self.default_high_temp = default_high_temp
    self.special_days = special_days if special_days else {}
    self.convection_coefficient = convection_coefficient
    self.timezone = timezone
    self.latitude = latitude
    self.longitude = longitude
    self.dewpoint_depression = dewpoint_depression  # Dry bulb - dew point (K)
    self.cloud_cover = cloud_cover
    self.cloud_cover_low = cloud_cover_low
    self.cloud_cover_high = cloud_cover_high
    self.irradiance_method = irradiance_method

    # Validate cloud_cover (static mode)
    if self.cloud_cover is not None:
      if self.cloud_cover < 0 or self.cloud_cover > 100:
        raise ValueError('cloud_cover must be between 0 and 100.')

    # Validate cloud_cover_low and cloud_cover_high (dynamic mode)
    if self.cloud_cover_low is not None or self.cloud_cover_high is not None:
      if self.cloud_cover_low is None or self.cloud_cover_high is None:
        raise ValueError(
            'Both cloud_cover_low and cloud_cover_high must be provided '
            'for dynamic cloud cover.'
        )
      if self.cloud_cover_low < 0 or self.cloud_cover_low > 100:
        raise ValueError('cloud_cover_low must be between 0 and 100.')
      if self.cloud_cover_high < 0 or self.cloud_cover_high > 100:
        raise ValueError('cloud_cover_high must be between 0 and 100.')
      if self.cloud_cover_low > self.cloud_cover_high:
        raise ValueError(
            'cloud_cover_low cannot be greater than cloud_cover_high.'
        )

    # Validate irradiance_method
    valid_methods = ('clearsky', 'linear', 'campbell_norman')
    if self.irradiance_method not in valid_methods:
      raise ValueError(
          f'irradiance_method must be one of {valid_methods}, '
          f'got {self.irradiance_method}.'
      )

    # Create location object if lat/lon are provided
    if self.latitude is not None and self.longitude is not None:
      self._location = location.Location(
          self.latitude, self.longitude, tz=self.timezone
      )
    else:
      self._location = None

    if self.default_low_temp > self.default_high_temp:
      raise ValueError(
          'default_low_temp cannot be greater than default_high_temp.'
      )

    for day, temps in self.special_days.items():
      low_temp, high_temp = temps
      if low_temp > high_temp:
        raise ValueError(
            f'Low temp cannot be greater than high temp for special day: {day}.'
        )

  def _ensure_timestamp_tz(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Ensure timestamp has timezone info, localizing if necessary.

    Args:
      timestamp: Pandas timestamp, may be naive or timezone-aware.

    Returns:
      Timezone-aware timestamp. If input was naive, localizes to self.tz.
      If input was already timezone-aware, converts to self.tz.
    """
    if timestamp.tzinfo is None:
      # Naive timestamp - localize to controller's timezone
      return timestamp.tz_localize(self.timezone)
    else:
      # Already timezone-aware - convert to controller's timezone
      return timestamp.tz_convert(self.timezone)

  def seconds_to_rads(self, seconds_in_day: int) -> float:
    """Returns radians corresponding to number of second in the day.

    Args:
      seconds_in_day: Seconds that have passed so far in the day.
    """
    return (seconds_in_day / _SECONDS_IN_A_DAY) * (
        _MAX_RADIANS - _MIN_RADIANS
    ) + _MIN_RADIANS

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns current temperature in K.

    Args:
      timestamp: Pandas timestamp to get temperature for. If naive (no tz),
        will be localized to the controller's timezone.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)
    today = timestamp.dayofyear
    tomorrow = (today + 1) % _DAYS_IN_A_YEAR

    if today in self.special_days:
      today_low, today_high = self.special_days[today]
    else:
      today_low, today_high = self.default_low_temp, self.default_high_temp

    if tomorrow in self.special_days:
      tomorrow_low, _ = self.special_days[tomorrow]
    else:
      tomorrow_low = self.default_low_temp

    high = today_high
    if timestamp.hour < 12:
      low = today_low
    else:
      low = tomorrow_low

    seconds_in_day = (timestamp - timestamp.normalize()).total_seconds()
    rad = self.seconds_to_rads(seconds_in_day)
    temp = 0.5 * (math.sin(rad) + 1) * (high - low) + low
    return temp

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    """Returns the convection coefficient (W/m2/K) based on the current wind.

    Args:
      timestamp: Pandas timestamp to get convection coefficient for. If naive
        (no timezone), will be localized to the controller's timezone.
    """
    return self.convection_coefficient

  def get_current_cloud_cover(self, timestamp: pd.Timestamp) -> float:
    """Returns current cloud cover in percent.

    Cloud cover can be:
    1. Static: Set via `cloud_cover` parameter (constant value)
    2. Dynamic: Set via `cloud_cover_low` and `cloud_cover_high` parameters
       (sinusoidal pattern: low at midnight, high at noon)
    3. Clearsky: If neither is set, returns 0.0

    Args:
      timestamp: Pandas timestamp to get cloud cover for. If naive (no timezone)
       , will be localized to the controller's timezone.

    Returns:
      Cloud cover in percent (0-100).
    """
    timestamp = self._ensure_timestamp_tz(timestamp)

    # Dynamic cloud cover mode (sinusoidal pattern like temperature)
    if self.cloud_cover_low is not None and self.cloud_cover_high is not None:
      seconds_in_day = (timestamp - timestamp.normalize()).total_seconds()
      rad = self.seconds_to_rads(seconds_in_day)
      # Same sinusoidal pattern as temperature: low at midnight, high at noon
      cloud_cover = (
          0.5
          * (math.sin(rad) + 1)
          * (self.cloud_cover_high - self.cloud_cover_low)
          + self.cloud_cover_low
      )
      return cloud_cover

    # Static cloud cover mode
    if self.cloud_cover is not None:
      return self.cloud_cover

    # Default: clearsky (no clouds)
    return 0.0

  def get_current_irradiance(
      self, timestamp: pd.Timestamp
  ) -> Mapping[str, float]:
    """Returns current irradiance (GHI, DNI, DHI) in W/m2.

    Uses clearsky model by default, or adjusts for cloud cover if specified.
    Supports both static cloud cover and dynamic cloud cover (sinusoidal).
    Consistent with ReplayWeatherController irradiance methods.

    Args:
      timestamp: Pandas timestamp to get irradiance for. If naive (no timezone),
        will be localized to the controller's timezone.

    Returns:
      Dictionary with 'ghi', 'dni', 'dhi', 'solar_zenith', and 'solar_azimuth'
      keys. Irradiance values in W/m2, angles in degrees.

    Raises:
      ValueError: If latitude/longitude not provided during initialization.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)

    if self._location is None:
      raise ValueError(
          'Latitude and longitude must be provided to calculate irradiance.'
      )

    # Get solar position (needed for all methods and output)
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    solar_zenith = float(solar_position['apparent_zenith'].iloc[0])
    solar_azimuth = float(solar_position['azimuth'].iloc[0])

    # Get current cloud cover (handles both static and dynamic modes)
    current_cloud_cover = self.get_current_cloud_cover(timestamp)

    # Check if we should use clearsky model
    has_cloud_cover = self.cloud_cover is not None or (
        self.cloud_cover_low is not None and self.cloud_cover_high is not None
    )

    # If no cloud cover configured or clearsky method, return clearsky irradiance # pylint: disable=line-too-long
    if not has_cloud_cover or self.irradiance_method == 'clearsky':
      clearsky = self._location.get_clearsky(pd.DatetimeIndex([timestamp]))
      return {
          'ghi': float(clearsky['ghi'].iloc[0]),
          'dni': float(clearsky['dni'].iloc[0]),
          'dhi': float(clearsky['dhi'].iloc[0]),
          'solar_zenith': solar_zenith,
          'solar_azimuth': solar_azimuth,
      }

    if self.irradiance_method == 'linear':
      # Get clear sky irradiance
      clearsky = self._location.get_clearsky(
          pd.DatetimeIndex([timestamp]), model='ineichen'
      )

      # Estimate GHI from cloud cover using linear relationship
      ghi = float(clearsky['ghi'].iloc[0]) * (
          1.0 - 0.8 * (current_cloud_cover / 100.0)
      )

      # Estimate DNI using DISC model
      dni_result = irradiance.disc(
          pd.Series([ghi], index=pd.DatetimeIndex([timestamp])),
          solar_position['zenith'],
          pd.DatetimeIndex([timestamp]),
      )
      dni = float(dni_result['dni'].iloc[0])

      # Calculate DHI
      zenith_rad = np.radians(solar_position['zenith'].iloc[0])
      dhi = ghi - dni * np.cos(zenith_rad)
      dhi = max(0, dhi)  # Ensure non-negative

    elif self.irradiance_method == 'campbell_norman':
      dni_extra = irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
      transmittance = 0.7 - 0.5 * (current_cloud_cover / 100.0)

      irrads = irradiance.campbell_norman(
          solar_position['apparent_zenith'].iloc[0],
          transmittance,
          dni_extra=dni_extra.iloc[0],
      )
      ghi = 0 if np.isnan(irrads['ghi']) else float(irrads['ghi'])
      dni = 0 if np.isnan(irrads['dni']) else float(irrads['dni'])
      dhi = 0 if np.isnan(irrads['dhi']) else float(irrads['dhi'])

    else:
      raise ValueError(f'Invalid irradiance_method: {self.irradiance_method}')

    return {
        'ghi': max(0, ghi),
        'dni': max(0, dni),
        'dhi': max(0, dhi),
        'solar_zenith': solar_zenith,
        'solar_azimuth': solar_azimuth,
    }

  def get_current_sky_temperature(self, timestamp: pd.Timestamp) -> float:
    """Returns sky temperature in K using Clark & Allen formula.

    Args:
      timestamp: Pandas timestamp to get sky temperature for. If naive
        (no timezone), will be localized to the controller's timezone.

    Returns:
      Sky temperature in K.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)

    # Stefan-Boltzmann constant
    sigma = 5.6697e-8  # W/(m^2*K^4)

    # Get dry bulb temperature (timestamp already localized)
    temp_k = self.get_current_temp(timestamp)

    # Estimate dew point temperature
    dp_k = temp_k - self.dewpoint_depression

    # Calculate sky emissivity (Clark & Allen)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)

    # Calculate horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k**4)

    # Calculate sky temperature
    temp_sky_k = (ir_h / sigma) ** 0.25

    return temp_sky_k


def get_replay_temperatures(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
) -> Mapping[str, float]:
  """Returns temperature replays from past observations.

  Args:
    observation_responses: array of observations to extract weather from

  Returns: map from timestamp to temp
  """

  def get_outside_air_temp(observation_response):
    for r in observation_response.single_observation_responses:
      if (
          r.single_observation_request.measurement_name
          == 'outside_air_temperature_sensor'
      ):
        return r.continuous_value
    return -1.0

  temps = {}
  for r in observation_responses:
    temp = get_outside_air_temp(r)
    time = utils.proto_to_pandas_timestamp(r.timestamp)
    temps[str(time)] = temp
  return temps


def get_replay_cloud_cover(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
) -> Mapping[str, float]:
  """Returns cloud cover replays from past observations.

  Args:
    observation_responses: array of observations to extract weather from

  Returns: map from timestamp to cloud cover (percent, 0-100)
  """

  def get_cloud_cover(observation_response):
    for r in observation_response.single_observation_responses:
      if r.single_observation_request.measurement_name == 'cloud_cover_sensor':
        return r.continuous_value
    return 0.0  # Default to clear sky

  cloud_covers = {}
  for r in observation_responses:
    cloud_cover = get_cloud_cover(r)
    time = utils.proto_to_pandas_timestamp(r.timestamp)
    cloud_covers[str(time)] = cloud_cover
  return cloud_covers


def get_replay_sky_temperature(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
    dewpoint_depression: float = 5.0,
) -> Mapping[str, float]:
  """Returns sky temperature replays from past observations.

  Calculates sky temperature using Clark & Allen formula from dry bulb
  temperature and dew point (estimated from dewpoint_depression if not
  available).

  Args:
    observation_responses: array of observations to extract weather from
    dewpoint_depression: Difference between dry bulb and dew point temperatures
      in K. Used if dew point sensor not available. Default is 5.0 K.

  Returns: map from timestamp to sky temperature (K)
  """
  # Stefan-Boltzmann constant
  sigma = 5.6697e-8  # W/(m^2*K^4)

  def get_value(observation_response, measurement_name):
    for r in observation_response.single_observation_responses:
      if r.single_observation_request.measurement_name == measurement_name:
        return r.continuous_value
    return None

  sky_temps = {}
  for r in observation_responses:
    # Get dry bulb temperature
    temp_k = get_value(r, 'outside_air_temperature_sensor')
    if temp_k is None:
      continue

    # Try to get dew point temperature, otherwise estimate from depression
    dp_k = get_value(r, 'dew_point_temperature_sensor')
    if dp_k is None:
      dp_k = temp_k - dewpoint_depression

    # Calculate sky emissivity (Clark & Allen)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)

    # Calculate horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k**4)

    # Calculate sky temperature
    temp_sky_k = (ir_h / sigma) ** 0.25

    time = utils.proto_to_pandas_timestamp(r.timestamp)
    sky_temps[str(time)] = temp_sky_k

  return sky_temps


def get_replay_irradiance(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
) -> Mapping[str, Mapping[str, float]]:
  """Returns irradiance replays from past observations.

  Args:
    observation_responses: array of observations to extract weather from

  Returns: map from timestamp to dict with 'ghi', 'dni', 'dhi' keys (W/m2)
  """

  def get_value(observation_response, measurement_name):
    for r in observation_response.single_observation_responses:
      if r.single_observation_request.measurement_name == measurement_name:
        return r.continuous_value
    return 0.0

  irradiances = {}
  for r in observation_responses:
    ghi = get_value(r, 'ghi_sensor')
    dni = get_value(r, 'dni_sensor')
    dhi = get_value(r, 'dhi_sensor')

    time = utils.proto_to_pandas_timestamp(r.timestamp)
    irradiances[str(time)] = {
        'ghi': ghi,
        'dni': dni,
        'dhi': dhi,
    }

  return irradiances


@gin.configurable
class ReplayWeatherController(BaseWeatherController):
  """Weather controller that interplolates real weather from past observations.

  Attributes:
    local_weather_path: Path to local weather CSV file.
    station_json_path: Optional path to station JSON file. When provided,
      latitude, longitude, and timezone are loaded from it. Explicit
      latitude, longitude, and timezone arguments take precedence.
    weather_df: Pandas dataframe of historical weather data.
    convection_coefficient: Air convection coefficient (W/m2/K).
    humidity_column: Column name of the humidity in the weather CSV file.
    timezone: IANA time zone string for the weather data. Loaded from
      station JSON when station_json_path is provided; defaults to 'UTC'.
    latitude: Latitude for the weather data.
    longitude: Longitude for the weather data.
    irradiance_method: Method for converting cloud cover to irradiance
      ('linear' or 'campbell_norman'). Defaults to 'campbell_norman'.
  """

  def __init__(
      self,
      local_weather_path: str,
      convection_coefficient: float = 12.0,
      station_json_path: str | None = None,
      timezone: str | None = None,
      latitude: float | None = None,
      longitude: float | None = None,
      irradiance_method: str = 'campbell_norman',
      humidity_column: str = 'Humidity',
  ):
    self.local_weather_path = local_weather_path
    self.convection_coefficient = convection_coefficient
    self.humidity_column = humidity_column

    # Load lat/lon/timezone from station JSON if provided; explicit args
    # override the values loaded from JSON.
    if station_json_path is not None:
      station_lat, station_lon, station_tz = load_station_info(
          station_json_path
      )
      self.latitude = latitude if latitude is not None else station_lat
      self.longitude = longitude if longitude is not None else station_lon
      self.timezone = timezone if timezone is not None else station_tz
    else:
      self.latitude = latitude
      self.longitude = longitude
      self.timezone = timezone or 'UTC'

    self._weather_data = self.read_weather_csv(self.local_weather_path)

    # Create location object if lat/lon are provided
    if self.latitude is not None and self.longitude is not None:
      self._location = location.Location(
          self.latitude, self.longitude, tz=self.timezone
      )
    else:
      self._location = None

    # Pre-calculate irradiance if location is provided
    if self._location is not None:
      self._calculate_irradiance_columns(irradiance_method)

    # Pre-calculate sky temperature
    self._calculate_sky_temperature_column()

  def _ensure_timestamp_tz(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Ensure timestamp has timezone info, localizing if necessary.

    Args:
      timestamp: Pandas timestamp, may be naive or timezone-aware.

    Returns:
      Timezone-aware timestamp. If input was naive, localizes to
      self.timezone. If input was already timezone-aware, converts to
      self.timezone.
    """
    if timestamp.tzinfo is None:
      # Naive timestamp - localize to controller's timezone
      return timestamp.tz_localize(self.timezone)
    else:
      # Already timezone-aware - convert to controller's timezone
      return timestamp.tz_convert(self.timezone)

  def _calculate_sky_temperature_column(self):
    """Pre-calculate sky temperature for all timestamps in weather data.

    Uses Clark & Allen formula with dry bulb and dew point temperatures.
    """
    # Stefan-Boltzmann constant
    sigma = 5.6697e-8  # W/(m^2*K^4)

    # Get dry bulb temperature in Kelvin (use numpy operations for arrays)
    if 'TempC' in self._weather_data:
      temp_k = self._weather_data['TempC'].values + 273.15
    elif 'TempF' in self._weather_data:
      temp_k = (self._weather_data['TempF'].values - 32) * 5.0 / 9.0 + 273.15
    else:
      raise ValueError(
          'Temperature column (TempC or TempF) not found in weather data.'
      )

    # Get dew point temperature in Kelvin (use numpy operations for arrays)
    if 'DewPointC' in self._weather_data:
      dp_k = self._weather_data['DewPointC'].values + 273.15
    elif 'DewPointF' in self._weather_data:
      dp_k = (self._weather_data['DewPointF'].values - 32) * 5.0 / 9.0 + 273.15
    else:
      raise ValueError(
          'Dew point temperature column (DewPointC or DewPointF) not found in'
          ' weather data.'
      )

    # Calculate sky emissivity (Clark & Allen)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)

    # Calculate horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k**4)

    # Calculate sky temperature
    temp_sky_k = (ir_h / sigma) ** 0.25

    # Store in dataframe
    self._weather_data['TempSkyC'] = temp_sky_k - 273.15  # Convert to Celsius

  def _calculate_irradiance_columns(self, method: str = 'campbell_norman'):
    """Pre-calculate irradiance (GHI, DNI, DHI) for all timestamps in weather
       data.

    Args:
      method: Method for converting cloud cover to irradiance
      ('linear' or 'campbell_norman').
    """
    # Handle SkyCoverage column
    if 'SkyCoverage' not in self._weather_data:
      # If no cloud cover data, use clearsky model
      clearsky = self._location.get_clearsky(
          self._weather_data['Time'], model='ineichen'
      )
      self._weather_data['ghi'] = clearsky['ghi'].values
      self._weather_data['dni'] = clearsky['dni'].values
      self._weather_data['dhi'] = clearsky['dhi'].values
      return

    # Replace invalid cloud coverage values with NaN
    self._weather_data.loc[
        (self._weather_data['SkyCoverage'] < 0)
        | (self._weather_data['SkyCoverage'] > 100),
        'SkyCoverage',
    ] = np.nan

    self._weather_data_ = self._weather_data.copy()
    self._weather_data_.set_index('Time', inplace=True)
    # If all SkyCoverage values are NaN, use clearsky model
    if self._weather_data['SkyCoverage'].isna().all():
      clearsky = self._location.get_clearsky(
          self._weather_data_['Time'], model='ineichen'
      )
      self._weather_data['ghi'] = clearsky['ghi'].values
      self._weather_data['dni'] = clearsky['dni'].values
      self._weather_data['dhi'] = clearsky['dhi'].values
      self._weather_data['SkyCoverage'] = 0
      return

    # Forward fill NaN values in cloud cover
    self._weather_data['SkyCoverage'] = self._weather_data[
        'SkyCoverage'
    ].ffill()

    # Get solar position for all timestamps
    solar_position = self._location.get_solarposition(self._weather_data_.index)

    if method == 'linear':
      # Create proper DatetimeIndex for pvlib functions
      datetime_index = pd.DatetimeIndex(self._weather_data['Time'])

      # Get clear sky irradiance
      clearsky = self._location.get_clearsky(datetime_index, model='ineichen')

      # Estimate GHI from cloud cover using linear relationship
      ghi = clearsky['ghi'].values * (
          1.0 - 0.8 * (self._weather_data['SkyCoverage'].values / 100.0)
      )

      # Estimate DNI using DISC model
      from pvlib.irradiance import disc  # pylint: disable=import-outside-toplevel

      dni_result = disc(
          pd.Series(ghi, index=datetime_index),
          solar_position['zenith'],
          datetime_index,
      )
      dni = dni_result['dni'].values

      # Calculate DHI
      zenith_rad = np.radians(solar_position['zenith'].values)
      dhi = ghi - dni * np.cos(zenith_rad)
      dhi = np.maximum(0, dhi)  # Ensure non-negative

    elif method == 'campbell_norman':
      from pvlib.irradiance import campbell_norman  # pylint: disable=import-outside-toplevel
      from pvlib.irradiance import get_extra_radiation  # pylint: disable=import-outside-toplevel

      dni_extra = get_extra_radiation(self._weather_data_.index)
      transmittance = 0.7 - 0.5 * (
          self._weather_data['SkyCoverage'].values / 100.0
      )

      # Calculate irradiance for each timestamp
      ghi_list = []
      dni_list = []
      dhi_list = []

      for i in range(len(self._weather_data)):
        irrads = campbell_norman(
            solar_position['apparent_zenith'].iloc[i],
            transmittance[i],
            dni_extra=dni_extra.iloc[i],
        )
        ghi_list.append(0 if np.isnan(irrads['ghi']) else irrads['ghi'])
        dni_list.append(0 if np.isnan(irrads['dni']) else irrads['dni'])
        dhi_list.append(0 if np.isnan(irrads['dhi']) else irrads['dhi'])

      ghi = np.array(ghi_list)
      dni = np.array(dni_list)
      dhi = np.array(dhi_list)

    else:
      raise ValueError(f'Invalid method: {method}')

    # Store in dataframe
    self._weather_data['ghi'] = np.maximum(0, ghi)
    self._weather_data['dni'] = np.maximum(0, dni)
    self._weather_data['dhi'] = np.maximum(0, dhi)

  @property
  def csv_filepath(self) -> str:
    """Alias for the local weather CSV file path."""
    return self.local_weather_path

  def read_weather_csv(self, csv_filepath: str) -> pd.DataFrame:
    """Loads time series weather data from the specified CSV file.

    The CSV file is expected to have at least the following columns:

      + `Time`: the time, as a string, in the format: `%Y%m%d-%H%M`
            (e.g. `20230701-0000`). Always interpreted as UTC regardless
            of the station timezone.
      + `TempF`: the temperature in Fahrenheit at the specified time.
      + `Humidity`: the relative humidity in percent at the specified time
            (0 to 100).

    Coerces the times to the station timezone (defaults to UTC). Falls
    back to UTC if the timezone causes DST ambiguity or non-existence
    errors (e.g. during clock changes). Updates the index to be seconds
    since epoch.

    Args:
      csv_filepath: Path to local weather CSV file.

    Returns:
      Pandas dataframe of weather data.
    """
    tz = self.timezone or 'UTC'
    df = pd.read_csv(csv_filepath)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    try:
      df['Time'] = [pd.Timestamp(t, tz=tz) for t in df['Time']]
    except (
        pytz.exceptions.AmbiguousTimeError,
        pytz.exceptions.NonExistentTimeError,
    ):
      df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df.index = (df['Time'] - _EPOCH).dt.total_seconds()
    df.index.name = 'SecondsSinceEpoch'
    return df

  @property
  def min_time(self) -> pd.Timestamp:
    """Earliest timestamp in the weather data."""
    return min(self._weather_data['Time'])

  @property
  def max_time(self) -> pd.Timestamp:
    """Latest timestamp in the weather data."""
    return max(self._weather_data['Time'])

  @property
  def times_in_seconds(self) -> pd.Index:
    """Returns the timestamps of the weather data, as seconds since epoch."""
    return self._weather_data.index

  @property
  def temps_f(self) -> pd.Series:
    """Returns the temperatures in Fahrenheit of the weather data."""
    return self._weather_data['TempF']

  @property
  def humidities(self) -> pd.Series:
    """Returns the humidities of the weather data."""
    return self._weather_data[self.humidity_column]

  def _get_interpolated_value(
      self, timestamp: pd.Timestamp, values: pd.Series
  ) -> float:
    """Helper to get interpolated value from a given series.

    The timestamp need not exactly appear in the weather data, but should be
    within the range of the data.
    If there is no exact match, linear interpolation is used to estimate the
    temperature between the nearest timestamps.

    Args:
      timestamp: Pandas timestamp to get temperature for interpolation. If naive
        (no timezone), will be localized to the controller's timezone. If the
        timestamp is timezone aware, it will be converted to UTC. If the
        timestamp is timezone naive, it will be localized to UTC. This allows
        for accurate comparisons against the min and max timestamps, as well as
        the epoch, which are always timezone aware (in UTC).
      values: Pandas series to interpolate from.

    Returns:
      The interpolated value from the series at the given timestamp.
    """
    # convert timestamp to UTC to enable proper comparisons:
    if timestamp.tzname() is not None:
      # timestamp is timezone aware, unable to localize, so convert to UTC:
      timestamp = self._ensure_timestamp_tz(timestamp)
    else:
      # timestamp is timezone naive, unable to convert, so localize to UTC:
      timestamp = timestamp.tz_localize('UTC')

    if timestamp < self.min_time:
      raise ValueError(
          f'Timestamp not in range. Timestamp {timestamp} is before the'
          f' earliest timestamp {self.min_time}.'
      )
    if timestamp > self.max_time:
      raise ValueError(
          f'Timestamp not in range. Timestamp {timestamp} is after the'
          f' latest timestamp {self.max_time}.'
      )

    time_in_seconds = (timestamp - _EPOCH).total_seconds()
    return np.interp(time_in_seconds, self.times_in_seconds, values)

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """For a given timestamp, returns the current temperature in Kelvin."""
    return utils.fahrenheit_to_kelvin(
        self._get_interpolated_value(timestamp, self.temps_f)
    )

  def get_current_humidity(self, timestamp: pd.Timestamp) -> float:
    """For a given timestamp, returns the current humidity level in percent."""
    return self._get_interpolated_value(timestamp, self.humidities)

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    """Returns the convection coefficient (W/m2/K).

    Args:
      timestamp: Pandas timestamp (unused but kept for API consistency).
    """
    return self.convection_coefficient

  def get_current_cloud_cover(self, timestamp: pd.Timestamp) -> float:
    """Returns current cloud cover in percent.

    Args:
      timestamp: Pandas timestamp to get cloud cover for. If naive (no tz),
        will be localized to the controller's timezone.

    Returns:
      Cloud cover in percent (0-100).

    Raises:
      ValueError: If timestamp is outside weather data range or no SkyCoverage
      column.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)
    min_time = min(self._weather_data['Time'])
    if timestamp < min_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, before the earliest'
          f' timestamp {min_time}.'
      )
    max_time = max(self._weather_data['Time'])
    if timestamp > max_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, after the latest'
          f' timestamp {max_time}.'
      )

    if 'SkyCoverage' not in self._weather_data:
      raise ValueError('SkyCoverage column not found in weather data.')

    times = np.array(self._weather_data.index)
    target_timestamp = (timestamp - _EPOCH).total_seconds()
    cloud_cover = np.interp(
        target_timestamp, times, self._weather_data['SkyCoverage']
    )
    return float(cloud_cover)

  def get_current_irradiance(
      self, timestamp: pd.Timestamp
  ) -> Mapping[str, float]:
    # pylint: disable=line-too-long
    """Returns current irradiance (GHI, DNI, DHI) and solar position by
       interpolating pre-calculated values.

    Args:
      timestamp: Pandas timestamp to get irradiance for. If naive (no timezone),
        will be localized to the controller's timezone.

    Returns:
      Dictionary with 'ghi', 'dni', 'dhi', 'solar_zenith', and 'solar_azimuth'
      keys. Irradiance values in W/m2, angles in degrees.

    Raises:
      ValueError: If latitude/longitude not provided or timestamp out of range
      or irradiance not calculated.

    Sources:
      https://pvlib-python.readthedocs.io/en/v0.6.1/_modules/pvlib/forecast.html#ForecastModel.cloud_cover_to_irradiance

    """
    # pylint: enable=line-too-long
    if self._location is None:
      raise ValueError(
          'Latitude and longitude must be provided to calculate irradiance.'
      )

    if (
        'ghi' not in self._weather_data
        or 'dni' not in self._weather_data
        or 'dhi' not in self._weather_data
    ):
      raise ValueError(
          'Irradiance data not available. Make sure latitude/longitude were'
          ' provided during initialization.'
      )

    timestamp = self._ensure_timestamp_tz(timestamp)
    min_time = min(self._weather_data['Time'])
    if timestamp < min_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, before the earliest'
          f' timestamp {min_time}.'
      )
    max_time = max(self._weather_data['Time'])
    if timestamp > max_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, after the latest'
          f' timestamp {max_time}.'
      )

    times = np.array(self._weather_data.index)
    target_timestamp = (timestamp - _EPOCH).total_seconds()

    # Interpolate GHI, DNI, DHI from pre-calculated values
    ghi = np.interp(target_timestamp, times, self._weather_data['ghi'])
    dni = np.interp(target_timestamp, times, self._weather_data['dni'])
    dhi = np.interp(target_timestamp, times, self._weather_data['dhi'])

    # Get solar position for this timestamp
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    return {
        'ghi': float(ghi),
        'dni': float(dni),
        'dhi': float(dhi),
        'solar_zenith': float(solar_position['apparent_zenith'].iloc[0]),
        'solar_azimuth': float(solar_position['azimuth'].iloc[0]),
    }

  def get_current_sky_temperature(self, timestamp: pd.Timestamp) -> float:
    """Returns sky temperature in K by interpolating pre-calculated values.

    Args:
      timestamp: Pandas timestamp to get sky temperature for. If naive
        (no timezone), will be localized to the controller's timezone.

    Returns:
      Sky temperature in K.

    Raises:
      ValueError: If timestamp is outside weather data range or sky temperature
      not calculated.
    """
    if 'TempSkyC' not in self._weather_data:
      raise ValueError(
          'Sky temperature data not available. This should have been calculated'
          ' during initialization.'
      )

    timestamp = self._ensure_timestamp_tz(timestamp)
    min_time = min(self._weather_data['Time'])
    if timestamp < min_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, before the earliest'
          f' timestamp {min_time}.'
      )
    max_time = max(self._weather_data['Time'])
    if timestamp > max_time:
      raise ValueError(
          f'Attempting to get weather data at {timestamp}, after the latest'
          f' timestamp {max_time}.'
      )

    times = np.array(self._weather_data.index)
    target_timestamp = (timestamp - _EPOCH).total_seconds()

    # Interpolate sky temperature from pre-calculated values
    temp_sky_c = np.interp(
        target_timestamp, times, self._weather_data['TempSkyC']
    )
    temp_sky_k = utils.celsius_to_kelvin(temp_sky_c)
    return temp_sky_k
