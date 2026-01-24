"""Controls ambient temperature in simulator."""

import abc
import math
from typing import Final, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib import location

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils as utils

TemperatureBounds = Tuple[float, float]

_SECONDS_IN_A_DAY: Final[float] = 24 * 3600
_DAYS_IN_A_YEAR: Final[int] = 365
_MIN_RADIANS: Final[float] = -math.pi / 2.0
_MAX_RADIANS: Final[float] = 3.0 * math.pi / 2.0
_EPOCH: Final[pd.Timestamp] = pd.Timestamp('1970-01-01', tz='UTC')


@gin.configurable
class BaseWeatherController(metaclass=abc.ABCMeta):
  """Represents the weather on any specific time."""

  @abc.abstractmethod
  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Gets outside temp at specified timestamp."""


@gin.configurable
class WeatherController(BaseWeatherController):
  """Represents the weather on any given day.

  Weather operates as a sinusoid: low at midnight and high at noon.

  Attributes:
    default_low_temp: Default low temperature in K at midnight.
    default_high_temp: Default high temperature in K at noon.
    special_days: Map of day of year (1-365) to 2-tuple (low_temp, high_temp).
    convection_coefficient: Air convection coefficient (W/m2/K).
    tz: Time zone for the weather data.
    latitude: Latitude for the weather data.
    longitude: Longitude for the weather data.
    dewpoint_depression: Difference between dry bulb and dew point temperatures
      in K. Default is 5.0 K.
    cloud_cover: Cloud cover in percent (0-100). If None, uses clearsky model.
    irradiance_method: Method for converting cloud cover to irradiance
      ('clearsky', 'linear', or 'campbell_norman'). Defaults to 'clearsky'.
  """

  def __init__(
      self,
      default_low_temp: float,
      default_high_temp: float,
      special_days: Optional[Mapping[int, TemperatureBounds]] = None,
      convection_coefficient: float = 12.0,
      tz: str = 'UTC',
      latitude: float | None = None,
      longitude: float | None = None,
      dewpoint_depression: float = 5.0,
      cloud_cover: float | None = None,
      irradiance_method: str = 'clearsky',
  ):
    self.default_low_temp = default_low_temp
    self.default_high_temp = default_high_temp
    self.special_days = special_days if special_days else {}
    self.convection_coefficient = convection_coefficient
    self.tz = tz
    self.latitude = latitude
    self.longitude = longitude
    self.dewpoint_depression = dewpoint_depression  # Dry bulb - dew point (K)
    self.cloud_cover = cloud_cover
    self.irradiance_method = irradiance_method

    # Validate cloud_cover
    if self.cloud_cover is not None:
      if self.cloud_cover < 0 or self.cloud_cover > 100:
        raise ValueError('cloud_cover must be between 0 and 100.')

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
          self.latitude, self.longitude, tz=self.tz
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
      timestamp: Pandas timestamp to get temperature for.
    """
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
      timestamp: Pandas timestamp to get convection coefficient for.
    """
    return self.convection_coefficient

  def get_current_cloud_cover(self, timestamp: pd.Timestamp) -> float:
    """Returns current cloud cover in percent.

    Args:
      timestamp: Pandas timestamp (unused, included for API consistency).

    Returns:
      Cloud cover in percent (0-100). Returns 0 if cloud_cover not set.
    """
    del timestamp  # Unused, but kept for API consistency with ReplayController
    return self.cloud_cover if self.cloud_cover is not None else 0.0

  def get_current_irradiance(
      self, timestamp: pd.Timestamp
  ) -> Mapping[str, float]:
    """Returns current irradiance (GHI, DNI, DHI) in W/m2.

    Uses clearsky model by default, or adjusts for cloud cover if specified.
    Consistent with ReplayWeatherController irradiance methods.

    Args:
      timestamp: Pandas timestamp to get irradiance for.

    Returns:
      Dictionary with 'ghi', 'dni', and 'dhi' keys.

    Raises:
      ValueError: If latitude/longitude not provided during initialization.
    """
    if self._location is None:
      raise ValueError(
          'Latitude and longitude must be provided to calculate irradiance.'
      )

    # If no cloud cover or clearsky method, return clearsky irradiance
    if self.cloud_cover is None or self.irradiance_method == 'clearsky':
      clearsky = self._location.get_clearsky(pd.DatetimeIndex([timestamp]))
      return {
          'ghi': float(clearsky['ghi'].iloc[0]),
          'dni': float(clearsky['dni'].iloc[0]),
          'dhi': float(clearsky['dhi'].iloc[0]),
      }

    # Get solar position
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    if self.irradiance_method == 'linear':
      # Get clear sky irradiance
      clearsky = self._location.get_clearsky(
          pd.DatetimeIndex([timestamp]), model='ineichen'
      )

      # Estimate GHI from cloud cover using linear relationship
      ghi = float(clearsky['ghi'].iloc[0]) * (
          1.0 - 0.8 * (self.cloud_cover / 100.0)
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
      transmittance = 0.7 - 0.5 * (self.cloud_cover / 100.0)

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
    }

  def get_irradiance_poa(
      self,
      timestamp: pd.Timestamp,
      surface_tilt: float,
      surface_azimuth: float,
  ) -> float:
    """Returns plane-of-array (POA) irradiance in W/m2.

    Args:
      timestamp: Pandas timestamp to get irradiance for.
      surface_tilt: Surface tilt angle in degrees.
      surface_azimuth: Surface azimuth angle in degrees.

    Returns:
      POA global irradiance in W/m2.

    Raises:
      ValueError: If latitude/longitude not provided during initialization.
    """
    if self._location is None:
      raise ValueError(
          'Latitude and longitude must be provided to calculate irradiance.'
      )

    # Get irradiance components
    irrad = self.get_current_irradiance(timestamp)

    # Get solar position
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    # Calculate POA irradiance
    poa_irrad = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=irrad['dni'],
        ghi=irrad['ghi'],
        dhi=irrad['dhi'],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )

    return float(poa_irrad['poa_global'])

  def get_current_sky_temperature(self, timestamp: pd.Timestamp) -> float:
    """Returns sky temperature in K using Clark & Allen formula.

    Args:
      timestamp: Pandas timestamp to get sky temperature for.

    Returns:
      Sky temperature in K.
    """
    # Stefan-Boltzmann constant
    sigma = 5.6697e-8  # W/(m^2*K^4)

    # Get dry bulb temperature
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


@gin.configurable
class ReplayWeatherController:
  """Weather controller that interplolates real weather from past observations.

  Attributes:
    local_weather_path: Path to local weather file.
    convection_coefficient: Air convection coefficient (W/m2/K).
    tz: Time zone for the weather data.
    latitude: Latitude for the weather data.
    longitude: Longitude for the weather data.
    irradiance_method: Method for converting cloud cover to irradiance
      ('linear' or 'campbell_norman'). Defaults to 'campbell_norman'.
  """

  def __init__(
      self,
      local_weather_path: str,
      convection_coefficient: float = 12.0,
      tz: str = 'UTC',
      latitude: float | None = None,
      longitude: float | None = None,
      irradiance_method: str = 'campbell_norman',
  ):
    self._weather_data = pd.read_csv(local_weather_path)
    self._weather_data['Time'] = [
        pd.Timestamp(t, tz=tz) for t in self._weather_data['Time']
    ]
    self._weather_data.index = [
        (t - _EPOCH).total_seconds() for t in self._weather_data['Time']
    ]
    self.convection_coefficient = convection_coefficient
    self.tz = tz
    self.latitude = latitude
    self.longitude = longitude

    # Create location object if lat/lon are provided
    if self.latitude is not None and self.longitude is not None:
      self._location = location.Location(
          self.latitude, self.longitude, tz=self.tz
      )
    else:
      self._location = None

    # Pre-calculate irradiance if location is provided
    if self._location is not None:
      self._calculate_irradiance_columns(irradiance_method)

    # Pre-calculate sky temperature (doesn't require location, only temp/dewpoint) # pylint: disable=line-too-long
    self._calculate_sky_temperature_column()

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

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns current temperature in K.

    Args:
      timestamp: Pandas timestamp to get temperature for interpolation.
    """
    timestamp = timestamp.tz_convert(self.tz)
    min_time = min(self._weather_data['Time'])
    if timestamp < min_time:

      raise ValueError(
          f'Attempting to get weather data at {timestamp}, before the latest'
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

    if 'TempC' in self._weather_data:
      temps = self._weather_data['TempC']
      temp_c = np.interp(target_timestamp, times, temps)
      return utils.celsius_to_kelvin(temp_c)
    else:
      temps = self._weather_data['TempF']
      temp_f = np.interp(target_timestamp, times, temps)
      return utils.fahrenheit_to_kelvin(temp_f)

  # pylint: disable=unused-argument
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    return self.convection_coefficient

  def get_current_cloud_cover(self, timestamp: pd.Timestamp) -> float:
    """Returns current cloud cover in percent.

    Args:
      timestamp: Pandas timestamp to get cloud cover for.

    Returns:
      Cloud cover in percent (0-100).

    Raises:
      ValueError: If timestamp is outside weather data range or no SkyCoverage
      column.
    """
    timestamp = timestamp.tz_convert(self.tz)
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
    """Returns current irradiance (GHI, DNI, DHI) in W/m2 by interpolating
       pre-calculated values.

    Args:
      timestamp: Pandas timestamp to get irradiance for.

    Returns:
      Dictionary with 'ghi', 'dni', and 'dhi' keys.

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

    timestamp = timestamp.tz_convert(self.tz)
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

    return {
        'ghi': float(ghi),
        'dni': float(dni),
        'dhi': float(dhi),
    }

  def get_irradiance_poa(
      self,
      timestamp: pd.Timestamp,
      surface_tilt: float,
      surface_azimuth: float,
  ) -> float:
    """Returns plane-of-array (POA) irradiance in W/m2.

    Args:
      timestamp: Pandas timestamp to get irradiance for.
      surface_tilt: Surface tilt angle in degrees.
      surface_azimuth: Surface azimuth angle in degrees.

    Returns:
      POA global irradiance in W/m2.

    Raises:
      ValueError: If latitude/longitude not provided or timestamp out of range.
    """
    if self._location is None:
      raise ValueError(
          'Latitude and longitude must be provided to calculate irradiance.'
      )

    # Get irradiance components from pre-calculated values
    irrad = self.get_current_irradiance(timestamp)

    # Get solar position
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )

    # Calculate POA irradiance
    poa_irrad = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=irrad['dni'],
        ghi=irrad['ghi'],
        dhi=irrad['dhi'],
        solar_zenith=solar_position['apparent_zenith'].iloc[0],
        solar_azimuth=solar_position['azimuth'].iloc[0],
    )

    return float(poa_irrad['poa_global'])

  def get_current_sky_temperature(self, timestamp: pd.Timestamp) -> float:
    """Returns sky temperature in K by interpolating pre-calculated values.

    Args:
      timestamp: Pandas timestamp to get sky temperature for.

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

    timestamp = timestamp.tz_convert(self.tz)
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
