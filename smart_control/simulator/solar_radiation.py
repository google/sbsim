"""Solar Radiation Calculations for Building Simulation.

For computing irradiance components, solar position, and sky temperature.
"""

import dataclasses
import math
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from pvlib import irradiance as pvlib_irradiance
from pvlib import location as pvlib_location

from smart_control.simulator import constants
from smart_control.simulator import weather_controller as wc_module
from smart_control.utils import conversion_utils as utils

# ---------------------------------------------------------------------------
# Valid irradiance method names
# ---------------------------------------------------------------------------
IRRADIANCE_METHODS = ('clearsky', 'linear', 'campbell_norman')

# ---------------------------------------------------------------------------
# Sensor measurement names used when reading observation responses
# ---------------------------------------------------------------------------
GHI_SENSOR: Final[str] = 'ghi_sensor'
DNI_SENSOR: Final[str] = 'dni_sensor'
DHI_SENSOR: Final[str] = 'dhi_sensor'

# Internal constants
_SECONDS_IN_A_DAY: Final[float] = 24 * 3600
_MIN_RADIANS: Final[float] = -math.pi / 2.0
_MAX_RADIANS: Final[float] = 3.0 * math.pi / 2.0

# Sensor measurement names used in replay helper functions
_OUTSIDE_AIR_TEMP_SENSOR: Final[str] = 'outside_air_temperature_sensor'
_DEW_POINT_TEMP_SENSOR: Final[str] = 'dew_point_temperature_sensor'
_CLOUD_COVER_SENSOR: Final[str] = 'cloud_cover_sensor'


# ---------------------------------------------------------------------------
# BuildingInfo
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class BuildingInfo:
  """Information about the building under control.

  On construction, `floor_plan_orientation` is validated to be within
  [0, 360] degrees; a `ValueError` is raised otherwise.

  Attributes:
    floor_plan_filepath: Path to the building's floor-plan `.npy` file.
    floor_plan_orientation: Compass angle (degrees) of the floor-plan's
      "up" direction.  0 / 360 = North, 90 = East, 180 = South, 270 = West.
      Must be between 0 and 360 inclusive.
    lat: Latitude of the building in decimal degrees.
    lon: Longitude of the building in decimal degrees.
    time_zone: IANA time-zone string (e.g. `"US/Pacific"`).
    altitude: Altitude above sea level in metres.  When *None* pvlib will
      attempt to look it up automatically.
  """

  floor_plan_filepath: str = ''
  floor_plan_orientation: float = 0.0
  lat: float = 37.4263
  lon: float = -122.0349
  time_zone: str = 'US/Pacific'
  altitude: float | None = None

  def __post_init__(self):
    self._validate_floor_plan_orientation()

  def _validate_floor_plan_orientation(self) -> None:
    """Raise if `floor_plan_orientation` is outside [0, 360]."""
    if self.floor_plan_orientation < 0 or self.floor_plan_orientation > 360:
      raise ValueError(
          'Expecting floor_plan_orientation to be between 0 and 360, '
          f'but got {self.floor_plan_orientation}.'
      )


# ---------------------------------------------------------------------------
# IrradianceComponents
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class IrradianceComponents:
  """Irradiance components and solar position at a given timestamp.

  Attributes:
    timestamp: Pandas timestamp for the measurement.  May be *None* when
      the timestamp is not available (e.g. when constructed from a weather
      controller that does not track it).
    ghi: Global Horizontal Irradiance in W/m².  Total solar radiation
      received on a horizontal surface.
    dni: Direct Normal Irradiance in W/m².  Solar radiation received
      perpendicular to the sun's rays.
    dhi: Diffuse Horizontal Irradiance in W/m².  Solar radiation received
      on a horizontal surface from the sky (excluding direct beam).
    solar_zenith: Solar zenith angle in degrees (angle from vertical).
      0 = sun directly overhead, 90 = sun at horizon.
    solar_azimuth: Solar azimuth angle in degrees (compass direction of
      the sun).  0 / 360 = North, 90 = East, 180 = South, 270 = West.
  """

  ghi: float
  dni: float
  dhi: float
  solar_zenith: float
  solar_azimuth: float
  timestamp: pd.Timestamp | None = None


@dataclasses.dataclass
class ExteriorRadiationData:
  """Full exterior radiation state for a single timestep.

  Bundles the ambient dry-bulb temperature, sky temperature (for longwave
  radiation), and irradiance components (for shortwave radiation) so that
  the simulator can request all exterior radiation quantities in a single
  call to :meth:`SolarRadiation.get_exterior_radiation`.

  Attributes:
    timestamp: Pandas timestamp for the observation.
    ambient_temp_k: Outdoor dry-bulb temperature in Kelvin.
    sky_temp_k: Sky temperature in Kelvin (Clark & Allen formula).  Used
      for exterior longwave radiative heat transfer (LWR).
    irradiance: Shortwave irradiance components (GHI, DNI, DHI) and solar
      position.  Used for solar gain through fenestrations.
  """

  timestamp: pd.Timestamp
  ambient_temp_k: float
  sky_temp_k: float
  irradiance: IrradianceComponents


# ---------------------------------------------------------------------------
# SolarRadiation calculator
# ---------------------------------------------------------------------------
class SolarRadiation:
  """Calculates solar radiation for a building location.

  Combines a :class:`simulator.weather_controller.BaseWeatherController`
  with cloud-cover and irradiance-method settings to produce
  :class:`IrradianceComponents` and sky-temperature values.

  Args:
    building_info: Location metadata.  Uses defaults (Mountain View, CA) if
      not provided.
    weather_controller: A weather controller instance used to obtain
      temperature (for sky-temperature calculations).
    dewpoint_depression: Difference between dry-bulb and dew-point
      temperatures in K.  Used when no dew-point sensor is available.
    cloud_cover: Static cloud cover in percent (0–100).  If *None* and no
      dynamic cloud cover is configured, the clearsky model is used.
    cloud_cover_low: Low cloud cover in percent at midnight (dynamic mode).
    cloud_cover_high: High cloud cover in percent at noon (dynamic mode).
    irradiance_method: One of `'clearsky'`, `'linear'`, or
      `'campbell_norman'`.
  """

  def __init__(
      self,
      building_info: BuildingInfo | None = None,
      weather_controller: wc_module.BaseWeatherController | None = None,
      dewpoint_depression: float = 5.0,
      cloud_cover: float | None = None,
      cloud_cover_low: float | None = None,
      cloud_cover_high: float | None = None,
      irradiance_method: str = 'clearsky',
  ):
    self.building_info = building_info or BuildingInfo()
    self.time_zone = self.building_info.time_zone
    self.lat = self.building_info.lat
    self.lon = self.building_info.lon
    self.weather_controller = weather_controller
    self.dewpoint_depression = dewpoint_depression
    self.cloud_cover = cloud_cover
    self.cloud_cover_low = cloud_cover_low
    self.cloud_cover_high = cloud_cover_high
    self.irradiance_method = irradiance_method

    self._validate_irradiance_method()
    self._validate_cloud_cover()
    self._location = self._get_pvlib_location()

  # ----- validation --------------------------------------------------------

  def _validate_irradiance_method(self) -> None:
    """Raise `ValueError` if `irradiance_method` is not recognised."""
    if self.irradiance_method not in IRRADIANCE_METHODS:
      raise ValueError(
          f'irradiance_method must be one of {IRRADIANCE_METHODS}, '
          f'got {self.irradiance_method!r}.'
      )

  def _validate_cloud_cover(self) -> None:
    """Raise `ValueError` if cloud-cover parameters are invalid."""
    if self.cloud_cover is not None:
      if self.cloud_cover < 0 or self.cloud_cover > 100:
        raise ValueError('cloud_cover must be between 0 and 100.')

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

  def _get_pvlib_location(self) -> pvlib_location.Location:
    """Construct and return the pvlib `Location` for this building."""
    kwargs: dict = dict(
        latitude=self.lat, longitude=self.lon, tz=self.time_zone
    )
    if self.building_info.altitude is not None:
      kwargs['altitude'] = self.building_info.altitude
    return pvlib_location.Location(**kwargs)

  # ----- timestamp helpers ------------------------------------------------

  def _ensure_timestamp_tz(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Ensure *timestamp* has timezone info, localising if necessary."""
    if timestamp.tzinfo is None:
      return timestamp.tz_localize(self.time_zone)
    return timestamp.tz_convert(self.time_zone)

  @staticmethod
  def _seconds_to_rads(seconds_in_day: float) -> float:
    """Map seconds-in-day to a sinusoidal radian value."""
    return (seconds_in_day / _SECONDS_IN_A_DAY) * (
        _MAX_RADIANS - _MIN_RADIANS
    ) + _MIN_RADIANS

  # ----- cloud cover ------------------------------------------------------

  def get_current_cloud_cover(self, timestamp: pd.Timestamp) -> float:
    """Return current cloud cover in percent (0–100).

    Supports static, dynamic (sinusoidal), and clearsky (returns 0.0) modes.

    Args:
      timestamp: Pandas timestamp.  If naive, will be localised to the
        building's timezone.

    Returns:
      Cloud cover in percent.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)

    # Dynamic cloud cover (sinusoidal like temperature)
    if self.cloud_cover_low is not None and self.cloud_cover_high is not None:
      seconds_in_day = (timestamp - timestamp.normalize()).total_seconds()
      rad = self._seconds_to_rads(seconds_in_day)
      return (
          0.5
          * (math.sin(rad) + 1)
          * (self.cloud_cover_high - self.cloud_cover_low)
          + self.cloud_cover_low
      )

    # Static cloud cover
    if self.cloud_cover is not None:
      return self.cloud_cover

    # Default: clearsky
    return 0.0

  # ----- irradiance -------------------------------------------------------

  def get_current_irradiance(
      self, timestamp: pd.Timestamp
  ) -> IrradianceComponents:
    """Return current irradiance (GHI, DNI, DHI) in W/m².

    Uses the clearsky model by default, or adjusts for cloud cover when
    configured.

    Args:
      timestamp: Pandas timestamp.  If naive, will be localised to the
        building's timezone.

    Returns:
      :class:`IrradianceComponents` with GHI, DNI, DHI and solar position.
    """
    timestamp = self._ensure_timestamp_tz(timestamp)

    # Solar position (needed for all methods and output)
    solar_position = self._location.get_solarposition(
        pd.DatetimeIndex([timestamp])
    )
    solar_zenith = float(solar_position['apparent_zenith'].iloc[0])
    solar_azimuth = float(solar_position['azimuth'].iloc[0])

    current_cloud_cover = self.get_current_cloud_cover(timestamp)

    has_cloud_cover = self.cloud_cover is not None or (
        self.cloud_cover_low is not None and self.cloud_cover_high is not None
    )

    # Clearsky model
    if not has_cloud_cover or self.irradiance_method == 'clearsky':
      clearsky = self._location.get_clearsky(pd.DatetimeIndex([timestamp]))
      return IrradianceComponents(
          ghi=float(clearsky['ghi'].iloc[0]),
          dni=float(clearsky['dni'].iloc[0]),
          dhi=float(clearsky['dhi'].iloc[0]),
          solar_zenith=solar_zenith,
          solar_azimuth=solar_azimuth,
          timestamp=timestamp,
      )

    if self.irradiance_method == 'linear':
      clearsky = self._location.get_clearsky(
          pd.DatetimeIndex([timestamp]), model='ineichen'
      )
      ghi = float(clearsky['ghi'].iloc[0]) * (
          1.0 - 0.8 * (current_cloud_cover / 100.0)
      )
      dni_result = pvlib_irradiance.disc(
          pd.Series([ghi], index=pd.DatetimeIndex([timestamp])),
          solar_position['zenith'],
          pd.DatetimeIndex([timestamp]),
      )
      dni = float(dni_result['dni'].iloc[0])
      zenith_rad = np.radians(solar_position['zenith'].iloc[0])
      dhi = max(0, ghi - dni * np.cos(zenith_rad))

    elif self.irradiance_method == 'campbell_norman':
      dni_extra = pvlib_irradiance.get_extra_radiation(
          pd.DatetimeIndex([timestamp])
      )
      transmittance = 0.7 - 0.5 * (current_cloud_cover / 100.0)
      irrads = pvlib_irradiance.campbell_norman(
          solar_position['apparent_zenith'].iloc[0],
          transmittance,
          dni_extra=dni_extra.iloc[0],
      )
      ghi = 0 if np.isnan(irrads['ghi']) else float(irrads['ghi'])
      dni = 0 if np.isnan(irrads['dni']) else float(irrads['dni'])
      dhi = 0 if np.isnan(irrads['dhi']) else float(irrads['dhi'])

    else:
      raise ValueError(f'Invalid irradiance_method: {self.irradiance_method}')

    return IrradianceComponents(
        ghi=max(0, ghi),
        dni=max(0, dni),
        dhi=max(0, dhi),
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        timestamp=timestamp,
    )

  # ----- sky temperature ---------------------------------------------------

  def get_current_sky_temperature(self, timestamp: pd.Timestamp) -> float:
    """Return sky temperature in K using the Clark & Allen formula.

    Requires `weather_controller` to have been set during construction so
    that the dry-bulb temperature can be obtained.

    Args:
      timestamp: Pandas timestamp.  Passed as-is to the weather controller's
        `get_current_temp` method.

    Returns:
      Sky temperature in K.

    Raises:
      ValueError: If no weather controller was provided.
    """
    if self.weather_controller is None:
      raise ValueError(
          'A weather_controller must be provided to compute sky temperature.'
      )

    sigma = constants.STEFAN_BOLTZMANN_CONSTANT

    # Dry-bulb temperature from the weather controller.
    # Pass the original timestamp so the controller can apply its own
    # timezone handling (the sinusoidal WeatherController expects naive
    # timestamps while ReplayWeatherController handles both).
    temp_k = self.weather_controller.get_current_temp(timestamp)

    # Estimate dew-point temperature
    dp_k = temp_k - self.dewpoint_depression

    # Sky emissivity (Clark & Allen)
    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)

    # Horizontal infrared radiation
    ir_h = epsilon_sky * sigma * (temp_k**4)

    # Sky temperature
    temp_sky_k = (ir_h / sigma) ** 0.25

    return float(temp_sky_k)

  # ----- combined exterior radiation state ---------------------------------

  def get_exterior_radiation(
      self, timestamp: pd.Timestamp
  ) -> ExteriorRadiationData:
    """Return ambient temperature, sky temperature, and irradiance at once.

    Convenience method for the simulator that needs all exterior radiation
    quantities (convection temperature, longwave sky temperature, and
    shortwave irradiance) in a single call per timestep.

    Args:
      timestamp: Pandas timestamp.  If naive, will be localised to the
        building's timezone for irradiance and sky-temperature calculations.
        Passed as-is to `weather_controller.get_current_temp()`.

    Returns:
      :class:`ExteriorRadiationData` with `ambient_temp_k`,
      `sky_temp_k`, and `irradiance`.

    Raises:
      ValueError: If no weather controller was provided.
    """
    if self.weather_controller is None:
      raise ValueError(
          'A weather_controller must be provided to compute exterior radiation.'
      )

    # Ambient temperature (naive-safe: each WC handles tz internally)
    ambient_temp_k = float(self.weather_controller.get_current_temp(timestamp))

    # Sky temperature (same tz rules as get_current_sky_temperature)
    sky_temp_k = self.get_current_sky_temperature(timestamp)

    # Irradiance components + solar position
    irrad = self.get_current_irradiance(timestamp)

    # Normalise timestamp for the return value
    ts = (
        self._ensure_timestamp_tz(timestamp)
        if timestamp.tzinfo is None
        else timestamp
    )

    return ExteriorRadiationData(
        timestamp=ts,
        ambient_temp_k=ambient_temp_k,
        sky_temp_k=sky_temp_k,
        irradiance=irrad,
    )


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def _get_observation_value(
    observation_response,
    measurement_name: str,
    default=None,
):
  """Return the continuous value for a named measurement in an observation.

  Searches the `single_observation_responses` of the given
  *observation_response* for an entry whose `measurement_name` matches the
  requested name.

  Args:
    observation_response: A single `ObservationResponse` proto.
    measurement_name: The sensor / measurement name to look up.
    default: Value to return when the measurement is not found.

  Returns:
    The `continuous_value` of the matching observation, or *default* if no
    matching measurement is found.
  """
  for r in observation_response.single_observation_responses:
    if r.single_observation_request.measurement_name == measurement_name:
      return r.continuous_value
  return default


def get_replay_irradiance(
    observation_responses: Sequence[object],
) -> Sequence[IrradianceComponents]:
  """Extract irradiance data from past observation protos.

  Iterates over *observation_responses* and reads the `ghi_sensor`,
  `dni_sensor`, and `dhi_sensor` measurements.  Solar zenith and azimuth
  are set to 0.0 because they are not typically recorded in observation
  protos.

  Args:
    observation_responses: Sequence of `ObservationResponse` protos.

  Returns:
    A list of :class:`IrradianceComponents`, one per observation.
  """
  irradiances: list[IrradianceComponents] = []
  for r in observation_responses:
    ghi = _get_observation_value(r, GHI_SENSOR, default=0.0)
    dni = _get_observation_value(r, DNI_SENSOR, default=0.0)
    dhi = _get_observation_value(r, DHI_SENSOR, default=0.0)

    timestamp = utils.proto_to_pandas_timestamp(r.timestamp)
    irradiances.append(
        IrradianceComponents(
            ghi=ghi,
            dni=dni,
            dhi=dhi,
            solar_zenith=0.0,
            solar_azimuth=0.0,
            timestamp=timestamp,
        )
    )

  return irradiances


def get_replay_temperatures(
    observation_responses: Sequence[object],
) -> Mapping[str, float]:
  """Return temperature replays from past observations.

  Args:
    observation_responses: Sequence of `ObservationResponse` protos.

  Returns:
    Mapping from timestamp string to temperature in Kelvin.  Entries missing
    the `outside_air_temperature_sensor` measurement are mapped to -1.0.
  """
  temps: dict[str, float] = {}
  for r in observation_responses:
    temp = _get_observation_value(r, _OUTSIDE_AIR_TEMP_SENSOR, default=-1.0)
    timestamp = utils.proto_to_pandas_timestamp(r.timestamp)
    temps[str(timestamp)] = temp
  return temps


def get_replay_cloud_cover(
    observation_responses: Sequence[object],
) -> Mapping[str, float]:
  """Return cloud cover replays from past observations.

  Args:
    observation_responses: Sequence of `ObservationResponse` protos.

  Returns:
    Mapping from timestamp string to cloud cover in percent (0–100).
    Entries missing the `cloud_cover_sensor` measurement default to 0.0
    (clear sky).
  """
  cloud_covers: dict[str, float] = {}
  for r in observation_responses:
    cloud_cover = _get_observation_value(r, _CLOUD_COVER_SENSOR, default=0.0)
    timestamp = utils.proto_to_pandas_timestamp(r.timestamp)
    cloud_covers[str(timestamp)] = cloud_cover
  return cloud_covers


def get_replay_sky_temperature(
    observation_responses: Sequence[object],
    dewpoint_depression: float = 5.0,
) -> Mapping[str, float]:
  """Return sky temperature replays from past observations.

  Calculates sky temperature using Clark & Allen formula from dry-bulb
  temperature and dew point.  If a dew-point sensor is not present in the
  observation, the dew point is estimated as
  `dry_bulb - dewpoint_depression`.  Observations that lack a dry-bulb
  temperature are silently skipped.

  Args:
    observation_responses: Sequence of `ObservationResponse` protos.
    dewpoint_depression: Difference between dry-bulb and dew-point
      temperatures in K.  Used when no dew-point sensor is available.

  Returns:
    Mapping from timestamp string to sky temperature in Kelvin.
  """
  sigma = constants.STEFAN_BOLTZMANN_CONSTANT
  sky_temps: dict[str, float] = {}

  for r in observation_responses:
    temp_k = _get_observation_value(r, _OUTSIDE_AIR_TEMP_SENSOR)
    if temp_k is None:
      continue

    dp_k = _get_observation_value(r, _DEW_POINT_TEMP_SENSOR)
    if dp_k is None:
      dp_k = temp_k - dewpoint_depression

    epsilon_sky = 0.787 + 0.764 * np.log(dp_k / 273.0)
    ir_h = epsilon_sky * sigma * (temp_k**4)
    temp_sky_k = (ir_h / sigma) ** 0.25

    timestamp = utils.proto_to_pandas_timestamp(r.timestamp)
    sky_temps[str(timestamp)] = float(temp_sky_k)

  return sky_temps


# ---------------------------------------------------------------------------
# POA irradiance helper (migrated from building_radiation_utils)
# ---------------------------------------------------------------------------


def calculate_poa_irradiance(
    irradiance_components: IrradianceComponents,
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: float,
    solar_azimuth: float,
) -> float:
  """Calculate plane-of-array (POA) global irradiance.

  Converts horizontal irradiance components (GHI, DNI, DHI) to the
  irradiance incident on a tilted surface using pvlib's
  `get_total_irradiance`.

  Args:
    irradiance_components: :class:`IrradianceComponents` with `ghi`,
      `dni`, and `dhi` fields (W/m²).
    surface_tilt: Surface tilt angle from horizontal in degrees
      (0 = horizontal, 90 = vertical).
    surface_azimuth: Surface azimuth angle in degrees (180 = south-facing
      in the Northern Hemisphere).
    solar_zenith: Solar zenith angle in degrees.
    solar_azimuth: Solar azimuth angle in degrees.

  Returns:
    POA global irradiance in W/m².

  Examples:
    ```python
    irrad = IrradianceComponents(
        ghi=800.0, dni=700.0, dhi=100.0,
        solar_zenith=30.0, solar_azimuth=180.0,
    )
    poa = calculate_poa_irradiance(
        irrad,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )
    ```
  """
  poa_irrad = pvlib_irradiance.get_total_irradiance(
      surface_tilt=surface_tilt,
      surface_azimuth=surface_azimuth,
      dni=irradiance_components.dni,
      ghi=irradiance_components.ghi,
      dhi=irradiance_components.dhi,
      solar_zenith=solar_zenith,
      solar_azimuth=solar_azimuth,
  )

  return float(poa_irrad['poa_global'])
