"""Data structures for response data returned by the Weather.gov API."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Self

import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast
from smart_buildings.smart_control.services.weather import base_forecast_period
from smart_buildings.smart_control.utils import temperature_conversion

assign_temp_unit = temperature_conversion.assign_temp_unit

ResponseData = Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class ForecastPeriod(base_forecast_period.BaseForecastPeriod):
  """Schema for a single forecast period.

  The Weather.gov API's 'forecasts' endpoint provides a list of 14 periods,
  spanning seven calendar dates (starting today), with two periods (one day-time
  and one night-time period) for each calendar date.

  One important note is that the timestamps for all day-time periods, or for all
  night-time periods, may not be consistant across different days. For example,
  the day-time period for today may be at 6am, but the day-time period for
  Saturday it may be at 9am. So this endpoint should essentially be used if you
  need a high-level seven day forecast.

  For more information, see the
  [API docs](https://www.weather.gov/documentation/services-web-api), as well as
  copies of real response data stored in the "weather_gov/test_data" directory.

  Attributes:
    number: Forecast period number.
    name: Forecast period name.
    start_time: Forecast period start time.
    end_time: Forecast period end time.
    is_daytime: Whether the forecast period is daytime.
    temp: Forecast temperature.
    temp_unit: Forecast temperature unit.
    temp_trend: Forecast temperature trend. May be null.
    chance_of_precip: Forecast chance of precipitation. Represented as a
      percentage value from 0 to 100.
    wind_speed: Forecast wind speed.
    wind_direction: Forecast wind direction.
    icon: Forecast icon URL.
    short_forecast: Short description of the forecast.
    detailed_forecast: Detailed description of the forecast.
  """
  number: int
  name: str
  start_timestamp: pd.Timestamp
  end_timestamp: pd.Timestamp
  is_daytime: bool
  temp: int
  temp_unit: temperature_conversion.TempUnit
  temp_trend: str | None
  chance_of_precip: int
  wind_speed: str
  wind_direction: str
  icon: str
  short_forecast: str
  detailed_forecast: str


@dataclasses.dataclass(frozen=True)
class HourlyForecastPeriod(ForecastPeriod):
  """Schema for a single hourly forecast period.

  The hourly forecast period is exactly one hour.

  The Weather.gov API's 'forecasts/hourly' endpoint provides a list of periods,
  spanning around seven days (starting now), with one record for each hour.

  For more information, see the
  [API docs](https://www.weather.gov/documentation/services-web-api), as well as
  copies of real response data stored in the "weather_gov/test_data" directory.

  Attributes:
    number: Forecast period number.
    name: Forecast period name. May be blank.
    start_timestamp: Forecast period start time.
    end_timestamp: Forecast period end time.
    is_daytime: Whether the forecast period is during the day-time.
    temp: Forecast temperature.
    temp_unit: Forecast temperature unit.
    temp_trend: Forecast temperature trend. May be null.
    chance_of_precip: Forecast chance of precipitation. Represented as a
      percentage value from 0 to 100.
    dewpoint: Forecast dewpoint, in Celsius.
    dewpoint_unit: Forecast dewpoint unit.
    relative_humidity: Forecast relative humidity. Represented as a percentage
      value from 0 to 100.
    relative_humidity_unit: Forecast relative humidity unit.
    wind_speed: Forecast wind speed.
    wind_direction: Forecast wind direction.
    icon: Forecast icon URL.
    short_forecast: Short description of the forecast.
    detailed_forecast: Detailed description of the forecast. May be blank.
  """
  dewpoint: float | None
  dewpoint_unit: str | None
  relative_humidity: int | None
  relative_humidity_unit: str | None


@dataclasses.dataclass(frozen=True)
class Gridpoint:
  """Gridpoint data from the Weather.gov API.

  A gridpoint is a location associated with a given set of latitude and
  longitude coordinates.

  Attributes:
    data: The JSON data returned by a request to the Gridpoints API endpoint
      (e.g. https://api.weather.gov/points/37.4263,-122.0349).
  """

  data: ResponseData

  @property
  def time_zone(self) -> str:
    """The time zone of the location."""
    return self.data["properties"]["timeZone"]

  @property
  def grid_id(self) -> str:
    """The unique identifier for the grid."""
    return self.data["properties"]["gridId"]

  @property
  def grid_x(self) -> int:
    """The X coordinate of the grid."""
    return self.data["properties"]["gridX"]

  @property
  def grid_y(self) -> int:
    """The Y coordinate of the grid."""
    return self.data["properties"]["gridY"]

  # LOCATION

  @property
  def location(self) -> Mapping[str, Any]:
    """Location data."""
    return self.data["properties"]["relativeLocation"]

  @property
  def city(self) -> str:
    """The city name."""
    return self.location["properties"]["city"]

  @property
  def state(self) -> str:
    """The state name."""
    return self.location["properties"]["state"]

  # URLS

  @property
  def forecast_url(self) -> str:
    """The URL for fetching forecast data."""
    return self.data["properties"]["forecast"]

  @property
  def hourly_forecast_url(self) -> str:
    """The URL for fetching hourly forecast data."""
    return self.data["properties"]["forecastHourly"]

  @property
  def stations_url(self) -> str:
    """The URL for fetching observation station data."""
    return self.data["properties"]["observationStations"]


@dataclasses.dataclass(frozen=True)
class Forecast(base_forecast.BaseForecast):
  """Forecast data from the Weather.gov API.

  Represents a seven-day forecast, with two forecast periods for each calendar
  date (one for daytime and one for night-time), starting today.

  Attributes:
    periods: Seven-day forecast periods (day and night for each calendar date).
  """

  periods: Sequence[ForecastPeriod]

  @classmethod
  def from_response_data(cls, data: ResponseData) -> Self:
    """Constructs a Forecast object from API response data.

    Args:
      data: The JSON data returned by a request to the Forecast API endpoint
        (e.g. https://api.weather.gov/gridpoints/MTR/95,87/forecast).

    Returns:
      A Forecast object.
    """
    periods = data.get("properties", {}).get("periods", [])
    forecast_periods = [
        ForecastPeriod(
            number=p.get("number"),
            name=p.get("name", ""),
            start_timestamp=pd.Timestamp(p.get("startTime")),
            end_timestamp=pd.Timestamp(p.get("endTime")),
            is_daytime=p.get("isDaytime"),
            temp=p.get("temperature"),
            temp_unit=assign_temp_unit(p.get("temperatureUnit")),
            temp_trend=p.get("temperatureTrend"),
            chance_of_precip=p.get("probabilityOfPrecipitation", {}).get("value"),  # pylint: disable=line-too-long
            wind_speed=p.get("windSpeed"),
            wind_direction=p.get("windDirection"),
            icon=p.get("icon"),
            short_forecast=p.get("shortForecast"),
            detailed_forecast=p.get("detailedForecast"),
        )
        for p in periods
    ]
    forecast_periods = sorted(forecast_periods, key=lambda p: p.start_timestamp)
    return cls(periods=forecast_periods)


@dataclasses.dataclass(frozen=True)
class HourlyForecast(Forecast):
  """Hourly forecast data from the Weather.gov API.

  Represents a forecast for each hour, spanning approximately seven days,
  starting from the current time.

  Attributes:
    periods: Hourly forecast periods.
  """

  periods: Sequence[HourlyForecastPeriod]

  @classmethod
  def from_response_data(
      cls,
      data: ResponseData,
  ) -> Self:
    """Constructs an HourlyForecast object from API response data.

    Args:
      data: The JSON data returned by a request to the Hourly Forecast API
        endpoint (e.g.
        https://api.weather.gov/gridpoints/MTR/95,87/forecast/hourly).

    Returns:
      An HourlyForecast object.
    """
    periods = data.get("properties", {}).get("periods", [])
    forecast_periods = [
        HourlyForecastPeriod(
            number=p.get("number"),
            name=p.get("name", ""),
            start_timestamp=pd.Timestamp(p.get("startTime")),
            end_timestamp=pd.Timestamp(p.get("endTime")),
            is_daytime=p.get("isDaytime"),
            temp=p.get("temperature"),
            temp_unit=assign_temp_unit(p.get("temperatureUnit")),
            temp_trend=p.get("temperatureTrend"),
            chance_of_precip=p.get("probabilityOfPrecipitation", {}).get("value"),  # pylint: disable=line-too-long
            dewpoint=p.get("dewpoint", {}).get("value"),
            dewpoint_unit=p.get("dewpoint", {}).get("unitCode"),
            relative_humidity=p.get("relativeHumidity", {}).get("value"),
            relative_humidity_unit=p.get("relativeHumidity", {}).get("unitCode"),  # pylint: disable=line-too-long
            wind_speed=p.get("windSpeed"),
            wind_direction=p.get("windDirection"),
            icon=p.get("icon"),
            short_forecast=p.get("shortForecast"),
            detailed_forecast=p.get("detailedForecast"),
        ) for p in periods
    ]
    forecast_periods = sorted(forecast_periods, key=lambda p: p.start_timestamp)
    return cls(periods=forecast_periods)
