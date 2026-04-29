"""Service class for interfacing with the Weather.gov API.

This API only provides weather data for locations within the United States.

API Docs: https://www.weather.gov/documentation/services-web-api

The API recommends the following authenticiation strategy:

> A User Agent is required to identify your application. This string can be
> anything, and the more unique to your application the less likely it will be
> affected by a security event. If you include contact information (website or
> email), we can contact you if your string is associated to a security event.
> This will be replaced with an API key in the future.
>
> User-Agent: (myweatherapp.com, contact@myweatherapp.com)
"""

import functools
import os
from typing import Any

import backoff
import immutabledict
import pandas as pd
import requests

from smart_buildings.smart_control.services.weather import base_weather_service
from smart_buildings.smart_control.services.weather import base_forecast
from smart_buildings.smart_control.services.weather_gov import models

MAX_TRIES = int(os.environ.get("WEATHER_GOV_MAX_TRIES", "3"))

SB1_COORDS = immutabledict.immutabledict({"lat": 37.4263, "lon": -122.0349})

USER_AGENT = os.environ.get(
    "WEATHER_GOV_USER_AGENT",
    "(oss-smart-buildings-control, https://github.com/google/sbsim)",
)


class WeatherService(base_weather_service.BaseWeatherService):
  """Service for fetching data from the Weather.gov API for a given US location.

  This service is initialized for a specific latitude and longitude, and the
  gridpoint data is cached. To fetch weather data for a different location, a
  new instance of WeatherService should be created.

  The intended use case is to make infrequent requests, for example, one request
  per hour. Based on this use case, we wouldn't get the benefits of keeping a
  persistent session, so we are choosing to use a new session for each request.

  The service supports an optional cache_max_age, which represents the
  duration of time for which fetched data should be considered valid. If the
  cache_max_age is specified, cached data will be used until the age of the
  cached data exceeds the cache_max_age. If cache_max_age is None, then new
  data will be fetched on every request. This helps avoid unnecessary API calls,
  since recently fetched forecast data will still be generally accurate, even a
  few hours later.

  Attributes:
    lat: Latitude of the location to fetch weather data for.
    lon: Longitude of the location to fetch weather data for.
    user_agent: User agent to use for the request.
    cache_max_age: The duration of time to wait before fetching a new forecast.
      If specified, cached data is used until the age of the cached data
      exceeds the cache_max_age. If None, new data is fetched on every request.
  """

  def __init__(
      self,
      *,
      lat: float = SB1_COORDS["lat"],
      lon: float = SB1_COORDS["lon"],
      user_agent: str = USER_AGENT,
      cache_max_age: pd.Timedelta | None = None,
  ):
    """Initializes the instance.

    Args:
      lat: Latitude of the location to fetch weather data for.
      lon: Longitude of the location to fetch weather data for.
      user_agent: User agent to use for the request.
      cache_max_age: The duration of time to wait before fetching a new
        forecast. If specified, cached data is used until the interval expires.
        If None, new data is fetched on every request.
    """
    self.lat = lat
    self.lon = lon
    self.user_agent = user_agent
    self.cache_max_age = cache_max_age

    self._cached_hourly_forecast_data: models.ResponseData | None = None
    self._cache_updated_timestamp: pd.Timestamp | None = None

  @property
  def session(self) -> requests.Session:
    """Provides a requests.Session instance.

    By default, this returns a new Session on each access, suitable for
    infrequent calls. Subclasses can override this property to provide
    a customized session.

    Returns:
      A requests.Session instance.
    """
    return requests.Session()

  @backoff.on_exception(
      backoff.expo, requests.exceptions.RequestException, max_tries=MAX_TRIES
  )
  def _get_data(
      self, request_url: str, timeout: int = 60
  ) -> models.ResponseData:
    """Makes a GET request for JSON data from the Weather.gov API.

    Because the use case is to make infrequent requests, like once every hour,
    we are choosing to initialize a new session for each request.

    We are also ensuring the session gets closed by using a context manager
    approach.

    Args:
      request_url: The URL to make the GET request to.
      timeout: Optional timeout in seconds for the request.

    Returns:
      The JSON data returned by the request.
    """
    with self.session as session:
      session.headers.update({"User-Agent": self.user_agent})
      response = session.get(request_url, timeout=timeout)
      response.raise_for_status()
      return response.json()

  @property
  def gridpoint_url(self) -> str:
    """The URL for fetching gridpoint data."""
    return f"https://api.weather.gov/points/{self.lat},{self.lon}"

  @functools.cached_property
  def gridpoint(self) -> models.Gridpoint:
    """A Gridpoint object for the given location."""
    return models.Gridpoint(self._get_data(self.gridpoint_url))

  @property
  def cached_data_is_valid(self) -> bool:
    """Whether or not cached data is available and recent enough to be reused.

    If we didn't specify a validity interval, or if cached data is not
    available, then we will always consider the cached data as invalid and
    fetch new data on every request.

    If the cache_max_age has been specified and cached data is available,
    then we determine if the cache age has exceeded the max age.

    Returns:
      True if the cache_max_age is specified, and the cached data is
      available and valid, False otherwise.
    """
    if (
        self.cache_max_age is None
        or self._cached_hourly_forecast_data is None
        or self._cache_updated_timestamp is None
    ):
      return False

    cache_age = pd.Timestamp.now(tz="UTC") - self._cache_updated_timestamp
    return cache_age < self.cache_max_age

  def get_forecast(self) -> models.Forecast:
    """Fetches forecast data for the given location."""
    data = self._get_data(self.gridpoint.forecast_url)
    return models.Forecast.from_response_data(data=data)

  def get_hourly_forecast(
      self,
      *,
      start_timestamp: pd.Timestamp | None = None,
      n_hours: int = 24,
      normalize_to_start: bool = False,
      **kwargs: Any,
  ) -> base_forecast.BaseForecast:
    """Fetches hourly forecast data for the given location.

    Args:
      start_timestamp: The timezone-aware start timestamp of the forecast.
      n_hours: The number of hours to fetch.
      normalize_to_start: Whether to normalize the forecast to the start
        timestamp.
      **kwargs: Additional keyword arguments for child classes.

    Returns:
      A BaseForecast object.

    Raises:
      requests.exceptions.RequestException: If the request fails, and no cached
        data is available.
      ValueError: If start_timestamp is provided and is not timezone-aware.
    """
    del kwargs  # Unused by this implementation.

    if start_timestamp and start_timestamp.tzinfo is None:
      raise ValueError("start_timestamp must be timezone-aware.")

    if self.cached_data_is_valid:
      # Use cached data if available and valid.
      forecast = models.HourlyForecast.from_response_data(
          data=self._cached_hourly_forecast_data,
      )
    else:
      try:
        # Fetch new data from the API, and cache it.
        data = self._get_data(self.gridpoint.hourly_forecast_url)
        forecast = models.HourlyForecast.from_response_data(data=data)
        self._cached_hourly_forecast_data = data
        self._cache_updated_timestamp = pd.Timestamp.now(tz="UTC")
      except requests.exceptions.RequestException as err:
        # Fallback to use cached data, if available.
        if self._cached_hourly_forecast_data:
          forecast = models.HourlyForecast.from_response_data(
              data=self._cached_hourly_forecast_data,
          )
        else:
          raise requests.exceptions.RequestException(
              "Failed to fetch forecast data. No cached data is available."
          ) from err

    if normalize_to_start and start_timestamp:
      forecast = forecast.resample(
          start_timestamp=start_timestamp,
          interval=pd.Timedelta(hours=1),
      )

    forecast = forecast.filter_periods(
        ends_after_timestamp=start_timestamp,
        max_periods=n_hours,
    )
    return forecast
