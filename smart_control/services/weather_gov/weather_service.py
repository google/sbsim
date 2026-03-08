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

import immutabledict
import requests

from smart_buildings.smart_control.services.weather_gov import models

ResponseData = models.ResponseData
Gridpoint = models.Gridpoint
Forecast = models.Forecast
HourlyForecast = models.HourlyForecast

SB1_COORDS = immutabledict.immutabledict({"lat": 37.4263, "lon": -122.0349})

USER_AGENT = "(oss-smart-buildings-control, https://github.com/google/sbsim)"


class WeatherService:
  """Service for fetching data from the Weather.gov API for a given US location.

  This service is initialized for a specific latitude and longitude, and the
  gridpoint data is cached. To fetch weather data for a different location, a
  new instance of WeatherService should be created.

  The intended use case is to make infrequent requests, for example, one request
  per hour. Based on this use case, we wouldn't get the benefits of keeping a
  persistent session, so we are choosing to use a new session for each request.

  Attributes:
    lat: Latitude of the location to fetch weather data for.
    lon: Longitude of the location to fetch weather data for.
    user_agent: User agent to use for the request.
  """

  def __init__(
      self,
      *,
      lat: float = SB1_COORDS["lat"],
      lon: float = SB1_COORDS["lon"],
      user_agent: str = USER_AGENT,
  ):
    """Initializes the instance.

    Args:
      lat: Latitude of the location to fetch weather data for.
      lon: Longitude of the location to fetch weather data for.
      user_agent: User agent to use for the request.
    """
    self.lat = lat
    self.lon = lon
    self.user_agent = user_agent

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

  def _get_data(self, request_url: str, timeout: int = 60) -> ResponseData:
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
  def gridpoint(self) -> Gridpoint:
    """A Gridpoint object for the given location."""
    return Gridpoint(self._get_data(self.gridpoint_url))

  def get_forecast(self) -> Forecast:
    """Fetches forecast data for the given location."""
    return Forecast(self._get_data(self.gridpoint.forecast_url))

  def get_hourly_forecast(self) -> HourlyForecast:
    """Fetches hourly forecast data for the given location."""
    return HourlyForecast(self._get_data(self.gridpoint.hourly_forecast_url))
