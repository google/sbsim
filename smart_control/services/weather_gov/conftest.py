"""Test fixtures and helpers for the Weather.gov API service."""

import json
import os
from unittest import mock

import pandas as pd
import requests
from smart_buildings.smart_control.services.weather_gov import models
from smart_buildings.smart_control.services.weather_gov import weather_service
from smart_buildings.smart_control.utils import constants
from smart_buildings.smart_control.utils import temperature_conversion

TempUnit = temperature_conversion.TempUnit

TEST_DATA_DIR = os.path.join(
    constants.REPO_DIRPATH, "services", "weather_gov", "test_data"
)

GRIDPOINT_URL = "https://api.weather.gov/points/37.4263,-122.0349"
FORECAST_URL = "https://api.weather.gov/gridpoints/MTR/95,87/forecast"
HOURLY_FORECAST_URL = f"{FORECAST_URL}/hourly"


# RESPONSE DATA


def _read_json_file(filename: str) -> models.ResponseData:
  filepath = os.path.join(TEST_DATA_DIR, filename)
  with open(filepath, encoding="utf-8") as f:
    return json.load(f)


def read_gridpoint_data() -> models.ResponseData:
  return _read_json_file("gridpoint.json")


def read_forecast_data() -> models.ResponseData:
  return _read_json_file("forecast.json")


def read_hourly_forecast_data() -> models.ResponseData:
  return _read_json_file("forecast_hourly.json")


# FACTORIES


def create_gridpoint() -> models.Gridpoint:
  """Returns a test Gridpoint object for test purposes."""
  return models.Gridpoint(read_gridpoint_data())


def create_forecast() -> models.Forecast:
  """Returns a test seven-day Forecast for test purposes."""
  return models.Forecast.from_response_data(read_forecast_data())


def create_hourly_forecast() -> models.HourlyForecast:
  """Returns a non-comprehensive HourlyForecast for test purposes.

  To keep the test data file size small, this data only includes the first three
  and the last three hourly periods. But in practice there are many more records
  returned by the API.

  Returns:
    A test HourlyForecast object.
  """
  return models.HourlyForecast.from_response_data(
      data=read_hourly_forecast_data()
  )


# EXAMPLE FORECAST RECORDS

FIRST_PERIOD = models.ForecastPeriod(
    number=1,
    name="Today",
    start_timestamp=pd.Timestamp("2026-02-06T09:00:00-08:00"),
    end_timestamp=pd.Timestamp("2026-02-06T18:00:00-08:00"),
    is_daytime=True,
    temp=67,
    temp_unit=TempUnit.FAHRENHEIT,
    temp_trend=None,
    chance_of_precip=6,
    wind_speed="2 to 7 mph",
    wind_direction="NNW",
    icon="https://api.weather.gov/icons/land/day/few?size=medium",
    short_forecast="Sunny",
    detailed_forecast=(
        "Sunny. High near 67, with temperatures falling to around 64 in the"
        " afternoon. North northwest wind 2 to 7 mph."
    ),
)

LAST_PERIOD = models.ForecastPeriod(
    number=14,
    name="Thursday Night",
    start_timestamp=pd.Timestamp("2026-02-12T18:00:00-08:00"),
    end_timestamp=pd.Timestamp("2026-02-13T06:00:00-08:00"),
    is_daytime=False,
    temp=44,
    temp_unit=TempUnit.FAHRENHEIT,
    temp_trend=None,
    chance_of_precip=11,
    wind_speed="2 to 9 mph",
    wind_direction="W",
    icon="https://api.weather.gov/icons/land/night/sct/fog?size=medium",
    short_forecast="Partly Cloudy then Patchy Fog",
    detailed_forecast=(
        "Patchy fog after 5am. Partly cloudy, with a low around 44."
    ),
)

# EXAMPLE HOURLY FORECAST RECORDS

FIRST_HOURLY_PERIOD = models.HourlyForecastPeriod(
    number=1,
    name="",
    start_timestamp=pd.Timestamp("2026-02-06T09:00:00-08:00"),
    end_timestamp=pd.Timestamp("2026-02-06T10:00:00-08:00"),
    is_daytime=True,
    temp=57,
    temp_unit=TempUnit.FAHRENHEIT,
    temp_trend=None,
    chance_of_precip=6,
    dewpoint=9.444444444444445,
    dewpoint_unit="wmoUnit:degC",
    relative_humidity=81,
    relative_humidity_unit="wmoUnit:percent",
    wind_speed="2 mph",
    wind_direction="W",
    icon="https://api.weather.gov/icons/land/day/sct?size=small",
    short_forecast="Mostly Sunny",
    detailed_forecast="",
)

LAST_HOURLY_PERIOD = models.HourlyForecastPeriod(
    number=156,
    name="",
    start_timestamp=pd.Timestamp("2026-02-12T20:00:00-08:00"),
    end_timestamp=pd.Timestamp("2026-02-12T21:00:00-08:00"),
    is_daytime=False,
    temp=52,
    temp_unit=TempUnit.FAHRENHEIT,
    temp_trend=None,
    chance_of_precip=9,
    dewpoint=8.88888888888889,
    dewpoint_unit="wmoUnit:degC",
    relative_humidity=83,
    relative_humidity_unit="wmoUnit:percent",
    wind_speed="7 mph",
    wind_direction="WNW",
    icon="https://api.weather.gov/icons/land/night/sct?size=small",
    short_forecast="Partly Cloudy",
    detailed_forecast="",
)


# MOCKING SETUP FOR WEATHER SERVICE


class MockResponse:
  """Mimics a requests.Response object."""

  def __init__(self, json_data: models.ResponseData, status_code: int):
    self.json_data = json_data
    self.status_code = status_code
    self.ok = status_code < 400

  def json(self):
    return self.json_data

  def raise_for_status(self):
    if not self.ok:
      raise requests.exceptions.RequestException(
          f"Request failed with status code {self.status_code}"
      )


def mock_requests_get(url: str, **kwargs) -> MockResponse:
  """Mimics requests.get() function, can be used to prevent network requests.

  For the default location, we have test data to return.

  For other locations, we swallow the request and return a custom error message.

  Args:
    url: The URL to make the GET request to.
    **kwargs: Keyword arguments to pass along with the request.

  Returns:
    A MockResponse object.
  """
  del kwargs  # needed for the timeout arg, but unused in the MockResponse
  if url == GRIDPOINT_URL:
    return MockResponse(read_gridpoint_data(), 200)
  elif url == FORECAST_URL:
    return MockResponse(read_forecast_data(), 200)
  elif url == HOURLY_FORECAST_URL:
    return MockResponse(read_hourly_forecast_data(), 200)
  return MockResponse(
      {"message": "Sorry, we don't have test data for the specified URL."}, 200
  )


def setup_mock_session(test_case: mock.Mock) -> mock.MagicMock:
  """Handles the standard mocking and patching of WeatherService.session.

  To be used within a setUp() method of a unittest / absltest test class.

  Args:
    test_case: The test case instance to register the patch with.

  Example:

    def setUp(self):
      super().setUp()
      self.mock_session = conftest.setup_mock_session(self)
      self.service = weather_service.WeatherService()

  Returns:
    The mock session object.
  """
  mock_session = mock.MagicMock(spec=requests.Session)
  mock_session.headers = mock.MagicMock()
  mock_session.get.side_effect = mock_requests_get
  mock_session.__enter__.return_value = mock_session
  test_case.enter_context(
      mock.patch.object(
          weather_service.WeatherService,
          "session",
          new_callable=mock.PropertyMock,
          return_value=mock_session,
      )
  )
  return mock_session
