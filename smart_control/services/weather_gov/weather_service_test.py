from unittest import mock

from absl.testing import absltest
import requests

from smart_buildings.smart_control.services.weather_gov import conftest
from smart_buildings.smart_control.services.weather_gov import weather_service

GRIDPOINT_URL = conftest.GRIDPOINT_URL
FORECAST_URL = conftest.FORECAST_URL
HOURLY_FORECAST_URL = conftest.HOURLY_FORECAST_URL


class MockResponse:

  def __init__(self, json_data: weather_service.ResponseData, status_code: int):
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
    return MockResponse(conftest.read_gridpoint_data(), 200)
  elif url == FORECAST_URL:
    return MockResponse(conftest.read_forecast_data(), 200)
  elif url == HOURLY_FORECAST_URL:
    return MockResponse(conftest.read_hourly_forecast_data(), 200)
  return MockResponse(
      {"message": "Sorry, we don't have test data for the specified URL."}, 200
  )


class WeatherServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_session = mock.MagicMock(spec=requests.Session)
    self.mock_session.headers = mock.MagicMock()
    self.mock_session.get.side_effect = mock_requests_get
    self.mock_session.__enter__.return_value = self.mock_session
    self.service = weather_service.WeatherService()
    self.enter_context(
        mock.patch.object(
            weather_service.WeatherService,
            "session",
            new_callable=mock.PropertyMock,
            return_value=self.mock_session,
        )
    )

  def test_attributes(self):
    self.assertEqual(self.service.lat, 37.4263)
    self.assertEqual(self.service.lon, -122.0349)

  def test_properties(self):
    self.assertEqual(self.service.gridpoint_url, GRIDPOINT_URL)

  def test_get_gridpoint(self):
    gridpoint = self.service.gridpoint
    self.assertIsInstance(gridpoint, weather_service.Gridpoint)

    self.mock_session.headers.update.assert_called_once_with(
        {"User-Agent": weather_service.USER_AGENT}
    )
    self.mock_session.get.assert_called_once_with(GRIDPOINT_URL, timeout=60)

  def test_get_forecast(self):
    forecast = self.service.get_forecast()
    self.assertIsInstance(forecast, weather_service.Forecast)

    # Makes two calls to get the gridpoint data and then the forecast data.
    self.assertEqual(2, self.mock_session.headers.update.call_count)
    self.mock_session.headers.update.assert_called_with(
        {"User-Agent": weather_service.USER_AGENT}
    )
    calls = [
        mock.call(GRIDPOINT_URL, timeout=60),
        mock.call(FORECAST_URL, timeout=60),
    ]
    self.mock_session.get.assert_has_calls(calls)
    self.mock_session.get.reset_mock()
    self.mock_session.headers.update.reset_mock()

    _ = self.service.get_forecast()
    with self.subTest(name="subsequent_calls_use_cached_gridpoint"):
      self.mock_session.headers.update.assert_called_once_with(
          {"User-Agent": weather_service.USER_AGENT}
      )
      self.mock_session.get.assert_called_once_with(FORECAST_URL, timeout=60)

  def test_get_hourly_forecast(self):
    hourly_forecast = self.service.get_hourly_forecast()
    self.assertIsInstance(hourly_forecast, weather_service.HourlyForecast)

    with self.subTest(name="first_call_fetches_gridpoint_then_hourly_forecast"):
      self.assertEqual(2, self.mock_session.headers.update.call_count)
      self.mock_session.headers.update.assert_called_with(
          {"User-Agent": weather_service.USER_AGENT}
      )
      calls = [
          mock.call(GRIDPOINT_URL, timeout=60),
          mock.call(HOURLY_FORECAST_URL, timeout=60),
      ]
      self.mock_session.get.assert_has_calls(calls)
      self.mock_session.get.reset_mock()
      self.mock_session.headers.update.reset_mock()

    _ = self.service.get_hourly_forecast()
    with self.subTest(name="subsequent_calls_use_cached_gridpoint"):
      self.mock_session.headers.update.assert_called_once_with(
          {"User-Agent": weather_service.USER_AGENT}
      )
      self.mock_session.get.assert_called_once_with(
          HOURLY_FORECAST_URL, timeout=60
      )

  def test_get_gridpoint_raises_for_status(self):
    self.mock_session.get.side_effect = lambda url, **kwargs: MockResponse(
        None, 404
    )
    with self.assertRaises(requests.exceptions.RequestException):
      _ = self.service.gridpoint


class WeatherServiceCustomLocationTest(absltest.TestCase):
  """Tests the WeatherService for a custom location.

  Since the weather service gets the forecast URL and hourly forecast URL from
  the gridpoint response data, we don't know what those URLs will be beforehand.
  """

  def setUp(self):
    super().setUp()
    self.lat = 40.7128
    self.lon = -74.0060
    self.mock_session = mock.MagicMock(spec=requests.Session)
    self.mock_session.headers = mock.MagicMock()
    self.mock_session.get.side_effect = mock_requests_get
    self.mock_session.__enter__.return_value = self.mock_session
    self.service = weather_service.WeatherService(lat=self.lat, lon=self.lon)
    self.enter_context(
        mock.patch.object(
            weather_service.WeatherService,
            "session",
            new_callable=mock.PropertyMock,
            return_value=self.mock_session,
        )
    )

  def test_get_gridpoint(self):
    _ = self.service.gridpoint
    self.mock_session.headers.update.assert_called_once_with(
        {"User-Agent": weather_service.USER_AGENT}
    )
    self.mock_session.get.assert_called_once_with(
        f"https://api.weather.gov/points/{self.lat},{self.lon}", timeout=60
    )


if __name__ == "__main__":
  absltest.main()
