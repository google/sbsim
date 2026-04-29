from unittest import mock

from absl.testing import absltest
import pandas as pd
import requests

from smart_buildings.smart_control.services.weather import base_forecast
from smart_buildings.smart_control.services.weather_gov import conftest
from smart_buildings.smart_control.services.weather_gov import models
from smart_buildings.smart_control.services.weather_gov import weather_service

GRIDPOINT_URL = conftest.GRIDPOINT_URL
FORECAST_URL = conftest.FORECAST_URL
HOURLY_FORECAST_URL = conftest.HOURLY_FORECAST_URL


class MockResponse:

  def __init__(
      self, json_data: models.ResponseData, status_code: int
  ):
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


def mock_requests_get_error(url: str, **kwargs) -> MockResponse:
  """Similar to mock_requests_get but raises for the given URL."""
  del url, kwargs  # Unused by this implementation.
  raise requests.exceptions.RequestException("Failed")


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
    self.assertIsInstance(gridpoint, models.Gridpoint)

    self.mock_session.headers.update.assert_called_once_with(
        {"User-Agent": self.service.user_agent}
    )
    self.mock_session.get.assert_called_once_with(GRIDPOINT_URL, timeout=60)

  def test_get_forecast(self):
    forecast = self.service.get_forecast()
    self.assertIsInstance(forecast, models.Forecast)

    # Makes two calls to get the gridpoint data and then the forecast data.
    self.assertEqual(2, self.mock_session.headers.update.call_count)
    self.mock_session.headers.update.assert_called_with(
        {"User-Agent": self.service.user_agent}
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
          {"User-Agent": self.service.user_agent}
      )
      self.mock_session.get.assert_called_once_with(FORECAST_URL, timeout=60)

  def test_get_hourly_forecast(self):
    hourly_forecast = self.service.get_hourly_forecast()
    self.assertIsInstance(hourly_forecast, models.HourlyForecast)
    self.assertLen(hourly_forecast.periods, 6)

    with self.subTest(name="first_call_fetches_gridpoint_then_hourly_forecast"):
      self.assertEqual(2, self.mock_session.headers.update.call_count)
      self.mock_session.headers.update.assert_called_with(
          {"User-Agent": self.service.user_agent}
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
          {"User-Agent": self.service.user_agent}
      )
      self.mock_session.get.assert_called_once_with(
          HOURLY_FORECAST_URL, timeout=60
      )

  def test_get_hourly_forecast_with_normalize_to_start(self):
    start_time = pd.Timestamp("2026-02-06 09:30:00-08:00")
    hourly_forecast = self.service.get_hourly_forecast(
        start_timestamp=start_time, n_hours=2, normalize_to_start=True
    )
    self.assertIsInstance(hourly_forecast, base_forecast.BaseForecast)
    self.assertLen(hourly_forecast.periods, 2)
    self.assertEqual(
        hourly_forecast.first_period.start_timestamp.minute,
        start_time.minute,
    )
    # Expect interpolated value between 57F (at 9:00) and 61F (at 10:00).
    self.assertEqual(hourly_forecast.first_period.temp, 59)
    # Expect interpolated value between 61F (at 10:00) and 63F (at 11:00).
    self.assertEqual(hourly_forecast.periods[1].temp, 62)

  def test_get_hourly_forecast_naive_start_timestamp_raises(self):
    with self.assertRaisesRegex(
        ValueError, "start_timestamp must be timezone-aware."
    ):
      self.service.get_hourly_forecast(
          start_timestamp=pd.Timestamp("2026-02-06 09:30:00")
      )

  def test_get_gridpoint_raises_for_status(self):
    self.mock_session.get.side_effect = lambda url, **kwargs: MockResponse(
        {}, 404
    )
    with self.assertRaises(requests.exceptions.RequestException):
      _ = self.service.gridpoint

  @mock.patch("time.sleep")
  def test_get_gridpoint_uses_backoff_on_request_exception(self, mock_sleep):
    del mock_sleep  # Unused by this test.
    self.mock_session.get.side_effect = [
        requests.exceptions.RequestException("Failed 1"),
        requests.exceptions.RequestException("Failed 2"),
        mock_requests_get(GRIDPOINT_URL, timeout=60),
    ]

    gridpoint = self.service.gridpoint

    self.assertIsInstance(gridpoint, models.Gridpoint)
    self.assertEqual(3, self.mock_session.get.call_count)

  @mock.patch("time.sleep")
  def test_get_forecast_uses_backoff_on_request_exception(self, mock_sleep):
    del mock_sleep  # Unused by this test.
    self.mock_session.get.side_effect = [
        mock_requests_get(GRIDPOINT_URL, timeout=60),
        requests.exceptions.RequestException("Failed 1"),
        requests.exceptions.RequestException("Failed 2"),
        mock_requests_get(FORECAST_URL, timeout=60),
    ]

    forecast = self.service.get_forecast()

    self.assertIsInstance(forecast, models.Forecast)
    self.assertEqual(4, self.mock_session.get.call_count)


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
        {"User-Agent": self.service.user_agent}
    )
    self.mock_session.get.assert_called_once_with(
        f"https://api.weather.gov/points/{self.lat},{self.lon}", timeout=60
    )


class WeatherServiceCachedDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_session = mock.MagicMock(spec=requests.Session)
    self.mock_session.headers = mock.MagicMock()
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

  def test_get_hourly_forecast_uses_cache_on_failure(self):
    # First call succeeds and caches data.
    self.mock_session.get.side_effect = mock_requests_get
    initial_forecast = self.service.get_hourly_forecast()
    self.assertIsNotNone(self.service._cached_hourly_forecast_data)
    self.assertLen(initial_forecast.periods, 6)

    # Second call fails, and returns cached data.
    self.mock_session.get.side_effect = mock_requests_get_error
    cached_forecast = self.service.get_hourly_forecast()
    self.assertIsInstance(cached_forecast, models.HourlyForecast)
    self.assertLen(cached_forecast.periods, len(initial_forecast.periods))
    self.assertEqual(
        initial_forecast.periods[0].start_timestamp,
        cached_forecast.periods[0].start_timestamp,
    )

  def test_get_hourly_forecast_raises_exception_on_failure_with_no_cache(self):
    self.assertIsNone(self.service._cached_hourly_forecast_data)

    self.mock_session.get.side_effect = mock_requests_get_error

    with self.assertRaisesRegex(
        requests.exceptions.RequestException,
        "Failed to fetch forecast data.",
    ):
      self.service.get_hourly_forecast()

  def test_get_hourly_forecast_with_forecast_start_filters_cache(self):
    # First call succeeds and caches data.
    self.mock_session.get.side_effect = mock_requests_get
    initial_forecast = self.service.get_hourly_forecast()
    self.assertIsNotNone(self.service._cached_hourly_forecast_data)

    # Second call fails, and returns cached data, filtered by start_timestamp.
    self.mock_session.get.side_effect = mock_requests_get_error
    start_timestamp = (
        initial_forecast.first_period.start_timestamp + pd.Timedelta(hours=1)
    )
    cached_forecast = self.service.get_hourly_forecast(
        start_timestamp=start_timestamp
    )
    self.assertIsInstance(cached_forecast, models.HourlyForecast)
    self.assertLen(cached_forecast.periods, 5)
    self.assertEqual(
        cached_forecast.first_period.start_timestamp,
        start_timestamp,
    )


class WeatherServiceValidityIntervalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_session = mock.MagicMock(spec=requests.Session)
    self.mock_session.headers = mock.MagicMock()
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

  def test_get_hourly_forecast_uses_validity_interval(self):
    # Setup service with cache_max_age
    self.service.cache_max_age = pd.Timedelta(hours=1)
    self.mock_session.get.side_effect = mock_requests_get

    t1 = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
    t2 = t1 + pd.Timedelta(minutes=30)  # Within cache_max_age
    t3 = t1 + pd.Timedelta(minutes=61)  # After cache_max_age expires

    with mock.patch("pandas.Timestamp.now") as mock_now:
      # Initial call, fetches data (no cached data is available).
      mock_now.return_value = t1
      self.service.get_hourly_forecast()
      # ... makes two calls (one for gridpoint, one for hourly forecast):
      self.assertEqual(2, self.mock_session.get.call_count)
      # ... updates the cache_updated_timestamp to the time of the call:
      self.assertEqual(t1, self.service._cache_updated_timestamp)
      self.mock_session.get.reset_mock()

      # Second call, within cache_max_age, uses cached data.
      mock_now.return_value = t2
      self.service.get_hourly_forecast()
      # ... makes no calls:
      self.assertEqual(0, self.mock_session.get.call_count)
      # ... does not update the cache_updated_timestamp:
      self.assertEqual(t1, self.service._cache_updated_timestamp)
      self.mock_session.get.reset_mock()

      # Third call, after cache_max_age expires, fetches new data.
      mock_now.return_value = t3
      self.service.get_hourly_forecast()
      # ... makes one call, for forecast (gridpoint has already been cached)
      self.assertEqual(1, self.mock_session.get.call_count)
      # ... updates the cache_updated_timestamp to the time of the call:
      self.assertEqual(t3, self.service._cache_updated_timestamp)


if __name__ == "__main__":
  absltest.main()
