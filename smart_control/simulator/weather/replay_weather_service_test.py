import dataclasses
import random

from absl.testing import absltest
import pandas as pd
from smart_buildings.smart_control.simulator.weather import conftest
from smart_buildings.smart_control.simulator.weather import replay_weather_service

ReplayHourlyForecast = replay_weather_service.ReplayHourlyForecast
ReplayHourlyForecastPeriod = replay_weather_service.ReplayHourlyForecastPeriod
ReplayWeatherService = replay_weather_service.ReplayWeatherService

START_TIMESTAMP = conftest.START_TIMESTAMP
EXPECTED_FORECAST_PERIODS = conftest.EXPECTED_FORECAST_PERIODS


class ReplayHourlyForecastTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.periods = [
        ReplayHourlyForecastPeriod(**p) for p in EXPECTED_FORECAST_PERIODS
    ]
    self.forecast = ReplayHourlyForecast(periods=self.periods)

  def test_initialization(self):
    self.assertIsInstance(self.forecast, ReplayHourlyForecast)
    self.assertEqual(self.forecast.periods, self.periods)

  def test_validates_periods_sorted(self):
    periods = self.periods.copy()
    random.shuffle(periods)  # out of order
    with self.assertRaisesRegex(
        ValueError,
        "Periods must be sorted by start_timestamp in ascending order.",
    ):
      ReplayHourlyForecast(periods=periods)


class ReplayWeatherServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.controller = conftest.create_replay_weather_controller()
    self.service = ReplayWeatherService(self.controller)

  def test_get_hourly_forecast(self):
    forecast = self.service.get_hourly_forecast(start_timestamp=START_TIMESTAMP)
    self.assertIsInstance(forecast, ReplayHourlyForecast)

    periods = forecast.periods

    with self.subTest(name="forecast_periods"):
      types = set([type(p) for p in periods])
      self.assertEqual(types, {ReplayHourlyForecastPeriod})

    with self.subTest(name="expected_values"):
      self.assertEqual(
          [dataclasses.asdict(p) for p in periods],
          EXPECTED_FORECAST_PERIODS,
      )

    with self.subTest(name="24_hours_by_default"):
      self.assertLen(periods, 24)

    with self.subTest(name="hourly_durations"):
      durations = set([p.duration for p in periods])
      self.assertEqual(durations, {pd.Timedelta(hours=1)})

    with self.subTest(name="sorted_by_start_time_ascending"):
      start_times = [p.start_timestamp for p in periods]
      self.assertEqual(start_times, sorted(start_times))

  def test_get_hourly_forecast_start_time_not_provided(self):
    with self.assertRaisesRegex(
        ValueError, "start_timestamp must be provided"
    ):
      self.service.get_hourly_forecast()

  def test_get_hourly_forecast_n_hours(self):
    n_hours = 5
    forecast = self.service.get_hourly_forecast(
        start_timestamp=START_TIMESTAMP, n_hours=n_hours
    )
    self.assertIsInstance(forecast, ReplayHourlyForecast)
    self.assertLen(forecast.periods, n_hours)

  def test_get_hourly_forecast_insufficient_data(self):
    start_timestamp = self.controller.max_time - pd.Timedelta(hours=1)
    with self.assertRaisesRegex(ValueError, "Timestamp not in range"):
      self.service.get_hourly_forecast(
          start_timestamp=start_timestamp, n_hours=3
      )

  def test_get_hourly_forecast_with_normalize_to_start(self):
    start_timestamp = START_TIMESTAMP + pd.Timedelta(minutes=5)
    forecast = self.service.get_hourly_forecast(
        start_timestamp=start_timestamp, n_hours=1, normalize_to_start=True
    )
    self.assertIsInstance(forecast, ReplayHourlyForecast)
    self.assertEqual(
        forecast.first_period.start_timestamp.minute,
        start_timestamp.minute,
    )


if __name__ == "__main__":
  absltest.main()
