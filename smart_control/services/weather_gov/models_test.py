import dataclasses

from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.services.weather_gov import conftest
from smart_buildings.smart_control.services.weather_gov import models


GRIDPOINT_URL = conftest.GRIDPOINT_URL
FORECAST_URL = conftest.FORECAST_URL
HOURLY_FORECAST_URL = conftest.HOURLY_FORECAST_URL


class ForecastPeriodTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.period = conftest.FIRST_PERIOD

  def test_properties(self):
    with self.subTest(name="timestamps"):
      self.assertEqual(
          self.period.start_timestamp, pd.Timestamp("2026-02-06 09:00:00-08:00")
      )
      self.assertEqual(
          self.period.end_timestamp, pd.Timestamp("2026-02-06 18:00:00-08:00")
      )

    with self.subTest(name="duration"):
      self.assertEqual(self.period.duration, pd.Timedelta(hours=9))

    with self.subTest(name="dates"):
      self.assertEqual(self.period.start_date, "2026-02-06")
      self.assertEqual(self.period.end_date, "2026-02-06")


class HourlyForecastPeriodTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.period = conftest.FIRST_HOURLY_PERIOD

  def test_properties(self):
    with self.subTest(name="timstamps"):
      self.assertEqual(
          self.period.start_timestamp, pd.Timestamp("2026-02-06 09:00:00-08:00")
      )
      self.assertEqual(
          self.period.end_timestamp, pd.Timestamp("2026-02-06 10:00:00-08:00")
      )

    with self.subTest(name="duration"):
      self.assertEqual(self.period.duration, pd.Timedelta(hours=1))

    with self.subTest(name="dates"):
      self.assertEqual(self.period.start_date, "2026-02-06")
      self.assertEqual(self.period.end_date, "2026-02-06")


class GridpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.gridpoint = conftest.create_gridpoint()

  def test_initialization(self):
    self.assertIsInstance(self.gridpoint, models.Gridpoint)

  def test_properties(self):
    self.assertEqual(self.gridpoint.time_zone, "America/Los_Angeles")

    with self.subTest(name="grid"):
      self.assertEqual(self.gridpoint.grid_id, "MTR")
      self.assertEqual(self.gridpoint.grid_x, 95)
      self.assertEqual(self.gridpoint.grid_y, 87)

    with self.subTest(name="location"):
      self.assertEqual(self.gridpoint.city, "Sunnyvale")
      self.assertEqual(self.gridpoint.state, "CA")

    with self.subTest(name="urls"):
      self.assertEqual(self.gridpoint.forecast_url, FORECAST_URL)
      self.assertEqual(self.gridpoint.hourly_forecast_url, HOURLY_FORECAST_URL)
      self.assertEqual(
          self.gridpoint.stations_url,
          "https://api.weather.gov/gridpoints/MTR/95,87/stations",
      )


class ForecastTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.forecast = conftest.create_forecast()

  def test_initialization(self):
    self.assertIsInstance(self.forecast, models.Forecast)

  def test_periods(self):
    periods = self.forecast.periods
    self.assertLen(periods, 14)

    with self.subTest(name="example_periods"):
      self.assertEqual(periods[0], conftest.FIRST_PERIOD)
      self.assertEqual(periods[-1], conftest.LAST_PERIOD)

    with self.subTest(name="spans_seven_days"):
      self.assertLen({r.start_date for r in periods}, 7)

    with self.subTest(name="two_periods_per_day"):
      self.assertEqual(
          pd.Series([r.is_daytime for r in periods]).value_counts().to_dict(),
          {True: 7, False: 7}
      )

  def test_df(self):
    forecast_df = self.forecast.df
    self.assertIsInstance(forecast_df, pd.DataFrame)
    self.assertLen(forecast_df, 14)

    with self.subTest(name="example_periods"):
      comparison_df = forecast_df.drop(
          columns=["start_timestamp", "end_timestamp", "duration", "start_date", "end_date"]  # pylint: disable=line-too-long
      )
      self.assertEqual(
          comparison_df.iloc[0].to_dict(),
          dataclasses.asdict(conftest.FIRST_PERIOD),
      )
      self.assertEqual(
          comparison_df.iloc[-1].to_dict(),
          dataclasses.asdict(conftest.LAST_PERIOD),
      )

    with self.subTest(name="spans_seven_days"):
      self.assertLen(forecast_df["start_date"].unique(), 7)

    with self.subTest(name="two_periods_per_day"):
      self.assertEqual(
          forecast_df["is_daytime"].value_counts().to_dict(),
          {True: 7, False: 7}
      )


class HourlyForecastTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.hourly_forecast = conftest.create_hourly_forecast()

  def test_initialization(self):
    self.assertIsInstance(self.hourly_forecast, models.HourlyForecast)

  def test_periods(self):
    periods = self.hourly_forecast.periods
    self.assertIsInstance(periods, list)

    with self.subTest(name="example_periods"):
      self.assertEqual(periods[0], conftest.FIRST_HOURLY_PERIOD)
      self.assertEqual(periods[-1], conftest.LAST_HOURLY_PERIOD)

    with self.subTest(name="spans_seven_days"):
      timedelta = periods[-1].end_timestamp - periods[0].start_timestamp
      self.assertEqual(
          timedelta,
          pd.Timedelta(days=6, hours=12),
      )

    with self.subTest(name="hourly_duration"):
      durations = {r.duration for r in periods}
      self.assertEqual(durations, {pd.Timedelta(hours=1)})

    with self.subTest(name="one_period_per_hour"):
      self.assertEqual(
          [r.start_timestamp for r in periods[0:3]],
          [
              pd.Timestamp("2026-02-06 09:00:00-08:00"),
              pd.Timestamp("2026-02-06 10:00:00-08:00"),
              pd.Timestamp("2026-02-06 11:00:00-08:00"),
          ],
      )

  def test_df(self):
    df = self.hourly_forecast.df
    self.assertIsInstance(df, pd.DataFrame)

    with self.subTest(name="example_periods"):
      comparison_df = df.drop(
          columns=["start_timestamp", "end_timestamp", "duration", "start_date", "end_date"]  # pylint: disable=line-too-long
      )
      self.assertEqual(
          comparison_df.iloc[0].to_dict(),
          dataclasses.asdict(conftest.FIRST_HOURLY_PERIOD),
      )
      self.assertEqual(
          comparison_df.iloc[-1].to_dict(),
          dataclasses.asdict(conftest.LAST_HOURLY_PERIOD),
      )

    with self.subTest(name="spans_seven_days"):
      timedelta = df["end_timestamp"].iloc[-1] - df["start_timestamp"].iloc[0]
      self.assertEqual(
          timedelta,
          pd.Timedelta(days=6, hours=12),
      )

    with self.subTest(name="hourly_duration"):
      durations = df["duration"].unique()
      self.assertEqual(durations, pd.Timedelta(hours=1))

    with self.subTest(name="one_period_per_hour"):
      self.assertEqual(
          df["start_timestamp"].iloc[0:3].to_list(),
          [
              pd.to_datetime("2026-02-06 09:00:00-08:00"),
              pd.to_datetime("2026-02-06 10:00:00-08:00"),
              pd.to_datetime("2026-02-06 11:00:00-08:00"),
          ],
      )


if __name__ == "__main__":
  absltest.main()
