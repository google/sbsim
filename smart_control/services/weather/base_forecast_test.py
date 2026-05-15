from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast
from smart_buildings.smart_control.services.weather import conftest


class BaseForecastTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.periods = conftest.create_hourly_periods()
    self.forecast = base_forecast.BaseForecast(periods=self.periods)

  def test_initialization(self):
    self.assertIsInstance(self.forecast, base_forecast.BaseForecast)

  def test_attributes(self):
    self.assertEqual(self.forecast.periods, self.periods)
    self.assertEqual(
        self.forecast.interpolation_interval,
        pd.Timedelta(minutes=1),
    )

  def test_validates_misordered_periods_raises(self):
    periods = list(self.periods).copy()
    # Manually misorder the periods by swapping the first two.
    periods[0], periods[1] = periods[1], periods[0]
    with self.assertRaisesRegex(
        ValueError,
        'Periods must be sorted by start_timestamp in ascending order.',
    ):
      base_forecast.BaseForecast(periods=periods)

  def test_validates_empty_periods_raises(self):
    with self.assertRaisesRegex(ValueError, 'Periods cannot be empty.'):
      base_forecast.BaseForecast(periods=[])

  def test_temp_unit(self):
    self.assertEqual(self.forecast.temp_unit, conftest.TEMP_UNIT)
    self.assertEqual(self.forecast.temp_unit.value, 'Fahrenheit')

  def test_df(self):
    df = self.forecast.df
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 24)
    self.assertEqual(
        df.columns.tolist(),
        [
            'start_timestamp',
            'end_timestamp',
            'temp',
            'temp_unit',
            'duration',
            'start_date',
            'end_date',
            'start_seconds',
            'end_seconds',
        ],
    )


class BaseForecastInterpolationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz='UTC')
    self.periods = [
        conftest.create_period(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            temp=10.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            temp=20.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=3),
            temp=30.0,
        ),
    ]
    self.forecast = base_forecast.BaseForecast(periods=self.periods)

  def test_first_period(self):
    self.assertEqual(self.forecast.first_period, self.periods[0])

  def test_last_period(self):
    self.assertEqual(self.forecast.last_period, self.periods[-1])

  def test_seconds(self):
    self.assertEqual(
        list(self.forecast.seconds),
        [p.start_seconds for p in self.periods],
    )

  def test_temps(self):
    self.assertEqual(list(self.forecast.temps), [10.0, 20.0, 30.0])

  def test_interpolation_interval(self):
    self.assertEqual(
        self.forecast.interpolation_interval,
        pd.Timedelta(minutes=1),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='start_of_forecast',
          timestamp=pd.Timestamp('2026-01-01 00:00:00', tz='UTC'),
          expected_temp=10.0,
      ),
      dict(
          testcase_name='middle_of_first_hour',
          timestamp=pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
          expected_temp=15.0,
      ),
      dict(
          testcase_name='start_of_second_hour',
          timestamp=pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
          expected_temp=20.0,
      ),
      dict(
          testcase_name='middle_of_second_hour',
          timestamp=pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
          expected_temp=25.0,
      ),
      dict(
          testcase_name='start_of_third_hour',
          timestamp=pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
          expected_temp=30.0,
      ),
      dict(
          testcase_name='middle_of_third_hour',
          timestamp=pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
          expected_temp=30.0,
      ),
      dict(
          testcase_name='end_of_forecast_range',
          timestamp=pd.Timestamp('2026-01-01 03:00:00', tz='UTC'),
          expected_temp=30.0,
      ),
  )
  def test_interpolate_period(self, timestamp, expected_temp):
    interpolated_period = self.forecast.interpolate_period(timestamp)

    with self.subTest('temp'):
      self.assertAlmostEqual(interpolated_period.temp, expected_temp)
      self.assertEqual(interpolated_period.temp_unit, self.forecast.temp_unit)

    with self.subTest('timestamps'):
      self.assertEqual(interpolated_period.start_timestamp, timestamp)
      self.assertEqual(
          interpolated_period.end_timestamp,
          timestamp + self.forecast.interpolation_interval,
      )

    with self.subTest('time_zone'):
      self.assertEqual(
          interpolated_period.start_timestamp.tzinfo,
          self.forecast.tzinfo,
      )

  def test_interpolate_period_different_tz_converts(self):
    timestamp_est = pd.Timestamp('2025-12-31 19:30:00', tz='America/New_York')
    # This is 2026-01-01 00:30:00 UTC, which is in the middle of the first hour.
    interpolated_period = self.forecast.interpolate_period(timestamp_est)

    self.assertAlmostEqual(interpolated_period.temp, 15.0)
    self.assertEqual(
        interpolated_period.start_timestamp,
        pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
    )
    self.assertEqual(
        interpolated_period.start_timestamp.tzinfo, self.forecast.tzinfo
    )

  def test_interpolate_period_naive_tz_raises(self):
    timestamp = pd.Timestamp('2026-01-01 00:30:00')
    with self.assertRaisesRegex(ValueError, 'must be timezone-aware'):
      self.forecast.interpolate_period(timestamp)

  @parameterized.named_parameters(
      dict(
          testcase_name='before_forecast',
          timestamp=pd.Timestamp('2025-12-31 23:59:59', tz='UTC'),
      ),
      dict(
          testcase_name='after_forecast',
          timestamp=pd.Timestamp('2026-01-01 03:00:01', tz='UTC'),
      ),
  )
  def test_interpolate_period_outside_range_raises(self, timestamp):
    with self.assertRaisesRegex(ValueError, 'is outside the forecast range'):
      self.forecast.interpolate_period(timestamp)

  def test_interpolated_forecast(self):
    interpolated_forecast = self.forecast.interpolated_forecast
    self.assertIsInstance(interpolated_forecast, base_forecast.BaseForecast)
    self.assertLen(interpolated_forecast.periods, 180)

    with self.subTest('first_period'):
      first_period = interpolated_forecast.first_period
      self.assertEqual(first_period.start_timestamp, self.start_timestamp)
      self.assertAlmostEqual(first_period.temp, 10.0)

    with self.subTest('last_period'):
      last_period = interpolated_forecast.last_period
      self.assertEqual(
          last_period.start_timestamp,
          self.start_timestamp
          + pd.Timedelta(hours=3)
          - pd.Timedelta(minutes=1),
      )
      self.assertAlmostEqual(last_period.temp, 30.0)

  def test_interp_df(self):
    interp_df = self.forecast.interp_df
    self.assertIsInstance(interp_df, pd.DataFrame)
    self.assertLen(interp_df, 180)

    with self.subTest('columns'):
      self.assertEqual(
          interp_df.columns.tolist(),
          [
              'start_timestamp',
              'end_timestamp',
              'temp',
              'temp_unit',
              'duration',
              'start_date',
              'end_date',
              'start_seconds',
              'end_seconds',
          ],
      )

    with self.subTest('first_row'):
      first_row = interp_df.iloc[0]
      self.assertEqual(
          first_row.start_timestamp,
          self.start_timestamp,
      )
      self.assertAlmostEqual(first_row.temp, 10.0)

    with self.subTest('last_row'):
      last_row = interp_df.iloc[-1]
      self.assertEqual(
          last_row.start_timestamp,
          self.start_timestamp
          + pd.Timedelta(hours=3)
          - pd.Timedelta(minutes=1),
      )
      self.assertAlmostEqual(last_row.temp, 30.0)


class BaseForecastResampleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz='UTC')
    self.periods = [
        conftest.create_period(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            temp=10.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            temp=20.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=3),
            temp=30.0,
        ),
    ]
    self.forecast = base_forecast.BaseForecast(periods=self.periods)

  def test_resample_interval_hourly_same_start(self):
    resampled = self.forecast.resample(start_timestamp=self.start_timestamp)
    self.assertLen(resampled.periods, 3)
    expected_values = [
        (
            pd.Timestamp('2026-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
            10.0,
        ),
        (
            pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
            20.0,
        ),
        (
            pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 03:00:00', tz='UTC'),
            30.0,
        ),
    ]
    self.assertEqual(
        [
            (p.start_timestamp, p.end_timestamp, p.temp)
            for p in resampled.periods
        ],
        expected_values
    )

  def test_resample_interval_hourly_offset_start(self):
    new_start_timestamp = self.start_timestamp + pd.Timedelta(minutes=30)
    resampled = self.forecast.resample(start_timestamp=new_start_timestamp)
    self.assertLen(resampled.periods, 2)
    expected_values = [
        (
            pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            15.0,
        ),
        (
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
            25.0,
        ),
    ]
    self.assertEqual(
        [
            (p.start_timestamp, p.end_timestamp, p.temp)
            for p in resampled.periods
        ],
        expected_values
    )

  def test_resample_interval_30m_same_start(self):
    resampled = self.forecast.resample(
        start_timestamp=self.start_timestamp, interval=pd.Timedelta(minutes=30)
    )
    self.assertLen(resampled.periods, 6)
    expected_values = [
        (
            pd.Timestamp('2026-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
            10.0,
        ),
        (
            pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
            15.0,
        ),
        (
            pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            20.0,
        ),
        (
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
            25.0,
        ),
        (
            pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
            pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
            30.0,
        ),
        (
            pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 03:00:00', tz='UTC'),
            30.0,
        ),
    ]
    self.assertEqual(
        [
            (p.start_timestamp, p.end_timestamp, p.temp)
            for p in resampled.periods
        ],
        expected_values
    )

  def test_resample_different_tz_converts(self):
    new_start_timestamp = pd.Timestamp(
        '2025-12-31 19:30:00', tz='America/New_York'
    )  # '2026-01-01 00:30:00' in UTC.
    resampled = self.forecast.resample(start_timestamp=new_start_timestamp)
    self.assertLen(resampled.periods, 2)
    expected_values = [
        (
            pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            15.0,
        ),
        (
            pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
            pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
            25.0,
        ),
    ]
    self.assertEqual(
        [
            (p.start_timestamp, p.end_timestamp, p.temp)
            for p in resampled.periods
        ],
        expected_values
    )

  def test_resample_naive_tz_raises(self):
    start_ts = pd.Timestamp('2026-01-01 00:00:00')
    with self.assertRaisesRegex(ValueError, 'must be timezone-aware'):
      self.forecast.resample(start_ts)

  def test_resample_invalid_interval_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'Interval must be a positive duration.'
    ):
      self.forecast.resample(
          self.start_timestamp, interval=pd.Timedelta(minutes=0)
      )


class BaseForecastCustomTimezoneInterpolationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tz = 'America/Los_Angeles'
    self.start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz=self.tz)
    self.periods = [
        conftest.create_period(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            temp=10.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            temp=20.0,
        ),
    ]
    self.forecast = base_forecast.BaseForecast(periods=self.periods)

  def test_interpolation_uses_utc_seconds(self):
    # Verify that the seconds used for interpolation are in UTC (epoch seconds).
    # '2026-01-01 00:00:00' in America/Los_Angeles is
    # '2026-01-01 08:00:00' in UTC.

    first_period = self.periods[0]
    self.assertEqual(
        str(first_period.start_timestamp),
        '2026-01-01 00:00:00-08:00',
    )

    expected_utc_start = pd.Timestamp('2026-01-01 08:00:00', tz='UTC')
    self.assertEqual(first_period.start_seconds, expected_utc_start.timestamp())

    # Interpolate at the middle of the first hour (30 minutes in).
    # 10.0 at t=0, 20.0 at t=3600. At t=1800, should be 15.0.
    timestamp = self.start_timestamp + pd.Timedelta(minutes=30)
    interpolated_period = self.forecast.interpolate_period(timestamp)

    self.assertAlmostEqual(interpolated_period.temp, 15.0)
    self.assertEqual(interpolated_period.start_timestamp, timestamp)
    self.assertEqual(
        interpolated_period.start_timestamp.tzinfo,
        first_period.start_timestamp.tzinfo,
    )


class BaseForecastFilterPeriodsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz='UTC')
    self.periods = [
        conftest.create_period(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            temp=10.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=1),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            temp=20.0,
        ),
        conftest.create_period(
            start_timestamp=self.start_timestamp + pd.Timedelta(hours=2),
            end_timestamp=self.start_timestamp + pd.Timedelta(hours=3),
            temp=30.0,
        ),
    ]
    self.forecast = base_forecast.BaseForecast(periods=self.periods)

  @parameterized.named_parameters(
      # DEFAULT PARAMS
      dict(
          testcase_name='defaults',
          timestamp=None,
          max_periods=24,
          expected_indices=[0, 1, 2],
      ),
      # MAX PERIODS FILTERING SCENARIOS
      dict(
          testcase_name='max_periods_less_than_count',
          timestamp=None,
          max_periods=2,
          expected_indices=[0, 1],
      ),
      dict(
          testcase_name='max_periods_greater_than_count',
          timestamp=None,
          max_periods=5,
          expected_indices=[0, 1, 2],
      ),
      dict(
          testcase_name='max_periods_is_none',
          timestamp=None,
          max_periods=None,
          expected_indices=[0, 1, 2],
      ),
      # TIMESTAMP FILTERING SCENARIOS > UTC
      dict(
          testcase_name='utc_ts_during_first_period',
          timestamp=pd.Timestamp('2026-01-01 00:30:00', tz='UTC'),
          max_periods=None,
          expected_indices=[0, 1, 2],
      ),
      dict(
          testcase_name='utc_ts_at_first_period_end',
          timestamp=pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
          max_periods=None,
          expected_indices=[1, 2],
      ),
      dict(
          testcase_name='utc_ts_at_first_period_end_with_max_periods',
          timestamp=pd.Timestamp('2026-01-01 01:00:00', tz='UTC'),
          max_periods=1,
          expected_indices=[1],
      ),
      dict(
          testcase_name='utc_ts_during_second_period',
          timestamp=pd.Timestamp('2026-01-01 01:30:00', tz='UTC'),
          max_periods=None,
          expected_indices=[1, 2],
      ),
      dict(
          testcase_name='utc_ts_at_second_period_end',
          timestamp=pd.Timestamp('2026-01-01 02:00:00', tz='UTC'),
          max_periods=None,
          expected_indices=[2],
      ),
      dict(
          testcase_name='utc_ts_during_third_period',
          timestamp=pd.Timestamp('2026-01-01 02:30:00', tz='UTC'),
          max_periods=None,
          expected_indices=[2],
      ),
      # TIMESTAMP FILTERING SCENARIOS > OTHER TIMEZONE
      dict(
          testcase_name='est_ts_during_first_period',  # 00:30 UTC
          timestamp=pd.Timestamp('2025-12-31 19:30:00', tz='America/New_York'),
          max_periods=None,
          expected_indices=[0, 1, 2],
      ),
      dict(
          testcase_name='est_ts_at_first_period_end',  # 01:00 UTC
          timestamp=pd.Timestamp('2025-12-31 20:00:00', tz='America/New_York'),
          max_periods=None,
          expected_indices=[1, 2],
      ),
      dict(
          testcase_name='est_ts_at_first_period_end_with_max_periods',
          timestamp=pd.Timestamp('2025-12-31 20:00:00', tz='America/New_York'),
          max_periods=1,
          expected_indices=[1],
      ),
      dict(
          testcase_name='est_ts_during_second_period',  # 01:30 UTC
          timestamp=pd.Timestamp('2025-12-31 20:30:00', tz='America/New_York'),
          max_periods=None,
          expected_indices=[1, 2],
      ),
      dict(
          testcase_name='est_ts_at_second_period_end',  # 02:00 UTC
          timestamp=pd.Timestamp('2025-12-31 21:00:00', tz='America/New_York'),
          max_periods=None,
          expected_indices=[2],
      ),
      dict(
          testcase_name='est_ts_during_third_period',  # 02:30 UTC
          timestamp=pd.Timestamp('2025-12-31 21:30:00', tz='America/New_York'),
          max_periods=None,
          expected_indices=[2],
      ),
  )
  def test_filter_periods(
      self, timestamp, max_periods, expected_indices
  ):
    new_forecast = self.forecast.filter_periods(
        ends_after_timestamp=timestamp, max_periods=max_periods
    )
    expected_periods = [self.periods[i] for i in expected_indices]
    self.assertEqual(list(new_forecast.periods), expected_periods)

  def test_filter_periods_naive_timezone_raises(self):
    ends_after_timestamp = pd.Timestamp('2026-01-01 01:00:00')
    with self.assertRaisesRegex(ValueError, 'must be timezone-aware'):
      self.forecast.filter_periods(ends_after_timestamp=ends_after_timestamp)

  def test_filter_periods_all_filtered_raises(self):
    ends_after_timestamp = pd.Timestamp('2026-01-01 03:01:00', tz='UTC')
    with self.assertRaisesRegex(ValueError, 'Periods cannot be empty.'):
      self.forecast.filter_periods(ends_after_timestamp=ends_after_timestamp)

  def test_filter_periods_max_periods_is_zero_raises(self):
    with self.assertRaisesRegex(ValueError, 'Periods cannot be empty.'):
      self.forecast.filter_periods(max_periods=0)


if __name__ == '__main__':
  absltest.main()
