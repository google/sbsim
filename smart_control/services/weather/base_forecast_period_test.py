import datetime
import re

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_buildings.smart_control.services.weather import base_forecast_period
from smart_buildings.smart_control.services.weather import conftest

EXPECTED_TZINFO = datetime.timezone.utc


class BaseForecastPeriodTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.period = conftest.create_period()

  def test_initialization(self):
    self.assertIsInstance(self.period, base_forecast_period.BaseForecastPeriod)

  def test_attributes(self):
    with self.subTest(name='timestamps'):
      self.assertEqual(self.period.start_timestamp, conftest.START_TIMESTAMP)
      self.assertEqual(self.period.end_timestamp, conftest.END_TIMESTAMP)

    with self.subTest(name='time_zones'):
      self.assertEqual(self.period.start_timestamp.tzinfo, EXPECTED_TZINFO)
      self.assertEqual(self.period.end_timestamp.tzinfo, EXPECTED_TZINFO)

    with self.subTest(name='temperature'):
      self.assertEqual(self.period.temp, conftest.TEMP)
      self.assertEqual(self.period.temp_unit, conftest.TEMP_UNIT)

  def test_properties(self):
    with self.subTest(name='duration'):
      self.assertEqual(self.period.duration, pd.Timedelta(minutes=1))

    with self.subTest(name='dates'):
      self.assertEqual(self.period.start_date, '2026-01-01')
      self.assertEqual(self.period.end_date, '2026-01-01')

    with self.subTest(name='seconds_since_epoch'):
      self.assertEqual(self.period.start_seconds, 1767226020.0)
      self.assertEqual(self.period.end_seconds, 1767226080.0)

  def _assert_timestamps_regex_raises(self, end_timestamp: pd.Timestamp):
    with self.assertRaisesRegex(
        ValueError,
        re.escape((
            f'start_timestamp ({conftest.START_TIMESTAMP}) must be before '
            f'end_timestamp ({end_timestamp}).'
        )),
    ):
      conftest.create_period(end_timestamp=end_timestamp)

  def test_validates_misordered_timestamps_raises(self):
    end_timestamp = conftest.START_TIMESTAMP - pd.Timedelta(minutes=1)
    self._assert_timestamps_regex_raises(end_timestamp)

  def test_validates_same_timestamps_raises(self):
    end_timestamp = conftest.START_TIMESTAMP  # end == start
    self._assert_timestamps_regex_raises(end_timestamp)

  @parameterized.named_parameters(
      dict(
          testcase_name='utc',
          time_zone='UTC',
      ),
      dict(testcase_name='est', time_zone='America/New_York'),
      dict(testcase_name='pst', time_zone='America/Los_Angeles'),
  )
  def test_validates_time_zones(self, time_zone):
    start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz=time_zone)
    end_timestamp = pd.Timestamp('2026-01-01 01:00:00', tz=time_zone)
    conftest.create_period(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )

  def test_validates_missing_time_zones_raises(self):
    start_timestamp = pd.Timestamp('2026-01-01 00:00:00')
    end_timestamp = pd.Timestamp('2026-01-01 01:00:00')
    with self.assertRaisesRegex(
        ValueError,
        re.escape((
            f'start_timestamp ({start_timestamp}) and end_timestamp'
            f' ({end_timestamp}) must have a time zone.'
        )),
    ):
      conftest.create_period(
          start_timestamp=start_timestamp,
          end_timestamp=end_timestamp,
      )

  def test_validates_mismatched_time_zones_raises(self):
    start_timestamp = pd.Timestamp('2026-01-01 00:00:00', tz='UTC')
    end_timestamp = pd.Timestamp('2026-01-01 01:00:00', tz='America/New_York')
    with self.assertRaisesRegex(
        ValueError,
        re.escape((
            f'start_timestamp ({start_timestamp}) must be in the same time'
            f' zone as end_timestamp ({end_timestamp}).'
        )),
    ):
      conftest.create_period(
          start_timestamp=start_timestamp,
          end_timestamp=end_timestamp,
      )

  def test_as_dict(self):
    self.assertEqual(
        self.period.as_dict,
        {
            'start_timestamp': conftest.START_TIMESTAMP,
            'end_timestamp': conftest.END_TIMESTAMP,
            'temp': conftest.TEMP,
            'temp_unit': conftest.TEMP_UNIT,
            'duration': self.period.duration,
            'start_date': self.period.start_date,
            'end_date': self.period.end_date,
            'start_seconds': self.period.start_seconds,
            'end_seconds': self.period.end_seconds,
        },
    )


if __name__ == '__main__':
  absltest.main()
