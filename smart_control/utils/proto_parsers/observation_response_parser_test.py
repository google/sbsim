"""Tests for the ObservationResponseParser class."""

from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.utils import test_utils
from smart_buildings.smart_control.utils.proto_parsers import observation_response_parser

Parser = observation_response_parser.ObservationResponseParser

NAIVE_TIMESTAMP = pd.Timestamp('2022-03-13 12:00:00')
UTC_TIMESTAMP = pd.Timestamp('2022-03-13 12:00:00', tz='UTC')
LOCAL_TIMESTAMP = pd.Timestamp('2022-03-13 05:00:00', tz='US/Pacific')


class ObservationResponseParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.observation_response = test_utils.get_test_observation_response(
        timestamp=UTC_TIMESTAMP,
    )
    self.parser = Parser(self.observation_response)

  def test_timestamp(self):
    self.assertEqual(self.parser.timestamp, UTC_TIMESTAMP)

  def test_get_local_time(self):
    self.assertEqual(self.parser.get_local_time(), LOCAL_TIMESTAMP)

  def test_single_observation_responses(self):
    self.assertEqual(
        self.parser.single_observation_responses,
        self.observation_response.single_observation_responses,
    )

  def test_observations_df(self):
    df = self.parser.observations_df
    self.assertIsInstance(df, pd.DataFrame)

    with self.subTest(name='columns'):
      self.assertEqual(df.columns.tolist(), [
          'device_id', 'measurement_name', 'is_valid', 'continuous_value'
      ])

    with self.subTest(name='rows'):
      self.assertLen(df, 4)

  def test_outside_air_temp(self):
    # consider parameterizing these tests

    with self.subTest(name='no outside air temp'):
      with self.assertRaises(ValueError):
        self.parser.outside_air_temp  # pylint: disable=pointless-statement

    with self.subTest(name='has outside air temp'):
      observation_response = test_utils.get_test_observation_response(
          timestamp=NAIVE_TIMESTAMP,
          device_measurement_values=[
              ('device_0', 'measurement_0', 7.0),
              ('device_0', 'measurement_1', 0.1),
              ('device_1', 'measurement_0', 10.0),
              ('device_1', 'measurement_1', -0.2),
              ('device_2', 'outside_air_temperature_sensor', 295.0),
          ],
      )
      parser = Parser(observation_response)
      self.assertEqual(parser.outside_air_temp, 295.0)

    with self.subTest(name='has multiple outside air temp'):
      observation_response = test_utils.get_test_observation_response(
          timestamp=NAIVE_TIMESTAMP,
          device_measurement_values=[
              ('device_0', 'measurement_0', 7.0),
              ('device_0', 'measurement_1', 0.1),
              ('device_1', 'measurement_0', 10.0),
              ('device_1', 'measurement_1', -0.2),
              ('device_2', 'ac1_outside_air_temperature_sensor', 295.0),
              ('device_3', 'ac2_outside_air_temperature_sensor', 305.0),
          ],
      )
      parser = Parser(observation_response)
      self.assertLen(parser.outside_air_temps_df, 2)
      self.assertEqual(parser.outside_air_temp, 300.0)  # averages them


if __name__ == '__main__':
  absltest.main()
