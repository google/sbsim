from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy

OccupancyStateEnum = randomized_arrival_departure_occupancy.OccupancyStateEnum
RandomizedArrivalDepartureOccupancy = randomized_arrival_departure_occupancy.RandomizedArrivalDepartureOccupancy  # pylint: disable=line-too-long
ZoneOccupant = randomized_arrival_departure_occupancy.ZoneOccupant


# fmt: off
# pylint: disable=bad-continuation
_EXPECTED_ZONE_OCCUPANCIES_PACIFIC = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0,
  4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0,
  8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
]

_EXPECTED_ZONE_OCCUPANCIES_EASTERN = [
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0,
  6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
  8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
  9.0, 8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
  5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
  3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
]

_EXPECTED_ZONE_OCCUPANCIES_UTC = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2,
  2, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
  7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8,
  8, 8, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
]
# pylint: disable=bad-continuation
# fmt: off


def create_zone_occupant(
    time_zone='US/Pacific',
    step_size=pd.Timedelta(5, unit='minute'),
    arrival_earliest=6,
    arrival_latest=11,
    departure_earliest=15,
    departure_latest=20,
):
  return ZoneOccupant(
      time_zone=time_zone,
      step_size=step_size,
      random_state=np.random.RandomState(seed=99),
      earliest_expected_arrival_hour=arrival_earliest,
      latest_expected_arrival_hour=arrival_latest,
      earliest_expected_departure_hour=departure_earliest,
      latest_expected_departure_hour=departure_latest,
  )


def create_occupancy(
    zone_assignment=1,
    earliest_arrival=6,
    latest_arrival=11,
    earliest_departure=15,
    latest_departure=20,
    time_step_sec=300,
    time_zone='US/Pacific',
    seed=99,
):
  """Creates an occupancy object, using default parameters, for use in tests."""
  return RandomizedArrivalDepartureOccupancy(
      zone_assignment=zone_assignment,
      earliest_expected_arrival_hour=earliest_arrival,
      latest_expected_arrival_hour=latest_arrival,
      earliest_expected_departure_hour=earliest_departure,
      latest_expected_departure_hour=latest_departure,
      time_step_sec=time_step_sec,
      time_zone=time_zone,
      seed=seed,
  )


class RandomizedArrivalDepartureOccupancyTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, _EXPECTED_ZONE_OCCUPANCIES_UTC),
      ('UTC', _EXPECTED_ZONE_OCCUPANCIES_UTC),
      ('US/Pacific', _EXPECTED_ZONE_OCCUPANCIES_PACIFIC),
      ('US/Eastern', _EXPECTED_ZONE_OCCUPANCIES_EASTERN),
  )
  def test_average_zone_occupancy_weekday(self, tz, expected_zone_occupancies):
    step_size = pd.Timedelta(5, unit='minute')

    occupancy = RandomizedArrivalDepartureOccupancy(
        10, 7, 11, 15, 20, step_size.total_seconds(), 511211, tz
    )
    current_time = pd.Timestamp('2021-09-01 00:00', tz='UTC')
    actual_occupancies = []
    while current_time < pd.Timestamp('2021-09-01 23:00', tz='UTC'):
      n = occupancy.average_zone_occupancy(
          'zone_0', current_time, current_time + step_size
      )
      actual_occupancies.append(n)

      current_time += step_size
    self.assertSequenceEqual(expected_zone_occupancies, actual_occupancies)

  def test_get_event_probability(self):
    occupant = ZoneOccupant(
        8,
        12,
        13,
        18,
        pd.Timedelta(5, unit='minute'),
        np.random.RandomState(seed=55213),
    )
    p = occupant._get_event_probability(8, 12)
    self.assertEqual(p, 1.0 / 24.0)

  @parameterized.parameters((None), 'UTC', 'US/Eastern', 'US/Pacific')
  def test_peek(self, tz):
    random_state = np.random.RandomState(seed=55213)
    occupant = ZoneOccupant(
        8, 12, 13, 18, pd.Timedelta(5, unit='minute'), random_state, tz
    )
    current_time = pd.Timestamp('2021-09-01 00:00', tz=tz)
    while current_time < pd.Timestamp('2021-09-01 23:00', tz=tz):
      state = occupant.peek(current_time=current_time)

      work_begin = pd.Timestamp('2021-09-01 11:10', tz=tz)
      work_end = pd.Timestamp('2021-09-01 17:00', tz=tz)
      if current_time < work_begin or current_time >= work_end:
        self.assertEqual(OccupancyStateEnum.AWAY, state)
      else:
        self.assertEqual(OccupancyStateEnum.WORK, state)

      current_time += pd.Timedelta(5, unit='minute')

  @parameterized.named_parameters(
      dict(
          testcase_name='naive',
          time_zone=None,
          expected_timestamp='2026-01-01 19:30:00-08:00',
      ),
      dict(
          testcase_name='utc',
          time_zone='UTC',
          expected_timestamp='2026-01-01 11:30:00-08:00',
      ),
      dict(
          testcase_name='eastern',
          time_zone='US/Eastern',
          expected_timestamp='2026-01-01 16:30:00-08:00',
      ),
      dict(
          testcase_name='pacific',
          time_zone='US/Pacific',
          expected_timestamp='2026-01-01 19:30:00-08:00',
      ),
  )
  def test_time_zone_conversion(self, time_zone, expected_timestamp):
    occupant = create_zone_occupant(time_zone='US/Pacific')

    timestamp = pd.Timestamp('2026-01-01 19:30', tz=time_zone)
    local_time = occupant._to_local_time(timestamp)

    self.assertEqual(str(local_time.tz), 'US/Pacific')
    self.assertEqual(str(local_time), expected_timestamp)

  def test_json_metadata(self):
    occupancy = create_occupancy()
    expected_metadata = {
        'type': 'RandomizedArrivalDepartureOccupancy',
        'zone_assignment': 1,
        'earliest_arrival': 6,
        'latest_arrival': 11,
        'earliest_departure': 15,
        'latest_departure': 20,
        'time_step_sec': 300.0,
        'time_zone': 'US/Pacific',
    }
    self.assertDictEqual(expected_metadata, occupancy.json_metadata)


if __name__ == '__main__':
  absltest.main()
