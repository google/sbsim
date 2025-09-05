"""Tests for randomized_arrival_departure_occupancy."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import randomized_arrival_departure_occupancy
from smart_control.simulator.randomized_arrival_departure_occupancy import OccupancyStateEnum
from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy

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


class RandomizedArrivalDepartureOccupancyTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, _EXPECTED_ZONE_OCCUPANCIES_UTC),
      ('UTC', _EXPECTED_ZONE_OCCUPANCIES_UTC),
      ('US/Pacific', _EXPECTED_ZONE_OCCUPANCIES_PACIFIC),
      ('US/Eastern', _EXPECTED_ZONE_OCCUPANCIES_EASTERN),
  )
  def test_average_zone_occupancy_weekday(self, tz, expected_zone_occupancies):
    step_size = pd.Timedelta(5, unit='minute')

    occupancy = randomized_arrival_departure_occupancy.RandomizedArrivalDepartureOccupancy(  # pylint: disable=line-too-long
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
    occupant = randomized_arrival_departure_occupancy.ZoneOccupant(
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
    occupant = randomized_arrival_departure_occupancy.ZoneOccupant(
        8, 12, 13, 18, pd.Timedelta(5, unit='minute'), random_state, tz
    )
    current_time = pd.Timestamp('2021-09-01 00:00', tz=tz)
    while current_time < pd.Timestamp('2021-09-01 23:00', tz=tz):
      state = occupant.peek(current_time=current_time)

      if current_time < pd.Timestamp(
          '2021-09-01 11:10', tz=tz
      ) or current_time >= pd.Timestamp('2021-09-01 17:00', tz=tz):
        self.assertEqual(
            randomized_arrival_departure_occupancy.OccupancyStateEnum.AWAY,
            state,
        )
      else:
        self.assertEqual(
            randomized_arrival_departure_occupancy.OccupancyStateEnum.WORK,
            state,
        )
      current_time += pd.Timedelta(5, unit='minute')

  def test_average_zone_occupancy_matches_manual_two_steps(self):
    """average_zone_occupancy should equal the mean of per-step counts."""
    step = pd.Timedelta(minutes=5)
    tz = 'UTC'

    occ = RandomizedArrivalDepartureOccupancy(
        zone_assignment=7,
        earliest_expected_arrival_hour=8,
        latest_expected_arrival_hour=12,
        earliest_expected_departure_hour=16,
        latest_expected_departure_hour=20,
        time_step_sec=step.total_seconds(),
        seed=55213,
        time_zone=tz,
    )

    t0 = pd.Timestamp('2021-09-01 10:00', tz=tz)
    t1 = t0 + 2 * step

    # initialise the zone
    _ = occ.average_zone_occupancy('zone_0', t0, t0 + step)

    manual_counts = []
    for cur in (t0, t0 + step):
      c = 0.0
      for zocc in occ._zone_occupants['zone_0']:
        if zocc.peek(cur) == OccupancyStateEnum.WORK:
          c += 1.0
      manual_counts.append(c)
    manual_avg = sum(manual_counts) / 2.0

    result = occ.average_zone_occupancy('zone_0', t0, t1)

    # In the old implementation this would have returned manual_counts[0]
    # and the assertion would fail. With the fix, it matches the average.
    assert result == manual_avg


if __name__ == '__main__':
  absltest.main()
