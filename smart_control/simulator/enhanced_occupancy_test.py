"""Test the enhanced occupancy model."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator.enhanced_occupancy import EnhancedOccupancy
from smart_control.simulator.enhanced_occupancy import MinuteLevelZoneOccupant
from smart_control.simulator.enhanced_occupancy import OccupancyStateEnum
from smart_control.simulator.enhanced_occupancy import WorkerType

DEBUG_PRINT = False
SEED = 511211
TIME_STEP = pd.Timedelta(minutes=5)
EARLIEST_EXPECTED_ARRIVAL_HOUR = 8
LATEST_EXPECTED_ARRIVAL_HOUR = 10
EARLIEST_EXPECTED_DEPARTURE_HOUR = 16
LATEST_EXPECTED_DEPARTURE_HOUR = 18
LUNCH_START_HOUR = 12
LUNCH_END_HOUR = 14
NUM_OCCUPANTS = 10
DEFAULT_TIMEZONE = 'UTC'
NO_WEEKEND_WORKERS_REGULAR_PCT = 0.0
NO_WEEKEND_WORKERS_OCCASIONAL_PCT = 0.0
NO_WEEKEND_WORKERS_DAILY_PROB = 0.0
REGULAR_WEEKEND_WORKERS_REGULAR_PCT = 0.2
REGULAR_WEEKEND_WORKERS_OCCASIONAL_PCT = 0.0
REGULAR_WEEKEND_WORKERS_DAILY_PROB = 0.0
TEST_SETUP_REGULAR_PCT = 0.1
TEST_SETUP_OCCASIONAL_PCT = 0.2
TEST_SETUP_DAILY_PROB = 0.3


class EnhancedOccupancyTest(parameterized.TestCase):

  @parameterized.parameters('UTC', 'US/Pacific', 'US/Eastern')
  def test_average_occupancy_weekday(self, tz):
    occupancy = EnhancedOccupancy(
        zone_assignment=NUM_OCCUPANTS,
        earliest_expected_arrival_hour=EARLIEST_EXPECTED_ARRIVAL_HOUR,
        latest_expected_arrival_hour=LATEST_EXPECTED_ARRIVAL_HOUR,
        earliest_expected_departure_hour=EARLIEST_EXPECTED_DEPARTURE_HOUR,
        latest_expected_departure_hour=LATEST_EXPECTED_DEPARTURE_HOUR,
        lunch_start_hour=LUNCH_START_HOUR,
        lunch_end_hour=LUNCH_END_HOUR,
        time_step=TIME_STEP,
        time_zone=tz,
        weekend_regular_pct=NO_WEEKEND_WORKERS_REGULAR_PCT,
        weekend_occasional_pct=NO_WEEKEND_WORKERS_OCCASIONAL_PCT,
        occasional_daily_prob=NO_WEEKEND_WORKERS_DAILY_PROB,
    )

    current_time = pd.Timestamp('2021-09-01 00:00', tz=tz)
    occupancies = []
    while current_time < pd.Timestamp('2021-09-02 00:00', tz=tz):
      n = occupancy.average_zone_occupancy(
          'zone_0', current_time, current_time + TIME_STEP
      )
      occupancies.append(n)
      current_time += TIME_STEP

    early_morning_avg = np.mean(occupancies[0:48])  # 48 time steps = 4 hours
    morning_avg = np.mean(occupancies[96:132])  # 8-11
    lunch_avg = np.mean(occupancies[144:168])  # 12-14
    afternoon_avg = np.mean(occupancies[180:204])  # 15-17
    evening_avg = np.mean(occupancies[240:288])  # 20-24

    self.assertEqual(early_morning_avg, 0.0)
    self.assertEqual(evening_avg, 0.0)
    self.assertGreater(morning_avg, 0.0)
    self.assertGreater(afternoon_avg, 0.0)
    self.assertLess(lunch_avg, NUM_OCCUPANTS)

  def test_weekend_occupancy(self):
    weekday_only_occupancy = EnhancedOccupancy(
        zone_assignment=100,
        earliest_expected_arrival_hour=EARLIEST_EXPECTED_ARRIVAL_HOUR,
        latest_expected_arrival_hour=LATEST_EXPECTED_ARRIVAL_HOUR,
        earliest_expected_departure_hour=EARLIEST_EXPECTED_DEPARTURE_HOUR,
        latest_expected_departure_hour=LATEST_EXPECTED_DEPARTURE_HOUR,
        lunch_start_hour=LUNCH_START_HOUR,
        lunch_end_hour=LUNCH_END_HOUR,
        time_step=TIME_STEP,
        time_zone=DEFAULT_TIMEZONE,
        weekend_regular_pct=NO_WEEKEND_WORKERS_REGULAR_PCT,
        weekend_occasional_pct=NO_WEEKEND_WORKERS_OCCASIONAL_PCT,
        occasional_daily_prob=NO_WEEKEND_WORKERS_DAILY_PROB,
    )

    weekend_regular_occupancy = EnhancedOccupancy(
        zone_assignment=100,
        earliest_expected_arrival_hour=EARLIEST_EXPECTED_ARRIVAL_HOUR,
        latest_expected_arrival_hour=LATEST_EXPECTED_ARRIVAL_HOUR,
        earliest_expected_departure_hour=EARLIEST_EXPECTED_DEPARTURE_HOUR,
        latest_expected_departure_hour=LATEST_EXPECTED_DEPARTURE_HOUR,
        lunch_start_hour=LUNCH_START_HOUR,
        lunch_end_hour=LUNCH_END_HOUR,
        time_step=TIME_STEP,
        time_zone=DEFAULT_TIMEZONE,
        weekend_regular_pct=REGULAR_WEEKEND_WORKERS_REGULAR_PCT,
        weekend_occasional_pct=REGULAR_WEEKEND_WORKERS_OCCASIONAL_PCT,
        occasional_daily_prob=REGULAR_WEEKEND_WORKERS_DAILY_PROB,
    )
    saturday_morning_start = pd.Timestamp(
        '2021-09-04 08:00', tz=DEFAULT_TIMEZONE
    )
    saturday_morning_end = pd.Timestamp('2021-09-04 12:00', tz=DEFAULT_TIMEZONE)
    weekday_only_occupancy = weekday_only_occupancy.average_zone_occupancy(
        'zone_0', saturday_morning_start, saturday_morning_end
    )
    weekend_regular_occupancy = (
        weekend_regular_occupancy.average_zone_occupancy(
            'zone_0', saturday_morning_start, saturday_morning_end
        )
    )
    self.assertEqual(weekday_only_occupancy, 0.0)
    self.assertGreater(weekend_regular_occupancy, 0.0)

  def test_worker_distribution(self):
    occupancy = EnhancedOccupancy(
        zone_assignment=1000,
        earliest_expected_arrival_hour=EARLIEST_EXPECTED_ARRIVAL_HOUR,
        latest_expected_arrival_hour=LATEST_EXPECTED_ARRIVAL_HOUR,
        earliest_expected_departure_hour=EARLIEST_EXPECTED_DEPARTURE_HOUR,
        latest_expected_departure_hour=LATEST_EXPECTED_DEPARTURE_HOUR,
        lunch_start_hour=LUNCH_START_HOUR,
        lunch_end_hour=LUNCH_END_HOUR,
        time_step=TIME_STEP,
        time_zone=DEFAULT_TIMEZONE,
        weekend_regular_pct=TEST_SETUP_REGULAR_PCT,
        weekend_occasional_pct=TEST_SETUP_OCCASIONAL_PCT,
        occasional_daily_prob=TEST_SETUP_DAILY_PROB,
    )
    distribution = occupancy.get_worker_distribution('zone_0')
    total_workers = sum(distribution.values())
    self.assertEqual(total_workers, 1000)

    expected_regular = 1000 * TEST_SETUP_REGULAR_PCT
    expected_occasional = 1000 * TEST_SETUP_OCCASIONAL_PCT
    expected_weekday = 1000 * (
        1.0 - TEST_SETUP_REGULAR_PCT - TEST_SETUP_OCCASIONAL_PCT
    )
    self.assertAlmostEqual(
        distribution['weekday_only'], expected_weekday, delta=100
    )
    self.assertAlmostEqual(
        distribution['weekend_regular'], expected_regular, delta=100
    )
    self.assertAlmostEqual(
        distribution['weekend_occasional'], expected_occasional, delta=100
    )

  def test_parameter_variation(self):
    occupant = MinuteLevelZoneOccupant(
        earliest_expected_arrival_min=EARLIEST_EXPECTED_ARRIVAL_HOUR * 60,
        latest_expected_arrival_min=LATEST_EXPECTED_ARRIVAL_HOUR * 60,
        earliest_expected_departure_min=EARLIEST_EXPECTED_DEPARTURE_HOUR * 60,
        latest_expected_departure_min=LATEST_EXPECTED_DEPARTURE_HOUR * 60,
        lunch_start_min=LUNCH_START_HOUR * 60,
        lunch_end_min=LUNCH_END_HOUR * 60,
        step_size=TIME_STEP,
        random_state=np.random.RandomState(seed=SEED),
        time_zone=DEFAULT_TIMEZONE,
        worker_type=WorkerType.WEEKDAY_ONLY,
        weekend_work_prob=NO_WEEKEND_WORKERS_DAILY_PROB,
        occupant_id=0,
    )
    day1_morning = pd.Timestamp('2021-09-01 09:00', tz=DEFAULT_TIMEZONE)
    day1_afternoon = pd.Timestamp('2021-09-01 15:00', tz=DEFAULT_TIMEZONE)
    day2_morning = pd.Timestamp('2021-09-02 09:00', tz=DEFAULT_TIMEZONE)
    params1_morning = occupant._get_daily_params(day1_morning)
    params1_afternoon = occupant._get_daily_params(day1_afternoon)
    params2_morning = occupant._get_daily_params(day2_morning)
    self.assertEqual(params1_morning, params1_afternoon)
    self.assertNotEqual(params1_morning, params2_morning)

  @parameterized.parameters('UTC', 'US/Eastern', 'US/Pacific')
  def test_occupant_peek(self, tz):
    occupant = MinuteLevelZoneOccupant(
        earliest_expected_arrival_min=EARLIEST_EXPECTED_ARRIVAL_HOUR * 60,
        latest_expected_arrival_min=LATEST_EXPECTED_ARRIVAL_HOUR * 60,
        earliest_expected_departure_min=EARLIEST_EXPECTED_DEPARTURE_HOUR * 60,
        latest_expected_departure_min=LATEST_EXPECTED_DEPARTURE_HOUR * 60,
        lunch_start_min=LUNCH_START_HOUR * 60,
        lunch_end_min=LUNCH_END_HOUR * 60,
        step_size=TIME_STEP,
        random_state=np.random.RandomState(seed=SEED),
        time_zone=tz,
        worker_type=WorkerType.WEEKDAY_ONLY,
        weekend_work_prob=NO_WEEKEND_WORKERS_DAILY_PROB,
        occupant_id=0,
    )
    day = pd.Timestamp('2021-09-01 00:00', tz=tz)
    params = occupant._get_daily_params(day)
    day1_early_morning = pd.Timestamp('2021-09-01 06:00', tz=tz)
    day1_work_morning = pd.Timestamp('2021-09-01 10:00', tz=tz)
    day1_afternoon = pd.Timestamp('2021-09-01 15:00', tz=tz)
    day1_evening = pd.Timestamp('2021-09-01 20:00', tz=tz)
    weekend = pd.Timestamp('2021-09-05 08:00', tz=tz)

    def expected_state(ts: pd.Timestamp):
      ts_local = ts.tz_convert(tz)
      minutes = ts_local.hour * 60 + ts_local.minute
      in_work = params['arrival_time'] <= minutes < params['departure_time']
      in_lunch = (
          params['lunch_start_time']
          <= minutes
          < params['lunch_start_time'] + params['lunch_duration']
      )
      return (
          OccupancyStateEnum.WORK
          if (in_work and not in_lunch)
          else OccupancyStateEnum.AWAY
      )

    self.assertEqual(occupant.peek(day1_early_morning), OccupancyStateEnum.AWAY)
    self.assertEqual(
        occupant.peek(day1_work_morning), expected_state(day1_work_morning)
    )
    self.assertEqual(
        occupant.peek(day1_afternoon), expected_state(day1_afternoon)
    )
    self.assertEqual(occupant.peek(day1_evening), OccupancyStateEnum.AWAY)
    self.assertEqual(occupant.peek(weekend), OccupancyStateEnum.AWAY)

  def test_occasional_worker(self):
    occupant = MinuteLevelZoneOccupant(
        earliest_expected_arrival_min=EARLIEST_EXPECTED_ARRIVAL_HOUR * 60,
        latest_expected_arrival_min=LATEST_EXPECTED_ARRIVAL_HOUR * 60,
        earliest_expected_departure_min=EARLIEST_EXPECTED_DEPARTURE_HOUR * 60,
        latest_expected_departure_min=LATEST_EXPECTED_DEPARTURE_HOUR * 60,
        lunch_start_min=LUNCH_START_HOUR * 60,
        lunch_end_min=LUNCH_END_HOUR * 60,
        step_size=TIME_STEP,
        random_state=np.random.RandomState(seed=SEED),
        time_zone=DEFAULT_TIMEZONE,
        worker_type=WorkerType.WEEKEND_OCCASIONAL,
        weekend_work_prob=0.5,
        occupant_id=13,
    )
    saturday_morning = pd.Timestamp('2021-09-04 08:00', tz=DEFAULT_TIMEZONE)
    saturday_afternoon = pd.Timestamp('2021-09-04 15:00', tz=DEFAULT_TIMEZONE)
    work_decision_morning = occupant._should_work_today(saturday_morning)
    work_decision_afternoon = occupant._should_work_today(saturday_afternoon)
    self.assertEqual(work_decision_morning, work_decision_afternoon)

  def test_minutes_precision(self):
    occupancy = EnhancedOccupancy(
        zone_assignment=NUM_OCCUPANTS,
        earliest_expected_arrival_hour=EARLIEST_EXPECTED_ARRIVAL_HOUR,
        latest_expected_arrival_hour=LATEST_EXPECTED_ARRIVAL_HOUR,
        earliest_expected_departure_hour=EARLIEST_EXPECTED_DEPARTURE_HOUR,
        latest_expected_departure_hour=LATEST_EXPECTED_DEPARTURE_HOUR,
        lunch_start_hour=LUNCH_START_HOUR,
        lunch_end_hour=LUNCH_END_HOUR,
        time_step=TIME_STEP,
        time_zone=DEFAULT_TIMEZONE,
        weekend_regular_pct=NO_WEEKEND_WORKERS_REGULAR_PCT,
        weekend_occasional_pct=NO_WEEKEND_WORKERS_OCCASIONAL_PCT,
        occasional_daily_prob=NO_WEEKEND_WORKERS_DAILY_PROB,
    )
    time_758 = pd.Timestamp('2021-09-01 07:58', tz=DEFAULT_TIMEZONE)
    time_759 = pd.Timestamp('2021-09-01 07:59', tz=DEFAULT_TIMEZONE)
    morning_start = pd.Timestamp('2021-09-01 08:30', tz=DEFAULT_TIMEZONE)
    morning_end = pd.Timestamp('2021-09-01 11:30', tz=DEFAULT_TIMEZONE)
    occ_758 = occupancy.average_zone_occupancy(
        'zone_0', time_758, time_758 + pd.Timedelta(minutes=1)
    )
    occ_759 = occupancy.average_zone_occupancy(
        'zone_0', time_759, time_759 + pd.Timedelta(minutes=1)
    )
    morning_occupancy = occupancy.average_zone_occupancy(
        'zone_0', morning_start, morning_end
    )
    self.assertEqual(occ_758, 0.0)
    self.assertEqual(occ_759, 0.0)
    self.assertGreater(morning_occupancy, 0.0)


if __name__ == '__main__':
  absltest.main()
