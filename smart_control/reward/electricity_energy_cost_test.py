"""Tests for electricity_energy_cost."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_control.reward import electricity_energy_cost


class ElectricityEnergyCostTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'zero_energy',
          pd.Timestamp('2021-05-06 10:00:00+0'),
          pd.Timestamp('2021-05-06 11:00:00+0'),
          0.0,
          0.0,
      ),
      (
          'peak_weekday',
          pd.Timestamp('2021-06-03 13:00-7'),
          pd.Timestamp('2021-06-03 16:00-7'),
          10000.0,
          6.0,
      ),
      (
          'partialpeak_weekday',
          pd.Timestamp('2021-06-03 07:00-7'),
          pd.Timestamp('2021-06-03 08:00-7'),
          20000.0,
          3.6,
      ),
      (
          'offpeak_weekday',
          pd.Timestamp('2021-06-03 23:00-7'),
          pd.Timestamp('2021-06-04 00:00-7'),
          40000.0,
          6.4,
      ),
      (
          'offpeak_weekend',
          pd.Timestamp('2021-06-05 07:00-7'),
          pd.Timestamp('2021-06-05 08:00-7'),
          20000.0,
          3.2,
      ),
      (
          'offpeak_holiday',
          pd.Timestamp('2022-05-31 11:00-7'),
          pd.Timestamp('2022-05-31 13:00-7'),
          20000.0,
          7.2,
      ),
      (
          'offpeak_holiday_negative',
          pd.Timestamp('2022-05-31 11:00-7'),
          pd.Timestamp('2022-05-31 13:00-7'),
          -20000.0,
          7.2,
      ),
  ])
  def test_cost(self, start_time, end_time, energy_rate, expected_cost):
    cost = electricity_energy_cost.ElectricityEnergyCost()
    self.assertAlmostEqual(
        expected_cost,
        cost.cost(
            start_time=start_time, end_time=end_time, energy_rate=energy_rate
        ),
        places=4,
    )

  @parameterized.parameters([
      (
          pd.Timestamp('2021-05-06 10:00:00+0'),
          pd.Timestamp('2021-05-06 11:00:00+0'),
          0.0,
          0.0,
      ),
      (
          pd.Timestamp('2021-06-03 13:00-7'),
          pd.Timestamp('2021-06-03 14:00-7'),
          10000.0,
          1.0344,
      ),
      (
          pd.Timestamp('2021-06-03 00:00-7'),
          pd.Timestamp('2021-06-03 00:30-7'),
          10000.0,
          0.4410,
      ),
      (
          pd.Timestamp('2021-06-03 00:00-7'),
          pd.Timestamp('2021-06-03 00:30-7'),
          -10000.0,
          0.4410,
      ),
  ])
  def test_carbon_emisison(
      self, start_time, end_time, energy_rate, expected_carbon
  ):
    cost = electricity_energy_cost.ElectricityEnergyCost()
    self.assertAlmostEqual(
        expected_carbon,
        cost.carbon(
            start_time=start_time, end_time=end_time, energy_rate=energy_rate
        ),
        places=4,
    )

  def test_invalid_weekday_energy_prices(self):
    with self.assertRaises(ValueError):
      _ = electricity_energy_cost.ElectricityEnergyCost(
          weekday_energy_prices=(
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              18.0,
              18.0,
              18.0,
              18.0,
              18.0,
              18.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              20.0,
              16.0,
              16.0,
              16.0,
              16.0,
          )
      )

  def test_invalid_weekend_energy_prices(self):
    with self.assertRaises(ValueError):
      _ = electricity_energy_cost.ElectricityEnergyCost(
          weekend_energy_prices=(
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              16.0,
              18.0,
          )
      )

  def test_invalid_carbon_emissions(self):
    with self.assertRaises(ValueError):
      _ = electricity_energy_cost.ElectricityEnergyCost(
          carbon_emission_rates=(
              88.19666493,
              87.79190866,
              87.87607686,
              87.83054163,
              88.00279618,
              88.19648183,
          )
      )


if __name__ == '__main__':
  absltest.main()
