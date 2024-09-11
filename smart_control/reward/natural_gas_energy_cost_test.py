"""Tests for natural_gas_energy_cost.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_control.reward import natural_gas_energy_cost
from smart_control.utils import constants


class NaturalGasEnergyCostTest(parameterized.TestCase):

  def test_zero_energy_use(self):
    start_time = pd.Timestamp('2021-05-06 10:00:00+0')
    end_time = pd.Timestamp('2021-05-06 11:00:00+0')

    cost = natural_gas_energy_cost.NaturalGasEnergyCost()
    self.assertEqual(
        0.0,
        cost.cost(start_time=start_time, end_time=end_time, energy_rate=0.0),
    )
    self.assertEqual(
        0.0,
        cost.carbon(start_time=start_time, end_time=end_time, energy_rate=0.0),
    )

  @parameterized.parameters(
      [(1, 9.02), (3, 7.77), (6, 6.86), (9, 6.99), (12, 8.98)]
  )
  def test_energy_cost(self, month, expected_cost):
    # Source: https://www.traditionaloven.com/tutorials/energy/
    # convert-cubic-foot-natural-gas-to-kilo-watt-hr-kwh.html
    # 1000 cubic feet = 293.071 kWh = 293071 Wh
    energy_rate = 293071.0  # W
    # Choose one hour to make it convertible.
    dt = pd.Timedelta(1.0, unit='hour')
    start_time = pd.Timestamp(year=2020, month=month, day=5, hour=8)
    end_time = start_time + dt
    cost = natural_gas_energy_cost.NaturalGasEnergyCost()
    cost_estimate = cost.cost(start_time, end_time, energy_rate)
    self.assertAlmostEqual(expected_cost, cost_estimate, 2)

  def test_carbon_emisison(self):
    # Source:
    # https://www.eia.gov/environment/emissions/co2_vol_mass.php
    # 1 million BTUs nat gas generate 53.1 kg C02.
    energy_rate = 1.0e6 * constants.JOULES_PER_BTU / 3600.0
    dt = pd.Timedelta(1.0, unit='hour')
    start_time = pd.Timestamp(year=2020, month=1, day=5, hour=8)
    end_time = start_time + dt

    cost = natural_gas_energy_cost.NaturalGasEnergyCost()
    carbon_estimate = cost.carbon(start_time, end_time, energy_rate)
    self.assertAlmostEqual(53.1, carbon_estimate, 1)

  def test_invalid_carbon_emission(self):
    dt = pd.Timedelta(1.0, unit='hour')
    start_time = pd.Timestamp(year=2020, month=1, day=5, hour=8)
    end_time = start_time + dt
    cost = natural_gas_energy_cost.NaturalGasEnergyCost()
    energy_rate = -1.0
    self.assertEqual(0.0, cost.carbon(start_time, end_time, energy_rate))

  def test_invalid_carbon_cost(self):
    dt = pd.Timedelta(1.0, unit='hour')
    start_time = pd.Timestamp(year=2020, month=1, day=5, hour=8)
    end_time = start_time + dt
    cost = natural_gas_energy_cost.NaturalGasEnergyCost()
    energy_rate = -1.0
    self.assertEqual(0.0, cost.cost(start_time, end_time, energy_rate))


if __name__ == '__main__':
  absltest.main()
