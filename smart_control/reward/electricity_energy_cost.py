"""Energy carbon and cost model for electricity.

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

from typing import Sequence

from absl import logging
import gin
import numpy as np
import pandas as pd
import pint
from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.utils import conversion_utils

UNIT = pint.UnitRegistry()
UNIT.define("cents_per_kWh = cents / kWh")
UNIT.define("usd_per_Ws = USD / W / s")
UNIT.define("kg_per_MWh = kg / MWh")
UNIT.define("Watt = J / s")

# Source:
# Google Carbon Free Reporting Dashboard
# US-SVL-BORD1212
# Units kg Carbon / MWh
CARBON_EMISSION_BY_HOUR = (
    88.19666493,
    87.79190866,
    87.87607686,
    87.83054163,
    88.00279618,
    88.19648183,
    89.70663283,
    93.97947901,
    98.85868291,
    100.7853521,
    101.3866866,
    101.7795612,
    102.5919168,
    103.4403736,
    104.1380294,
    104.7359292,
    102.0714466,
    97.04226176,
    93.57895651,
    92.46355045,
    91.72914657,
    90.69209747,
    89.76552213,
    88.99950995,
) * UNIT.kg_per_MWh

# Time-of use schedule source (PG&E) for commercial/industrial:
# https://www.pge.com/includes/docs/pdfs/mybusiness/energysavingsrebates/economicdevelopment/factsheet/ed-comind_e_rates_v4.pdf
# Actual values estimated from Joint Rate Comparisons PG&E - MCE,
# Large Commercial and Industrial
# https://www.pge.com/pge_global/common/pdfs/customer-service/other-services/alternative-energy-providers/community-choice-aggregation/mce_rateclasscomparison.pdf
# Units cents / kWh
WEEKDAY_PRICE_BY_HOUR = (
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
    16.0,
) * UNIT.cents_per_kWh
WEEKEND_PRICE_BY_HOUR = (
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
) * UNIT.cents_per_kWh


@gin.configurable()
class ElectricityEnergyCost(BaseEnergyCost):
  """Energy cost and carbon emission model for reward function."""

  def __init__(
      self,
      weekday_energy_prices: Sequence[float] = WEEKDAY_PRICE_BY_HOUR,
      weekend_energy_prices: Sequence[float] = WEEKEND_PRICE_BY_HOUR,
      carbon_emission_rates: Sequence[float] = CARBON_EMISSION_BY_HOUR,
  ):
    if len(weekday_energy_prices) != 24:
      raise ValueError("Energy cost rates must have 24 entries.")

    if len(weekend_energy_prices) != 24:
      raise ValueError("Energy cost rates must have 24 entries.")

    if len(carbon_emission_rates) != 24:
      raise ValueError("Carbon emission rates must have 24 entries.")

    # Convert the emission rates from kg / MWh to kg / Ws.
    self._carbon_emission_rates = (
        np.array(carbon_emission_rates) / 1.0e6 / 3600.0
    )

    # Convert the energy rates to USD / W / s
    self._weekday_energy_prices = (
        np.array(weekday_energy_prices)
        / 100.0
        / 1000.0
        / 3600.0
        * UNIT.usd_per_Ws
    )
    self._weekend_energy_prices = (
        np.array(weekend_energy_prices)
        / 100.0
        / 1000.0
        / 3600.0
        * UNIT.usd_per_Ws
    )

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Returns the cost of energy from this time step.

    Args:
      start_time: start of window
      end_time: end of window
      energy_rate: power applies in W, if negative then energy is drawn away
        (i.e., cooling), positive energy_rate means heating.

    Returns:
      cost in USD for the energy consumed over the interval.
    """
    dt = (end_time - start_time).total_seconds()
    if dt > 3600.0:
      logging.warn(
          "Queries greater than an hour may yield incorrect price estimates."
      )

    hour_index = start_time.hour
    if conversion_utils.is_work_day(start_time):
      current_price = self._weekday_energy_prices[hour_index]
    else:
      current_price = self._weekend_energy_prices[hour_index]
    return (
        current_price * np.abs(energy_rate) * UNIT.Watt * dt * UNIT.second
    ).magnitude

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Returns the carbon produced in this time step.

    Args:
      start_time: start of window
      end_time: end of window
      energy_rate: power applies in W, if negative then energy is drawn away
        (i.e., cooling), positive energy_rate means heating.

    Returns:
      carbon emitted [kg] for the energy consumed over the interval.
    """
    dt = (end_time - start_time).total_seconds()

    if dt > 3600.0:
      logging.warn(
          "Queries greater than an hour may yield incorrect carbon estimates."
      )

    hour_index = start_time.hour
    # Return carbon mass [kg] generated by the energy consumed [J].
    return (
        self._carbon_emission_rates[hour_index]
        * np.abs(energy_rate)
        * UNIT.Watt
        * dt
        * UNIT.second
    ).magnitude
