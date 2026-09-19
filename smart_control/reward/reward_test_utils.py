"""Shared test utilities for reward tests."""

import pandas as pd

from smart_control.models.base_energy_cost import BaseEnergyCost


class TestEnergyCost(BaseEnergyCost):
  """Calculates energy cost and carbon emissions using fixed rates."""

  def __init__(self, usd_per_kwh: float, kg_per_kwh: float):
    # Convert USD/kWh and kg/kWh to USD/W/s and kg/W/s, respectively.
    self._energy_price = usd_per_kwh / 3600.0 / 1000.0
    self._carbon_rate = kg_per_kwh / 3600.0 / 1000.0

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    dt = (end_time - start_time).total_seconds()
    return self._energy_price * energy_rate * dt

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    dt = (end_time - start_time).total_seconds()
    return self._carbon_rate * energy_rate * dt
