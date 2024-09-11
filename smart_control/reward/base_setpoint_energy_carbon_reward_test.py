"""Tests for setpoint_energy_carbon_reward.

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
from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward import base_setpoint_energy_carbon_reward
from smart_control.utils import conversion_utils


class BaseSetpointEnergyCarbonRewardTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('occupied_in_setpoint', 293.0, 296.0, 294.0, 3600.0, 10.0, 5000.0),
      ('not_occupied_in_setpoint', 293.0, 296.0, 292.0, 3600.0, 0.0, 0.0),
      ('occupied_above_setpoint', 293.0, 296.0, 297.1, 3600.0, 10.0, 4240.6441),
      ('occupied_below_setpoint', 293.0, 296.0, 291.2, 3600.0, 10.0, 1079.2640),
  ])
  def test_get_zone_productivity_reward(
      self,
      heating_setpoint,
      cooling_setpoint,
      zone_temp,
      time_interval_sec,
      average_occupancy,
      expected_productivity,
  ):
    reward_fn = self._get_test_reward_function()
    productivity = reward_fn._get_zone_productivity_reward(
        heating_setpoint,
        cooling_setpoint,
        zone_temp,
        time_interval_sec,
        average_occupancy,
    )
    self.assertAlmostEqual(expected_productivity, productivity, delta=0.001)

  def test_sum_zone_productivities(self):
    info = self._get_test_reward_info()

    reward_fn = self._get_test_reward_function()
    productivity_reward, occupancy = reward_fn._sum_zone_productivities(info)
    self.assertEqual(10.0, occupancy)
    self.assertAlmostEqual(5000.0 / 12.0, productivity_reward, delta=0.001)

  def test_sum_electricity_energy_rate(self):
    info = self._get_test_reward_info()
    reward_fn = self._get_test_reward_function()
    energy_rate = reward_fn._sum_electricity_energy_rate(info)
    # Expected = 1 units x (pump + a/c + blower)
    self.assertAlmostEqual((250.0 + 4500.0 + 800.0), energy_rate, delta=0.001)

  def test_sum_natural_gas_energy_rate(self):
    info = self._get_test_reward_info()
    reward_fn = self._get_test_reward_function()
    energy_rate = reward_fn._sum_natural_gas_energy_rate(info)
    # Expected = 1 units x nat_gas_heater
    self.assertAlmostEqual(5000.0, energy_rate, delta=0.001)

  def test_get_time_delta_sec(self):
    info = self._get_test_reward_info()
    reward_fn = self._get_test_reward_function()
    delta_sec = reward_fn._get_delta_time_sec(info)
    self.assertEqual(300.0, delta_sec)

  def _get_test_reward_function(self):
    max_productivity_personhour_usd = 500.0

    productivity_decay_stiffness = 4.3
    productivity_midpoint_delta = 1.5

    return base_setpoint_energy_carbon_reward.BaseSetpointEnergyCarbonRewardFunction(
        max_productivity_personhour_usd=max_productivity_personhour_usd,
        productivity_midpoint_delta=productivity_midpoint_delta,
        productivity_decay_stiffness=productivity_decay_stiffness,
    )

  def _get_test_reward_info(
      self,
      zone_air_temperature=294.0,
      average_occupancy=5.0,
      blower_electrical_energy_rate=800.0,
      air_conditioning_electrical_energy_rate=4500.0,
      natural_gas_heating_energy_rate=5000.0,
      pump_electrical_energy_rate=250.0,
  ):
    heating_setpoint_temperature = 293.0
    cooling_setpoint_temperature = 297.0
    zone_air_flow_rate_setpoint = 0.013
    zone_air_flow_rate = 0.012
    info = smart_control_reward_pb2.RewardInfo()
    info.agent_id = 'test_agent'
    info.scenario_id = 'test_scenario'
    info.start_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(
            pd.Timestamp('2021-05-03 12:13:00-5')
        )
    )
    info.end_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(
            pd.Timestamp('2021-05-03 12:18:00-5')
        )
    )
    zone_info = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo()
    zone_info.heating_setpoint_temperature = heating_setpoint_temperature
    zone_info.cooling_setpoint_temperature = cooling_setpoint_temperature
    zone_info.zone_air_temperature = zone_air_temperature
    zone_info.average_occupancy = average_occupancy
    zone_info.air_flow_rate_setpoint = zone_air_flow_rate_setpoint
    zone_info.air_flow_rate = zone_air_flow_rate
    info.zone_reward_infos['0,0'].CopyFrom(zone_info)
    info.zone_reward_infos['1,1'].CopyFrom(zone_info)

    air_handler_info = (
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo()
    )
    air_handler_info.blower_electrical_energy_rate = (
        blower_electrical_energy_rate
    )
    air_handler_info.air_conditioning_electrical_energy_rate = (
        air_conditioning_electrical_energy_rate
    )
    info.air_handler_reward_infos['air_handler_0'].CopyFrom(air_handler_info)

    boiler_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo()
    boiler_info.natural_gas_heating_energy_rate = (
        natural_gas_heating_energy_rate
    )
    boiler_info.pump_electrical_energy_rate = pump_electrical_energy_rate
    info.boiler_reward_infos['boiler_0'].CopyFrom(boiler_info)
    return info


class TestEnergyCost(BaseEnergyCost):

  def __init__(self, usd_per_kwh: float, kg_per_kwh: float):
    # Energy price in USD/Watt second (fixed schedule)
    # To convert denominator units hours to seconds, divide by 3600.0, and to
    # convert kW to W, divide by 1000. This leaves us with an enegy price
    # in USD /W /s and carbon rate of kg /W /s.
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


if __name__ == '__main__':
  absltest.main()
