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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward import setpoint_energy_carbon_reward
from smart_control.utils import conversion_utils


class SetpointEnergyCarbonRewardTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'occupied_in_setpoint',
          293.0,
          10.0,
          500.0,
          2500.0,
          20000.0,
          250.0,
          0.11660,
          5000.0 / 6,
          0.102916665,
          0.1,
          0.63208336,
          0.12641667,
      ),
      (
          'not_occupied_in_setpoint',
          293.0,
          0.0,
          500.0,
          2500.0,
          20000.0,
          250.0,
          -0.050065,
          0.0,
          0.102916665,
          0.1,
          0.63208336,
          0.12641667,
      ),
      (
          'occupied_below_setpoint',
          292.0,
          10.0,
          500.0,
          2500.0,
          20000.0,
          250.0,
          0.099212,
          746.3906,
          0.102916665,
          0.1,
          0.63208336,
          0.12641667,
      ),
      (
          'occupied_above_setpoint',
          299.0,
          10.0,
          500.0,
          2500.0,
          20000.0,
          250.0,
          -0.0326773,
          86.942687,
          0.102916665,
          0.1,
          0.63208336,
          0.12641667,
      ),
      (
          'occupied_in_setpoint_no_energy',
          293.0,
          10.0,
          0.0,
          00.0,
          0.0,
          0.0,
          0.116666,
          5000.0 / 6,
          0.0,
          0.0,
          0.0,
          0.0,
      ),
  ])
  def test_compute_reward(
      self,
      zone_air_temperature,
      average_occupancy,
      blower_electrical_energy_rate,
      air_conditioning_electrical_energy_rate,
      natural_gas_heating_energy_rate,
      pump_electrical_energy_rate,
      expected_reward,
      expected_productivity,
      expected_electrical_energy_cost,
      expected_natural_gas_cost,
      expected_carbon_emitted,
      expected_carbon_cost,
  ):
    info = self._get_test_reward_info(
        zone_air_temperature,
        average_occupancy,
        blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate,
        natural_gas_heating_energy_rate,
        pump_electrical_energy_rate,
    )

    reward_fn = self._get_test_reward_function()
    response = reward_fn.compute_reward(info)

    logging.info(response)
    self.assertAlmostEqual(expected_reward, response.agent_reward_value, 4)
    self.assertAlmostEqual(
        expected_productivity, response.productivity_reward, 4
    )

    self.assertAlmostEqual(
        expected_electrical_energy_cost, response.electricity_energy_cost, 4
    )
    self.assertAlmostEqual(
        expected_natural_gas_cost, response.natural_gas_energy_cost, 4
    )
    self.assertAlmostEqual(expected_carbon_emitted, response.carbon_emitted, 4)
    self.assertAlmostEqual(expected_carbon_cost, response.carbon_cost, 4)

  def _get_test_reward_function(self):
    person_productivity_hour = 500.0
    productivity_decay_stiffness = 4.3
    productivity_midpoint_delta = 1.5
    electricity_energy_cost = TestEnergyCost(0.19, 0.01)
    natural_gas_energy_cost = TestEnergyCost(0.03, 0.188)

    energy_cost_weight = 1.0
    carbon_cost_weight = 1.0
    carbon_cost_factor = 0.2
    reward_normalization_shift = 250.0
    reward_normalization_scale = 500.0 * 10
    return setpoint_energy_carbon_reward.SetpointEnergyCarbonRewardFunction(
        person_productivity_hour,
        productivity_midpoint_delta,
        productivity_decay_stiffness,
        electricity_energy_cost,
        natural_gas_energy_cost,
        energy_cost_weight,
        carbon_cost_weight,
        carbon_cost_factor,
        reward_normalization_shift,
        reward_normalization_scale,
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
    info.air_handler_reward_infos['air_handler_1'].CopyFrom(air_handler_info)

    boiler_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo()
    boiler_info.natural_gas_heating_energy_rate = (
        natural_gas_heating_energy_rate
    )
    boiler_info.pump_electrical_energy_rate = pump_electrical_energy_rate
    info.boiler_reward_infos['boiler_0'].CopyFrom(boiler_info)
    info.boiler_reward_infos['boiler_2'].CopyFrom(boiler_info)
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
