"""Reward (Regret) Function for Smart Buildings.

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

The reward function provides a feedback signal to the reinforcement learning
agent that indicates the benefit of the action taken. During training, the
agent learns an action policy to maximize the cumulative, or long-term reward.

For this pilot there are three principal factors that contribute to the
reward function:
  * Setpoint: Maintaining the zone temperatures within heating and cooling
  setpoints  results in a positive reward, and any temperature outside of
  setpoints may also result in a negative reward (i.e., penalty).
  * Cost: The cost of electricity and natural gas is a negative reward (cost).
  Then by minimizing negative rewards/maximizing positive reward, the agent
  will reduce overall energy cost. To compute the cost, both energy consumption
  and the energy cost schedules are required.
  * Carbon: By receiving negative reward for consuming natural gas, the agent
  will learn to shift energy use to renewable sources. This factor requires an
  energy-to-carbon conversion formula/table.

The three factors can be scaled and combined into a single regret function:
        r_i = [u x (s(setpoint) - s_max)/s_max
            - v x f(cost)/f_max - w x g(carbon)/g_max]
            / [u + v + w]

        r_i -> [-1, 0]
where:
  r_i is the incremental reward at step i
  s(setpoint) is the reward for maintining temperature inside setpoint
  s_max = occupancy x productivity, the maximum possible reward
  f(cost) is the cost of consuming electrical and natural gas energy
  f_max: maximum momentary cost that occurs at max energy use
  g(carbon) is the cost of emitting carbon,
  g_max
  and u, w, w are weighing factors for the policy.

The fundamental metric unit of energy is the Joule (J), and the unit of energy
applied over a fixed time interval (energy rate) is power measured in J/sec or
Watts. However, energy is expressed based on diverse traditional units.
For example, electrical energy unit is one hour of 1,000 W, or kWh.
However, natural gas energy is measured in British thermal units (Btu) or
cubic feet. So coordinate conversions are necessary.

Notes on Setpoint Reward:
Setpoint reward is the incremental reward (beneficial feedback) for maintaining
comfort conditions inside the zone.

We postulate that productivity is adversely affected when the zone air
temperature is outside the deadband. Near the deadband, individual productivity
decreases a little, but decreases smoothly and monotonically the farther the
zone air temperature is away from the deadband.

Cumulative productivity is the maximum potential reward, and is parameterized by
how many persons occupy the zones, and the average hourly per-person
productivity.

Two other parameters are added to describe how productivity decays outside the
deadband.

productivity_midpoint_delta_temp: The difference in temperature
beyond the setpoint, at which productivity decays to 50%.
decay_stiffness: Parameter that controls the slope of the decay, the higher
the value, the steeper the slope.

The function for setpoint reward is based on a piecewise logistic regression.
Maximum/full productivity occurs when the zone is occupied and inside its
deadband. Productivity decays smoothly on a logistic curve outside the deadband.
"""

import gin
from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward.base_setpoint_energy_carbon_reward import BaseSetpointEnergyCarbonRewardFunction
from smart_control.utils import conversion_utils

_HOUR_SEC = 3600.0


@gin.configurable()
class SetpointEnergyCarbonRegretFunction(
    BaseSetpointEnergyCarbonRewardFunction
):
  """Reward function based on productivity, energy cost and carbon emission.

  Attributes:
    max_productivity_personhour_usd: max occupant hourly productivity in $
    min_productivity_personhour_usd: min occupant hourly productivity in $
    productivity_midpoint_delta: temp difference from setpoint of half prod.
    productivity_decay_stiffness: midpoint slope of the decay curve
    electricity_energy_cost: cost and carbon model for electricity
    natural_gas_energy_cost: cost and carbon model for natural gas
    energy_cost_weight: u-coefficient described above
    carbon_emission_weight: w-coefficient described above
    carbon_cost_factor: cost value in $ per kg carbon emitted
  """

  @gin.configurable()
  def __init__(
      self,
      max_productivity_personhour_usd: float,
      min_productivity_personhour_usd: float,
      max_electricity_rate: float,
      max_natural_gas_rate: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
      electricity_energy_cost: BaseEnergyCost,
      natural_gas_energy_cost: BaseEnergyCost,
      productivity_weight: float,
      energy_cost_weight: float,
      carbon_emission_weight: float,
  ):
    self._max_productivity_personhour_usd = max_productivity_personhour_usd
    self._min_productivity_personhour_usd = min_productivity_personhour_usd
    self._max_electricity_rate = max_electricity_rate
    self._max_natural_gas_rate = max_natural_gas_rate
    self._productivity_midpoint_delta = productivity_midpoint_delta
    self._productivity_decay_stiffness = productivity_decay_stiffness
    self._electricity_energy_cost = electricity_energy_cost
    self._natural_gas_energy_cost = natural_gas_energy_cost
    self._productivity_weight = productivity_weight
    self._energy_cost_weight = energy_cost_weight
    self._carbon_emission_weight = carbon_emission_weight

    assert (
        self._max_productivity_personhour_usd
        > self._min_productivity_personhour_usd
    )

  def compute_reward(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Returns the real-valued reward for the current state of the building."""

    start_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.start_timestamp
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.end_timestamp
    )

    delta_time_sec = (end_time - start_time).total_seconds()

    actual_productivity, total_occupancy = self._sum_zone_productivities(
        energy_reward_info
    )

    max_productivity = (
        self._max_productivity_personhour_usd
        * total_occupancy
        * delta_time_sec
        / _HOUR_SEC
    )
    min_productivity = (
        self._min_productivity_personhour_usd
        * total_occupancy
        * delta_time_sec
        / _HOUR_SEC
    )

    actual_productivity = max(actual_productivity, min_productivity)

    if total_occupancy > 0.0:
      normalized_productivity_regret = (
          actual_productivity - min_productivity
      ) / (max_productivity - min_productivity) - 1.0
    else:
      normalized_productivity_regret = 0.0

    capped_electricity_energy_rate = min(
        self._sum_electricity_energy_rate(energy_reward_info),
        self._max_electricity_rate,
    )

    actual_electricity_energy_cost = self._electricity_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_electricity_energy_rate,
    )

    max_electricity_energy_cost = self._electricity_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_electricity_rate,
    )

    actual_electricity_carbon_emission = self._electricity_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_electricity_energy_rate,
    )

    max_electricity_carbon_emission = self._electricity_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_electricity_rate,
    )

    capped_natural_gas_energy_rate = min(
        self._sum_natural_gas_energy_rate(energy_reward_info),
        self._max_natural_gas_rate,
    )

    actual_natural_gas_energy_cost = self._natural_gas_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_natural_gas_energy_rate,
    )

    max_natural_gas_energy_cost = self._natural_gas_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_natural_gas_rate,
    )

    actual_natural_gas_carbon_emission = self._natural_gas_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_natural_gas_energy_rate,
    )

    max_natural_gas_carbon_emission = self._natural_gas_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_natural_gas_rate,
    )

    response = smart_control_reward_pb2.RewardResponse()
    response.productivity_reward = actual_productivity
    response.natural_gas_energy_cost = actual_natural_gas_energy_cost
    response.electricity_energy_cost = actual_electricity_energy_cost

    combined_energy_cost = (
        actual_electricity_energy_cost + actual_natural_gas_energy_cost
    )
    normalized_energy_cost = combined_energy_cost / (
        max_electricity_energy_cost + max_natural_gas_energy_cost
    )

    combined_carbon_emission = (
        actual_electricity_carbon_emission + actual_natural_gas_carbon_emission
    )

    normalized_carbon_emission = combined_carbon_emission / (
        max_electricity_carbon_emission + max_natural_gas_carbon_emission
    )

    response.carbon_emitted = combined_carbon_emission

    response.productivity_weight = self._productivity_weight
    response.energy_cost_weight = self._energy_cost_weight
    response.carbon_emission_weight = self._carbon_emission_weight

    response.person_productivity = self._max_productivity_personhour_usd
    response.total_occupancy = total_occupancy

    response.reward_scale = 1.0
    response.reward_shift = 0.0
    response.productivity_regret = actual_productivity - max_productivity
    response.normalized_productivity_regret = normalized_productivity_regret
    response.normalized_energy_cost = normalized_energy_cost
    response.normalized_carbon_emission = normalized_carbon_emission
    response.start_timestamp.CopyFrom(energy_reward_info.start_timestamp)
    response.end_timestamp.CopyFrom(energy_reward_info.end_timestamp)

    raw_reward_value = (
        normalized_productivity_regret * self._productivity_weight
        - normalized_energy_cost * self._energy_cost_weight
        - normalized_carbon_emission * self._carbon_emission_weight
    )

    # Return a weighted average.
    response.agent_reward_value = raw_reward_value / (
        self._productivity_weight
        + self._energy_cost_weight
        + self._carbon_emission_weight
    )

    return response
