"""Base Reward Function for Smart Buildings.

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

from typing import Tuple

import gin
import numpy as np
from smart_control.models.base_reward_function import BaseRewardFunction
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import conversion_utils


@gin.configurable()
class BaseSetpointEnergyCarbonRewardFunction(BaseRewardFunction):
  """Reward function based on productivity, energy cost and carbon emission.

  Attributes:
    max_productivity_personhour_usd: max productivity for average occupancy in $
    productivity_midpoint_delta: temp difference from setpoint of half prod.
    productivity_decay_stiffness: midpoint slope of the decay curve
  """

  @gin.configurable()
  def __init__(
      self,
      max_productivity_personhour_usd: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
  ):
    self._max_productivity_personhour_usd = max_productivity_personhour_usd
    self._productivity_midpoint_delta = productivity_midpoint_delta
    self._productivity_decay_stiffness = productivity_decay_stiffness

  def compute_reward(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Returns the real-valued reward for the current state of the building."""
    raise NotImplementedError()

  def _sum_zone_productivities(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> Tuple[float, float]:
    time_interval_sec = self._get_delta_time_sec(energy_reward_info)
    cumulative_productivity = 0.0
    total_occupancy = 0.0

    for zid in energy_reward_info.zone_reward_infos:
      occupancy = energy_reward_info.zone_reward_infos[zid].average_occupancy
      total_occupancy += occupancy
      cumulative_productivity += self._get_zone_productivity_reward(
          energy_reward_info.zone_reward_infos[
              zid
          ].heating_setpoint_temperature,
          energy_reward_info.zone_reward_infos[
              zid
          ].cooling_setpoint_temperature,
          energy_reward_info.zone_reward_infos[zid].zone_air_temperature,
          time_interval_sec,
          occupancy,
      )

    return cumulative_productivity, total_occupancy

  def _get_zone_productivity_reward(
      self,
      heating_setpoint: float,
      cooling_setpoint,
      zone_temp: float,
      time_interval_sec: float,
      average_occupancy,
  ) -> float:
    """Computes the productivity for person hour from the zone temp."""

    x0low = heating_setpoint - self._productivity_midpoint_delta  # pytype: disable=attribute-error  # trace-all-classes
    x0high = cooling_setpoint + self._productivity_midpoint_delta  # pytype: disable=attribute-error  # trace-all-classes
    if zone_temp < heating_setpoint:
      productivity = (
          self._max_productivity_personhour_usd
          / (  # pytype: disable=attribute-error  # trace-all-classes
              1.0
              + np.exp(
                  -self._productivity_decay_stiffness
                  * (  # pytype: disable=attribute-error  # trace-all-classes
                      zone_temp - x0low
                  )
              )
          )
      )
    elif zone_temp > cooling_setpoint:
      productivity = (
          self._max_productivity_personhour_usd
          * (  # pytype: disable=attribute-error  # trace-all-classes
              1.0
              - 1.0
              / (
                  1.0
                  + np.exp(
                      -self._productivity_decay_stiffness
                      * (  # pytype: disable=attribute-error  # trace-all-classes
                          zone_temp - x0high
                      )
                  )
              )
          )
      )
    else:
      productivity = self._max_productivity_personhour_usd  # pytype: disable=attribute-error  # trace-all-classes

    return productivity * average_occupancy * time_interval_sec / 3600.0

  def _get_delta_time_sec(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Gets the time interval in seconds."""
    start_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.start_timestamp
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.end_timestamp
    )
    return (end_time - start_time).total_seconds()

  def _sum_electricity_energy_rate(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Returns the sum of electrical energy rate over the interval in W."""

    # Sum up the power in Watts for the total power. Take the abs of the
    # AC to ensure both heating (positive), and cooling (negative) are assessed
    # as energy consumed.
    electrical_energy_rate = 0.0
    for ahid in energy_reward_info.air_handler_reward_infos:
      electrical_energy_rate += energy_reward_info.air_handler_reward_infos[
          ahid
      ].blower_electrical_energy_rate + np.abs(
          energy_reward_info.air_handler_reward_infos[
              ahid
          ].air_conditioning_electrical_energy_rate
      )

    for bid in energy_reward_info.boiler_reward_infos:
      electrical_energy_rate += energy_reward_info.boiler_reward_infos[
          bid
      ].pump_electrical_energy_rate
    return electrical_energy_rate

  def _sum_natural_gas_energy_rate(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Returns the sum of nat gas energy rate over the interval in W."""

    # Sum up the power in Watts for the total power.
    gas_energy_rate = 0.0
    for bid in energy_reward_info.boiler_reward_infos:
      gas_energy_rate += energy_reward_info.boiler_reward_infos[
          bid
      ].natural_gas_heating_energy_rate
    return gas_energy_rate
