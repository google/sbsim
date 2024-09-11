"""Shared test utiltiles for environment tests.

Copyright 2022 Google LLC

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

import collections
from typing import Sequence

import pandas as pd
from smart_control.models import base_building
from smart_control.models import base_reward_function
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import conversion_utils


class SimpleBuilding(base_building.BaseBuilding):
  """Building implementation for unit tests."""

  def __init__(self):
    self.layout = {
        "zone_1": {
            "boiler_1": ["setpoint_1", "measurement_1"],
            "vav_2": [
                "setpoint_2",
                "setpoint_3",
                "setpoint_4",
                "measurement_2",
            ],
        },
        "zone_2": {
            "boiler_3": ["measurement_3", "measurement_4"],
            "vav_4": ["setpoint_5", "measurement_5"],
            "air_handler_5": ["setpoint_6"],
        },
    }
    self.values = collections.defaultdict(int)
    self.reset_called = False
    self.step_count = 0

  @property
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    # For the purposes of this test, we can return a dummy value
    return smart_control_reward_pb2.RewardInfo()

  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_timestamp: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries the building for observations between start and end times."""
    raise NotImplementedError()

  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state."""
    response = smart_control_building_pb2.ObservationResponse(
        request=observation_request
    )
    for (
        single_observation_request
    ) in observation_request.single_observation_requests:
      response.single_observation_responses.append(
          smart_control_building_pb2.SingleObservationResponse(
              single_observation_request=single_observation_request,
              observation_valid=True,
              continuous_value=self.values[
                  single_observation_request.measurement_name
              ],
          )
      )
    return response

  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    """Issues a command to the building to change one or more setpoints."""
    timestamp = conversion_utils.pandas_to_proto_timestamp(
        pd.Timestamp.utcnow()
    )
    response = smart_control_building_pb2.ActionResponse(
        request=action_request, timestamp=timestamp
    )
    for single_action_request in action_request.single_action_requests:
      self.values[single_action_request.setpoint_name] = (
          single_action_request.continuous_value
      )
      response.single_action_responses.append(
          smart_control_building_pb2.SingleActionResponse(
              request=single_action_request,
              response_type=smart_control_building_pb2.SingleActionResponse.ActionResponseType.ACCEPTED,
              additional_info="test",
          )
      )
    return response

  def reset(self) -> None:
    self.reset_called = True

  def wait_time(self) -> None:
    """Returns after a certain amount of time."""
    self.step_count += 1

  @property
  def current_timestamp(self) -> pd.Timestamp:
    return pd.Timestamp("2021-06-07 12:00:01") + pd.Timedelta(
        5.0 * self.step_count, unit="minute"
    )

  def render(self, path: str) -> None:
    """Renders the current state of the building."""
    raise NotImplementedError()

  @property
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Lists the devices that can be queried and/or controlled."""
    devices = []
    for zone, info in self.layout.items():
      for device, fields in info.items():
        zone_id = zone
        device_id = device
        device_type = None
        if "boiler" in device:
          device_type = smart_control_building_pb2.DeviceInfo.DeviceType.BLR
        elif "vav" in device:
          device_type = smart_control_building_pb2.DeviceInfo.DeviceType.VAV
        elif "air_handler" in device:
          device_type = smart_control_building_pb2.DeviceInfo.DeviceType.AHU
        observable_fields = {}
        action_fields = {}
        for field in fields:
          if "setpoint" in field:
            action_fields[field] = (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            )
          if "measurement" in field:
            observable_fields[field] = (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            )
        device_info = smart_control_building_pb2.DeviceInfo(
            zone_id=zone_id,
            device_id=device_id,
            device_type=device_type,
            observable_fields=observable_fields,
            action_fields=action_fields,
        )
        devices.append(device_info)
    return devices

  @property
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Lists the zones in the building managed by the RL agent."""
    zones = []
    for zone, info in self.layout.items():
      zone_id = zone
      devices = info.keys()
      zone_info = smart_control_building_pb2.ZoneInfo(
          zone_id=zone_id,
          building_id="SimpleBuilding",
          zone_description=zone_id,
      )
      for device in devices:
        zone_info.devices.append(device)
      zones.append(zone_info)
    return zones

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""
    return True

  @property
  def num_occupants(self) -> int:
    return 10

  @property
  def time_step_sec(self) -> float:
    return 300.0


class SimpleRewardFunction(base_reward_function.BaseRewardFunction):
  """Test reward function."""

  def __init__(self):
    self.counter = 0
    self.values = [0, 1, 6, 43, 0.8, -1, 54, 12, -50]

  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Returns the real-valued reward for the current state of the building."""
    value = self.values[self.counter]
    self.counter = (self.counter + 1) % len(self.values)
    reward_response = smart_control_reward_pb2.RewardResponse(
        agent_reward_value=value
    )
    return reward_response
