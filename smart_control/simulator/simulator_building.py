"""Extension of BaseBuilding for a simulated building.

This file is used to build an RL environment with a simulator controlling the
thermodynamics and observation/action space.

Copyright 2023 Google LLC

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

from typing import Sequence, Type, Union
import uuid

from absl import logging
import gin
import pandas as pd
from smart_control.models.base_building import BaseBuilding
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import simulator as simulator_py
from smart_control.simulator import simulator_flexible_floor_plan
from smart_control.simulator import smart_device
from smart_control.simulator import tf_simulator
from smart_control.utils import conversion_utils

_ValueType = smart_control_building_pb2.DeviceInfo.ValueType
_ActionResponseType = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)


@gin.configurable
class SimulatorBuilding(BaseBuilding):
  """Base class for a controllable building for reinforcement learning."""

  def __init__(
      self,
      simulator: Union[
          simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
          simulator_py.Simulator,
          tf_simulator.TFSimulator,
      ],
      occupancy: BaseOccupancy,
  ):
    """Creates SimulatorBuilding.

    Args:
      simulator: Simulator to run for the RL environment. This can take in
        either the floor_plan based simulator or the rectangular sim
        (deprecated).
      occupancy: a function to determine building occupancy by zone.
    """

    self._simulator = simulator

    self._occupancy = occupancy
    hvac = self._simulator.hvac

    # List of tuple (device, device_info)
    all_devices = [
        (hvac.boiler, self._create_device_info(hvac.boiler)),
        (hvac.air_handler, self._create_device_info(hvac.air_handler)),
    ]
    all_devices.extend(
        [
            (vav, self._create_device_info(vav, vav.zone_id()))
            for vav in hvac.vavs.values()
        ]
    )

    # List of device infos to return in devices().
    self._device_infos = [device_info for _, device_info in all_devices]

    # Mapping from device_id to smart_device for making observations/actions.
    self._device_map = {
        device_info.device_id: smart_device
        for smart_device, device_info in all_devices
    }

  def _class_to_value_type(self, clazz: Type[object]) -> _ValueType:
    """Returns a ValueType that corresponds to a given class/type.

    Args:
      clazz: Class/Type to convert.
    """
    if clazz == float:
      return _ValueType.VALUE_CONTINUOUS
    elif clazz == int:
      # TODO(gusatb): Handle non continuous values.
      return _ValueType.VALUE_CONTINUOUS
    else:
      return _ValueType.VALUE_TYPE_UNDEFINED

  def _create_device_info(
      self, device: smart_device.SmartDevice, zone_id: str = "default_zone_id"
  ) -> smart_control_building_pb2.DeviceInfo:
    """Returns DeviceInfo based on device.

    Args:
      device: SmartDevice to create info object for.
      zone_id: Zone Id of the device.
    """
    observable_fields = device.observable_field_names()
    action_fields = device.action_field_names()

    device_info = smart_control_building_pb2.DeviceInfo()
    device_info.device_id = device.device_id()
    device_info.namespace = f"device_namespace_{uuid.uuid4()}"
    device_info.code = f"device_code_{uuid.uuid4()}"
    device_info.zone_id = zone_id
    device_info.device_type = device.device_type()
    for observable_field in observable_fields:
      observable_class = device.get_observable_type(observable_field)
      device_info.observable_fields[observable_field] = (
          self._class_to_value_type(observable_class)
      )

    for action_field in action_fields:
      action_class = device.get_action_type(action_field)
      device_info.action_fields[action_field] = self._class_to_value_type(
          action_class
      )

    return device_info

  @property
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    return self._simulator.reward_info(self._occupancy)

  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_time: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries the building for observations between start and end times."""
    raise NotImplementedError()

  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state."""
    observation_response = smart_control_building_pb2.ObservationResponse()
    observation_response.request.CopyFrom(observation_request)
    observation_response.timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(
            self._simulator.current_timestamp
        )
    )
    for single_request in observation_request.single_observation_requests:
      single_response = smart_control_building_pb2.SingleObservationResponse()

      single_response.single_observation_request.CopyFrom(single_request)
      single_response.timestamp.CopyFrom(
          conversion_utils.pandas_to_proto_timestamp(
              self._simulator.current_timestamp
          )
      )
      single_response.observation_valid = True

      if single_request.device_id not in self._device_map:
        single_response.observation_valid = False
        observation_response.single_observation_responses.append(
            single_response
        )
        logging.warning(
            "Device was not found. Requested device id: %s",
            single_request.device_id,
        )
        continue

      device = self._device_map[single_request.device_id]
      try:
        observed_value = device.get_observation(
            single_request.measurement_name, self._simulator.current_timestamp
        )
        # TODO(gusatb): Extend this to handle non-continuous types.
        single_response.continuous_value = observed_value
      except AttributeError as e:
        single_response.observation_valid = False
        logging.warning(
            "Could not get requested observation. Device id: %s, measurement"
            " name: %s. Error: %s",
            single_request.device_id,
            single_request.measurement_name,
            e,
        )

      observation_response.single_observation_responses.append(single_response)
    return observation_response

  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    """Issues a command to the building to change one or more setpoints."""
    # Set up default building behavior
    self._simulator.setup_step_sim()

    action_response = smart_control_building_pb2.ActionResponse()
    action_response.request.CopyFrom(action_request)
    action_response.timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(
            self._simulator.current_timestamp
        )
    )
    for single_request in action_request.single_action_requests:
      single_response = smart_control_building_pb2.SingleActionResponse()

      single_response.request.CopyFrom(single_request)
      single_response.response_type = _ActionResponseType.ACCEPTED

      if single_request.device_id not in self._device_map:
        single_response.response_type = (
            _ActionResponseType.REJECTED_INVALID_DEVICE
        )
        action_response.single_action_responses.append(single_response)
        logging.warning(
            "Device was not found. Requested device id: %s",
            single_request.device_id,
        )
        continue

      device = self._device_map[single_request.device_id]

      set_field = single_request.WhichOneof("setpoint_value")
      if set_field:
        set_value = getattr(single_request, set_field)
      else:
        # Note: None value will allow most accurate error to be caught below.
        set_value = None

      try:
        device.set_action(
            single_request.setpoint_name,
            set_value,
            self._simulator.current_timestamp,
        )
      except (AttributeError, ValueError) as e:
        single_response.response_type = (
            _ActionResponseType.REJECTED_NOT_ENABLED_OR_AVAILABLE
        )
        logging.warning(
            "Could not perform action. Device id: %s, setpoint name: %s."
            " Error: %s",
            single_request.device_id,
            single_request.setpoint_name,
            e,
        )
      action_response.single_action_responses.append(single_response)

    return action_response

  def wait_time(self) -> None:
    """Returns after a certain amount of time."""
    # Update the building state.
    self._simulator.execute_step_sim()

  def reset(self) -> None:
    """Resets the building, throwing a RuntimeError if this is impossible."""
    self._simulator.reset()

  @property
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Lists the devices that can be queried and/or controlled."""
    return self._device_infos

  @property
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Lists the zones in the building managed by the RL agent."""

    return list(self._simulator.hvac.zone_infos.values())

  @property
  def time_step_sec(self) -> float:
    """Returns the amount of time between time steps."""
    return self._simulator.time_step_sec

  @property
  def current_timestamp(self) -> pd.Timestamp:
    """Lists the current local time of the building."""
    return self._simulator.current_timestamp

  def render(self, path: str) -> None:
    """Renders the current state of the building."""
    raise NotImplementedError(
        "Rendering not currently supported on simulated building"
    )

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Returns True if building is in comfort mode."""
    return self._simulator.hvac.is_comfort_mode(current_time)

  @property
  def num_occupants(self) -> int:
    num_occupants = 0.0
    for zone in self.zones:
      zone_id = zone.zone_id
      current_timestamp = self.current_timestamp
      start_timestamp = current_timestamp - pd.Timedelta(5, unit="minute")
      num_occupants += self._occupancy.average_zone_occupancy(
          zone_id, start_timestamp, current_timestamp
      )
    return int(num_occupants)
