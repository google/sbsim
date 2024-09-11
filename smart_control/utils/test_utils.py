"""Test utilities for replay_building.

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
from typing import Sequence, Tuple

import pandas as pd
from smart_control.environment.environment import ActionConfig
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import setpoint_schedule
from smart_control.utils import conversion_utils
from smart_control.utils.bounded_action_normalizer import BoundedActionNormalizer
from smart_control.utils.reader_lib import BaseReader


def get_test_setpoint_schedule() -> setpoint_schedule.SetpointSchedule:
  morning_start_hour = 9
  evening_start_hour = 18
  comfort_temp_window = (292, 295)
  eco_temp_window = (290, 297)

  return setpoint_schedule.SetpointSchedule(
      morning_start_hour,
      evening_start_hour,
      comfort_temp_window,
      eco_temp_window,
  )


def get_test_action_response(
    timestamp: pd.Timestamp,
    device_setpoint_values: Sequence[Tuple[str, str, float]],
) -> smart_control_building_pb2.ActionResponse:
  """Returns an ActionResponse for unit testing."""

  request_ts = conversion_utils.pandas_to_proto_timestamp(
      pd.Timestamp(timestamp)
  )
  response_ts = conversion_utils.pandas_to_proto_timestamp(
      pd.Timestamp(timestamp)
  )

  single_action_requests = []
  single_action_responses = []

  for device_setpoint_value in device_setpoint_values:
    device_id, setpoint_name, value = device_setpoint_value

    single_request = smart_control_building_pb2.SingleActionRequest(
        device_id=device_id, setpoint_name=setpoint_name, continuous_value=value
    )
    single_action_requests.append(single_request)
    single_response = smart_control_building_pb2.SingleActionResponse(
        request=single_request,
        response_type=smart_control_building_pb2.SingleActionResponse.ACCEPTED,
    )
    single_action_responses.append(single_response)

  request = smart_control_building_pb2.ActionRequest(
      timestamp=request_ts, single_action_requests=single_action_requests
  )
  return smart_control_building_pb2.ActionResponse(
      timestamp=response_ts,
      request=request,
      single_action_responses=single_action_responses,
  )


def get_zone_infos() -> Sequence[smart_control_building_pb2.ZoneInfo]:
  """Provides test zone infos."""
  z0 = smart_control_building_pb2.ZoneInfo(
      zone_id='z0',
      building_id='US-BLDG-0000',
      zone_description='microkitchen',
      area=900.0,
      zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
      floor=2,
  )
  z1 = smart_control_building_pb2.ZoneInfo(
      zone_id='z1',
      building_id='US-BLDG-0000',
      zone_description='work area 01',
      area=500.0,
      zone_type=smart_control_building_pb2.ZoneInfo.ROOM,
      floor=1,
  )
  return [z0, z1]


def get_action_config() -> ActionConfig:
  action_normalizer_inits = {
      'a0': BoundedActionNormalizer(0, 100, -1, 1),
      'a1': BoundedActionNormalizer(-10, 10, -1, 1),
  }

  return ActionConfig(action_normalizer_inits)


def get_replay_action_responses() -> (
    Sequence[smart_control_building_pb2.ActionResponse]
):
  return [
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:00:00'),
          [
              ('d0', 'a0', 30.0),
              ('d0', 'a1', 1.0),
              ('d1', 'a0', 100.0),
              ('d1', 'a1', 2.0),
          ],
      ),
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:05:00'),
          [
              ('d0', 'a0', 40.0),
              ('d0', 'a1', 2.0),
              ('d1', 'a0', 90.0),
              ('d1', 'a1', 1.0),
          ],
      ),
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:10:00'),
          [
              ('d0', 'a0', 50.0),
              ('d0', 'a1', 3.0),
              ('d1', 'a0', 80.0),
              ('d1', 'a1', 0.0),
          ],
      ),
  ]


def get_agent_action_responses() -> (
    Sequence[smart_control_building_pb2.ActionResponse]
):
  return [
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:00:00'),
          [
              ('d0', 'a0', 32.0),
              ('d0', 'a1', 1.7),
              ('d1', 'a0', 90.0),
              ('d1', 'a1', 1.0),
          ],
      ),
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:05:00'),
          [
              ('d0', 'a0', 39.0),
              ('d0', 'a1', 1.0),
              ('d1', 'a0', 88.0),
              ('d1', 'a1', 0.0),
          ],
      ),
      get_test_action_response(
          pd.Timestamp('2022-03-13 00:10:00'),
          [
              ('d0', 'a0', 51.0),
              ('d0', 'a1', 5.0),
              ('d1', 'a0', 83.0),
              ('d1', 'a1', 2.0),
          ],
      ),
  ]


def get_test_observation_request() -> (
    smart_control_building_pb2.ObservationRequest
):
  return get_observation_request(
      [('d0', 'm0'), ('d0', 'm1'), ('d1', 'm0'), ('d1', 'm1')]
  )


def get_observation_request(
    device_measurements: Sequence[Tuple[str, str]]
) -> smart_control_building_pb2.ObservationRequest:
  """Returns a test observation request."""
  single_observation_requests = []
  for device_id, measurement_name in device_measurements:
    single_request = smart_control_building_pb2.SingleObservationRequest(
        device_id=device_id, measurement_name=measurement_name
    )
    single_observation_requests.append(single_request)

  return smart_control_building_pb2.ObservationRequest(
      single_observation_requests=single_observation_requests
  )


def get_test_observation_response(
    timestamp: pd.Timestamp,
    device_measurement_values: Sequence[Tuple[str, str, float]],
) -> smart_control_building_pb2.ObservationResponse:
  """Returns test observation responses."""
  request_ts = conversion_utils.pandas_to_proto_timestamp(
      pd.Timestamp(timestamp)
  )
  response_ts = conversion_utils.pandas_to_proto_timestamp(
      pd.Timestamp(timestamp)
  )

  single_observation_requests = []
  single_observation_responses = []
  for device_measurement_value in device_measurement_values:
    device_id, measurement_name, value = device_measurement_value
    single_request = smart_control_building_pb2.SingleObservationRequest(
        device_id=device_id, measurement_name=measurement_name
    )
    single_observation_requests.append(single_request)
    single_response = smart_control_building_pb2.SingleObservationResponse(
        timestamp=response_ts,
        single_observation_request=single_request,
        observation_valid=True,
        continuous_value=value,
    )
    single_observation_responses.append(single_response)

  request = smart_control_building_pb2.ObservationRequest(
      timestamp=request_ts,
      single_observation_requests=single_observation_requests,
  )
  return smart_control_building_pb2.ObservationResponse(
      timestamp=response_ts,
      request=request,
      single_observation_responses=single_observation_responses,
  )


def get_observation_responses() -> (
    Sequence[smart_control_building_pb2.ObservationResponse]
):
  """Returns pre-defined ObservationResponses for unit test."""
  return [
      get_test_observation_response(
          pd.Timestamp('2022-03-13 00:00:00'),
          [
              ('d0', 'm0', 7.0),
              ('d0', 'm1', 0.1),
              ('d1', 'm0', 10.0),
              ('d1', 'm1', -0.2),
          ],
      ),
      get_test_observation_response(
          pd.Timestamp('2022-03-13 00:05:00'),
          [
              ('d0', 'm0', 7.0),
              ('d0', 'm1', 0.1),
              ('d1', 'm0', 10.0),
              ('d1', 'm1', -0.2),
          ],
      ),
      get_test_observation_response(
          pd.Timestamp('2022-03-13 00:10:00'),
          [
              ('d0', 'm0', 7.0),
              ('d0', 'm1', 0.1),
              ('d1', 'm0', 10.0),
              ('d1', 'm1', -0.2),
          ],
      ),
  ]


def get_test_reward_responses() -> (
    Sequence[smart_control_reward_pb2.RewardResponse]
):
  """Returns pre-defined RewardResponses for unit testing."""
  return [
      get_test_reward_response(
          start_timestamp=pd.Timestamp('2022-03-13 00:00:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:05:00'),
          agent_reward_value=-0.2,
          productivity_reward=200.0,
          electricity_energy_cost=120.0,
          natural_gas_energy_cost=1.0,
          carbon_emitted=0.20,
          carbon_cost=20.0,
      ),
      get_test_reward_response(
          start_timestamp=pd.Timestamp('2022-03-13 00:05:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:10:00'),
          agent_reward_value=-0.0,
          productivity_reward=10.0,
          electricity_energy_cost=5.0,
          natural_gas_energy_cost=10.0,
          carbon_emitted=0.20,
          carbon_cost=10.0,
      ),
      get_test_reward_response(
          start_timestamp=pd.Timestamp('2022-03-13 00:10:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:15:00'),
          agent_reward_value=-0.2,
          productivity_reward=20.0,
          electricity_energy_cost=20.0,
          natural_gas_energy_cost=1.0,
          carbon_emitted=0.20,
          carbon_cost=2.0,
      ),
  ]


def get_test_reward_response(
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    agent_reward_value: float,
    productivity_reward: float,
    electricity_energy_cost: float,
    natural_gas_energy_cost: float,
    carbon_emitted: float,
    carbon_cost: float,
) -> smart_control_reward_pb2.RewardResponse:
  """Returns a RewardResponse for unit testing."""
  return smart_control_reward_pb2.RewardResponse(
      agent_reward_value=agent_reward_value,
      productivity_reward=productivity_reward,
      electricity_energy_cost=electricity_energy_cost,
      natural_gas_energy_cost=natural_gas_energy_cost,
      carbon_emitted=carbon_emitted,
      carbon_cost=carbon_cost,
      start_timestamp=conversion_utils.pandas_to_proto_timestamp(
          start_timestamp
      ),
      end_timestamp=conversion_utils.pandas_to_proto_timestamp(end_timestamp),
  )


def get_device_infos() -> Sequence[smart_control_building_pb2.DeviceInfo]:
  """Returns pre-defined DeviceInfos for unit test."""

  d0 = smart_control_building_pb2.DeviceInfo(
      device_id='d0',
      namespace='test',
      code='code0',
      zone_id='z0',
      device_type=smart_control_building_pb2.DeviceInfo.VAV,
      observable_fields={
          'm0': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
          ),
          'm1': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
          ),
      },
      action_fields={
          'a0': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CATEGORICAL
          ),
          'a1': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CATEGORICAL
          ),
      },
  )

  d1 = smart_control_building_pb2.DeviceInfo(
      device_id='d1',
      namespace='test',
      code='code0',
      zone_id='z1',
      device_type=smart_control_building_pb2.DeviceInfo.VAV,
      observable_fields={
          'm0': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
          ),
          'm1': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
          ),
      },
      action_fields={
          'a0': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CATEGORICAL
          ),
          'a1': (
              smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CATEGORICAL
          ),
      },
  )
  return [d0, d1]


def get_test_reward_infos() -> Sequence[smart_control_reward_pb2.RewardInfo]:
  """Creates pre-defined RewardInfos for unit testing."""

  return [
      get_test_reward_info(
          zone_temp_occupancies=[('z0', 295.0, 8.0), ('z1', 292.0, 3.0)],
          air_handler_energies=[('ac0', 23.0, 15.0)],
          boiler_energies=[('b0', 200.1, 2.3)],
          start_timestamp=pd.Timestamp('2022-03-13 00:00:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:05:00'),
      ),
      get_test_reward_info(
          zone_temp_occupancies=[('z0', 293.0, 5.0), ('z1', 291.0, 8.0)],
          air_handler_energies=[('ac0', 26.0, 22.0)],
          boiler_energies=[('b0', 202.0, 2.1)],
          start_timestamp=pd.Timestamp('2022-03-13 00:05:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:10:00'),
      ),
      get_test_reward_info(
          zone_temp_occupancies=[('z0', 291.0, 3.0), ('z1', 290.0, 10.0)],
          air_handler_energies=[('ac0', 28.0, 21.0)],
          boiler_energies=[('b0', 190.0, 8.1)],
          start_timestamp=pd.Timestamp('2022-03-13 00:10:00'),
          end_timestamp=pd.Timestamp('2022-03-13 00:15:00'),
      ),
  ]


def get_test_reward_info(
    zone_temp_occupancies: Sequence[Tuple[str, float, float]],
    air_handler_energies: Sequence[Tuple[str, float, float]],
    boiler_energies: Sequence[Tuple[str, float, float]],
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
) -> smart_control_reward_pb2.RewardInfo:
  """Creates RewardInfos for unit tests."""
  heating_setpoint_temperature = 293.0
  cooling_setpoint_temperature = 297.0
  zone_air_flow_rate_setpoint = 0.013
  zone_air_flow_rate = 0.012
  info = smart_control_reward_pb2.RewardInfo(
      agent_id='test_agent',
      scenario_id='test_scenario',
      start_timestamp=conversion_utils.pandas_to_proto_timestamp(
          pd.Timestamp(start_timestamp)
      ),
      end_timestamp=conversion_utils.pandas_to_proto_timestamp(
          pd.Timestamp(end_timestamp)
      ),
  )

  for zone_temp_occupancy in zone_temp_occupancies:
    zone_id, zone_air_temp, zone_occupancy = zone_temp_occupancy

    zone_info = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heating_setpoint_temperature,
        cooling_setpoint_temperature=cooling_setpoint_temperature,
        zone_air_temperature=zone_air_temp,
        average_occupancy=zone_occupancy,
        air_flow_rate_setpoint=zone_air_flow_rate_setpoint,
        air_flow_rate=zone_air_flow_rate,
    )

    info.zone_reward_infos[zone_id].CopyFrom(zone_info)

  for air_handler_energy in air_handler_energies:
    (
        air_handler_id,
        blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate,
    ) = air_handler_energy
    air_handler_info = smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
        blower_electrical_energy_rate=blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate,
    )
    info.air_handler_reward_infos[air_handler_id].CopyFrom(air_handler_info)

  for boiler_energy in boiler_energies:
    boiler_id, natural_gas_heating_energy_rate, pump_electrical_energy_rate = (
        boiler_energy
    )
    boiler_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
        natural_gas_heating_energy_rate=natural_gas_heating_energy_rate,
        pump_electrical_energy_rate=pump_electrical_energy_rate,
    )
    info.boiler_reward_infos[boiler_id].CopyFrom(boiler_info)

  return info


class TestReader(BaseReader):
  """Implementation of BaseReader for test."""

  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads observation_responses from endpoint bounded by start & end time."""
    return get_observation_responses()

  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads action_responses from endpoint bounded by start and end time."""
    return get_replay_action_responses()

  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward infos from endpoint bounded by start and end time."""

    return get_test_reward_infos()

  def read_reward_responses(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardResponse]:
    """Reads reward responses from endpoint bounded by start and end time."""
    return get_test_reward_responses()

  def read_normalization_info(self):
    raise NotImplementedError()  # pragma: no cover

  def read_zone_infos(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Reads the zone infos for the Building."""
    return get_zone_infos()

  def read_device_infos(
      self,
  ) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Reads the device infos for the Building."""
    return get_device_infos()
