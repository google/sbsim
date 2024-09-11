"""Utility functions for the Regression Building.

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

import collections
import datetime
import itertools
from typing import Any, List, Mapping, Sequence, Set, Tuple, Union

from absl import logging
import gin
import numpy as np
import pandas as pd
from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator.setpoint_schedule import SetpointSchedule
from smart_control.utils import conversion_utils

_ValueType = smart_control_building_pb2.DeviceInfo.ValueType
_ActionResponseType = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)
_ACTION_DEVICE_TYPES = [
    smart_control_building_pb2.DeviceInfo.AHU,
    smart_control_building_pb2.DeviceInfo.BLR,
    smart_control_building_pb2.DeviceInfo.AC,
]
_ACTION_PREFIX = 'action'
_TIMESTAMP = 'timestamp'
_REWARD_INFO = 'reward_info'
_START = 'start'
_END = 'end'
_BLOWER_ELECTRICAL_ENERGY_RATE = 'blower_electrical_energy_rate'
_AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE = (
    'air_conditioning_electrical_energy_rate'
)
_NATURAL_GAS_HEATING_ENERGY_RATE = 'natural_gas_heating_energy_rate'
_PUMP_ELECTRICAL_ENERGY_RATE = 'pump_electrical_energy_rate'
_ZONE_AIR_TEMPERATURE_SENSOR = 'zone_air_temperature_sensor'
_ZONE_AIR_COOLING_TEMPERATURE_SETPOINT = 'zone_air_cooling_temperature_setpoint'
_ZONE_AIR_HEATING_TEMPERATURE_SETPOINT = 'zone_air_heating_temperature_setpoint'
_DAY_OF_WEEK = 'dow'
_HOUR_OF_DAY = 'hod'
_SIN_RAD = 'sin'
_COS_RAD = 'cos'


@gin.configurable
def get_consolidated_time_features(
    n_hod: int, n_dow: int
) -> List[Union[str, Tuple[str, str]]]:
  """Macro for expanding time feature names."""
  return (
      [_TIMESTAMP]
      + get_time_feature_names(n=n_hod, label=_HOUR_OF_DAY)
      + get_time_feature_names(n=n_dow, label=_DAY_OF_WEEK)
  )


def get_time_feature_names(
    n: int, label: str = _HOUR_OF_DAY
) -> list[tuple[str, str]]:
  """Returns labels for phase-shifted time signals, 0...n-1 cos, 0...n-1 sin.

  Args:
    n: Number of sines and cosines.
    label: Feature label, i.e., dow or hod, etc.

  Returns:
    List of feature names (label, cos_0), (label, cos_1)...(label, sin_n-1).
  """

  l1 = _COS_RAD + '_%03d'
  l2 = _SIN_RAD + '_%03d'
  return [(label, l1 % i) for i in range(n)] + [
      (label, l2 % i) for i in range(n)
  ]
  # return [(label, '%s_%0.3d' % (_COS_RAD, i)) for i in range(n)
  # ] + [(label, '%s_%0.3d' % (_SIN_RAD, i)) for i in range(n)]


def expand_time_features(
    n: int, rad: float, label: str = _HOUR_OF_DAY
) -> Mapping[tuple[str, str], float]:
  """Generates 2n phase-shifted time signals: 0...n-1 cos and 0...n-1 sin.

  Args:
    n: Number of sines and cosines.
    rad: Radian value of the time signal.
    label: Feature label, i.e., dow or hod, etc.

  Returns:
    Dict with {(label, cos_0): cos(theta0), ..., (label, sin_n): sin(thetan-1)}
  """

  feature_names = get_time_feature_names(n, label)

  phase = rad + (np.arange(n) / n * 2.0 * np.pi)
  cos_component = np.cos(phase)
  sin_component = np.sin(phase)

  feature_names = get_time_feature_names(n, label)

  assert len(feature_names) == len(sin_component) + len(cos_component)
  return {
      feature_name: value
      for feature_name, value in zip(
          feature_names, itertools.chain(cos_component, sin_component)
      )
  }


def get_observation_sequence(
    observation_responses: Sequence[
        smart_control_building_pb2.ObservationResponse
    ],
    feature_tuples: set[tuple[str, str]],
    time_zone: str = 'UTC',
    n_hod: int = 1,
    n_dow: int = 1,
) -> pd.DataFrame:
  """Converts observation responses into a pandas DF table.

  Args:
    observation_responses: list of ObservationResponses
    feature_tuples: set of desired (device_id: measurement_name) pairs
    time_zone: time_zone, defaulting to UTC
    n_hod: Number of cos/sin feature pairs for hour of day.
    n_dow: Number of cos/sin feature pairs for day or week.

  Returns:
    a pandas DF, one row for each ObservationResponse and columns for each
    (device, meausrement) pair, and also timestamp and Day of Week (dow)
    and Hour of Day (hod) features.
  """

  cols = (
      [_TIMESTAMP]
      + get_time_feature_names(n_hod, _HOUR_OF_DAY)
      + get_time_feature_names(n_dow, _DAY_OF_WEEK)
      + sorted(feature_tuples)
  )

  dfs = []
  for index, observation_response in enumerate(observation_responses):
    feature_map = get_feature_map(observation_response, time_zone, n_hod, n_dow)
    dfs.append(pd.DataFrame(feature_map, columns=cols, index=[index]))
  return pd.concat(dfs)


def get_feature_map(
    observation_response: smart_control_building_pb2.ObservationResponse,
    time_zone: str = 'UTC',
    n_hod: int = 1,
    n_dow: int = 1,
) -> Mapping[Any, float]:
  """Converts ObservationResponse to {(feature tuple): value} w/ time vals.

  Args:
    observation_response: an ObservationResponse object
    time_zone: time_zone, defaulting to UTC
    n_hod: Number of cos/sin feature pairs for hour of day.
    n_dow: Number of cos/sin feature pairs for day or week.

  Returns:
    a mapping  {(feature tuple): value} appending timestamp, Day of Week and
    Hour of Day sine/cosine features.
  """
  feature_map = {}
  timestamp = conversion_utils.proto_to_pandas_timestamp(
      observation_response.timestamp
  ).tz_convert(time_zone)
  ts_rad_hod = conversion_utils.get_radian_time(
      timestamp, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
  )
  ts_rad_dow = conversion_utils.get_radian_time(
      timestamp, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
  )
  feature_map[_TIMESTAMP] = timestamp
  feature_map.update(
      expand_time_features(n=n_hod, rad=ts_rad_hod, label=_HOUR_OF_DAY)
  )
  feature_map.update(
      expand_time_features(n=n_dow, rad=ts_rad_dow, label=_DAY_OF_WEEK)
  )
  for (
      single_observation_response
  ) in observation_response.single_observation_responses:
    request = single_observation_response.single_observation_request

    if single_observation_response.observation_valid:
      feature_map[(request.device_id, request.measurement_name)] = (
          single_observation_response.continuous_value
      )

  return feature_map


def get_action_tuples(
    action_response: smart_control_building_pb2.ActionResponse,
) -> Set[Tuple[str, str, str]]:
  """Returns the tuples (_ACTION_PREFIX, device_id, setpoint) from ActionResponse."""
  action_tuples = set()
  for request in action_response.request.single_action_requests:
    action_tuples.add(
        (_ACTION_PREFIX, request.device_id, request.setpoint_name)
    )
  return action_tuples


def get_feature_tuples(
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> Set[Tuple[str, str]]:
  """Returns tuples (device_id, measurement) from ObservationResponse."""
  feature_tuples = set()
  for (
      single_observation_response
  ) in observation_response.single_observation_responses:
    if single_observation_response.observation_valid:
      request = single_observation_response.single_observation_request
      feature_tuples.add((request.device_id, request.measurement_name))
  return feature_tuples


def get_action_map(
    action_response: smart_control_building_pb2.ActionResponse,
    time_zone: Union[str, datetime.tzinfo] = 'UTC',
) -> Mapping[Any, Any]:
  """Returns a map {(action_tuple): value} from an ActionResponse."""
  action_map = {}

  timestamp = conversion_utils.proto_to_pandas_timestamp(
      action_response.timestamp
  ).tz_convert(time_zone)
  action_map[_TIMESTAMP] = timestamp

  for single_action_response in action_response.single_action_responses:
    if (
        single_action_response.response_type
        == smart_control_building_pb2.SingleActionResponse.ACCEPTED
    ):
      request = single_action_response.request

      action_map[(_ACTION_PREFIX, request.device_id, request.setpoint_name)] = (
          request.continuous_value
      )
    else:
      action_map[(_ACTION_PREFIX, request.device_id, request.setpoint_name)] = (
          np.nan
      )

  return action_map


def get_reward_info_tuples(
    reward_info: smart_control_reward_pb2.RewardInfo,
) -> Set[Tuple[str, str, str]]:
  """Returns tuples (_REWARD_INFO, device, field) for a RewardInfo.

  Args:
    reward_info: a RewardInfo

  Returns: an enumeration of devices and associated energy use. Note that
    ZoneInfo components can be calculated with the observation values.
  """
  reward_info_tuples = set()

  reward_info_tuples.add((_REWARD_INFO, _TIMESTAMP, _START))
  reward_info_tuples.add((_REWARD_INFO, _TIMESTAMP, _END))
  for air_handler_id in reward_info.air_handler_reward_infos:
    reward_info_tuples.add(
        (_REWARD_INFO, air_handler_id, _BLOWER_ELECTRICAL_ENERGY_RATE)
    )
    reward_info_tuples.add(
        (_REWARD_INFO, air_handler_id, _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE)
    )
  for boiler_id in reward_info.boiler_reward_infos:
    reward_info_tuples.add(
        (_REWARD_INFO, boiler_id, _NATURAL_GAS_HEATING_ENERGY_RATE)
    )
    reward_info_tuples.add(
        (_REWARD_INFO, boiler_id, _PUMP_ELECTRICAL_ENERGY_RATE)
    )
  return reward_info_tuples


def get_reward_info_map(
    reward_info: smart_control_reward_pb2.RewardInfo,
    time_zone: Union[str, datetime.tzinfo] = 'UTC',
) -> Mapping[Tuple[str, str, str], float]:
  """Returns a {reward_info tuple: value} for a RewardInfo.

  Args:
    reward_info: A RewardInfo.
    time_zone: The local time zone.

  Returns: an mapping of devices and associated energy use.
  """
  reward_info_map = {}

  start_timestamp = conversion_utils.proto_to_pandas_timestamp(
      reward_info.start_timestamp
  ).tz_convert(time_zone)
  reward_info_map[(_REWARD_INFO, _TIMESTAMP, _START)] = start_timestamp

  end_timestamp = conversion_utils.proto_to_pandas_timestamp(
      reward_info.end_timestamp
  ).tz_convert(time_zone)
  reward_info_map[(_REWARD_INFO, _TIMESTAMP, _END)] = end_timestamp

  for air_handler_id in reward_info.air_handler_reward_infos:
    reward_info_map[
        (_REWARD_INFO, air_handler_id, _BLOWER_ELECTRICAL_ENERGY_RATE)
    ] = reward_info.air_handler_reward_infos[
        air_handler_id
    ].blower_electrical_energy_rate
    reward_info_map[
        (_REWARD_INFO, air_handler_id, _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE)
    ] = reward_info.air_handler_reward_infos[
        air_handler_id
    ].air_conditioning_electrical_energy_rate

  for boiler_id in reward_info.boiler_reward_infos:
    reward_info_map[
        (_REWARD_INFO, boiler_id, _NATURAL_GAS_HEATING_ENERGY_RATE)
    ] = reward_info.boiler_reward_infos[
        boiler_id
    ].natural_gas_heating_energy_rate
    reward_info_map[(_REWARD_INFO, boiler_id, _PUMP_ELECTRICAL_ENERGY_RATE)] = (
        reward_info.boiler_reward_infos[boiler_id].pump_electrical_energy_rate
    )

  return reward_info_map


def get_matching_indexes(
    raw_input_sequence: pd.DataFrame,
    raw_output_sequence: pd.DataFrame,
    step_interval: pd.Timedelta,
) -> Tuple[Sequence[pd.Timestamp], Sequence[pd.Timestamp]]:
  """Matches input and output DataFrames, offset by one timestep.

  Both input and output dataframes have timestamp indexes, that are
  separated by step_interval. If t1 - t0 = time_interval, then the
  resultant indexes will be:
    input   output
    t0      t1
    t1      t2
    t2      t3
    ...

  Args:
    raw_input_sequence: dataframe with timestamp indices
    raw_output_sequence: dataframe with timestamp indices
    step_interval: time delay between updates

  Returns:
    matched input and output indices
  """

  raw_input_sequence = raw_input_sequence.dropna()
  raw_output_sequence = raw_output_sequence.dropna()

  input_deque = collections.deque(list(raw_input_sequence.index))
  output_deque = collections.deque(list(raw_output_sequence.index))

  input_indexes = []
  output_indexes = []

  ts_output = output_deque.popleft()
  while len(input_deque) and len(output_deque):
    ts_input = input_deque.popleft()
    while ts_output <= ts_input:
      ts_output = output_deque.popleft()
    if ts_output - ts_input <= step_interval:
      input_indexes.append(ts_input)
      output_indexes.append(ts_output)

  assert len(output_indexes) == len(input_indexes)
  return input_indexes, output_indexes


def get_reward_info_sequence(
    reward_infos: Sequence[smart_control_reward_pb2.RewardInfo],
    reward_info_tuples: set[tuple[str, str, str]],
    time_zone: Union[str, datetime.tzinfo] = 'UTC',
) -> pd.DataFrame:
  """Converts a list of RewardInfos into a dataframe."""
  return pd.DataFrame(
      (
          get_reward_info_map(reward_info, time_zone)
          for reward_info in reward_infos
      ),
      columns=sorted(list(reward_info_tuples)),
  )


def get_action_sequence(
    action_responses: Sequence[smart_control_building_pb2.ActionResponse],
    action_tuples: set[tuple[str, str, str]],
    time_zone: Union[str, datetime.tzinfo] = 'UTC',
) -> pd.DataFrame:
  """Converts a list of ActionResponses in to a dataframe."""
  df = pd.DataFrame(columns=[_TIMESTAMP] + sorted(list(action_tuples)))
  for action_response in action_responses:
    action_map_all = get_action_map(action_response, time_zone)
    action_map = {k: v for k, v in action_map_all.items() if k in df.columns}
    df = pd.concat([df, pd.DataFrame([action_map])], ignore_index=True)

  return df


def get_device_action_tuples(
    devices: Sequence[smart_control_building_pb2.DeviceInfo],
) -> Sequence[Tuple[str, str, str]]:
  """Converts DeviceInfos into action tuples: (_ACTION_PREFIX, device, setpoint)."""
  device_action_tuples = []
  for device_info in devices:
    device_id = device_info.device_id
    if device_info.device_type in _ACTION_DEVICE_TYPES:
      for action in device_info.action_fields:
        device_action_tuples.append((_ACTION_PREFIX, device_id, action))
  return device_action_tuples


def get_observation_response(
    observation_request: smart_control_building_pb2.ObservationRequest,
    native_observation_mapping: Mapping[Tuple[str, str], float],
    current_timestamp: pd.Timestamp,
) -> smart_control_building_pb2.ObservationResponse:
  """Creates an observation response, combining a request and obs mapping.

  Args:
    observation_request: agent's observation request
    native_observation_mapping: mapping of {(device, field): value}
    current_timestamp: validity timestamp of the device

  Returns:
      ObservationResponse
  """

  observation_response = smart_control_building_pb2.ObservationResponse()
  observation_response.request.CopyFrom(observation_request)
  observation_response.timestamp.CopyFrom(
      conversion_utils.pandas_to_proto_timestamp(current_timestamp)
  )

  for single_request in observation_request.single_observation_requests:
    single_response = smart_control_building_pb2.SingleObservationResponse()

    single_response.single_observation_request.CopyFrom(single_request)
    single_response.timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(current_timestamp)
    )

    device_id = single_request.device_id
    measurement_name = single_request.measurement_name

    if (device_id, measurement_name) in native_observation_mapping:
      single_response.continuous_value = native_observation_mapping[
          (device_id, measurement_name)
      ]
      single_response.observation_valid = True

    else:
      single_response.observation_valid = False

    observation_response.single_observation_responses.append(single_response)
  return observation_response


def observation_response_to_observation_mapping(
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> Mapping[Tuple[str, str], float]:
  """Converts an ObservationResponse to a device, measurement mapping.

  Args:
    observation_response: an ObservationResponse

  Returns:
    Dictionary of {(device, measurement_name): measurement_value}
  """

  native_observation_mapping = {}
  for (
      single_observation_response
  ) in observation_response.single_observation_responses:
    device_id = single_observation_response.single_observation_request.device_id
    measurement_name = (
        single_observation_response.single_observation_request.measurement_name
    )
    if single_observation_response.observation_valid:
      native_observation_mapping[(device_id, measurement_name)] = (
          single_observation_response.continuous_value
      )

  return native_observation_mapping


def create_action_response(
    action_request: smart_control_building_pb2.ActionRequest,
    current_timestamp: pd.Timestamp,
    device_action_tuples: Sequence[Tuple[str, str, str]],
) -> smart_control_building_pb2.ActionResponse:
  """Given an action request from the agent create an action response.

  Args:
    action_request: ActionRequest from agent
    current_timestamp: response timestamp
    device_action_tuples: list of [(_ACTION_PREFIX, device_id, setpoint_name)]

  Returns:
    An ActionResponse
  """

  action_response = smart_control_building_pb2.ActionResponse()
  action_response.request.CopyFrom(action_request)
  action_response.timestamp.CopyFrom(
      conversion_utils.pandas_to_proto_timestamp(current_timestamp)
  )
  for single_request in action_request.single_action_requests:
    single_response = smart_control_building_pb2.SingleActionResponse()

    single_response.request.CopyFrom(single_request)
    single_response.response_type = _ActionResponseType.ACCEPTED
    action_tuple = (
        _ACTION_PREFIX,
        single_request.device_id,
        single_request.setpoint_name,
    )

    if action_tuple not in device_action_tuples:
      single_response.response_type = (
          _ActionResponseType.REJECTED_INVALID_DEVICE
      )
      action_response.single_action_responses.append(single_response)
      continue

    action_response.single_action_responses.append(single_response)

  return action_response


def split_output_into_observations_and_reward_info_mapping(
    output_mapping: Mapping[Tuple[str, ...], float]
) -> Tuple[
    Mapping[Tuple[str, str], float], Mapping[Tuple[str, str, str], float]
]:
  """Splits the prediction output into a reward_info and observation mappings."""
  reward_info_mapping = {
      k: output_mapping[k] for k in output_mapping if k[0] == _REWARD_INFO
  }
  observation_mapping = {
      k: output_mapping[k] for k in output_mapping if k[0] != _REWARD_INFO
  }
  return observation_mapping, reward_info_mapping


def get_reward_info_devices(
    reward_info_mapping: Mapping[Tuple[str, str, str], float]
) -> Mapping[str, Mapping[str, float]]:
  """Combines the reward infos by device (e.g., by air handler).

  Args:
    reward_info_mapping: [(_REWARD_INFO, device, fieldname): value]

  Returns:
    mapping by device: {device: {fieldname:value}}
  """
  device_mapping = collections.defaultdict(dict)
  for tup in reward_info_mapping:
    device_mapping[tup[1]].update({tup[2]: reward_info_mapping[tup]})
  return device_mapping


def action_request_to_action_mapping(
    action_request: smart_control_building_pb2.ActionRequest,
    device_action_tuples: Sequence[Tuple[str, str, str]],
) -> Mapping[Tuple[str, str, str], float]:
  """Converts the action request proto to an action mapping.

  Args:
    action_request: ActionRequest from the agent
    device_action_tuples: (_ACTION_PREFIX, device_id, setpoint_name)

  Returns:
    mapping of {(_ACTION_PREFIX, device_id, setpoint_name): value}

  Raises:
    ValueError if agent does not change all setpoints.
  """
  device_action_map = {}
  for single_action_request in action_request.single_action_requests:
    tup = (
        _ACTION_PREFIX,
        single_action_request.device_id,
        single_action_request.setpoint_name,
    )
    if tup in device_action_tuples:
      device_action_map[tup] = single_action_request.continuous_value

  incomplete_actions = set(device_action_tuples).difference(
      set(device_action_map.keys())
  )

  if incomplete_actions:
    logging.warning(
        'The following %d actions were not set: %s',
        len(incomplete_actions),
        incomplete_actions,
    )
  return device_action_map


def get_boiler_reward_infos(
    reward_info_devices: Mapping[str, Mapping[str, float]]
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.BoilerRewardInfo]:
  """Converts the reward info devices in to a map of BoilerRewardInfos.

  Args:
    reward_info_devices: Mapping {device_id: {field_id: field_value}}

  Returns:
    a Mapping {device_id: BoilerRewardInfo}, one for each boiler.
  """
  boiler_reward_infos = {}
  for device_id in reward_info_devices:
    # Determine this device is a boiler by its fields, not its name.
    # To be a boiler (HW system) is must reoprt both a natural gas heating and
    # pump electric power.
    if _NATURAL_GAS_HEATING_ENERGY_RATE in reward_info_devices[device_id]:
      natural_has_heating_energy_rate = reward_info_devices[device_id][
          _NATURAL_GAS_HEATING_ENERGY_RATE
      ]
    else:
      natural_has_heating_energy_rate = np.nan

    if _PUMP_ELECTRICAL_ENERGY_RATE in reward_info_devices[device_id]:
      pump_electrical_energy_rate = reward_info_devices[device_id][
          _PUMP_ELECTRICAL_ENERGY_RATE
      ]
    else:
      pump_electrical_energy_rate = np.nan

    if not np.isnan(pump_electrical_energy_rate) and not np.isnan(
        natural_has_heating_energy_rate
    ):
      boiler_reward_info = smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
          natural_gas_heating_energy_rate=natural_has_heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
      boiler_reward_infos[device_id] = boiler_reward_info

  return boiler_reward_infos


def get_air_handler_reward_infos(
    reward_info_devices: Mapping[str, Mapping[str, float]]
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo]:
  """Converts the reward_info_devices into a map of AirHandlerRewardInfos.

  Args:
    reward_info_devices: Mapping {device_id: {field_id: field_value}}

  Returns:
    a Mapping {device_id: AirHandlerRewardInfo}, one for each air handler.
  """
  air_handler_reward_infos = {}

  for device_id in reward_info_devices:
    if _BLOWER_ELECTRICAL_ENERGY_RATE in reward_info_devices[device_id]:
      blower_electrical_energy_rate = reward_info_devices[device_id][
          _BLOWER_ELECTRICAL_ENERGY_RATE
      ]
    else:
      blower_electrical_energy_rate = np.nan

    if (
        _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE
        in reward_info_devices[device_id]
    ):
      air_conditioning_electrical_energy_rate = reward_info_devices[device_id][
          _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE
      ]
    else:
      air_conditioning_electrical_energy_rate = np.nan

    if not np.isnan(air_conditioning_electrical_energy_rate) and not np.isnan(
        blower_electrical_energy_rate
    ):
      air_handler_reward_info = smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
          blower_electrical_energy_rate=blower_electrical_energy_rate,
          air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate,
      )
      air_handler_reward_infos[device_id] = air_handler_reward_info

  return air_handler_reward_infos


def get_current_device_observations(
    current_observations: Mapping[Tuple[str, str], float], device_id
) -> Mapping[str, float]:
  """Returns the current observations for just this device.

  Args:
    current_observations: Mapping {(device_id, measurement_name): float}
    device_id: specific device of interest.

  Returns: Mapping {measurement_name: float} for just that device_id.
  """
  return {
      tup[1]: current_observations[tup]
      for tup in current_observations
      if tup[0] == device_id
  }


def get_zone_reward_infos(
    current_timestamp: pd.Timestamp,
    step_interval: pd.Timedelta,
    current_observation_mapping: Mapping[Tuple[str, str], float],
    occupancy_function: BaseOccupancy,
    setpoint_schedule: SetpointSchedule,
    zone_infos: Sequence[smart_control_building_pb2.ZoneInfo],
    device_infos: Sequence[smart_control_building_pb2.DeviceInfo],
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.ZoneRewardInfo]:
  """Returns a map of messages with zone data to compute the instantaneous reward."""
  zone_reward_infos = {}
  zone_device_mapping = {
      zone_info.zone_id: zone_info.devices for zone_info in zone_infos
  }

  device_mapping = {
      device_info.device_id: device_info for device_info in device_infos
  }
  (
      zone_air_heating_temperature_setpoint,
      zone_air_cooling_temperature_setpoint,
  ) = setpoint_schedule.get_temperature_window(current_timestamp)

  if (
      zone_air_heating_temperature_setpoint
      > zone_air_cooling_temperature_setpoint
  ):
    raise ValueError(
        'Bad setpoints: zone_air_heating_temperature_setpoint'
        f' {zone_air_heating_temperature_setpoint} >'
        ' zone_air_cooling_temperature_setpoint'
        f' {zone_air_cooling_temperature_setpoint}'
    )

  for zone_info in zone_infos:
    zone_id = zone_info.zone_id

    average_occupancy = occupancy_function.average_zone_occupancy(
        zone_id=zone_id,
        start_time=current_timestamp - step_interval,
        end_time=current_timestamp,
    )
    zone_devices = zone_device_mapping[zone_id]

    for device_id in zone_devices:
      device_info = device_mapping[device_id]
      device_observations = get_current_device_observations(
          current_observation_mapping, device_id
      )

      if (
          device_info.device_type == smart_control_building_pb2.DeviceInfo.VAV
          and _ZONE_AIR_TEMPERATURE_SENSOR in device_observations
      ):
        zone_air_temperature = device_observations[_ZONE_AIR_TEMPERATURE_SENSOR]

        # In the real building, VAVs generate setpoint and sensor
        # measurements in F rather than K. So, here we make the adjustment
        # from F to K.
        zone_air_temperature = conversion_utils.fahrenheit_to_kelvin(
            zone_air_temperature
        )

        if _ZONE_AIR_COOLING_TEMPERATURE_SETPOINT in device_observations:
          zone_air_cooling_temperature_setpoint = (
              conversion_utils.fahrenheit_to_kelvin(
                  device_observations[_ZONE_AIR_COOLING_TEMPERATURE_SETPOINT]
              )
          )

        if _ZONE_AIR_HEATING_TEMPERATURE_SETPOINT in device_observations:
          zone_air_heating_temperature_setpoint = (
              conversion_utils.fahrenheit_to_kelvin(
                  device_observations[_ZONE_AIR_HEATING_TEMPERATURE_SETPOINT]
              )
          )

        zone_reward_infos[zone_id] = (
            smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
                heating_setpoint_temperature=zone_air_heating_temperature_setpoint,
                cooling_setpoint_temperature=zone_air_cooling_temperature_setpoint,
                zone_air_temperature=zone_air_temperature,
                average_occupancy=average_occupancy,
            )
        )
        break

  return zone_reward_infos
