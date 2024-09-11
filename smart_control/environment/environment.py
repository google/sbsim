"""Controllable building RL environment to interact with TF-Agents.

RL environment where the agent is able to control various
setpoints with the goal of making the HVAC system more efficient.

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

import collections
import copy
import os
import time
from typing import Final, Mapping, NewType, Optional, Sequence, Tuple

from absl import logging
import bidict
import gin
import numpy as np
import pandas as pd
from smart_control.models import base_building
from smart_control.models import base_normalizer
from smart_control.models import base_reward_function
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import building_image_generator
from smart_control.utils import constants
from smart_control.utils import conversion_utils
from smart_control.utils import histogram_reducer
from smart_control.utils import plot_utils
from smart_control.utils import regression_building_utils
from smart_control.utils import run_command_predictor
from smart_control.utils import writer_lib
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


ACTION_REJECTION_REWARD: Final[float] = -np.inf

ValueType = smart_control_building_pb2.DeviceInfo.ValueType
DeviceInfo = smart_control_building_pb2.DeviceInfo

DeviceFieldId = NewType("DeviceFieldId", str)
DeviceId = NewType("DeviceId", str)
FieldName = NewType("FieldName", str)

COMFORT_MODE_NOW: Final[str] = "comfort_mode_now"
COMFORT_MODE_SOON: Final[str] = "comfort_mode_soon"
NUM_OCCUPANTS: Final[str] = "num_occupants"
DOW_LABEL: Final[str] = "dow"
HOD_LABEL: Final[str] = "hod"

DeviceFieldId = NewType("DeviceFieldId", str)
FieldName = NewType("FieldName", str)
ActionNormalizerMap = Mapping[
    DeviceFieldId, base_normalizer.BaseActionNormalizer
]

DefaultActions = Mapping[DeviceFieldId, float]

DeviceCode = str
Setpoint = str
MeasurementName = str
DeviceActionTuple = Tuple[DeviceCode, Setpoint]
DeviceMeasurementTuple = Tuple[DeviceCode, MeasurementName]


def all_actions_accepted(
    action_response: smart_control_building_pb2.ActionResponse,
) -> bool:
  """Returns true if all single action requests have response code ACCEPTED."""

  return all(
      single_action_response.response_type
      == smart_control_building_pb2.SingleActionResponse.ACCEPTED
      for single_action_response in action_response.single_action_responses
  )


def replace_missing_observations_past(
    current_observation_response: smart_control_building_pb2.ObservationResponse,
    past_observation_response: Optional[
        smart_control_building_pb2.ObservationResponse
    ],
) -> smart_control_building_pb2.ObservationResponse:
  """Replaces any missing observations with a past ObservationResponse.

  Sometimes, the building doesn't report all the observations; however,
  the agent requires all fields to be populated. When a missing observation is
  encountered, impute the value from the most recent observation.

  Args:
    current_observation_response: Current observations from the building.
    past_observation_response: Use this observation to fill in any missing
      observations.

  Returns:
    A merged ObservationResponsem filled in from the past observation.

   Raises:
     ValueError when a missing observation exists and there is no past
       observation.
  """

  def get_observation_request_tuples(
      observation_request: smart_control_building_pb2.ObservationRequest,
  ) -> set[DeviceMeasurementTuple]:
    return set(
        [
            (request.device_id, request.measurement_name)
            for request in observation_request.single_observation_requests
        ]
    )

  def get_observation_response_mapping(
      observation_response: smart_control_building_pb2.ObservationResponse,
  ) -> dict[
      DeviceMeasurementTuple,
      smart_control_building_pb2.SingleObservationResponse,
  ]:
    """Converts an ObservationResponse into a dict of single observations."""
    # pylint: disable=g-complex-comprehension
    return {
        (
            response.single_observation_request.device_id,
            response.single_observation_request.measurement_name,
        ): response
        for response in observation_response.single_observation_responses
        if response.observation_valid
    }

  def check_valid_past_observation(
      past_observation_response: Optional[
          smart_control_building_pb2.ObservationResponse
      ],
      missing_observations: set[DeviceMeasurementTuple],
  ) -> None:
    """Checks that the past observation is available, and raises ValueError."""
    if not past_observation_response:
      # If there is not a past response, then provide a detailed log entry and
      # raise a ValueError.
      for missing_observation in missing_observations:
        logging.error(
            "Missing or invalid observation response for %s %s; no past"
            " observation to replace with.",
            missing_observation[0],
            missing_observation[1],
        )

      raise ValueError(
          f"Missing {len(missing_observations)} observations, and no past"
          " observation available to replace with."
      )

  def get_missing_observations(
      observation_response: smart_control_building_pb2.ObservationResponse,
  ) -> set[DeviceMeasurementTuple]:
    """Returns device/measurements set for requests that weren't provided."""

    observation_request_tuples = get_observation_request_tuples(
        observation_response.request
    )
    observation_response_map = get_observation_response_mapping(
        observation_response
    )
    return observation_request_tuples - set(observation_response_map.keys())

  def update_single_observation_response(
      single_observation_response: smart_control_building_pb2.SingleObservationResponse,
      past_observation_response_mapping: dict[
          DeviceMeasurementTuple,
          smart_control_building_pb2.SingleObservationResponse,
      ],
  ) -> smart_control_building_pb2.SingleObservationResponse:
    """Checks a single observation response and fills in when invalid."""
    if single_observation_response.observation_valid:
      updated_single_observation_response = single_observation_response
    # If it's not valid, then use the past observation to fill in the gap.
    else:
      missing_observation = (
          single_observation_response.single_observation_request.device_id,
          single_observation_response.single_observation_request.measurement_name,
      )
      updated_single_observation_response = past_observation_response_mapping[
          missing_observation
      ]

      logging.warning(
          "Missing or invalid observation response for %s %s; replacing it with"
          " past observation.",
          missing_observation[0],
          missing_observation[1],
      )
    return updated_single_observation_response

  # Compare what's in the request to what was returned in the response.
  # Put any missing or invalid responses into the missing observations list.
  missing_observations = get_missing_observations(current_observation_response)

  if missing_observations:
    # If there are missing observations and we have a past ObservationRespose,
    # filling the missing values from the past response.
    # If there are no missing observations, just return the original
    # ObservationResponse.
    check_valid_past_observation(
        past_observation_response, missing_observations
    )

    updated_single_observation_responses = []

    past_observation_response_mapping = get_observation_response_mapping(
        past_observation_response
    )

    # Maintain the same ordering between the requests and responses.
    for (
        single_observation_response
    ) in current_observation_response.single_observation_responses:
      # If the observation is valid, just add it to the updated list.
      updated_single_observation_response = update_single_observation_response(
          single_observation_response, past_observation_response_mapping
      )

      updated_single_observation_responses.append(
          updated_single_observation_response
      )
      # Create a new observation response that combines both the valid current
      # observations and the valid past observations when the current is
      # invalid.
      current_observation_response = copy.deepcopy(current_observation_response)
      del current_observation_response.single_observation_responses[:]
      current_observation_response.single_observation_responses.extend(
          updated_single_observation_responses
      )

  return current_observation_response


def compute_action_regularization_cost(
    action_history: Sequence[np.ndarray],
) -> float:
  """Applies a smoothing cost based on recent action history.

    Returns the L2 Norm of the actions as a penalty term for large changes.

  Args:
    action_history: Seqential array of actions taken in the episode.

  Returns:
    A smoothing cost applied to the reward function for applying big changes.
  """

  if len(action_history) > 1:
    if action_history[-2].shape != action_history[-1].shape:
      raise ValueError("Action history shapes do not match.")
    return np.linalg.norm(
        action_history[-2] - action_history[-1], axis=0, ord=2
    )
  else:
    return 0.0


@gin.configurable
class ActionConfig:
  """Configures BaseActionNormalizers for each setpoint.

  This class allows the user to configure a BaseActionNormalizer for any
  device_id/setpoint name tuple.

  Only setpoints given as part of this config will be part of the action space.

  Example:
    action_normalizers = {
      ('boiler_0', 'supply_water_setpoint'):
      ContinuousBaseActionNormalizer(args)
    }

  This would set a ContinuousBaseActionNormalizer for the supply_water_setpoint
  setpoint on the device with id boiler_0.
  """

  def __init__(self, action_normalizers: ActionNormalizerMap):
    self._action_normalizers = action_normalizers

  def get_action_normalizer(
      self, setpoint_name: FieldName
  ) -> Optional[base_normalizer.BaseActionNormalizer]:
    """Returns corresponding action normalizer if it exists.

    Args:
      setpoint_name: Name of setpoint to get action normalizer for.
    """
    return self._action_normalizers.get(DeviceFieldId(setpoint_name))


def generate_field_id(
    device: DeviceId, field: FieldName, id_map: bidict.bidict
) -> DeviceFieldId:
  """Returns new Id not already present in id_map.

  Ids are created by joining the device and field: device_field.

  If the same device and field are added again, the same id will be returned.

  If a unique device/field generates the same id as a different device/field,
  the id will be concatenated with an integer if the id already exists.

  Examples for clarity:
    generate_field_id(device='a_b', field='c') -> a_b_c
    generate_field_id(device='a_b', field='c') -> a_b_c
    generate_field_id(device='a', field='b_c') -> a_b_c_1

  The first id is a_b_c. The second call is an exact duplicate of the first,
  so the same id is returned. When the third call is made, because a_b_c is
  already taken, an int is concatenated and the returned id is a_b_c_1.

  Args:
    device: Device id.
    field: Measurement or setpoint name.
    id_map: Current mapping of device fields to ids.
  """
  if (device, field) in id_map:
    # May happen if observable and action have the same field name.
    return id_map[(device, field)]

  new_id = f"{device}_{field}"
  counter = 0

  # Check for duplicates.
  while new_id in id_map.inv:
    counter += 1
    new_id = f"{device}_{field}_{counter}"

  return DeviceFieldId(new_id)


@gin.configurable
class Environment(py_environment.PyEnvironment):
  """Controllable building RL environment to interact with TF-Agents."""

  def __init__(
      self,
      building: base_building.BaseBuilding,
      reward_function: base_reward_function.BaseRewardFunction,
      observation_normalizer: base_normalizer.BaseObservationNormalizer,
      action_config: ActionConfig,
      discount_factor: float = 1,
      metrics_path: str | None = None,
      num_days_in_episode: int = 3,
      device_action_tuples: Sequence[DeviceActionTuple] | None = None,
      default_actions: DefaultActions | None = None,
      metrics_reporting_interval: float = 100,
      label: str = "episode_metrics",
      num_hod_features: int = 1,
      num_dow_features: int = 1,
      occupancy_normalization_constant: float = 0.0,
      run_command_predictors: (
          Sequence[run_command_predictor.BaseRunCommandPredictor] | None
      ) = None,
      observation_histogram_reducer: (
          histogram_reducer.HistogramReducer | None
      ) = None,
      time_zone: str = "US/Pacific",
      image_generator: (
          building_image_generator.BuildingImageGenerator | None
      ) = None,
      step_interval: pd.Timedelta = pd.Timedelta(5, unit="minutes"),
      writer_factory: writer_lib.BaseWriterFactory | None = None,
  ) -> None:
    """Environment constructor.

    Args:
      building: An implementation of BaseBuilding.
      reward_function: An implementation of BaseRewardFunction.
      observation_normalizer: Normalizer parameters for observations.
      action_config: Parameters for actions: min, max, type, etc.
      discount_factor: Future reward discount, i.e., gamma.
      metrics_path: CNS directory to write environment data.
      num_days_in_episode: Episode duration.
      device_action_tuples: List of (device, setpoint) pairs for control.
      default_actions: Initial actions.
      metrics_reporting_interval: Frequency of TensorBoard metrics.
      label: Episode label prepended to the episode output directory.
      num_hod_features: Number of sin/cos pairs of time features for hour.
      num_dow_features: Number of sin/cos pairs of time features for day.
      occupancy_normalization_constant: Value used to normalize occupancy sig.
      run_command_predictors: Predictors for setting on/off in RunCommands
      observation_histogram_reducer: Add histogram reduction to observations.
      time_zone: Time zone of the building/environment.
      image_generator: Building image generator that generates image encodings
        from observation responses.
      step_interval: amount of time between env steps.
      writer_factory: Used with metrics_path, factory for metrics writers.
    """
    self.building: base_building.BaseBuilding = building
    self._time_zone = time_zone
    self._device_action_tuples: Optional[Sequence[DeviceActionTuple]] = (
        device_action_tuples
    )
    self.reward_function: base_reward_function.BaseRewardFunction = (
        reward_function
    )
    self._observation_histogram_reducer = observation_histogram_reducer
    self.discount_factor: float = discount_factor
    self._step_count: int = 0
    self._global_step_count: int = 0
    self._episode_count: int = 0
    self._episode_cumulative_reward: float = 0
    self._last_log_timestamp: float = 0.0
    self._observation_normalizer: base_normalizer.BaseObservationNormalizer = (
        observation_normalizer
    )
    self._start_timestamp: pd.Timestamp = self.building.current_timestamp
    self._action_history = []
    self._end_timestamp: pd.Timestamp = self._start_timestamp + pd.Timedelta(
        num_days_in_episode, unit="days"
    )
    self._step_interval = step_interval
    self._num_timesteps_in_episode = int(
        (self._end_timestamp - self._start_timestamp) / self._step_interval
    )
    self._metrics = plot_utils.init_metrics()
    logging.info(
        "Episode starts at %s and ends at %s; % d timesteps.",
        self._start_timestamp,
        self._end_timestamp,
        self._num_timesteps_in_episode
    )

    self._id_map = bidict.bidict()

    if self.discount_factor <= 0 or self.discount_factor > 1:
      raise ValueError("Discount factor must be in (0,1]")

    self._metrics_path: Optional[str] = metrics_path
    self._writer_factory: Optional[writer_lib.BaseWriterFactory] = (
        writer_factory
    )
    self._metrics_writer: Optional[writer_lib.BaseWriter] = None
    self._summary_writer = None
    self._label = label
    self._num_dow_features = num_dow_features
    self._num_hod_features = num_hod_features
    # Retain the last observation to fill in missing or invalid values.
    self._last_observation_response: Optional[
        smart_control_building_pb2.ObservationResponse
    ] = None

    if self.discount_factor <= 0 or self.discount_factor > 1:
      raise ValueError("Discount factor must be in (0,1]")

    if device_action_tuples is not None:
      self._action_spec, self._action_normalizers, self._action_names = (
          self._get_action_spec_and_normalizers_from_device_action_tuples(
              action_config=action_config,
              device_action_tuples=device_action_tuples,
          )
      )
    else:
      self._action_spec, self._action_normalizers, self._action_names = (
          self._get_action_spec_and_normalizers(action_config, building.devices)
      )

    logging.info("Action Names %s", self._action_names)

    self._auxiliary_features = self._get_auxiliary_features_labels(
        self._num_hod_features, self._num_dow_features
    )
    logging.info("Auxiliary Features %s", self._auxiliary_features)

    self._observation_spec, self._field_names = self._get_observation_spec(
        building.devices
    )
    logging.info("Observation Spec %s", self._observation_spec)

    logging.info("%s FIELD NAMES (%d)", self._label, len(self._field_names))
    for i, fn in enumerate(self._field_names):
      logging.info("Field %d: %s", i, fn)

    self._episode_ended = False
    self._episode_start_time = time.time()

    self._default_policy_values = (
        self._normalize_default_actions(default_actions)
        if default_actions
        else tf.constant([])
    )

    self._accumulator = collections.defaultdict(list)
    self._metrics_reporting_interval = metrics_reporting_interval
    # Since the request will not change (i.e., feature vector is fixed),
    # just define a single ObservationRequest as a template for all requests.
    self._observation_request = self._get_observation_request(building.devices)
    self._occupancy_normalization_constant = occupancy_normalization_constant
    if run_command_predictors is None:
      self._run_command_predictors = None
    else:
      self._run_command_predictors = list(run_command_predictors)

    self._building_image_generator = image_generator

  def set_summary_writer(self, summary_path: str) -> None:
    self._summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_path, flush_millis=10000
    )

  @property
  def steps_per_episode(self) -> int:
    return (
        self._end_timestamp - self._start_timestamp
    ).total_seconds() // self.building.time_step_sec

  @property
  def start_timestamp(self) -> pd.Timestamp:
    return self._start_timestamp

  @property
  def end_timestamp(self) -> pd.Timestamp:
    return self._end_timestamp

  @end_timestamp.setter
  def end_timestamp(self, value: pd.Timestamp):
    self._end_timestamp = value

  @property
  def default_policy_values(self):
    return self._default_policy_values

  def _get_observation_request(
      self, devices: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> smart_control_building_pb2.ObservationRequest:
    observation_request = smart_control_building_pb2.ObservationRequest()
    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = device.device_id
        observation_request.single_observation_requests.add(
            device_id=device_id, measurement_name=measurement_name
        )
    return observation_request

  def _get_auxiliary_features_labels(
      self, num_hod_features: int, num_dow_features: int
  ) -> Sequence[str]:
    """Returns the labels of the auxiliary features."""
    return (
        [
            "%s_%s" % (tup[0], tup[1])
            for tup in regression_building_utils.get_time_feature_names(
                num_hod_features, HOD_LABEL
            )
        ]
        + [
            "%s_%s" % (tup[0], tup[1])
            for tup in regression_building_utils.get_time_feature_names(
                num_dow_features, DOW_LABEL
            )
        ]
        + [COMFORT_MODE_NOW, COMFORT_MODE_SOON, NUM_OCCUPANTS]
    )

  def _normalize_default_actions(self, default_actions: DefaultActions):
    """Converts the default actions into a normalized action array."""

    fixed_actions = []
    for field_id in self._action_names:
      # assert action_name in default_actions

      _, setpoint_name = self._id_map.inv[field_id]
      native_setpoint_value = default_actions[setpoint_name]
      normalized_agent_value = self._action_normalizers[field_id].agent_value(
          native_setpoint_value
      )
      fixed_actions.append(normalized_agent_value)

    return tf.constant(fixed_actions)

  def _get_action_spec_and_normalizers(
      self,
      action_config: ActionConfig,
      devices: Sequence[smart_control_building_pb2.DeviceInfo],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[str]]:
    """Returns an action spec, action normalizers, and the order of actions.

    Args:
      action_config: action config object for action normalization.
      devices: list of controllable devices in the building.

    Returns:
      ArraySpec the action spec as a bounded array,
      ActionNormalizerMap: mapping between field name and its normalization,
      Sequence of fields names that indicate the field in the bounded array.
    """

    def _check_value_type_continuous(value: ValueType) -> None:
      if value == ValueType.VALUE_TYPE_UNDEFINED:
        raise ValueError("Value Type Undefined")
      elif value != ValueType.VALUE_CONTINUOUS:
        raise NotImplementedError("Value Type not supported")

    action_spec = {}
    action_normalizers = {}
    action_names = []
    logging.info(
        "Loading device-setpoint pairs from %d device_infos.", len(devices)
    )
    for device in devices:
      # We need to apply an arbitrary, but consistent ordering the actions
      # within a device. Since device.action_fields is a map and has a random
      # order, we choose to sort the actions within a device alphabetically.
      for setpoint_name in sorted(device.action_fields.keys()):
        value = device.action_fields[setpoint_name]

        device_id = DeviceId(device.device_id)
        setpoint_name = FieldName(setpoint_name)

        # Get BaseActionNormalizer based on device and setpoint_name
        action_normalizer = action_config.get_action_normalizer(setpoint_name)

        # Do not add to action_spec without an action_normalizer.
        if not action_normalizer:
          continue

        field_id = generate_field_id(device_id, setpoint_name, self._id_map)
        self._id_map[(device.device_id, setpoint_name)] = field_id
        action_names.append(field_id)

        _check_value_type_continuous(value)
        field_array_spec = action_normalizer.get_array_spec(field_id)

        action_spec[field_id] = field_array_spec
        action_normalizers[field_id] = action_normalizer

    action_spec = array_spec.BoundedArraySpec(
        shape=(len(action_names),),
        dtype=np.float32,
        name="action",
        minimum=-1.0,
        maximum=1.0,
    )
    logging.info(
        "The action_spec contains %d actions: %s.",
        len(action_names),
        ", ".join(action_names),
    )

    return action_spec, action_normalizers, action_names

  def _get_action_spec_and_normalizers_from_device_action_tuples(
      self,
      action_config: ActionConfig,
      device_action_tuples: Sequence[DeviceActionTuple],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[str]]:
    """Applies the device_action_tuples to the action configurations."""
    action_spec = {}
    action_normalizers = {}
    action_names = []
    logging.info(
        "Loading device-setpoint pairs from %d device_action_tuples.",
        len(device_action_tuples),
    )
    for device_action_tuple in device_action_tuples:
      device_id = DeviceId(device_action_tuple[0])
      setpoint_name = FieldName(device_action_tuple[1])

      # Get BaseActionNormalizer based on device and setpoint_name
      action_normalizer = action_config.get_action_normalizer(setpoint_name)

      # Do not add to action_spec without an action_normalizer.
      # TODO(sipple) Include a unit test.
      if not action_normalizer:
        raise ValueError("Missing a normalizer")

      field_id = generate_field_id(device_id, setpoint_name, self._id_map)
      self._id_map[(device_id, setpoint_name)] = field_id
      action_names.append(field_id)

      field_array_spec = action_normalizer.get_array_spec(field_id)
      action_spec[field_id] = field_array_spec
      action_normalizers[field_id] = action_normalizer

    action_spec = array_spec.BoundedArraySpec(
        shape=(len(action_names),),
        dtype=np.float32,
        name="action",
        minimum=-1.0,
        maximum=1.0,
    )
    logging.info(
        "The action_spec from device_action_tuples contains %d actions: %s.",
        len(action_names),
        ", ".join(action_names),
    )
    return action_spec, action_normalizers, action_names

  def _get_observation_spec(
      self, devices: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Returns an observation spec and a list of field names."""

    # TODO(sipple): Desuplicate the else case of
    # _get_observation_spec_histogram_reducer if the same as
    # _get_observation_spec_single_timeseries.

    if self._observation_histogram_reducer is None:
      obs_spec, observable_fields = (
          self._get_observation_spec_single_timeseries(devices)
      )
    else:
      obs_spec, observable_fields = (
          self._get_observation_spec_histogram_reducer(devices)
      )

    logging.info("There are %d observable fields.", len(observable_fields))
    logging.info("observable_fields: %s", observable_fields)
    return obs_spec, observable_fields

  def _get_observation_spec_histogram_reducer(
      self, devices: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Returns an observation spec and a list of field names as histogram."""

    assert self._observation_histogram_reducer is not None

    observable_fields = []

    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = DeviceId(device.device_id)
        measurement_name = FieldName(measurement_name)
        if (
            measurement_name
            in self._observation_histogram_reducer.histogram_parameters.keys()
        ):
          for v in self._observation_histogram_reducer.histogram_parameters[
              measurement_name
          ]:
            bin_id = "h_%.2f" % v
            if (measurement_name, bin_id) not in self._id_map.keys():
              field_id = DeviceFieldId(f"{measurement_name}_{bin_id}")

              self._id_map[(measurement_name, bin_id)] = field_id
              logging.info(
                  "Histogram feature: %s %s added to the id_map.",
                  measurement_name,
                  bin_id,
              )
              observable_fields.append(field_id)

        else:
          field_id = generate_field_id(
              device_id, measurement_name, self._id_map
          )
          self._id_map[(device_id, measurement_name)] = field_id
          logging.info(
              "Passthrough feature: %s %s",
              device_id,
              measurement_name,
          )
          observable_fields.append(field_id)

    # Include the temporal features.
    observable_fields.extend(self._auxiliary_features)

    obs_spec = array_spec.ArraySpec(
        shape=(len(observable_fields),), dtype=np.float32, name="observation"
    )
    return obs_spec, observable_fields

  def _get_observation_spec_single_timeseries(
      self, devices: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Returns an observation spec and a list of field names."""

    observable_fields = []
    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = DeviceId(device.device_id)
        measurement_name = FieldName(measurement_name)

        field_id = generate_field_id(device_id, measurement_name, self._id_map)
        self._id_map[(device_id, measurement_name)] = field_id
        observable_fields.append(field_id)

    # Include the temporal features.
    observable_fields.extend(self._auxiliary_features)

    # Multiple attempts to use a map of field_name:values for
    # the observation spec failed in various locations, including
    # (a) the ActorDistributionNetwork with various combinations
    # of preprocessing combiners, and (b) the replay buffer when adding
    # trajectories. By mapping to a simple flat ArraySpec, the failures
    # were reliably prevented and allowed the agent to train.

    logging.info("There are %d observable fields.", len(observable_fields))

    obs_spec = array_spec.ArraySpec(
        shape=(len(observable_fields),), dtype=np.float32, name="observation"
    )
    return obs_spec, observable_fields

  @property
  def current_simulation_timestamp(self):
    return self.building.current_timestamp

  def _get_action_value_type(self, field_id) -> ValueType:
    if field_id in self._action_names:
      spec = self.action_spec()[field_id]
    else:
      spec = self.observation_spec()[field_id]

    if spec.dtype == array_spec.ArraySpec((), int):
      return ValueType.VALUE_INTEGER
    if spec.dtype == array_spec.ArraySpec((), bool):
      return ValueType.VALUE_BINARY
    if spec.dtype == array_spec.ArraySpec((), np.float32):
      return ValueType.VALUE_CONTINUOUS
    # categorical not supported
    return ValueType.VALUE_TYPE_UNDEFINED

  def _create_action_request(
      self, action_array
  ) -> smart_control_building_pb2.ActionRequest:
    timestamp = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    action_request = smart_control_building_pb2.ActionRequest(
        timestamp=timestamp
    )

    action = {}
    for i in range(len(self._action_names)):
      action[self._action_names[i]] = action_array[i]

    # Append the action to the action history for use in computing cost/penalty
    # for large changes in the action.
    self._action_history.append(
        np.array(np.fromiter(action.values(), dtype=np.float32))
    )

    for field_id, _ in action.items():
      device_id, setpoint_name = self._id_map.inv[field_id]

      agent_action = action[field_id]

      action_normalizer = self._action_normalizers[field_id]

      action_value = action_normalizer.setpoint_value(agent_action)

      single_action_request = smart_control_building_pb2.SingleActionRequest(
          device_id=device_id,
          setpoint_name=setpoint_name,
          continuous_value=action_value,
      )

      action_request.single_action_requests.append(single_action_request)

    return action_request

  def _get_observation(self) -> np.ndarray:
    timestamp = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    observation_request = smart_control_building_pb2.ObservationRequest()
    observation_request.CopyFrom(self._observation_request)
    observation_request.timestamp.CopyFrom(timestamp)

    observation_response = self.building.request_observations(
        observation_request
    )

    observation_response = replace_missing_observations_past(
        current_observation_response=observation_response,
        past_observation_response=self._last_observation_response,
    )
    self._last_observation_response = observation_response

    if self._metrics_writer:
      self._metrics_writer.write_observation_response(
          observation_response, self.current_simulation_timestamp
      )
      if self._building_image_generator:
        building_image = self._building_image_generator.generate_building_image(
            observation_response
        )
        self._metrics_writer.write_building_image(
            building_image, self.current_simulation_timestamp
        )

    normalized_observation_response = self._observation_normalizer.normalize(
        observation_response
    )

    if self._observation_histogram_reducer is None:
      observation = self._normalized_observation_response_to_observation_map_single_timeseries(
          normalized_observation_response
      )
    else:
      observation = self._normalized_observation_response_to_observation_map_histogram_reducer(
          normalized_observation_response
      )

    hod_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.HOUR_OF_DAY,
    )

    hod_features = regression_building_utils.expand_time_features(
        self._num_hod_features, hod_rad, HOD_LABEL
    )
    for hod_feature_name in hod_features:
      observation["%s_%s" % (hod_feature_name[0], hod_feature_name[1])] = (
          np.array(hod_features[hod_feature_name], dtype=np.float32)
      )

    dow_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.DAY_OF_WEEK,
    )

    dow_features = regression_building_utils.expand_time_features(
        self._num_dow_features, dow_rad, DOW_LABEL
    )
    for dow_feature_name in dow_features:
      observation["%s_%s" % (dow_feature_name[0], dow_feature_name[1])] = (
          np.array(dow_features[dow_feature_name], dtype=np.float32)
      )

    observation[COMFORT_MODE_NOW] = np.array(
        self.building.is_comfort_mode(self.current_simulation_timestamp),
        dtype=np.float32,
    )
    observation[COMFORT_MODE_SOON] = np.array(
        self.building.is_comfort_mode(
            self.current_simulation_timestamp + pd.Timedelta(60, unit="minute")
        ),
        dtype=np.float32,
    )
    observation[NUM_OCCUPANTS] = np.array(
        (self.building.num_occupants - self._occupancy_normalization_constant)
        / (self._occupancy_normalization_constant + 1),
        dtype=np.float32,
    )
    # Return observation as a flat array.
    if len(self._field_names) > len(observation):
      dif_set = set(self._field_names) - observation.keys()
      dif_set_str = ", ".join(dif_set)
      logging.error("Difference: %s", dif_set_str)
      raise ValueError(
          f"Observation of length ({len(observation)}) is missing"
          f" {len(dif_set)} fields from expected fields size"
          f" ({len(self._field_names)})."
      )

    obsarray = np.array(
        [observation[field_id] for field_id in self._field_names],
        dtype=np.float32,
    )
    nan_ix = np.squeeze(np.argwhere(np.isnan(obsarray)), axis=1)
    if nan_ix.size > 0:
      nan_fields = [self._field_names[i] for i in nan_ix]
      logging.warning(
          "Observation vector contains Nans at %s.", ", ".join(nan_fields)
      )
    inf_ix = np.squeeze(np.argwhere(np.isinf(obsarray)), axis=1)
    # TODO(sipple) Add a unit test for the logging below.
    if inf_ix.size > 0:
      inf_fields = [self._field_names[i] for i in inf_ix]
      logging.warning(
          "Observation vector contains Infs at %s.", ", ".join(inf_fields)
      )
    return obsarray

  def _normalized_observation_response_to_observation_map_single_timeseries(
      self,
      normalized_observation_response: smart_control_building_pb2.ObservationResponse,
  ) -> dict[str, np.ndarray]:
    """Converts an ObservationResponse to (device, field): measurement.

    Single timeseries, since every measurement translates directly to
    its own feature, without any reduction.

    Args:
      normalized_observation_response: A normalized ObservationResponse.

    Returns:
      Dict of (device, field): measurement
    """
    observation_map = {}
    for (
        single_observation_response
    ) in normalized_observation_response.single_observation_responses:
      device_id = (
          single_observation_response.single_observation_request.device_id
      )
      measurement_name = (
          single_observation_response.single_observation_request.measurement_name
      )
      continuous_value = single_observation_response.continuous_value

      if not single_observation_response.observation_valid:
        logging.warn(
            "Invalid observation reported %s %s %f",
            device_id,
            measurement_name,
            continuous_value,
        )
        continue

      field_id = self._id_map[(device_id, measurement_name)]

      value = np.array(
          single_observation_response.continuous_value, dtype=np.float32
      )

      observation_map[field_id] = value
    return observation_map

  def _normalized_observation_response_to_observation_map_histogram_reducer(
      self,
      normalized_observation_response: smart_control_building_pb2.ObservationResponse,
  ) -> dict[str, np.ndarray]:
    """Converts an ObservationResponse to (device, field): measurement.

    This method uses a HistogramReducer to reduce multiple timeseries
    into a binned counts array.

    Args:
      normalized_observation_response: A normalized ObservationResponse.

    Returns:
      Dict of (device, field): measurement
    """

    assert self._observation_histogram_reducer is not None

    feature_tuples = regression_building_utils.get_feature_tuples(
        normalized_observation_response
    )

    observation_sequence = regression_building_utils.get_observation_sequence(
        [normalized_observation_response],
        feature_tuples,
        self._time_zone,
        self._num_hod_features,
        self._num_dow_features,
    )
    rs = self._observation_histogram_reducer.reduce(
        observation_sequence
    ).reduced_sequence

    observation_map = rs.iloc[0].to_dict()
    observation_map = {
        "_".join(k): observation_map[k]
        for k in observation_map
        if isinstance(k, tuple)
    }
    return observation_map

  def _get_reward(self) -> float:
    """Computes the immediate reward for the last action taken by the agent."""

    # Get the reward input (RewardInfo) from the building.
    reward_info = self.building.reward_info
    # Using the reward function, compute the reward value.
    reward_response = self.reward_function.compute_reward(reward_info)

    # Write both RewardInfo and RewardResponse if a metrics writer is
    # enabled.
    if self._metrics_writer:
      self._metrics_writer.write_reward_info(
          reward_info, self.current_simulation_timestamp
      )
      self._metrics_writer.write_reward_response(
          reward_response, self.current_simulation_timestamp
      )

    # Summary writer commits additional metrics to TensorBoard.
    if self._summary_writer:
      self._write_summary_reward_info_metrics(reward_info)
      self._write_summary_reward_response_metrics(reward_response)
      self._commit_reward_metrics()

    return reward_response.agent_reward_value

  def _write_summary_reward_info_metrics(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> None:
    """Writes reward input metrics into the TensorBoard logs."""
    energy_use = conversion_utils.get_reward_info_energy_use(reward_info)

    self._accumulator["electrical_energy"].append(
        energy_use["air_handler_blower_electricity"]
        + energy_use["air_handler_air_conditioning"]
        + energy_use["boiler_pump_electrical_energy"]
    )
    self._accumulator["natural_gas_energy"].append(
        energy_use["boiler_natural_gas_heating_energy"]
    )

  def _write_summary_reward_response_metrics(
      self, reward_response: smart_control_reward_pb2.RewardResponse
  ) -> None:
    """Writes reward output metrics into the TensorBoard logs."""
    self._accumulator["electricity_energy_cost"].append(
        reward_response.electricity_energy_cost
    )
    self._accumulator["natural_gas_energy_cost"].append(
        reward_response.natural_gas_energy_cost
    )
    self._accumulator["carbon_emitted"].append(reward_response.carbon_emitted)
    self._accumulator["total_occupancy"].append(reward_response.total_occupancy)
    self._accumulator["productivity_regret"].append(
        reward_response.productivity_regret
    )
    self._accumulator["normalized_productivity_regret"].append(
        reward_response.normalized_productivity_regret
    )
    self._accumulator["normalized_energy_cost"].append(
        reward_response.normalized_energy_cost
    )
    self._accumulator["normalized_carbon_emission"].append(
        reward_response.normalized_carbon_emission
    )
    self._accumulator["step_duration_sec"].append(
        reward_response.normalized_productivity_regret
    )

  def _commit_reward_metrics(self) -> None:
    """Aggregates and writes reward metrics, and resets accumulator."""
    assert self._summary_writer is not None

    if self._global_step_count % self._metrics_reporting_interval == 0:
      with (
          self._summary_writer.as_default(),
          tf.compat.v2.summary.record_if(True),
          tf.name_scope("RewardInfo/"),
      ):
        for key in self._accumulator:
          tf.compat.v2.summary.scalar(
              name=key,
              data=np.mean(self._accumulator[key]),
              step=self._global_step_count,
          )

        self._accumulator = collections.defaultdict(list)

  @property
  def label(self) -> str:
    return self._label

  def _reset(self) -> ts.TimeStep:
    self.building.reset()

    self._accumulator = collections.defaultdict(list)

    self._episode_ended = False
    self._episode_count += 1
    self._episode_cumulative_reward = 0

    observation = self._get_observation()
    self._action_history = []

    now = pd.Timestamp.utcnow()

    self._metrics_writer = None

    if self._metrics_path and self._writer_factory:
      episode_metrics_id = f"{self._label}_{now:%y%m%d_%H%M%S}"
      output_dir = os.path.join(self._metrics_path, episode_metrics_id)

      logging.info("Writing metric files to %s", output_dir)
      self._metrics_writer = self._writer_factory.create(output_dir)

      if self._building_image_generator:
        img_file_path = os.path.join(
            output_dir, constants.BUILDING_IMAGE_CSV_FILE
        )
        logging.info("Writing building image files to %s", img_file_path)

    if self._metrics_writer:
      logging.info("Writing %d device_infos.", len(self.building.devices))
      self._metrics_writer.write_device_infos(self.building.devices)
      logging.info("Writing %d zone_infos.", len(self.building.zones))
      self._metrics_writer.write_zone_infos(self.building.zones)

    self._episode_start_time = time.time()
    self._step_count = 0
    self._start_timestamp = self.building.current_timestamp
    self._end_timestamp = (
        self._start_timestamp
        + self._num_timesteps_in_episode * self._step_interval
    )
    logging.info(
        "Restarting the environment for %s to %s",
        self._start_timestamp,
        self._end_timestamp,
    )
    return ts.restart(observation)

  @gin.configurable
  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  @gin.configurable
  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def _format_action(
      self, action: types.NestedArray, action_names: Sequence[str]
  ) -> types.NestedArray:
    """Enables extension classes to reformat actions into base format."""
    return action

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Individual time step calculations.

    Args:
      action: action array performed by the agent.

    Returns:
      A Timestep containing current state, action and reward.
    """

    def _action_strings(
        action_request: smart_control_building_pb2.ActionRequest,
    ) -> Sequence[str]:
      """Create a list of actions from an ActionRequest for logging."""
      action_strings = []
      for single_action_request in action_request.single_action_requests:
        action_string = "%s %s: %3.2f" % (
            single_action_request.device_id,
            single_action_request.setpoint_name,
            single_action_request.continuous_value,
        )
        action_strings.append(action_string)
      return action_strings

    if self._episode_ended:
      return self.reset()

    t0 = time.time()
    reward_value = 0.0
    observation = None
    last_timestamp = self.current_simulation_timestamp

    # Reformat actions if necessary.
    action = self._format_action(action, self._action_names)

    # Convert the action from normalized to native values.
    action_request = self._create_action_request(action)

    try:
      # Send the action request to the building.
      action_response = self.building.request_action(action_request)

    except RuntimeError as err:
      # If the building rejects the request, create an action response
      # indicating that the request was rejected.
      action_accepted = False

      action_response = _apply_action_response(
          action_request,
          response_timestamp=self.current_simulation_timestamp,
          action_response_type=smart_control_building_pb2.SingleActionResponse.ActionResponseType.REJECTED_NOT_ENABLED_OR_AVAILABLE,
          additional_info=str(err),
      )
      logging.exception(
          "Action REJECTED at %s: %s.",
          self.current_simulation_timestamp,
          ", ".join(_action_strings(action_request)),
      )

    else:
      action_accepted = all_actions_accepted(action_response)

    if self._metrics_writer and action_response is not None:
      self._metrics_writer.write_action_response(
          action_response, self.current_simulation_timestamp
      )

    last_timestamp = self.current_simulation_timestamp

    self.building.wait_time()

    observation = self._get_observation()

    # We need to signal to the Actor that action was rejected and not to
    # append this observation/action request to the trajectory.
    # Since TimeStep cannot be extended and it is checked for NaNs,
    # we apply -inf as a reward to indicate the rejection.
    # This requires a specialized Actor extension class to handle the
    # rejection.
    reward_value = self._get_reward()
    if not action_accepted:
      reward_value = ACTION_REJECTION_REWARD

    # Exit when the episode has ended and return terminal step information.
    # We still need to get the final observation to add to the transition.
    self._episode_ended = self._has_episode_ended(last_timestamp)

    self._episode_cumulative_reward += reward_value

    t1 = time.time()
    episode_dt = t1 - self._episode_start_time
    step_dt = t1 - t0

    if self._episode_ended:
      logging.info(
          "%s: Terminating episode=%d step=%d current_time=%s step_reward=%4.2f"
          " cumulative_reward=%5.2f episode_time=%5.2fs step_time=%3.2fs",
          self._label,
          self._episode_count,
          self._step_count,
          self.building.current_timestamp,
          reward_value,
          self._episode_cumulative_reward,
          episode_dt,
          step_dt,
      )
      termination = ts.termination(observation, reward_value)
      return termination

    else:
      transition = ts.transition(
          observation, reward_value, self.discount_factor
      )

      if self._step_count % 100 == 0:
        logging.info(
            (
                "%s: episode=%d step=%d current_time=%s step_reward=%4.2f"
                " cumulative_reward=%5.2f episode_time=%5.2fs step_time=%3.2fs"
            ),
            self._label,
            self._episode_count,
            self._step_count,
            self.building.current_timestamp,
            reward_value,
            self._episode_cumulative_reward,
            episode_dt,
            step_dt,
        )

      self._step_count += 1
      self._global_step_count += 1
      return transition

  def render(self, mode: str = "rgb_array") -> Optional[types.NestedArray]:
    raise NotImplementedError("Rendering not supported yet.")

  def _has_episode_ended(self, last_timestamp: pd.Timestamp) -> bool:
    """Flag to indicate the episode has ended."""

    return self._step_count >= self._num_timesteps_in_episode


def _apply_action_response(
    action_request: smart_control_building_pb2.ActionRequest,
    action_response_type: smart_control_building_pb2.SingleActionResponse.ActionResponseType,
    response_timestamp: pd.Timestamp,
    additional_info: Optional[str] = None,
) -> smart_control_building_pb2.ActionResponse:
  """Returns an ActionResponse if not passed by the Building."""

  single_action_responses = [
      _apply_single_action_response(
          single_action_request, action_response_type, additional_info
      )
      for single_action_request in action_request.single_action_requests
  ]
  return smart_control_building_pb2.ActionResponse(
      timestamp=conversion_utils.pandas_to_proto_timestamp(response_timestamp),
      request=action_request,
      single_action_responses=single_action_responses,
  )


def _apply_single_action_response(
    single_action_request: smart_control_building_pb2.SingleActionRequest,
    action_response_type: smart_control_building_pb2.SingleActionResponse.ActionResponseType,
    additional_info: Optional[str] = None,
) -> smart_control_building_pb2.SingleActionResponse:
  """Creates a SingleActionResponse if not passed by the Building."""
  return smart_control_building_pb2.SingleActionResponse(
      request=single_action_request,
      response_type=action_response_type,
      additional_info=additional_info,
  )
