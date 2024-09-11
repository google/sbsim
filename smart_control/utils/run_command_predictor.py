"""Run command predictor predicts whether the device is On or Off.

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

Some RL Agents, like Soft-Actor-Critic, are designed for continuous action
spaces. However, SM control requires a hybrid action space (continuous pressures
and temperatures, and a discrete on/off run command.

Multiple attempts have failed to train an agent to manage the hybrid action
space effectively, but given pressure and temperature setting, it is easy to
train a classifier to estimate the run command for both air conditioners and
hot water systems. This CL adds in the RunCommand predictor that predicts if
the device is on or off based on its temp and pressure features.

We have trained and evaluated linear, SVM, Random Forest, and dense
NN architectures, and found that the Random Forest classifiers performed
best and are very efficiently trained.
"""

import abc
import collections
from typing import Sequence
from absl import logging
import gin
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils
from smart_control.utils import reader_lib


_SEED = 191
_TEST_PROPORTION = 0.1
_LABELS = ['On', 'Off']
_SUPERVISOR_RUN_COMMAND = 'supervisor_run_command'


class BaseRunCommandPredictor(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def predict(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.SingleActionRequest:
    """Returns a Single ActionRequest indicating the device RunCommand state."""


class BaseLearningAlgorithm(metaclass=abc.ABCMeta):
  """Base class for a Supervised Learning Algorithm."""

  @abc.abstractmethod
  def fit(
      self,
      x_train: np.ndarray,
      y_train: np.ndarray,
  ) -> None:
    """Trains a model from labels y and examples X."""

  @abc.abstractmethod
  def predict(self, x_pred: np.ndarray) -> np.ndarray:
    """Predicts on an unlabeled sample, X."""


def _action_request_to_setpoint_features(
    action_request: smart_control_building_pb2.ActionRequest,
    device_id: str,
    setpoint_names: Sequence[str],
) -> pd.DataFrame:
  """Converts an ActionRequest into a DataFrame for prediction."""
  feature_map = collections.defaultdict(list)

  for single_action_request in action_request.single_action_requests:
    if (
        single_action_request.device_id == device_id
        and single_action_request.setpoint_name in setpoint_names
    ):
      feature_map[single_action_request.setpoint_name].append(
          single_action_request.continuous_value
      )

  # Explicitly specify setpoint_names to ensure the order is retained, since
  # single_action_requests can arrive in any arbitrary order.
  return pd.DataFrame(feature_map)[setpoint_names]


def _train_eval(
    learning_algo: BaseLearningAlgorithm,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: Sequence[str],
):
  """Trains and evaluates the generic model."""

  learning_algo.fit(x_train, y_train)
  y_pred = learning_algo.predict(x_test)
  logging.info(classification_report(y_test, y_pred, target_names=labels))


def _split_data(actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
  """Splits the dataframe into features and labels."""
  y_labels = np.array((actions[_SUPERVISOR_RUN_COMMAND] == 1).astype(int))
  x_vals = np.array(actions.drop(columns=[_SUPERVISOR_RUN_COMMAND]))
  return x_vals, y_labels


def _get_one_step_action_timeseries(
    action_response: smart_control_building_pb2.ActionResponse,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
  """Converts a single ActionResponse into a dataframe."""
  devices = []
  setpoints = []
  values = []
  acknowledgements = []
  for single_action_response in action_response.single_action_responses:
    if '_id_' in single_action_response.request.device_id:
      device_id = single_action_response.request.device_id.split('_id_')[0]
    else:
      device_id = single_action_response.request.device_id
    devices.append(device_id)
    setpoints.append(single_action_response.request.setpoint_name)
    values.append(float(single_action_response.request.continuous_value))
    acknowledgements.append(
        smart_control_building_pb2.SingleActionResponse.ActionResponseType.Name(
            single_action_response.response_type
        )
    )
  return pd.DataFrame({
      'timestamp': [timestamp] * len(action_response.single_action_responses),
      'device': devices,
      'setpoint': setpoints,
      'value': values,
      'acknowledgement': acknowledgements,
  })


def get_action_timeseries(
    action_responses: Sequence[smart_control_building_pb2.ActionResponse],
    time_zone: str,
) -> pd.DataFrame:
  """Converts the ActionResponses into a dataframe."""
  action_responses_dfs = []

  for action_response in action_responses:
    timestamp = conversion_utils.proto_to_pandas_timestamp(
        action_response.timestamp
    ).tz_convert(time_zone)

    actions_response_df = _get_one_step_action_timeseries(
        action_response,
        timestamp,
    )
    action_responses_dfs.append(actions_response_df)
  return pd.concat(action_responses_dfs)


@gin.configurable
class RandomForestLearningAlgorithm(BaseLearningAlgorithm):
  """Wrapper class for Random Forest."""

  def __init__(
      self,
      n_estimators: int = 100,
      criterion: str = 'gini',
      max_depth: int = 5,
      min_samples_split: int = 2,
  ):
    self._model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )

  def fit(
      self,
      x_train: np.ndarray,
      y_train: np.ndarray,
  ) -> None:
    self._model.fit(x_train, y_train)

  def predict(self, x_pred: np.ndarray) -> np.ndarray:
    return self._model.predict(x_pred)


@gin.configurable
class RunCommandPredictor(BaseRunCommandPredictor):
  """Predicts the RunCommand from other continuous actions."""

  def __init__(
      self,
      device_id: str,
      reader: reader_lib.BaseReader,
      learning_algo: BaseLearningAlgorithm,
      time_zone: str,
  ) -> None:
    logging.info('Training the run command predictor for device %s', device_id)
    self._device_id = device_id

    self._learning_algo = learning_algo

    action_responses = reader.read_action_responses(
        start_time=pd.Timestamp.min, end_time=pd.Timestamp.max
    )
    action_timeseries = get_action_timeseries(action_responses, time_zone)
    device_action_timeseries = action_timeseries[
        action_timeseries['device'] == device_id
    ]
    device_actions = pd.pivot_table(
        device_action_timeseries,
        values='value',
        columns=['setpoint'],
        index=['timestamp'],
        aggfunc=np.sum,
    )
    self._column_order = device_actions.columns.tolist()
    self._column_order.remove(_SUPERVISOR_RUN_COMMAND)
    x, y = _split_data(device_actions)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=_TEST_PROPORTION, random_state=_SEED
    )
    _train_eval(
        self._learning_algo,
        x_train,
        y_train,
        x_test,
        y_test,
        _LABELS,
    )

  def predict(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.SingleActionRequest:
    df_input = _action_request_to_setpoint_features(
        action_request=action_request,
        device_id=self._device_id,
        setpoint_names=self._column_order,
    )
    x_pred = df_input.to_numpy()

    # Adjust the range [0,1] to [-1,1].
    logging.info(x_pred)
    continuous_value = self._learning_algo.predict(x_pred)[0] * 2 - 1

    return smart_control_building_pb2.SingleActionRequest(
        device_id=self._device_id,
        setpoint_name=_SUPERVISOR_RUN_COMMAND,
        continuous_value=continuous_value,
    )
