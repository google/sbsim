"""Unit test for run_command_predictor.

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

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import conversion_utils
from smart_control.utils import reader_lib
from smart_control.utils import run_command_predictor


# Create 2 dimensions, with ON and OFF examples
_TEST_SAMPLE_SIZE_ON = 100
_TEST_SAMPLE_SIZE_OFF = 100
_DEVICE_ID = 'X'
_TEST_DIM_1 = [293.0] * _TEST_SAMPLE_SIZE_ON + [280.0] * _TEST_SAMPLE_SIZE_OFF
_TEST_DIM_2 = [100.0] * _TEST_SAMPLE_SIZE_ON + [0.0] * _TEST_SAMPLE_SIZE_OFF
_RUN_COMMAND = [1.0] * _TEST_SAMPLE_SIZE_ON + [-1.0] * _TEST_SAMPLE_SIZE_OFF
_SUPERVISOR_RUN_COMMAND = 'supervisor_run_command'


def _create_single_action_response(setpoint_name, continuous_value):
  single_action_request = smart_control_building_pb2.SingleActionRequest(
      device_id=_DEVICE_ID,
      setpoint_name=setpoint_name,
      continuous_value=continuous_value,
  )
  single_action_response = smart_control_building_pb2.SingleActionResponse(
      request=single_action_request,
      response_type=smart_control_building_pb2.SingleActionResponse.ACCEPTED,
  )
  return single_action_response


class TestReader(reader_lib.BaseReader):
  """Test reader for histogram reducer returning just observation responses."""

  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads observation_responses from endpoint bounded by start & end time."""
    raise NotImplementedError()

  def _collate_action_response(self, single_action_responses, timestamp):
    single_action_requests = [resp.request for resp in single_action_responses]
    action_request = smart_control_building_pb2.ActionRequest(
        single_action_requests=single_action_requests,
        timestamp=conversion_utils.pandas_to_proto_timestamp(timestamp),
    )
    return smart_control_building_pb2.ActionResponse(
        request=action_request,
        single_action_responses=single_action_responses,
        timestamp=conversion_utils.pandas_to_proto_timestamp(timestamp),
    )

  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads action_responses from endpoint bounded by start and end time."""

    action_responses = []
    timestamp = pd.Timestamp('2023-04-09 10:00:00')
    for i in range(_TEST_SAMPLE_SIZE_ON + _TEST_SAMPLE_SIZE_OFF):
      single_action_responses = []

      respd1 = _create_single_action_response(
          'dim_1',
          _TEST_DIM_1[i],
      )
      respd2 = _create_single_action_response(
          'dim_2',
          _TEST_DIM_2[i],
      )
      resprc = _create_single_action_response(
          _SUPERVISOR_RUN_COMMAND, _RUN_COMMAND[i]
      )

      single_action_responses.append(resprc)
      single_action_responses.append(respd1)
      single_action_responses.append(respd2)

      action_response = self._collate_action_response(
          single_action_responses, timestamp
      )
      action_responses.append(action_response)
      timestamp += pd.Timedelta(seconds=300)
    return action_responses

  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward infos from endpoint bounded by start and end time."""
    raise NotImplementedError()

  def read_reward_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward responses from endpoint bounded by start and end time."""
    raise NotImplementedError()

  def read_normalization_info(
      self,
  ):
    """Reads variable normalization info from RecordIO."""
    raise NotImplementedError()

  def read_zone_infos(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Reads the zone infos for the Building."""
    raise NotImplementedError()

  def read_device_infos(
      self,
  ) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Reads the device infos for the Building."""
    raise NotImplementedError()


class RunCommandPredictorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('ON', 292.0, 99.0, 1.0), ('OFF', 272.0, 8.0, -1.0)
  )
  def test_run_command_predict(self, d1, d2, expected):
    learning_algo = run_command_predictor.RandomForestLearningAlgorithm()
    reader = TestReader()
    predictor = run_command_predictor.RunCommandPredictor(
        _DEVICE_ID, reader, learning_algo, 'UTC'
    )

    action_request = smart_control_building_pb2.ActionRequest(
        single_action_requests=[
            _create_single_action_response('dim_1', d1).request,
            _create_single_action_response('dim_2', d2).request,
        ]
    )
    expected_single_action_request = (
        smart_control_building_pb2.SingleActionRequest(
            device_id=_DEVICE_ID,
            continuous_value=expected,
            setpoint_name=_SUPERVISOR_RUN_COMMAND,
        )
    )

    single_action_response = predictor.predict(action_request)
    self.assertEqual(single_action_response, expected_single_action_request)

  def test_get_action_timeseries(self):
    reader = TestReader()
    action_responses = reader.read_action_responses(
        pd.Timestamp.min, pd.Timestamp.max
    )

    action_timeseries = run_command_predictor.get_action_timeseries(
        action_responses, 'UTC'
    )
    self.assertIsInstance(action_timeseries, pd.DataFrame)
    self.assertEqual(
        list(action_timeseries['setpoint']),
        ['supervisor_run_command', 'dim_1', 'dim_2'] * 200,
    )
    self.assertEqual(list(action_timeseries['device']), ['X'] * 600)
    self.assertEqual(
        list(action_timeseries['acknowledgement']),
        ['ACCEPTED'] * 600,
    )
    self.assertEqual(
        list(action_timeseries['value']),
        [1.0, 293.0, 100] * 100 + [-1.0, 280.0, 0] * 100,
    )


if __name__ == '__main__':
  absltest.main()
