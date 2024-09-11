"""Unit Tests for HistogramReducer.

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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import histogram_reducer
from smart_control.utils import reader_lib


class TestReader(reader_lib.BaseReader):
  """Test reader for histogram reducer returning just observation responses."""

  def __init__(
      self,
      observation_responses: Sequence[
          smart_control_building_pb2.ObservationResponse
      ],
  ):
    self._observation_responses = observation_responses

  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads observation_responses from endpoint bounded by start & end time."""
    return self._observation_responses

  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads action_responses from endpoint bounded by start and end time."""
    raise NotImplementedError()

  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads reward infos from endpoint bounded by start and end time."""

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


class HistogramReducerTest(parameterized.TestCase):

  def _get_native_reduced_sequence(self):
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-04-21 00:00:00'),
            pd.Timestamp('2023-04-21 00:05:00'),
            pd.Timestamp('2023-04-21 00:10:00'),
        ],
        ('A', 'm2'): np.array([11.0, 12.0, 13.0], dtype=np.float64),
        ('m0', 'h_70.00'): np.array([2.0] * 3, dtype=np.float32),
        ('m0', 'h_72.00'): np.array([2.0] * 3, dtype=np.float32),
        ('m0', 'h_74.00'): np.array([2.0] * 3, dtype=np.float32),
        ('m0', 'h_76.00'): np.array([0.0] * 3, dtype=np.float32),
        ('m1', 'h_120.00'): np.array([2.0] * 3, dtype=np.float32),
        ('m1', 'h_130.00'): np.array([2.0] * 3, dtype=np.float32),
        ('m1', 'h_140.00'): np.array([1.0] * 3, dtype=np.float32),
        ('m1', 'h_150.00'): np.array([1.0] * 3, dtype=np.float32),
    })

  def _get_normalized_reduced_sequence(self):
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-04-21 00:00:00'),
            pd.Timestamp('2023-04-21 00:05:00'),
            pd.Timestamp('2023-04-21 00:10:00'),
        ],
        ('A', 'm2'): np.array([11.0, 12.0, 13.0], dtype=np.float64),
        ('m0', 'h_70.00'): np.array([0.33333334] * 3, dtype=np.float32),
        ('m0', 'h_72.00'): np.array([0.33333334] * 3, dtype=np.float32),
        ('m0', 'h_74.00'): np.array([0.33333334] * 3, dtype=np.float32),
        ('m0', 'h_76.00'): np.array([0.0] * 3, dtype=np.float32),
        ('m1', 'h_120.00'): np.array([0.33333334] * 3, dtype=np.float32),
        ('m1', 'h_130.00'): np.array([0.33333334] * 3, dtype=np.float32),
        ('m1', 'h_140.00'): np.array([0.16666667] * 3, dtype=np.float32),
        ('m1', 'h_150.00'): np.array([0.16666667] * 3, dtype=np.float32),
    })

  def _get_test_observation_sequence(self):
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-04-21 00:00:00'),
            pd.Timestamp('2023-04-21 00:05:00'),
            pd.Timestamp('2023-04-21 00:10:00'),
        ],
        ('A', 'm0'): [70.0] * 3,
        ('B', 'm0'): [72.0] * 3,
        ('C', 'm0'): [74.0] * 3,
        ('D', 'm0'): [75.0] * 3,
        ('E', 'm0'): [71.0] * 3,
        ('F', 'm0'): [72.0] * 3,
        ('A', 'm1'): [120.0] * 3,
        ('B', 'm1'): [130.0] * 3,
        ('C', 'm1'): [140.0] * 3,
        ('D', 'm1'): [150.0] * 3,
        ('E', 'm1'): [125.0] * 3,
        ('F', 'm1'): [135.0] * 3,
        ('A', 'm2'): [11.0, 12.0, 13.0],
    })

  def _get_expanded_sequence(self):
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-04-21 00:00:00'),
            pd.Timestamp('2023-04-21 00:05:00'),
            pd.Timestamp('2023-04-21 00:10:00'),
        ],
        ('A', 'm0'): [70.0] * 3,
        ('B', 'm0'): [72.0] * 3,
        ('C', 'm0'): [74.0] * 3,
        ('D', 'm0'): [74.0] * 3,
        ('E', 'm0'): [70.0] * 3,
        ('F', 'm0'): [72.0] * 3,
        ('A', 'm1'): [120.0] * 3,
        ('B', 'm1'): [130.0] * 3,
        ('C', 'm1'): [140.0] * 3,
        ('D', 'm1'): [150.0] * 3,
        ('E', 'm1'): [120.0] * 3,
        ('F', 'm1'): [130.0] * 3,
        ('A', 'm2'): [11.0, 12.0, 13.0],
    })

  @parameterized.named_parameters(
      (
          'native values',
          False,
      ),
      (
          'normalized values',
          True,
      ),
  )
  def test_histogram_reducer_reduce(self, is_normalized):
    devices = ['A', 'B', 'C', 'D', 'E', 'F', 'A', 'B', 'C', 'D', 'E', 'F', 'A']
    measurements = ['m0'] * 6 + ['m1'] * 6 + ['m2']
    values = [
        70,
        72.0,
        74.0,
        75.0,
        71.0,
        72.0,
        120.0,
        130.0,
        140.0,
        150.0,
        125.3,
        135.1,
        10.0,
    ]
    observation_response = self._get_test_observation_response(
        devices, measurements, values
    )

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=[
            ('m0', np.arange(70.0, 78.0, 2.0)),
            ('m1', np.arange(120.0, 160.0, 10.0)),
        ],
        reader=TestReader([observation_response]),
        normalize_reduce=is_normalized,
    )

    observation_sequence = self._get_test_observation_sequence()

    if is_normalized:
      expected_reduced_sequence = self._get_normalized_reduced_sequence()
    else:
      expected_reduced_sequence = self._get_native_reduced_sequence()

    rs = hr.reduce(observation_sequence)

    logging.info(rs.reduced_sequence.to_string())
    logging.info(expected_reduced_sequence.to_string())
    pd.testing.assert_frame_equal(
        rs.reduced_sequence, expected_reduced_sequence
    )

  def test_get_clipped_histogram_clipped(self):
    test_array = np.array([
        68.5,
        70.19999694824219,
        66.5,
        69.19999694824219,
        71.19999694824219,
        70.5,
        70.0,
        69.5,
        68.69999694824219,
        71.5,
        65.30000305175781,
        71.9000015258789,
        68.0,
        66.0999984741211,
        71.9000015258789,
        69.69999694824219,
        68.5,
        65.30000305175781,
        66.0,
        63.5,
        95.80000305175781,
    ])
    expected_array = np.array([
        0,
        0,
        0,
        1,
        0,
        2,
        3,
        0,
        4,
        3,
        3,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ])

    hist_array = histogram_reducer.get_clipped_histogram(
        test_array, bins=np.arange(60, 90), clip=True
    )
    self.assertEqual(list(hist_array), list(expected_array))

  def test_get_clipped_histogram_not_clipped(self):
    test_array = np.array([
        68.5,
        70.19999694824219,
        66.5,
        69.19999694824219,
        71.19999694824219,
        70.5,
        70.0,
        69.5,
        68.69999694824219,
        71.5,
        65.30000305175781,
        71.9000015258789,
        58.0,
        66.0999984741211,
        71.9000015258789,
        69.69999694824219,
        68.5,
        65.30000305175781,
        66.0,
        63.5,
        95.80000305175781,
    ])
    expected_array = np.array([
        0,
        0,
        0,
        1,
        0,
        2,
        3,
        0,
        3,
        3,
        3,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])

    hist_array = histogram_reducer.get_clipped_histogram(
        test_array, bins=np.arange(60, 90), clip=False
    )

    self.assertEqual(list(hist_array), list(expected_array))

  def test_histogram_reducer_expand(self):
    devices = ['A', 'B', 'C', 'D', 'E', 'F', 'A', 'B', 'C', 'D', 'E', 'F', 'A']
    measurements = ['m0'] * 6 + ['m1'] * 6 + ['m2']
    values = [
        70,
        72.0,
        74.0,
        75.0,
        71.0,
        72.0,
        120.0,
        130.0,
        140.0,
        150.0,
        125.3,
        135.1,
        10.0,
    ]

    observation_response = self._get_test_observation_response(
        devices, measurements, values
    )

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=[
            ('m0', np.arange(70.0, 78.0, 2.0)),
            ('m1', np.arange(120.0, 160.0, 10.0)),
        ],
        reader=TestReader([observation_response]),
    )

    observation_sequence = self._get_test_observation_sequence()

    rs = hr.reduce(observation_sequence)
    expanded_sequence = rs.expand()
    expected_expanded_sequence = self._get_expanded_sequence()

    pd.testing.assert_frame_equal(
        expanded_sequence[expected_expanded_sequence.columns],
        expected_expanded_sequence,
    )

  def test_histogram_reducer_expand_with_update(self):
    devices = ['A', 'B', 'C', 'D', 'E', 'F', 'A', 'B', 'C', 'D', 'E', 'F', 'A']
    measurements = ['m0'] * 6 + ['m1'] * 6 + ['m2']
    values_original = [
        70,
        72.0,
        74.0,
        75.0,
        71.0,
        72.0,
        120.0,
        130.0,
        140.0,
        150.0,
        125.3,
        135.1,
        10.0,
    ]

    observation_response_original = self._get_test_observation_response(
        devices, measurements, values_original
    )

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=[
            ('m0', np.arange(70.0, 78.0, 2.0)),
            ('m1', np.arange(120.0, 160.0, 10.0)),
        ],
        reader=TestReader([observation_response_original]),
    )

    observation_sequence = self._get_test_observation_sequence()

    rs = hr.reduce(observation_sequence)

    reduced_sequence = rs.reduced_sequence

    # Override the values with updetd predictions:
    predicted_measurements = [15.0, 16.0, 17.0]
    reduced_sequence[('A', 'm2')] = predicted_measurements

    expanded_sequence = rs.expand()
    expected_expanded_sequence = self._get_expanded_sequence()
    expected_expanded_sequence[('A', 'm2')] = predicted_measurements

    pd.testing.assert_frame_equal(
        expanded_sequence[expected_expanded_sequence.columns],
        expected_expanded_sequence,
    )

  def test_reassign_nodes(self):
    current_assignment = [
        [],
        [],
        ['A'],
        ['C', 'F', 'I'],
        [],
        ['D', 'G'],
        ['B', 'H'],
        ['E'],
        ['J'],
        [],
    ]
    node_counts_next = [5, 0, 0, 0, 0, 0, 0, 0, 0, 5]

    expected_next_assignment = [
        ['A', 'C', 'F', 'I', 'D'],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        ['J', 'E', 'B', 'H', 'G'],
    ]

    next_assignment = histogram_reducer.reassign_nodes(
        current_assignment, node_counts_next
    )

    self.assertEqual(next_assignment, expected_next_assignment)

  def test_reassign_bin_count_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            "Number of bins don't match. node_counts_current 10 and"
            ' histogram_counts_next 11'
        ),
    ):
      current_assignment = [
          [],
          [],
          ['A'],
          ['C', 'F', 'I'],
          [],
          ['D', 'G'],
          ['B', 'H'],
          ['E'],
          ['J'],
          [],
      ]
      node_counts_next = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]

      _ = histogram_reducer.reassign_nodes(current_assignment, node_counts_next)

  def test_reassign_node_count_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        (
            'Assignment has 11 nodes, but histogram_counts_next has 10 nodes.'
            ' The counts must match.'
        ),
    ):
      current_assignment = [
          [],
          [],
          ['A'],
          ['C', 'F', 'I'],
          [],
          ['D', 'G'],
          ['B', 'H'],
          ['E'],
          ['J', 'K'],
          [],
      ]
      node_counts_next = [5, 0, 0, 0, 0, 0, 0, 0, 0, 5]

      _ = histogram_reducer.reassign_nodes(current_assignment, node_counts_next)

  def test_assign_devices_to_bins(self):
    bins = np.arange(70.0, 75.0)

    observation_response = self._get_test_observation_response(
        ['A', 'B', 'C', 'D'], ['m0', 'm0', 'm0', 'm1'], [68, 71, 99.0, 0.1]
    )

    assignment = histogram_reducer.assign_devices_to_bins(
        'm0', bins, observation_response
    )

    # Since D is associated with m1, and this assignment is for m0,
    # D should not appear in the expectec assignment.
    expected_assignmant = [['A'], ['B'], [], [], ['C']]

    self.assertEqual(assignment, expected_assignmant)

  def _get_test_observation_response(
      self, device_ids, measurement_names, values
  ):
    if (len(device_ids) != len(measurement_names)) or (
        len(device_ids) != len(values)
    ):
      raise ValueError(
          'Lengths of device_ids, measurement_names, and values must be the'
          ' same.'
      )

    single_observation_responses = []
    single_observation_requests = []

    for device_id, measurement_name, value in zip(
        device_ids, measurement_names, values
    ):
      single_request = smart_control_building_pb2.SingleObservationRequest(
          device_id=device_id, measurement_name=measurement_name
      )
      single_response = smart_control_building_pb2.SingleObservationResponse(
          single_observation_request=single_request,
          observation_valid=True,
          continuous_value=value,
      )
      single_observation_responses.append(single_response)
      single_observation_requests.append(single_request)
    request = smart_control_building_pb2.ObservationRequest(
        single_observation_requests=single_observation_requests
    )
    return smart_control_building_pb2.ObservationResponse(
        request=request,
        single_observation_responses=single_observation_responses,
    )


if __name__ == '__main__':
  absltest.main()
