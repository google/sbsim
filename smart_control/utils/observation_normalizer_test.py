"""Tests for observation_normalizer.

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

from absl.testing import absltest
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.utils import observation_normalizer

_DEVICES = ['a', 'a', 'b', 'b', 'c', 'c']
_MEASUREMENTS = [
    'zone_air_temperature',
    'damper_percentage',
    'supply_air_flowrate',
    'supply_water_setpoint',
    'heating_percentage',
    'discharge_air_flowrate',
]
_NATIVE_VALUES = [294.0, 0.6, 0.25, 310.0, 60.0, -0.5]
_NORMALIZED_VALUES = [-0.32, 0.20000005, -0.125, 0.0, 0.0, -0.5]


class ObservationNormalizerTest(absltest.TestCase):

  def test_denormalize(self):
    normalized_response = self._get_test_observation_response(
        _DEVICES, _MEASUREMENTS, _NORMALIZED_VALUES
    )
    expected_native_response = self._get_test_observation_response(
        _DEVICES, _MEASUREMENTS, _NATIVE_VALUES
    )
    normalization_constants = self._get_normalization_constants()
    normalizer = observation_normalizer.StandardScoreObservationNormalizer(
        normalization_constants
    )
    native_response = normalizer.denormalize(normalized_response)

    self.assertEqual(expected_native_response, native_response)

  def test_normalize(self):

    expected_normalized_response = self._get_test_observation_response(
        _DEVICES, _MEASUREMENTS, _NORMALIZED_VALUES
    )
    normalization_constants = self._get_normalization_constants()
    native_response = self._get_test_observation_response(
        _DEVICES, _MEASUREMENTS, _NATIVE_VALUES
    )
    normalizer = observation_normalizer.StandardScoreObservationNormalizer(
        normalization_constants
    )
    normalized_response = normalizer.normalize(native_response)

    self.assertEqual(expected_normalized_response, normalized_response)

  def _get_test_observation_response(self, devices, measurements, values):
    assert len(devices) == len(measurements) == len(values)
    observation_request = smart_control_building_pb2.ObservationRequest()
    single_observation_responses = []
    for device, measurement, value in zip(devices, measurements, values):
      single_observation_request = (
          smart_control_building_pb2.SingleObservationRequest(
              device_id=device, measurement_name=measurement
          )
      )
      observation_request.single_observation_requests.append(
          single_observation_request
      )
      single_observation_responses.append(
          smart_control_building_pb2.SingleObservationResponse(
              single_observation_request=single_observation_request,
              continuous_value=value,
          )
      )
    response = smart_control_building_pb2.ObservationResponse(
        single_observation_responses=single_observation_responses,
        request=observation_request,
    )
    return response

  def _get_normalization_constants(self):
    return {
        'zone_air_temperature': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='temperature', sample_mean=310.0, sample_variance=50 * 50
            )
        ),
        'supply_water_setpoint': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='supply_water_setpoint',
                sample_mean=310.0,
                sample_variance=50 * 50,
            )
        ),
        'discharge_air_flowrate': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='discharge_air_flowrate',
                sample_mean=0.5,
                sample_variance=4.0,
            )
        ),
        'supply_air_flowrate': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='discharge_air_flowrate',
                sample_mean=0.5,
                sample_variance=4.0,
            )
        ),
        'differential_pressure': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='differential_pressure',
                sample_mean=20000.0,
                sample_variance=100000.0,
            )
        ),
        'damper_percentage': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='damper_percentage', sample_mean=0.5, sample_variance=0.25
            )
        ),
        'heating_percentage': (
            smart_control_normalization_pb2.ContinuousVariableInfo(
                id='heating_percentage', sample_mean=60.0, sample_variance=0.0
            )
        ),
        'request_count': smart_control_normalization_pb2.ContinuousVariableInfo(
            id='request_count', sample_mean=9, sample_variance=25.0
        ),
    }


if __name__ == '__main__':
  absltest.main()
