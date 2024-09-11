"""Normalizes observations by standardized shifting and scaling.

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

import math
from typing import Callable, Mapping, NewType
import gin
from smart_control.models import base_normalizer
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2

FieldNameKeyword = NewType('FieldNameKeyword', str)
FieldName = NewType('FieldName', str)


@gin.configurable
class StandardScoreObservationNormalizer(
    base_normalizer.BaseObservationNormalizer
):
  """Normalizes and denormalizes ObservationResponses and ActionResponses.

  Normalization is simply a shift by mean and scale by sqrt(variance).
  Native variable: variable value in original form, before normlization.
  Normalized variable: variable shifted and scaled, after normalization.

  Attributes:
    normalization_constants: Mapping from field name to the normalization
      constants.

  Raises:
    ValueError if the field name is not matched in the normalization_constants.
  """

  def __init__(
      self,
      normalization_constants: Mapping[
          FieldNameKeyword,
          smart_control_normalization_pb2.ContinuousVariableInfo,
      ],
  ):
    self._normalization_constants = normalization_constants

  def _get_normalization_constants(
      self, field_name: FieldName
  ) -> smart_control_normalization_pb2.ContinuousVariableInfo:
    """Returns normalization data for exact match or keyword-contain match."""
    if field_name in self._normalization_constants:
      return self._normalization_constants[field_name]
    else:
      return smart_control_normalization_pb2.ContinuousVariableInfo(
          sample_mean=0.0, sample_variance=1.0
      )

  def _normalize_one(self, field_name: FieldName, value: float) -> float:
    """Shifts and scales a native value based on its field name.

    There are multiple messages that should be converted in the same way. For
    example, all temperatures are in Kelvin, so the same normalization should
    apply to all fields with temperature (e.g., zone_air_temperature sensor,
    exhaust_air_temperature_sensor should be normalized the same way.) For
    this reason, we apply a keyword match of the field_name rather than and
    exact match.

    Args:
      field_name: name of the field to be normalized.
      value: native value to be shifted and scaled

    Returns:
      Normalized value w/o units.
    """

    normalization_constants = self._get_normalization_constants(field_name)
    if normalization_constants.sample_variance > 0.0:
      return (value - normalization_constants.sample_mean) / math.sqrt(
          normalization_constants.sample_variance
      )
    return 0.0

  def _denormalize_one(self, field_name: FieldName, value: float) -> float:
    """Converts a normalized variable back into its native value."""
    normalization_constants = self._get_normalization_constants(field_name)
    return (
        value * math.sqrt(normalization_constants.sample_variance)
    ) + normalization_constants.sample_mean

  def normalize(
      self, native: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Shifts/scales a ObservationResponse from native to normalized."""

    return self._transform_observation(native, self._normalize_one)

  def denormalize(
      self, normalized: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Scales/Shifts a ObservationResponse from normalized to native."""

    return self._transform_observation(normalized, self._denormalize_one)

  def _transform_observation(
      self,
      obs_in: smart_control_building_pb2.ObservationResponse,
      transform_func: Callable[[FieldName, float], float],
  ) -> smart_control_building_pb2.ObservationResponse:
    """Applies a (de-)normalization transformation to an ObservationResponse.

    Args:
      obs_in: input ObservationResponse
      transform_func: normallization or denormlization function

    Returns:
      an ObservationResponse with the same fields, but transformed values.
    """

    obs_out = smart_control_building_pb2.ObservationResponse()
    obs_out.CopyFrom(obs_in)

    for single_observation_response in obs_out.single_observation_responses:
      field_name = (
          single_observation_response.single_observation_request.measurement_name
      )
      value = single_observation_response.continuous_value
      single_observation_response.continuous_value = transform_func(
          FieldName(field_name), value
      )
    return obs_out
