"""Reinforcement learning environment utility functions."""

import gin
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_normalization_pb2
from smart_buildings.smart_control.utils import bounded_action_normalizer


@gin.configurable
def to_timestamp(date_str: str) -> pd.Timestamp:
  """Utilty macro for gin config."""
  return pd.Timestamp(date_str)


@gin.configurable
def local_time(time_str: str) -> pd.Timedelta:
  """Utilty macro for gin config."""
  return pd.Timedelta(time_str)


@gin.configurable
def set_observation_normalization_constants(
    field_id: str, sample_mean: float, sample_variance: float
) -> smart_control_normalization_pb2.ContinuousVariableInfo:
  return smart_control_normalization_pb2.ContinuousVariableInfo(
      id=field_id, sample_mean=sample_mean, sample_variance=sample_variance
  )


@gin.configurable
def set_action_normalization_constants(
    min_native_value,
    max_native_value,
    min_normalized_value,
    max_normalized_value,
) -> bounded_action_normalizer.BoundedActionNormalizer:
  return bounded_action_normalizer.BoundedActionNormalizer(
      min_native_value,
      max_native_value,
      min_normalized_value,
      max_normalized_value,
  )
