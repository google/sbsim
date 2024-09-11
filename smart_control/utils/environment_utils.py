from typing import Sequence
import pandas as pd
import gin
from smart_control.proto import smart_control_normalization_pb2, smart_control_building_pb2
from smart_control.utils import bounded_action_normalizer

@gin.configurable
def to_timestamp(date_str: str) -> pd.Timestamp:
  """Utilty macro for gin config."""
  return pd.Timestamp(date_str)


@gin.configurable
def local_time(time_str: str) -> pd.Timedelta:
  """Utilty macro for gin config."""
  return pd.Timedelta(time_str)


@gin.configurable
def enumerate_zones(
    n_building_x: int, n_building_y: int
) -> Sequence[tuple[int, int]]:
  """Utilty macro for gin config."""
  zone_coordinates = []
  for x in range(n_building_x):
    for y in range(n_building_y):
      zone_coordinates.append((x, y))
  return zone_coordinates


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


@gin.configurable
def get_zones_from_config(
    configuration_path: str,
) -> Sequence[smart_control_building_pb2.ZoneInfo]:
  """Loads up the zones as a gin macro."""
  with gin.unlock_config():
    reader = reader_lib_google.RecordIoReader(input_dir=configuration_path)
    zone_infos = reader.read_zone_infos()
    return zone_infos


@gin.configurable
def get_devices_from_config(
    configuration_path: str,
) -> Sequence[smart_control_building_pb2.DeviceInfo]:
  """Loads up HVAC devices as a gin macro."""
  with gin.unlock_config():
    reader = reader_lib_google.RecordIoReader(input_dir=configuration_path)
    device_infos = reader.read_device_infos()
    return device_infos
