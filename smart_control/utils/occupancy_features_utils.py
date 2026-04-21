"""Utility functions for processing smart building occupancy and observation data.

This module provides tools to bridge raw building telemetry (handled via
Protocol Buffers) and downstream machine learning models or reinforcement
learning agents. It focuses on two primary areas of data transformation:

1. Observation Context Enrichment:
   Modifies incoming building observation responses by injecting spatial context
   (floor numbers) into measurement names. For example, it appends a suffix like
   '@floor3' to generic 'zone_air_temperature_sensor' readings. This allows
   downstream spatial-aware agents to distinguish identical sensor types across
   different levels of the building.

2. Occupancy Feature Extraction:
   Flattens multi-dimensional pandas DataFrames (representing building occupancy
   histograms) into 1D feature arrays (numpy) suitable for direct input into ML
   pipelines. It extracts both standard occupancy metrics (people per floor per
   temperature bin) and "exposed" occupancy metrics (people explicitly
   experiencing temperatures outside of defined comfort setpoints).

Typical usage involves calling `assign_floor_to_observation_response` on
incoming telemetry streams and
`get_occupancy_features_from_zone_conditions_histogram` on
aggregated reward/state representations.
"""

from collections.abc import Sequence

from absl import logging
import numpy as np
import pandas as pd
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils.proto_parsers import reward_info_parser

OCCUPANCY_AT_FLOOR_PREFIX = reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX
FLOOR_PREFIX = reward_info_parser.FLOOR_PREFIX

# Default temperature bins (in Kelvin) used to group zone conditions.
OCCUPANCY_TEMPERATURE_HISTOGRAM_BINS = [
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    301,
    302,
]


def get_zone_conditions_histogram(
    reward_info: smart_control_reward_pb2.RewardInfo,
    temperature_bins: Sequence[float],
    zones: Sequence[smart_control_building_pb2.ZoneInfo],
) -> pd.DataFrame:
  """Generates a histogram DataFrame of building zone conditions over temp bins."""
  return reward_info_parser.RewardInfoParser(
      reward_info=reward_info,
      zone_temp_bins=temperature_bins,
      temp_unit="K",
  ).get_zone_conditions_histogram_by_floor(zones)


def append_floor_to_measurement_name(measurement_name: str, floor: int) -> str:
  """Appends the floor suffix to a given measurement name.

  Args:
      measurement_name: The original name of the measurement (e.g.,
        "zone_air_temp").
      floor: The integer floor number where the measurement device is located.

  Returns:
      A new string with the floor appended (e.g., "zone_air_temp@floor3").
  """
  return f"{measurement_name}{reward_info_parser.FLOOR_PREFIX}{floor}"


def get_zone_info_mapping(
    zone_infos: Sequence[smart_control_building_pb2.ZoneInfo],
) -> dict[str, smart_control_building_pb2.ZoneInfo]:
  """Creates a flat mapping from individual device IDs to their parent ZoneInfo.

  A single zone can contain multiple devices. To quickly look up which zone
  (and therefore which floor) a specific device belongs to, we flatten the
  hierarchy into a direct dictionary lookup.

  Args:
      zone_infos: A sequence of ZoneInfo protobuf objects representing the
        building.

  Returns:
      A dictionary where the key is a string `device_id` and the value is the
      `ZoneInfo` object that contains that device.
  """
  device_zone_mapping = {}
  for zone_info in zone_infos:
    for device in zone_info.devices:
      device_zone_mapping[device] = zone_info
  return device_zone_mapping


def get_occupancy_features_from_zone_conditions_histogram(
    zone_conditions_histogram: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
  """Extracts 1D occupancy feature arrays from a 2D conditions histogram.

  This function flattens the matrix of floor-by-temperature occupancies into
  two sets of 1D arrays suitable for ML model inputs:
  1. Standard occupancy features (occupancy at floor X at temperature Y).
  2. Exposed occupancy features (occupants explicitly outside comfort bounds).

  Args:
      zone_conditions_histogram: A DataFrame where the index represents
        temperature bins and columns represent specific metrics (including floor
        occupancies and setpoint masks).

  Returns:
      A tuple containing two numpy arrays:
      - feature_names (1D array of strings): The generated names for each
      feature.
      - feature_values (1D array of np.float32): The corresponding numerical
      values.
  """
  # Isolate only the columns that contain floor-level occupancy data.
  floor_cols = [
      col
      for col in zone_conditions_histogram.columns
      if col.startswith(reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX)
  ]

  names = []
  values = []

  # --- Pass 1: Extract Standard Occupancy Features ---
  # For every floor, iterate through the temperature bins and record the
  # occupancy.
  for col in floor_cols:
    for temp, row in zone_conditions_histogram.iterrows():
      names.append(f"{col}_h{temp:.1f}")
      values.append(float(row[col]))

  # --- Pass 2: Extract Exposed Occupancy Features ---
  # Exposed occupancy multiplies the actual occupancy by the setpoint mask.
  # If the mask is 0 (comfortable), the exposed value is 0.
  # If the mask is -1 or 1 (too cold/hot), the exposed value reflects the
  # occupants affected.
  for col in floor_cols:
    for temp, row in zone_conditions_histogram.iterrows():
      names.append(f"{col}_exposed_h{temp:.1f}")
      exposed_val = float(row[col]) * float(row["setpoint_mask"])
      values.append(exposed_val)

  # Cast the numerical values to float32, which is standard for TensorFlow/ML
  # pipelines.
  return np.array(names), np.array(values, dtype=np.float32)


def _assign_floor_to_single_request(
    single_request: smart_control_building_pb2.SingleObservationRequest,
    device_zone_mapping: dict[str, smart_control_building_pb2.ZoneInfo],
) -> None:
  """Modifies a SingleObservationRequest in-place by appending its floor number.

  This is a private helper function to keep the main observation response
  processing logic clean and flat.

  Args:
      single_request: The protobuf request object to modify in-place.
      device_zone_mapping: A dictionary mapping device IDs to their parent
        zones.
  """
  # If the device isn't mapped to a zone, we can't determine its floor.
  if single_request.device_id not in device_zone_mapping:
    logging.warning(
        "A device with zone temperature '%s' is not assigned to a zone",
        single_request.device_id,
    )
    return

  # Extract the floor and update the measurement name directly on the protobuf.
  floor = device_zone_mapping[single_request.device_id].floor
  single_request.measurement_name = append_floor_to_measurement_name(
      single_request.measurement_name, floor
  )


def assign_floor_to_observation_response(
    observation_response_in: smart_control_building_pb2.ObservationResponse,
    zone_infos: Sequence[smart_control_building_pb2.ZoneInfo],
) -> smart_control_building_pb2.ObservationResponse:
  """Injects floor identifiers into the measurement names of an obs response.

  Because the downstream reinforcement learning agent often needs spatial
  context (which floor a temperature reading came from), this function
  intercepts the response payload and appends the floor number to specific
  sensor names.

  Args:
      observation_response_in: The original, unmodified observation response
        from the building.
      zone_infos: A sequence of building zones used to determine which device is
        on which floor.

  Returns:
      A completely new ObservationResponse protobuf with modified measurement
      names.
      The original input object is left unmutated.
  """
  # Generate the lookup table to quickly find the floor for any given device ID.
  device_zone_mapping = get_zone_info_mapping(zone_infos)

  # Create a copy of the input response to avoid mutating the original data.
  observation_response_out = smart_control_building_pb2.ObservationResponse()
  observation_response_out.CopyFrom(observation_response_in)

  # --- Phase 1: Update the Parent Request ---
  # The response object echoes back the original request. We must update the
  # measurement names in this echoed request.
  for (
      single_req
  ) in observation_response_out.request.single_observation_requests:
    if (
        single_req.measurement_name
        == reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR
    ):
      _assign_floor_to_single_request(single_req, device_zone_mapping)

  # --- Phase 2: Update the Individual Responses ---
  # Now we update the actual array of responses returned by the building.
  for single_resp in observation_response_out.single_observation_responses:
    # Check the nested request inside the response to see if it's a temperature
    # sensor.
    if (
        single_resp.single_observation_request.measurement_name
        == reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR
    ):
      _assign_floor_to_single_request(
          single_resp.single_observation_request, device_zone_mapping
      )

  return observation_response_out
