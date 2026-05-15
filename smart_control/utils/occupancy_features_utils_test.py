import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils import occupancy_features_utils
from smart_buildings.smart_control.utils.proto_parsers import reward_info_parser

from google3.net.proto2.contrib.pyutil import compare


def _create_zone_info(
    zone_id: str, floor: int, devices: list[str] | None = None
):
  """Helper to create an actual ZoneInfo protobuf with optional devices."""
  return smart_control_building_pb2.ZoneInfo(
      zone_id=zone_id, floor=floor, devices=devices or []
  )


class TestGetZoneConditionsHistogram(
    compare.Proto2Assertions, unittest.TestCase
):

  def setUp(self):
    super().setUp()
    # Define 6 temperature bins: [290, 292, 294, 296, 298, 300]
    # Indices: 0: 290, 1: 292, 2: 294, 3: 296, 4: 298, 5: 300
    self.temperature_bins = [290.0, 292.0, 294.0, 296.0, 298.0, 300.0]

  def _create_zone_reward_info(
      self, temp: float, heat_set: float, cool_set: float, occupancy: float
  ):
    """Helper to create a populated RewardInfo.ZoneRewardInfo proto."""
    zone_info = occupancy_features_utils.smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
        zone_air_temperature=temp,
        heating_setpoint_temperature=heat_set,
        cooling_setpoint_temperature=cool_set,
        average_occupancy=occupancy,
        # Fields from proto not used by histogram logic, added for structural
        # parity
        air_flow_rate_setpoint=0.5,
        air_flow_rate=0.5,
    )
    return zone_info

  def _create_single_obs_request(self, device_id: str, measurement_name: str):
    """Helper to create a SingleObservationRequest."""
    req = smart_control_building_pb2.SingleObservationRequest(
        device_id=device_id, measurement_name=measurement_name
    )
    return req

  # ====================================================================
  # get_zone_conditions_histogram Tests
  # ====================================================================

  def test_get_zone_conditions_histogram_standard_behavior(self):
    """Tests standard aggregation across multiple floors and out-of-bounds temps."""
    zones = [
        _create_zone_info("zone_1", floor=1),
        _create_zone_info("zone_2", floor=1),
        _create_zone_info("zone_3", floor=2),
    ]
    reward_info = smart_control_reward_pb2.RewardInfo()

    # Zone 1: Temp 292 (Too cold). Occ: 5 -> Exposed: -5
    reward_info.zone_reward_infos["zone_1"].CopyFrom(
        self._create_zone_reward_info(292.1, 294.0, 296.0, 5.0)
    )
    # Zone 2: Temp 296 (Comfort). Occ: 10 -> Exposed: 0
    reward_info.zone_reward_infos["zone_2"].CopyFrom(
        self._create_zone_reward_info(296.2, 294.0, 296.0, 10.0)
    )
    # Zone 3: Temp 298 (Too hot). Occ: 3 -> Exposed: 3
    reward_info.zone_reward_infos["zone_3"].CopyFrom(
        self._create_zone_reward_info(297.9, 294.0, 296.0, 3.0)
    )
    # Missing Zone: Temp 294 (Comfort). Occ: 2 -> Exposed: 0
    reward_info.zone_reward_infos["zone_missing"].CopyFrom(
        self._create_zone_reward_info(293.8, 294.0, 296.0, 2.0)
    )

    df = occupancy_features_utils.get_zone_conditions_histogram(
        reward_info, self.temperature_bins, zones
    )

    # Build Expected DataFrame using the proper _OCCUPANCY_AT_FLOOR_PREFIX
    expected_data = {
        "occupancy_count": [0, 5, 2, 10, 3, 0],
        "setpoint_mask": [-1, -1, 0, 0, 1, 1],
        "setpoint_range": ["-", "-", "+", "+", "-", "-"],
        "exposed_count": [0, -5, 0, 0, 3, 0],
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}0": [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1": [
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.0,
        ],
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2": [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    }
    expected_df = pd.DataFrame(expected_data, index=self.temperature_bins)

    pdt.assert_frame_equal(df, expected_df, check_dtype=False)

  def test_empty_reward_info(self):
    """Tests behavior when no telemetry data is provided."""
    reward_info = smart_control_reward_pb2.RewardInfo()
    zones = [_create_zone_info("zone_1", floor=1)]

    df = occupancy_features_utils.get_zone_conditions_histogram(
        reward_info, self.temperature_bins, zones
    )

    expected_data = {
        "occupancy_count": [0, 0, 0, 0, 0, 0],
        "setpoint_mask": [0, 0, 0, 0, 0, 0],
        "setpoint_range": ["-", "-", "-", "-", "-", "-"],
        "exposed_count": [0, 0, 0, 0, 0, 0],
    }
    expected_df = pd.DataFrame(expected_data, index=self.temperature_bins)
    pdt.assert_frame_equal(df, expected_df, check_dtype=False)

  def test_wide_setpoint_range(self):
    """Tests that the global setpoint mask adapts to the widest zone requirements."""
    zones = [
        _create_zone_info("zone_1", floor=1),
        _create_zone_info("zone_2", floor=1),
    ]
    reward_info = smart_control_reward_pb2.RewardInfo()

    reward_info.zone_reward_infos["zone_1"].CopyFrom(
        self._create_zone_reward_info(294.0, 294.0, 296.0, 0.0)
    )
    reward_info.zone_reward_infos["zone_2"].CopyFrom(
        self._create_zone_reward_info(294.0, 290.0, 300.0, 0.0)
    )

    df = occupancy_features_utils.get_zone_conditions_histogram(
        reward_info, self.temperature_bins, zones
    )

    expected_setpoint_range = pd.Series(
        ["+"] * 6, index=self.temperature_bins, name="setpoint_range"
    )
    pdt.assert_series_equal(df["setpoint_range"], expected_setpoint_range)

    expected_setpoint_mask = pd.Series(
        [0] * 6, index=self.temperature_bins, name="setpoint_mask"
    )
    pdt.assert_series_equal(df["setpoint_mask"], expected_setpoint_mask)

  # ====================================================================
  # Utility Mapping Tests
  # ====================================================================

  def test_append_floor_to_measurement_name(self):
    name = "zone_air_temp"
    floor = 3
    expected = f"zone_air_temp{reward_info_parser.FLOOR_PREFIX}3"
    result = occupancy_features_utils.append_floor_to_measurement_name(
        name, floor
    )
    self.assertEqual(result, expected)

  def test_get_zone_info_mapping(self):
    zone1 = _create_zone_info("zone_1", floor=1, devices=["dev_A", "dev_B"])
    zone2 = _create_zone_info("zone_2", floor=2, devices=["dev_C"])
    mapping = occupancy_features_utils.get_zone_info_mapping([zone1, zone2])

    expected_mapping = {"dev_A": zone1, "dev_B": zone1, "dev_C": zone2}
    self.assertDictEqual(mapping, expected_mapping)

  # ====================================================================
  # Occupancy Feature Extraction Tests
  # ====================================================================

  def test_get_occupancy_features_from_zone_conditions_histogram(self):
    index = [290.0, 292.0]
    data = {
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1": [10.0, 0.0],
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2": [5.0, 20.0],
        "setpoint_mask": [-1.0, 0.0],
    }
    df = pd.DataFrame(data, index=index)

    names, values = (
        occupancy_features_utils.get_occupancy_features_from_zone_conditions_histogram(
            df
        )
    )

    expected_names = [
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1_h290.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1_h292.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2_h290.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2_h292.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1_exposed_h290.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}1_exposed_h292.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2_exposed_h290.0",
        f"{reward_info_parser.OCCUPANCY_AT_FLOOR_PREFIX}2_exposed_h292.0",
    ]
    expected_values = [
        10.0,
        0.0,
        5.0,
        20.0,
        -10.0,
        0.0,
        -5.0,
        0.0,
    ]

    np.testing.assert_array_equal(names, expected_names)
    np.testing.assert_array_almost_equal(values, expected_values)

  # ====================================================================
  # Observation Response Modification Tests
  # ====================================================================

  def test_assign_floor_to_observation_response(self):
    zone_infos = [
        _create_zone_info("z1", floor=1, devices=["dev_1"]),
        _create_zone_info("z2", floor=2, devices=["dev_2"]),
    ]

    obs_response_in = smart_control_building_pb2.ObservationResponse()

    req1 = self._create_single_obs_request(
        "dev_1", reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR
    )
    req2 = self._create_single_obs_request("dev_2", "water_temperature")
    req3 = self._create_single_obs_request(
        "dev_missing", reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR
    )

    obs_response_in.request.single_observation_requests.extend(
        [req1, req2, req3]
    )

    resp1 = obs_response_in.single_observation_responses.add()
    resp1.single_observation_request.CopyFrom(req1)
    resp2 = obs_response_in.single_observation_responses.add()
    resp2.single_observation_request.CopyFrom(req2)

    obs_response_out = (
        occupancy_features_utils.assign_floor_to_observation_response(
            obs_response_in, zone_infos
        )
    )

    # Build the expected ObservationResponse
    expected_obs_response = smart_control_building_pb2.ObservationResponse()
    req1_expected = self._create_single_obs_request(
        "dev_1",
        f"{reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR}{reward_info_parser.FLOOR_PREFIX}1",
    )
    req2_expected = self._create_single_obs_request(
        "dev_2", "water_temperature"
    )
    req3_expected = self._create_single_obs_request(
        "dev_missing", reward_info_parser.ZONE_AIR_TEMPERATURE_SENSOR
    )
    expected_obs_response.request.single_observation_requests.extend(
        [req1_expected, req2_expected, req3_expected]
    )

    resp1_expected = expected_obs_response.single_observation_responses.add()
    resp1_expected.single_observation_request.CopyFrom(req1_expected)
    resp2_expected = expected_obs_response.single_observation_responses.add()
    resp2_expected.single_observation_request.CopyFrom(req2_expected)

    self.assertProto2Equal(obs_response_out, expected_obs_response)


if __name__ == "__main__":
  unittest.main()
