"""Tests for the base building."""

from typing import Sequence

from absl.testing import absltest
import pandas as pd
from smart_buildings.smart_control.models import base_building
from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.proto import smart_control_reward_pb2


class _MockBuilding(base_building.BaseBuilding):
  # consider moving the environment_test_utils.SimpleBuilding class here
  # and updating all references
  def __init__(self, devices, zones):
    self._devices = devices
    self._zones = zones

  @property
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    pass

  def request_observations(self, observation_request):
    pass

  def request_observations_within_time_interval(
      self, observation_request, start_timestamp, end_timestamp
  ):
    pass

  def request_action(self, action_request):
    pass

  def wait_time(self):
    pass

  def reset(self):
    pass

  @property
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    return self._devices

  @property
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    return self._zones

  @property
  def current_timestamp(self) -> pd.Timestamp:
    pass

  def render(self, path: str) -> None:
    pass

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    pass

  @property
  def num_occupants(self) -> int:
    pass

  @property
  def time_step_sec(self) -> float:
    pass


class BaseBuildingTest(absltest.TestCase):

  def test_devices_df(self):
    device = smart_control_building_pb2.DeviceInfo(
        device_id='d1',
        namespace='ns1',
        code='c1',
        zone_id='z1',
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.VAV,
    )
    device.observable_fields['obs1'] = (
        smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
    )
    device.action_fields['act1'] = (
        smart_control_building_pb2.DeviceInfo.ValueType.VALUE_BINARY
    )
    building = _MockBuilding(devices=[device], zones=[])

    expected_df = pd.DataFrame([{
        'device_id': 'd1',
        'namespace': 'ns1',
        'code': 'c1',
        'zone_id': 'z1',
        'device_type': 'VAV',
        'observable_fields': ['obs1'],
        'action_fields': ['act1'],
        'observable_field_types': {'obs1': 'VALUE_CONTINUOUS'},
        'action_field_types': {'act1': 'VALUE_BINARY'},
    }])

    pd.testing.assert_frame_equal(building.devices_df, expected_df)

  def test_zones_df(self):
    zone = smart_control_building_pb2.ZoneInfo(
        zone_id='z1',
        building_id='b1',
        zone_description='desc1',
        area=100.0,
        devices=['d1', 'd2'],
        zone_type=smart_control_building_pb2.ZoneInfo.ZoneType.ROOM,
        floor=1,
    )
    building = _MockBuilding(devices=[], zones=[zone])

    expected_df = pd.DataFrame([{
        'zone_id': 'z1',
        'building_id': 'b1',
        'zone_description': 'desc1',
        'area': 100.0,
        'devices': ['d1', 'd2'],
        'zone_type': 'ROOM',
        'floor': 1,
    }])

    pd.testing.assert_frame_equal(building.zones_df, expected_df)


if __name__ == '__main__':
  absltest.main()
