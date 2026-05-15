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
    super().__init__(zones=zones)
    self._devices = devices

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
        observable_fields={
            'obs1': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            )
        },
        action_fields={
            'act1': smart_control_building_pb2.DeviceInfo.ValueType.VALUE_BINARY
        },
    )
    building = _MockBuilding(devices=[device], zones=[])

    expected_records = [{
        'device_id': 'd1',
        'namespace': 'ns1',
        'code': 'c1',
        'zone_id': 'z1',
        'device_type_id': 4,
        'device_type': 'VAV',
    }]

    self.assertEqual(building.devices_df.to_dict('records'), expected_records)

  def test_device_fields_dfs(self):
    device = smart_control_building_pb2.DeviceInfo(
        device_id='d1',
        namespace='ns1',
        code='c1',
        zone_id='z1',
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.VAV,
        observable_fields={
            'obs1': (
                smart_control_building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS
            )
        },
        action_fields={
            'act1': smart_control_building_pb2.DeviceInfo.ValueType.VALUE_BINARY
        },
    )
    building = _MockBuilding(devices=[device], zones=[])

    expected_observable_fields = [{
        'device_id': 'd1',
        'field_name': 'obs1',
        'field_type_id': 1,
        'field_type': 'VALUE_CONTINUOUS',
    }]
    expected_action_fields = [{
        'device_id': 'd1',
        'field_name': 'act1',
        'field_type_id': 4,
        'field_type': 'VALUE_BINARY',
    }]
    expected_fields = [
        {
            'device_id': 'd1',
            'field_name': 'act1',
            'field_type_id': 4,
            'field_type': 'VALUE_BINARY',
            'is_observable': False,
            'is_actionable': True,
        },
        {
            'device_id': 'd1',
            'field_name': 'obs1',
            'field_type_id': 1,
            'field_type': 'VALUE_CONTINUOUS',
            'is_observable': True,
            'is_actionable': False,
        },
    ]

    self.assertEqual(
        building.observable_fields_df.to_dict('records'),
        expected_observable_fields
    )
    self.assertEqual(
        building.action_fields_df.to_dict('records'), expected_action_fields
    )
    self.assertEqual(
        building.fields_df.to_dict('records'), expected_fields
    )

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

    expected_records = [{
        'building_id': 'b1',
        'zone_id': 'z1',
        'zone_type_id': 1,
        'zone_type': 'ROOM',
        'description': 'desc1',
        'area': 100.0,
        'floor': 1,
        'device_ids': ['d1', 'd2'],
    }]

    self.assertEqual(building.zones_df.to_dict('records'), expected_records)

  def test_zone_floor_mappings(self):
    building = _MockBuilding(
        devices=[],
        zones=[
            smart_control_building_pb2.ZoneInfo(zone_id='z1', floor=5),
            smart_control_building_pb2.ZoneInfo(zone_id='z2', floor=10),
        ],
    )
    self.assertEqual(building.zones_df['floor'].tolist(), [5, 10])

  def test_json_metadata(self):
    devices = [smart_control_building_pb2.DeviceInfo(device_id='device_1')]
    zones = [
        smart_control_building_pb2.ZoneInfo(zone_id='zone_1'),
        smart_control_building_pb2.ZoneInfo(zone_id='zone_2'),
    ]
    building = _MockBuilding(devices=devices, zones=zones)
    expected_metadata = {
        'n_devices': 1,
        'n_zones': 2,
        'device_ids': ['device_1'],
        'zone_ids': ['zone_1', 'zone_2'],
    }
    self.assertEqual(building.json_metadata, expected_metadata)


if __name__ == '__main__':
  absltest.main()
