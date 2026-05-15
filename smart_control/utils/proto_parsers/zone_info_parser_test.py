from absl.testing import absltest
from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.utils.proto_parsers import zone_info_parser


class ZoneInfoParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.zone_info = building_pb2.ZoneInfo(
        zone_id='z1',
        building_id='b1',
        zone_description='desc1',
        area=100.0,
        devices=['d1', 'd2'],
        zone_type=building_pb2.ZoneInfo.ZoneType.ROOM,
        floor=1,
    )
    self.parser = zone_info_parser.ZoneInfoParser(self.zone_info)

  def test_initialization(self):
    self.assertIsInstance(self.parser, zone_info_parser.ZoneInfoParser)

  def test_zone_id(self):
    self.assertEqual(self.parser.zone_id, 'z1')

  def test_building_id(self):
    self.assertEqual(self.parser.building_id, 'b1')

  def test_description(self):
    self.assertEqual(self.parser.description, 'desc1')

  def test_area(self):
    self.assertEqual(self.parser.area, 100.0)

  def test_device_ids(self):
    self.assertEqual(list(self.parser.device_ids), ['d1', 'd2'])

  def test_zone_type_id(self):
    self.assertEqual(
        self.parser.zone_type_id, int(building_pb2.ZoneInfo.ZoneType.ROOM)
    )

  def test_zone_type(self):
    self.assertEqual(self.parser.zone_type, 'ROOM')

  def test_floor(self):
    self.assertEqual(self.parser.floor, 1)

  def test_as_dict(self):
    expected_dict = {
        'building_id': 'b1',
        'zone_id': 'z1',
        'zone_type_id': 1,
        'zone_type': 'ROOM',
        'description': 'desc1',
        'area': 100.0,
        'floor': 1,
        'device_ids': ['d1', 'd2'],
    }
    self.assertEqual(self.parser.as_dict, expected_dict)

  def test_devices_df(self):
    expected_records = [
        {'building_id': 'b1', 'zone_id': 'z1', 'device_id': 'd1'},
        {'building_id': 'b1', 'zone_id': 'z1', 'device_id': 'd2'},
    ]
    self.assertCountEqual(
        self.parser.devices_df.to_dict('records'), expected_records
    )


if __name__ == '__main__':
  absltest.main()
