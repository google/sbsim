from absl.testing import absltest

from smart_buildings.smart_control.proto import smart_control_building_pb2 as building_pb2
from smart_buildings.smart_control.utils.proto_parsers import device_info_parser


class DeviceInfoParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device_info = building_pb2.DeviceInfo(
        device_id='d1',
        namespace='ns1',
        code='c1',
        zone_id='z1',
        device_type=building_pb2.DeviceInfo.DeviceType.VAV,
        observable_fields={
            'obs_act': building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS,
            'obs_only': building_pb2.DeviceInfo.ValueType.VALUE_INTEGER,
        },
        action_fields={
            'obs_act': building_pb2.DeviceInfo.ValueType.VALUE_CONTINUOUS,
            'act_only': building_pb2.DeviceInfo.ValueType.VALUE_BINARY,
        },
    )
    self.parser = device_info_parser.DeviceInfoParser(self.device_info)
    self.expected_observable_fields = [
        device_info_parser.ObservableField(
            device_id='d1',
            field_name='obs_act',
            field_type_id=building_pb2.DeviceInfo.VALUE_CONTINUOUS,
        ),
        device_info_parser.ObservableField(
            device_id='d1',
            field_name='obs_only',
            field_type_id=building_pb2.DeviceInfo.VALUE_INTEGER,
        ),
    ]
    self.expected_action_fields = [
        device_info_parser.ActionField(
            device_id='d1',
            field_name='act_only',
            field_type_id=building_pb2.DeviceInfo.VALUE_BINARY,
        ),
        device_info_parser.ActionField(
            device_id='d1',
            field_name='obs_act',
            field_type_id=building_pb2.DeviceInfo.VALUE_CONTINUOUS,
        ),
    ]
    self.expected_fields = [
        device_info_parser.DeviceField(
            device_id='d1',
            field_name='act_only',
            field_type_id=building_pb2.DeviceInfo.VALUE_BINARY,
            is_observable=False,
            is_actionable=True,
        ),
        device_info_parser.DeviceField(
            device_id='d1',
            field_name='obs_act',
            field_type_id=building_pb2.DeviceInfo.VALUE_CONTINUOUS,
            is_observable=True,
            is_actionable=True,
        ),
        device_info_parser.DeviceField(
            device_id='d1',
            field_name='obs_only',
            field_type_id=building_pb2.DeviceInfo.VALUE_INTEGER,
            is_observable=True,
            is_actionable=False,
        ),
    ]

  def test_initialization(self):
    self.assertIsInstance(self.parser, device_info_parser.DeviceInfoParser)

  def test_device_id(self):
    self.assertEqual(self.parser.device_id, 'd1')

  def test_namespace(self):
    self.assertEqual(self.parser.namespace, 'ns1')

  def test_code(self):
    self.assertEqual(self.parser.code, 'c1')

  def test_zone_id(self):
    self.assertEqual(self.parser.zone_id, 'z1')

  def test_device_type_id(self):
    self.assertEqual(
        self.parser.device_type_id, int(building_pb2.DeviceInfo.DeviceType.VAV)
    )

  def test_device_type(self):
    self.assertEqual(self.parser.device_type, 'VAV')

  def test_as_dict(self):
    expected_dict = {
        'device_id': 'd1',
        'device_type_id': building_pb2.DeviceInfo.DeviceType.VAV,
        'device_type': 'VAV',
        'namespace': 'ns1',
        'code': 'c1',
        'zone_id': 'z1',
    }
    self.assertEqual(self.parser.as_dict, expected_dict)

  # DEVICE FIELD TESTS

  def test_observable_fields(self):
    self.assertCountEqual(
        self.parser.observable_fields, self.expected_observable_fields
    )

  def test_observable_fields_df(self):
    self.assertCountEqual(
        self.parser.observable_fields_df.to_dict('records'),
        [r.as_dict for r in self.expected_observable_fields],
    )

  def test_action_fields(self):
    self.assertCountEqual(
        self.parser.action_fields, self.expected_action_fields
    )

  def test_action_fields_df(self):
    self.assertCountEqual(
        self.parser.action_fields_df.to_dict('records'),
        [r.as_dict for r in self.expected_action_fields],
    )

  def test_fields(self):
    self.assertCountEqual(self.parser.fields, self.expected_fields)

  def test_fields_df(self):
    self.assertCountEqual(
        self.parser.fields_df.to_dict('records'),
        [r.as_dict for r in self.expected_fields],
    )


if __name__ == '__main__':
  absltest.main()
