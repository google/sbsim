"""Tests for smart_device.

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
import pandas as pd
from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device


class SmartDeviceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    class Heater(smart_device.SmartDevice):

      def __init__(self):
        observable_fields = {
            'obs_temp': smart_device.AttributeInfo('temperature', float),
            'obs_heat_setting': smart_device.AttributeInfo(
                'heat_setting', float
            ),
            'obs_seconds_active': smart_device.AttributeInfo('seconds', int),
            'obs_bad': smart_device.AttributeInfo('fake_field', float),
        }

        action_fields = {
            'act_heat_setting': smart_device.AttributeInfo(
                'heat_setting', float
            ),
            'act_bad': smart_device.AttributeInfo('fake_field', float),
        }

        super().__init__(
            observable_fields,
            action_fields,
            smart_control_building_pb2.DeviceInfo.DeviceType.OTHER,
            device_id='heater_id',
            zone_id='zone_id',
        )

        self.temperature = 282.5
        self.heat_setting = 300.0
        self.seconds = 7

    self.heater_class = Heater

  def test_device_id(self):
    heater = self.heater_class()

    self.assertEqual(heater.device_id(), 'heater_id')

  def test_zone_id(self):
    heater = self.heater_class()

    self.assertEqual(heater.zone_id(), 'zone_id')

  def test_observable_field_names(self):
    heater = self.heater_class()

    self.assertSameElements(
        heater.observable_field_names(),
        ['obs_temp', 'obs_heat_setting', 'obs_seconds_active', 'obs_bad'],
    )

  def test_action_field_names(self):
    heater = self.heater_class()

    self.assertSameElements(
        heater.action_field_names(), ['act_heat_setting', 'act_bad']
    )

  def test_observable_type(self):
    heater = self.heater_class()

    self.assertEqual(heater.get_observable_type('obs_temp'), float)

  def test_action_type(self):
    heater = self.heater_class()

    self.assertEqual(heater.get_action_type('act_heat_setting'), float)

  def test_get_observation(self):
    heater = self.heater_class()

    observed_temp = heater.get_observation(
        'obs_temp', pd.Timestamp('2021-09-01 10:00')
    )

    self.assertEqual(observed_temp, 282.5)

  def test_set_action(self):
    heater = self.heater_class()

    heat_setting = 275.0

    heater.set_action(
        'act_heat_setting', heat_setting, pd.Timestamp('2021-09-01 10:00')
    )

    self.assertEqual(heater.heat_setting, heat_setting)

  def test_non_declared_observable_raises_attribute_error(self):
    heater = self.heater_class()

    with self.assertRaises(AttributeError):
      heater.get_observation('pressure', pd.Timestamp('2021-09-01 10:00'))

  def test_declared_observable_not_an_attribute_raises_attribute_error(self):
    heater = self.heater_class()

    with self.assertRaises(AttributeError):
      heater.get_observation('obs_bad', pd.Timestamp('2021-09-01 10:00'))

  def test_non_declared_action_raises_attribute_error(self):
    heater = self.heater_class()

    with self.assertRaises(AttributeError):
      heater.set_action('pressure', 100.0, pd.Timestamp('2021-09-01 10:00'))

  def test_declared_action_not_an_attribute_raises_attribute_error(self):
    heater = self.heater_class()

    with self.assertRaises(AttributeError):
      heater.set_action('act_bad', 999, pd.Timestamp('2021-09-01 10:00'))

  def test_declared_action_incorrect_value_type_raises_value_error(self):
    heater = self.heater_class()

    with self.assertRaises(ValueError):
      heater.set_action(
          'act_heat_setting', 'hello', pd.Timestamp('2021-09-01 10:00')
      )

  def test_device_type(self):
    heater = self.heater_class()

    device_type = heater.device_type()

    self.assertEqual(
        device_type, smart_control_building_pb2.DeviceInfo.DeviceType.OTHER
    )


if __name__ == '__main__':
  absltest.main()
