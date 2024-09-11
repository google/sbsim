"""Tests for conversion_utils.

Copyright 2022 Google LLC

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
from absl.testing import parameterized
import numpy as np
import pandas as pd
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import conversion_utils


class ConversionUtilsTest(parameterized.TestCase):

  def test_pandas_proto_timestamp_conversion(self):
    start_timestamp = pd.Timestamp('2021-05-03 10:00:02+0')
    proto_ts = conversion_utils.pandas_to_proto_timestamp(start_timestamp)
    pandas_ts = conversion_utils.proto_to_pandas_timestamp(proto_ts)

    self.assertLessEqual(
        np.abs((start_timestamp - pandas_ts).total_seconds()), 1e-9
    )

  def test_is_workday_workday(self):
    self.assertTrue(conversion_utils.is_work_day(pd.Timestamp('2021-05-06')))

  def test_is_workday_weekend(self):
    self.assertFalse(conversion_utils.is_work_day(pd.Timestamp('2021-05-08')))

  def test_is_workday_holiday(self):
    self.assertFalse(conversion_utils.is_work_day(pd.Timestamp('2022-03-06')))

  def test_zone_id_conversion(self):
    coords = (0, 0)
    self.assertEqual(
        coords,
        conversion_utils.zone_id_to_coordinates(
            conversion_utils.zone_coordinates_to_id(coords)
        ),
    )

  def test_zone_id_bad_conversion(self):
    with self.assertRaises(ValueError):
      conversion_utils.zone_id_to_coordinates('zone_id(x, y)')

  def test_normalize_hod(self):
    self.assertEqual(conversion_utils.normalize_hod(0), -1.0)
    self.assertEqual(conversion_utils.normalize_hod(23), 1.0)

  def test_normalize_dow(self):
    self.assertEqual(conversion_utils.normalize_dow(0), -1.0)
    self.assertEqual(conversion_utils.normalize_dow(6), 1.0)

  @parameterized.parameters(
      (pd.Timestamp('2021-09-27 10:00:00-08:00'), 0),
      (pd.Timestamp('2021-10-10 18:25:00+02:00'), 6.0 / 7.0 * 2 * np.pi),
      (pd.Timestamp('2021-10-01 00:05:00-5:00'), 4.0 / 7.0 * 2.0 * np.pi),
  )
  def test_get_radian_dow(self, current_time, expected_radian):
    self.assertEqual(
        conversion_utils.get_radian_time(
            current_time, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
        ),
        expected_radian,
    )

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_kelvin_to_fahrenheit(self, fahrenheit, kelvin):
    self.assertAlmostEqual(
        fahrenheit, conversion_utils.kelvin_to_fahrenheit(kelvin), places=2
    )

  def test_kelvin_to_fahrenheit_invalid(self):
    with self.assertRaises(ValueError):
      _ = conversion_utils.kelvin_to_fahrenheit(0.0)

  @parameterized.parameters(
      (32.0, 273.15), (-10.0, 249.817), (70.0, 294.261), (110.0, 316.483)
  )
  def test_fahrenheit_to_kelvin(self, fahrenheit, kelvin):
    self.assertAlmostEqual(
        kelvin, conversion_utils.fahrenheit_to_kelvin(fahrenheit), places=2
    )

  def test_fahrenheit_to_kelvin_invalid(self):
    with self.assertRaises(ValueError):
      _ = conversion_utils.fahrenheit_to_kelvin(-495.67)

  @parameterized.parameters(
      (pd.Timestamp('2021-09-27 00:00:00+01'), 0),
      (pd.Timestamp('2021-10-10 23:59:59-07'), 6.28311258512742),
      (pd.Timestamp('2021-09-30 12:00:00+3'), np.pi),
  )
  def test_get_radian_hod(self, current_time, expected_radian):
    self.assertEqual(
        conversion_utils.get_radian_time(
            current_time, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
        ),
        expected_radian,
    )

  def test_get_reward_info_energy_use(self):
    dt = 300
    start_time = pd.Timestamp('2021-05-03 12:13:00-5')
    end_time = start_time + pd.Timedelta(dt, unit='second')
    to_kwh = dt / 3600.0 / 1000.0
    reward_info = smart_control_reward_pb2.RewardInfo()
    reward_info.start_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(start_time)
    )
    reward_info.end_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(end_time)
    )
    reward_info.air_handler_reward_infos['air_handler_0'].CopyFrom(
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=100.0,
            air_conditioning_electrical_energy_rate=20.0,
        )
    )
    reward_info.air_handler_reward_infos['air_handler_1'].CopyFrom(
        smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
            blower_electrical_energy_rate=10.0,
            air_conditioning_electrical_energy_rate=30.0,
        )
    )

    reward_info.boiler_reward_infos['boiler_0'].CopyFrom(
        smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=250.0,
            pump_electrical_energy_rate=30.0,
        )
    )
    reward_info.boiler_reward_infos['boiler_1'].CopyFrom(
        smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
            natural_gas_heating_energy_rate=50.0,
            pump_electrical_energy_rate=100.0,
        )
    )

    energy_use = conversion_utils.get_reward_info_energy_use(reward_info)

    expected_energy_use = {
        'air_handler_blower_electricity': 110.0 * to_kwh,
        'air_handler_air_conditioning': 50.0 * to_kwh,
        'boiler_natural_gas_heating_energy': 300.0 * to_kwh,
        'boiler_pump_electrical_energy': 130 * to_kwh,
    }

    for field in expected_energy_use:
      self.assertAlmostEqual(
          expected_energy_use[field], energy_use[field], places=5
      )


if __name__ == '__main__':
  absltest.main()
