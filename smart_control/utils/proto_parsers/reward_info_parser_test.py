"""Tests for the RewardInfoParser class."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils import conversion_utils
from smart_buildings.smart_control.utils import test_utils
from smart_buildings.smart_control.utils.proto_parsers import reward_info_parser

get_comfort_diffs = reward_info_parser.get_comfort_diffs


TIME_ZONE = 'US/Pacific'
START_TIMESTAMP = pd.Timestamp('2022-03-13 00:00:00', tz=TIME_ZONE)
END_TIMESTAMP = pd.Timestamp('2022-03-13 00:05:00', tz=TIME_ZONE)

HISTOGRAM_INDEX_NAMES = [
    'count of zones',
    'count of occupants',
    'temperature setpoint range',
    'count of occupants exposed',
]


COMFORT_DIFFS_TEST_PARAMS = [
    # zone_air_temp, expected_diff, expected_label (with magnitude)
    (287.0, -6.0, 'TOO_COLD_5'),  # label uses capped diff value of 5
    (288.0, -5.0, 'TOO_COLD_5'),
    (289.0, -4.0, 'TOO_COLD_4'),
    (290.0, -3.0, 'TOO_COLD_3'),
    (291.0, -2.0, 'TOO_COLD_2'),
    (292.0, -1.0, 'TOO_COLD_1'),
    (292.4, -0.6, 'TOO_COLD_1'),  # label uses rounded diff value of -1
    (292.6, -0.4, 'TOO_COLD_0'),  # label uses rounded diff value of 0
    (293.0, 0.0, 'IN_RANGE'),
    (294.0, 0.0, 'IN_RANGE'),
    (295.0, 0.0, 'IN_RANGE'),
    (296.0, 0.0, 'IN_RANGE'),
    (297.0, 0.0, 'IN_RANGE'),
    (297.4, 0.4, 'TOO_HOT_0'),  # label uses rounded diff value of 0
    (297.6, 0.6, 'TOO_HOT_1'),  # label uses rounded diff value of 1
    (298.0, 1.0, 'TOO_HOT_1'),
    (299.0, 2.0, 'TOO_HOT_2'),
    (300.0, 3.0, 'TOO_HOT_3'),
    (301.0, 4.0, 'TOO_HOT_4'),
    (302.0, 5.0, 'TOO_HOT_5'),
    (303.0, 6.0, 'TOO_HOT_5'),  # label uses capped diff value of 5
]


class ComfortDiffsTest(parameterized.TestCase):
  """Tests for comfort differential functions."""

  @parameterized.parameters(COMFORT_DIFFS_TEST_PARAMS)
  def test_comfort_diffs_without_magnitude_labels(
      self, zone_air_temp, expected_diff, expected_label
  ):
    row = pd.Series({
        'zone_air_temp': zone_air_temp,
        'heating_setpoint_temp': 293.0,  # comfort range min
        'cooling_setpoint_temp': 297.0,  # comfort range max
    })

    label_no_magnitude = expected_label
    if label_no_magnitude != 'IN_RANGE':
      label_no_magnitude = label_no_magnitude.rsplit('_', 1)[0]

    diff, label = get_comfort_diffs(row, use_magnitude_labels=False)
    self.assertAlmostEqual(diff, expected_diff, places=4)
    self.assertEqual(label, label_no_magnitude)

  @parameterized.parameters(COMFORT_DIFFS_TEST_PARAMS)
  def test_comfort_diffs_with_magnitude_labels(
      self, zone_air_temp, expected_diff, expected_label
  ):
    row = pd.Series({
        'zone_air_temp': zone_air_temp,
        'heating_setpoint_temp': 293.0,  # comfort range min
        'cooling_setpoint_temp': 297.0,  # comfort range max
    })

    diff, label = get_comfort_diffs(row, use_magnitude_labels=True)
    self.assertAlmostEqual(diff, expected_diff, places=4)
    self.assertEqual(label, expected_label)

  @parameterized.parameters(
      # without a cap, the label can exceed the default of 5:
      (287.0, None, -6.0, 'TOO_COLD_6'),
      (303.0, None, 6.0, 'TOO_HOT_6'),
      # with a custom cap, the label is capped to the custom value:
      (289.0, 3, -4.0, 'TOO_COLD_3'),  # capped to 3
      (290.0, 3, -3.0, 'TOO_COLD_3'),
      (291.0, 3, -2.0, 'TOO_COLD_2'),
      (301.0, 3, 4.0, 'TOO_HOT_3'),  # capped to 3
      (300.0, 3, 3.0, 'TOO_HOT_3'),
      (299.0, 3, 2.0, 'TOO_HOT_2'),
  )
  def test_magnitude_labels_max_degrees(
      self, zone_air_temp, max_degrees, expected_diff, expected_label
  ):
    row = pd.Series({
        'zone_air_temp': zone_air_temp,
        'heating_setpoint_temp': 293.0,  # comfort range min
        'cooling_setpoint_temp': 297.0,  # comfort range max
    })
    diff, label = get_comfort_diffs(
        row, use_magnitude_labels=True, label_max_degrees=max_degrees
    )
    self.assertAlmostEqual(diff, expected_diff, places=4)
    self.assertEqual(label, expected_label)

  def test_comfort_diffs_invalid_setpoint_range(self):
    invalid_row = pd.Series({
        'zone_air_temp': 295.0,
        'heating_setpoint_temp': 298.0,  # comfort range min
        'cooling_setpoint_temp': 296.0,  # comfort range max
    })  # min >= max is invalid

    error_message = 'Invalid setpoint range. Expecting heating < cooling.'

    with self.assertRaisesRegex(ValueError, error_message):
      get_comfort_diffs(invalid_row)

    with self.assertRaisesRegex(ValueError, error_message):
      get_comfort_diffs(invalid_row, use_magnitude_labels=True)

  def test_comfort_diffs_invalid_temperature_values(self):
    invalid_row = pd.Series({
        'zone_air_temp': None,   # invalid value
        'heating_setpoint_temp': 293.0,
        'cooling_setpoint_temp': 297.0,
    })

    with self.subTest(name='without magnitude labels'):
      with self.assertRaisesRegex(ValueError, 'Invalid temperature values.'):
        get_comfort_diffs(invalid_row)

    with self.subTest(name='with magnitude labels'):
      with self.assertRaisesRegex(ValueError, 'Invalid temperature values.'):
        get_comfort_diffs(invalid_row, use_magnitude_labels=True)


class RewardInfoParserTest(absltest.TestCase):
  def setUp(self):
    super().setUp()

    # FYI the usual comfort range is between 293 and 297 K
    zone_temp_occupancies = [
        # zone_id, zone_air_temp, zone_occupancy
        ('zone_0', 295.0, 8.0),  # IN_RANGE  (71.33°F)
        ('zone_1', 292.0, 4.0),  # TOO_COLD_1  (65.93°F)
        ('zone_2', 299.0, 2.0),  # TOO_HOT_2  (78.53°F)
    ]
    air_handler_energies = [
        # ahu_id, blower_electrical_energy_rate, ac_electrical_energy_rate
        ('air_handler_0', 23.0, 15.0),
        ('air_handler_1', 26.0, 22.0),
    ]
    boiler_energies = [
        # hws_id, natural_gas_heating_energy_rate, pump_electrical_energy_rate
        ('boiler_0', 200.1, 2.3),
    ]

    self.reward_info = test_utils.get_test_reward_info(
        zone_temp_occupancies=zone_temp_occupancies,
        air_handler_energies=air_handler_energies,
        boiler_energies=boiler_energies,
        start_timestamp=START_TIMESTAMP,
        end_timestamp=END_TIMESTAMP,
    )

    self.parser = reward_info_parser.RewardInfoParser(
        self.reward_info, comfort_diff_params={'use_magnitude_labels': True}
    )

  # PROPERTIES AND ALIASES

  def test_reward_info(self):
    self.assertEqual(self.parser.reward_info, self.reward_info)

  def test_timestamps(self):
    self.assertEqual(self.parser.start_timestamp, START_TIMESTAMP)
    self.assertEqual(self.parser.end_timestamp, END_TIMESTAMP)

  def test_duration(self):
    self.assertEqual(self.parser.dt, 300)

  def test_zone_reward_infos(self):
    self.assertEqual(
        self.parser.zone_reward_infos, self.reward_info.zone_reward_infos
    )

  def test_air_handler_reward_infos(self):
    self.assertEqual(
        self.parser.air_handler_reward_infos,
        self.reward_info.air_handler_reward_infos,
    )

  def test_boiler_reward_infos(self):
    self.assertEqual(
        self.parser.boiler_reward_infos,
        self.reward_info.boiler_reward_infos,
    )

  # ZONE INFO

  def test_zone_conditions_histogram(self):
    # It calls and returns the result of get_zone_conditions_histogram() method:
    pd.testing.assert_frame_equal(
        self.parser.get_zone_conditions_histogram(),
        self.parser.zone_conditions_histogram,
    )
    # See test_get_zone_conditions_histogram() below for more details.

  def test_get_zone_conditions_histogram(self):
    histogram = self.parser.get_zone_conditions_histogram()
    self.assertIsInstance(histogram, pd.DataFrame)

    with self.subTest(name='index'):
      self.assertEqual(histogram.index.tolist(), HISTOGRAM_INDEX_NAMES)

    with self.subTest(name='columns'):
      expected_columns = [f'{temp}°K' for temp in reward_info_parser.TEMP_BINS]
      self.assertEqual(histogram.columns.tolist(), expected_columns)

    with self.subTest(name='zone counts'):
      # number of zones in each temperature bin:
      zone_counts = histogram.loc['count of zones',].to_dict()
      expected = {'290°K': 0, '291°K': 0, '292°K': 1, '293°K': 0, '294°K': 0,
                  '295°K': 1, '296°K': 0, '297°K': 0, '298°K': 0, '299°K': 1,
                  '300°K': 0}
      self.assertEqual(zone_counts, expected)

    with self.subTest(name='occupant counts'):
      # number of occupants in each temperature bin:
      occupant_counts = histogram.loc['count of occupants',].to_dict()
      expected = {'290°K': 0, '291°K': 0, '292°K': 4, '293°K': 0, '294°K': 0,
                  '295°K': 8, '296°K': 0, '297°K': 0, '298°K': 0, '299°K': 2,
                  '300°K': 0}
      self.assertEqual(occupant_counts, expected)

    with self.subTest(name='setpoint range'):
      # labels indicating whether each bin is in the comfort range or not:
      comfort_labels = histogram.loc['temperature setpoint range',].to_dict()
      expected = {'290°K': '-', '291°K': '-', '292°K': '-', '293°K': '+',
                  '294°K': '+', '295°K': '+', '296°K': '+', '297°K': '+',
                  '298°K': '-', '299°K': '-', '300°K': '-'}
      self.assertEqual(comfort_labels, expected)

    with self.subTest(name='occupant exposure'):
      # number of occupants outside of the comfort range (0 if in range):
      occupant_exposure = histogram.loc['count of occupants exposed',].to_dict()
      expected = {'290°K': 0, '291°K': 0, '292°K': 4, '293°K': 0, '294°K': 0,
                  '295°K': 0, '296°K': 0, '297°K': 0, '298°K': 0, '299°K': 2,
                  '300°K': 0}
      self.assertEqual(occupant_exposure, expected)

  def test_get_zone_conditions_histogram_with_custom_params(self):
    histogram = self.parser.get_zone_conditions_histogram(
        temp_unit='F', temp_bins=[60, 65, 70]
    )
    self.assertIsInstance(histogram, pd.DataFrame)

    with self.subTest(name='index'):
      self.assertEqual(histogram.index.tolist(), HISTOGRAM_INDEX_NAMES)

    with self.subTest(name='columns'):
      self.assertEqual(histogram.columns.tolist(), ['60°F', '65°F', '70°F'])

    with self.subTest(name='zone counts'):
      # Zone temps in F are: [65.93, 71.33, 78.53]
      # This row shows the number of zones in each temperature bin:
      self.assertEqual(
          histogram.loc['count of zones'].to_dict(),
          {'60°F': 0, '65°F': 1, '70°F': 2},
      )

  def test_get_zone_conditions_histogram_with_long_form_temp_unit(self):
    histogram = self.parser.get_zone_conditions_histogram(
        temp_unit='Kelvin',
    )
    self.assertIsInstance(histogram, pd.DataFrame)
    expected_columns = [f'{temp}°K' for temp in reward_info_parser.TEMP_BINS]
    self.assertEqual(histogram.columns.tolist(), expected_columns)

  def test_zone_occupancies_df(self):
    df = self.parser.zone_occupancies_df
    self.assertIsInstance(df, pd.DataFrame)

    with self.subTest(name='indexed on zone_id'):
      self.assertEqual(df.index.name, 'zone_id')
      self.assertEqual(df.index.tolist(), ['zone_0', 'zone_1', 'zone_2'])

    with self.subTest(name='columns'):
      self.assertEqual(df.columns.tolist(), [
          'average_occupancy', 'heating_setpoint_temp',
          'cooling_setpoint_temp', 'zone_air_temp', 'comfort_diff',
          'comfort_label'
      ])

    with self.subTest(name='occupancy'):
      # number of occupants in each zone:
      self.assertEqual(df['average_occupancy'].to_dict(), {
          'zone_0': 8.0,
          'zone_1': 4.0,
          'zone_2': 2.0,
      })

    with self.subTest(name='conditions'):
      self.assertEqual(df['heating_setpoint_temp'].to_dict(), {
          'zone_0': 293.0,
          'zone_1': 293.0,
          'zone_2': 293.0,
      })

      self.assertEqual(df['cooling_setpoint_temp'].to_dict(), {
          'zone_0': 297.0,
          'zone_1': 297.0,
          'zone_2': 297.0,
      })

      self.assertEqual(df['zone_air_temp'].to_dict(), {
          'zone_0': 295.0,
          'zone_1': 292.0,
          'zone_2': 299.0,
      })

    with self.subTest(name='comfort diffs'):
      # how far each zone's temp is from being in range (0 if in range):
      self.assertEqual(df['comfort_diff'].to_dict(), {
          'zone_0': 0.0,
          'zone_1': -1.0,
          'zone_2': 2.0,
      })

    with self.subTest(name='comfort labels'):
      # category label for each zone:
      self.assertEqual(df['comfort_label'].to_dict(), {
          'zone_0': 'IN_RANGE',
          'zone_1': 'TOO_COLD_1',
          'zone_2': 'TOO_HOT_2',
      })

  def test_zone_occupancies_df_with_default_comfort_diff_params(self):
    parser = reward_info_parser.RewardInfoParser(self.reward_info)
    self.assertEqual(
        parser.zone_occupancies_df['comfort_label'].to_dict(),
        {
            'zone_0': 'IN_RANGE',
            'zone_1': 'TOO_COLD',
            'zone_2': 'TOO_HOT',
        },
    )

  def test_zone_occupancies_df_with_custom_comfort_diff_params(self):
    custom_params = {'use_magnitude_labels': True, 'label_max_degrees': 1}
    parser = reward_info_parser.RewardInfoParser(
        self.reward_info, comfort_diff_params=custom_params,
    )
    self.assertEqual(
        parser.zone_occupancies_df['comfort_label'].to_dict(),
        {
            'zone_0': 'IN_RANGE',
            'zone_1': 'TOO_COLD_1',
            'zone_2': 'TOO_HOT_1',  # 2.0 diff is capped at 1 for label
        },
    )

  def test_num_zones(self):
    self.assertEqual(self.parser.num_zones, 3)

  def test_total_occupancy(self):
    self.assertEqual(self.parser.total_occupancy, 14)

  def test_occupant_comfort_counts(self):
    self.assertEqual(self.parser.num_occupants_comfortable, 8)
    self.assertEqual(self.parser.num_occupants_uncomfortable, 6)

    self.assertEqual(self.parser.occupant_comfort_histogram, {
        'IN_RANGE': 8,
        'TOO_COLD_1': 4,
        'TOO_HOT_2': 2,
    })

  # ENERGY CONSUMPTION

  def _assert_device_energy_consumption(
      self,
      df: pd.DataFrame,
      device_id: str,
      expected_values: list[dict[str, float]],
  ):
    rows = df[df['device_id'] == device_id]
    metrics = rows[['metric', 'rate_watts', 'consumption_kwh']]

    expected = pd.DataFrame(expected_values)

    pd.testing.assert_frame_equal(
        metrics.sort_values('metric').reset_index(drop=True),
        expected.sort_values('metric').reset_index(drop=True),
        check_dtype=False,
        check_index_type=False,
    )

  def test_energy_consumption_df(self):
    df = self.parser.energy_consumption_df
    self.assertIsInstance(df, pd.DataFrame)

    with self.subTest(name='row per device_id per metric (composite key)'):
      no_dups = not df.duplicated(subset=['device_id', 'metric']).any()
      self.assertTrue(no_dups)

    with self.subTest(name='columns include generic units and values'):
      self.assertEqual(df.columns.tolist(), [
          'device_type', 'device_id',
          'metric', 'description',
          'value', 'unit'
      ])

  def test_energy_consumption_df_watts(self):
    df = self.parser.energy_consumption_df_watts
    self.assertIsInstance(df, pd.DataFrame)

    with self.subTest(name='row per device_id per metric (composite key)'):
      no_dups = not df.duplicated(subset=['device_id', 'metric']).any()
      self.assertTrue(no_dups)

    with self.subTest(name='columns are watts-specific'):
      self.assertEqual(df.columns.tolist(), [
          'device_type', 'device_id',
          'metric', 'description',
          'rate_watts', 'consumption_kwh'
      ])

    with self.subTest(name='unique devices'):
      unique_devices = df[['device_type', 'device_id']].drop_duplicates()
      unique_devices.sort_values(by=['device_type', 'device_id'], inplace=True)
      expected_devices = [
          {'device_type': 'AHU', 'device_id': 'air_handler_0'},
          {'device_type': 'AHU', 'device_id': 'air_handler_1'},
          {'device_type': 'HWS', 'device_id': 'boiler_0'},
      ]
      self.assertEqual(unique_devices.to_dict('records'), expected_devices)

    with self.subTest(name='consumption metrics (air_handler_0)'):
      rows = df[df['device_id'] == 'air_handler_0']
      metrics = rows[['metric', 'rate_watts', 'consumption_kwh']]
      expected = pd.DataFrame([
          {
              'metric': 'blower_electrical_energy_rate',
              'rate_watts': 23.0,
              'consumption_kwh': 0.0019166666666666666
          },
          {
              'metric': 'air_conditioning_electrical_energy_rate',
              'rate_watts': 15.0,
              'consumption_kwh': 0.0012499999999999998
          }
      ])
      pd.testing.assert_frame_equal(
          metrics.sort_values('metric').reset_index(drop=True),
          expected.sort_values('metric').reset_index(drop=True),
          check_dtype=False,
          check_index_type=False,
      )

    with self.subTest(name='consumption metrics (air_handler_1)'):
      expected = [
          {
              'metric': 'blower_electrical_energy_rate',
              'rate_watts': 26.0,
              'consumption_kwh': 0.0021666666666666666
          },
          {
              'metric': 'air_conditioning_electrical_energy_rate',
              'rate_watts': 22.0,
              'consumption_kwh': 0.0018333333333333333
          }
      ]
      self._assert_device_energy_consumption(df, 'air_handler_1', expected)

    with self.subTest(name='consumption metrics (boiler_0)'):
      expected = [
          {
              'metric': 'pump_electrical_energy_rate',
              'rate_watts': 2.3,
              'consumption_kwh': 0.00019166666666666668,
          },
          {
              'metric': 'natural_gas_heating_energy_rate',
              'rate_watts': 200.1,
              'consumption_kwh': 0.016675,
          },
      ]
      self._assert_device_energy_consumption(df, 'boiler_0', expected)


class RewardInfoParserLegacyEnergyConsumptionTest(absltest.TestCase):
  """This uses the same setup as the original conversion_utils test."""

  def setUp(self):
    super().setUp()

    self.dt = 300
    start_time = pd.Timestamp('2021-05-03 12:13:00-5')
    end_time = start_time + pd.Timedelta(self.dt, unit='second')
    self.to_kwh = self.dt / 3600.0 / 1000.0

    reward_info = smart_control_reward_pb2.RewardInfo()
    # TIMESTAMPS:
    reward_info.start_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(start_time)
    )
    reward_info.end_timestamp.CopyFrom(
        conversion_utils.pandas_to_proto_timestamp(end_time)
    )
    # AIR HANDLERS:
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
    # BOILERS:
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

    self.reward_info = reward_info
    self.parser = reward_info_parser.RewardInfoParser(self.reward_info)

  def test_get_energy_consumption(self):
    energy_use = self.parser.get_energy_consumption()

    expected_energy_use = {
        'air_handler_blower_electricity': 110.0 * self.to_kwh,
        'air_handler_air_conditioning': 50.0 * self.to_kwh,
        'boiler_natural_gas_heating_energy': 300.0 * self.to_kwh,
        'boiler_pump_electrical_energy': 130 * self.to_kwh,
    }

    for field in expected_energy_use:
      self.assertAlmostEqual(
          expected_energy_use[field], energy_use[field], places=5
      )


if __name__ == '__main__':
  absltest.main()
