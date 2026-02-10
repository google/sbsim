"""Tests for BuildingDataset class."""

import os
import tempfile
import unittest

from absl.testing import absltest
import numpy as np
import pandas as pd
import pytest

from smart_control.dataset.conftest import DATASET_DIRPATH
from smart_control.dataset.conftest import SKIP_REASON
from smart_control.dataset.conftest import TEST_DATASET
from smart_control.dataset.conftest import ZIP_FILEPATH
from smart_control.dataset.dataset import BuildingDataset
from smart_control.dataset.dataset import DATA_DIR

#
# EXAMPLE BUILDING LEVEL DATA
#

_DATASET_ID = 'sb1'
_PARTITION_IDS = ['2022_a', '2022_b', '2023_a', '2023_b', '2024_a']

_DEVICE_LAYOUT_IDS = [
    'VAV CO 1-1-06',
    'VAV CO 1-1-07 CO2',
    'VAV CO 1-1-08 CO2',
    'VAV CO 1-1-10 CO2',
    'VAV CO 1-1-13 CO2',
    'VAV CO 1-1-16 CO2',
    'VAV CO 1-1-17 CO2',
    'VAV CO 1-1-18 CO2',
    'VAV CO 1-1-26',
    'VAV CO 1-1-27',
    'VAV CO 1-1-35 CO2',
    'VAV CO 1-1-43',
    'VAV CO 1-1-51',
    'VAV RH 1-1-01',
    'VAV RH 1-1-02',
    'VAV RH 1-1-03',
    'VAV RH 1-1-04',
    'VAV RH 1-1-05',
    'VAV RH 1-1-09 CO2',
    'VAV RH 1-1-11',
    'VAV RH 1-1-12 CO2',
    'VAV RH 1-1-14 CO2',
    'VAV RH 1-1-15',
    'VAV RH 1-1-19',
    'VAV RH 1-1-20',
    'VAV RH 1-1-21',
    'VAV RH 1-1-22',
    'VAV RH 1-1-23',
    'VAV RH 1-1-25 (MK # 1F9)',
    'VAV RH 1-1-28 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-29 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-30 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-31 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-32 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-33',
    'VAV RH 1-1-34',
    'VAV RH 1-1-36',
    'VAV RH 1-1-37',
    'VAV RH 1-1-38',
    'VAV RH 1-1-39',
    'VAV RH 1-1-40',
    'VAV RH 1-1-41',
    'VAV RH 1-1-42',
    'VAV RH 1-1-44',
    'VAV RH 1-1-45',
    'VAV RH 1-1-46',
    'VAV RH 1-1-47',
    'VAV RH 1-1-48',
    'VAV RH 1-1-49',
    'VAV RH 1-1-50',
    'VAV RH 1-1-52 CO2',
    'VAV RH 1-1-53',
    'VAV RH 1-1-54',
    'VAV RH 1-1-55',
]

_FIRST_ZONE_INFO = {
    'zone_id': 'rooms/1002000133978',
    'building_id': 'buildings/3616672508',
    'zone_description': 'SB1-2-C2054',
    'area': 0.0,
    'zone_type': 1,
    'floor': 2,
    'devices': ['2618581107144046', '2696593986887004'],
}

_LAST_ZONE_INFO = {
    'zone_id': 'rooms/11312312488',
    'building_id': 'buildings/3616672508',
    'zone_description': 'SB1-1-1J7B',
    'area': 0.0,
    'zone_type': 1,
    'floor': 1,
    'devices': ['2802781341872564'],
}

_FIRST_DEVICE_INFO = {
    'device_id': '202194278473007104',
    'namespace': 'PHRED',
    'code': 'SB1:AHU:AC-2',
    'zone_id': '',
    'device_type': 6,
    'observable_fields': {
        'building_air_static_pressure_sensor': 1,
        'outside_air_flowrate_sensor': 1,
        'supply_fan_speed_percentage_command': 1,
        'supply_air_temperature_sensor': 1,
        'supply_fan_speed_frequency_sensor': 1,
        'supply_air_static_pressure_setpoint': 1,
        'return_air_temperature_sensor': 1,
        'mixed_air_temperature_setpoint': 1,
        'exhaust_fan_speed_percentage_command': 1,
        'exhaust_fan_speed_frequency_sensor': 1,
        'outside_air_damper_percentage_command': 1,
        'mixed_air_temperature_sensor': 1,
        'exhaust_air_damper_percentage_command': 1,
        'cooling_percentage_command': 1,
        'outside_air_flowrate_setpoint': 1,
        'supply_air_temperature_setpoint': 1,
        'building_air_static_pressure_setpoint': 1,
        'supply_air_static_pressure_sensor': 1,
    },
    'actionable_fields': {
        'exhaust_air_damper_percentage_command': 1,
        'supply_air_temperature_setpoint': 1,
        'supply_fan_speed_percentage_command': 1,
        'outside_air_flowrate_setpoint': 1,
        'cooling_percentage_command': 1,
        'mixed_air_temperature_setpoint': 1,
        'exhaust_fan_speed_percentage_command': 1,
        'outside_air_damper_percentage_command': 1,
        'supply_air_static_pressure_setpoint': 1,
        'building_air_static_pressure_setpoint': 1,
    },
}

_LAST_DEVICE_INFO = {
    'device_id': '2640423556868160',
    'namespace': 'CDM',
    'code': 'VAV RH 2-2-68',
    'zone_id': '',
    'device_type': 4,
    'observable_fields': {
        'supply_air_flowrate_setpoint': 1,
        'discharge_air_temperature_setpoint': 1,
        'discharge_air_temperature_sensor': 1,
        'zone_air_heating_temperature_setpoint': 1,
        'heating_water_valve_percentage_command': 1,
        'supply_air_flowrate_sensor': 1,
        'zone_air_temperature_sensor': 1,
        'zone_air_cooling_temperature_setpoint': 1,
        'supply_air_damper_percentage_command': 1,
    },
    'actionable_fields': {
        'zone_air_cooling_temperature_setpoint': 1,
        'discharge_air_temperature_setpoint': 1,
        'heating_water_valve_percentage_command': 1,
        'supply_air_flowrate_setpoint': 1,
        'zone_air_heating_temperature_setpoint': 1,
        'supply_air_damper_percentage_command': 1,
    },
}

_DEVICE_OBSERVABLE_FIELD_NAMES = [
    'building_air_static_pressure_sensor',
    'building_air_static_pressure_setpoint',
    'cooling_percentage_command',
    'differential_pressure_sensor',
    'differential_pressure_setpoint',
    'discharge_air_temperature_sensor',
    'discharge_air_temperature_setpoint',
    'exhaust_air_damper_percentage_command',
    'exhaust_air_damper_percentage_sensor',
    'exhaust_fan_speed_frequency_sensor',
    'exhaust_fan_speed_percentage_command',
    'heating_water_valve_percentage_command',
    'mixed_air_temperature_sensor',
    'mixed_air_temperature_setpoint',
    'outside_air_damper_percentage_command',
    'outside_air_dewpoint_temperature_sensor',
    'outside_air_flowrate_sensor',
    'outside_air_flowrate_setpoint',
    'outside_air_relative_humidity_sensor',
    'outside_air_specificenthalpy_sensor',
    'outside_air_temperature_sensor',
    'outside_air_wetbulb_temperature_sensor',
    'program_differential_pressure_setpoint',
    'program_supply_air_static_pressure_setpoint',
    'program_supply_air_temperature_setpoint',
    'program_supply_water_temperature_setpoint',
    'return_air_temperature_sensor',
    'return_water_temperature_sensor',
    'run_status',
    'speed_frequency_sensor',
    'speed_percentage_command',
    'supervisor_supply_air_static_pressure_setpoint',
    'supervisor_supply_air_temperature_setpoint',
    'supervisor_supply_water_temperature_setpoint',
    'supply_air_damper_percentage_command',
    'supply_air_flowrate_sensor',
    'supply_air_flowrate_setpoint',
    'supply_air_static_pressure_sensor',
    'supply_air_static_pressure_setpoint',
    'supply_air_temperature_sensor',
    'supply_air_temperature_setpoint',
    'supply_fan_run_status',
    'supply_fan_speed_frequency_sensor',
    'supply_fan_speed_percentage_command',
    'supply_water_temperature_sensor',
    'supply_water_temperature_setpoint',
    'zone_air_co2_concentration_sensor',
    'zone_air_co2_concentration_setpoint',
    'zone_air_cooling_temperature_setpoint',
    'zone_air_heating_temperature_setpoint',
    'zone_air_temperature_sensor',
]

_DEVICE_ACTIONABLE_FIELD_NAMES = [
    'building_air_static_pressure_setpoint',
    'cooling_percentage_command',
    'differential_pressure_setpoint',
    'discharge_air_temperature_setpoint',
    'exhaust_air_damper_percentage_command',
    'exhaust_fan_speed_percentage_command',
    'heating_water_valve_percentage_command',
    'mixed_air_temperature_setpoint',
    'outside_air_damper_percentage_command',
    'outside_air_flowrate_setpoint',
    'program_differential_pressure_setpoint',
    'program_supply_air_static_pressure_setpoint',
    'program_supply_air_temperature_setpoint',
    'program_supply_water_temperature_setpoint',
    'speed_percentage_command',
    'supervisor_supply_air_static_pressure_setpoint',
    'supervisor_supply_air_temperature_setpoint',
    'supervisor_supply_water_temperature_setpoint',
    'supply_air_damper_percentage_command',
    'supply_air_flowrate_setpoint',
    'supply_air_static_pressure_setpoint',
    'supply_air_temperature_setpoint',
    'supply_fan_speed_percentage_command',
    'supply_water_temperature_setpoint',
    'zone_air_co2_concentration_setpoint',
    'zone_air_cooling_temperature_setpoint',
    'zone_air_heating_temperature_setpoint',
]


#
# BUILDING-SPECIFIC TESTS
#


class TestDataDirectory(absltest.TestCase):
  """Tests for the data directory."""

  def test_data_dir(self):
    self.assertTrue(os.path.isdir(DATA_DIR))


@pytest.mark.usefixtures('set_dataset')
class TestBuildingDataset(absltest.TestCase):
  """Tests for the BuildingDataset class."""

  def test_building_validations(self):
    with self.assertRaises(ValueError):
      invalid_id = 'OOPS'
      BuildingDataset(dataset_id=invalid_id, download=False)

  def test_dataset_id(self):
    self.assertEqual(self.ds.dataset_id, _DATASET_ID)

  def test_partition_ids(self):
    self.assertEqual(self.ds.partition_ids, _PARTITION_IDS)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_download(self):
    self.assertIn('download', dir(self.ds))
    self.assertTrue(os.path.isdir(DATASET_DIRPATH))
    self.assertTrue(os.path.exists(ZIP_FILEPATH))

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_floorplan(self):
    floorplan = self.ds.floorplan

    self.assertIsInstance(floorplan, np.ndarray)
    self.assertEqual(floorplan.shape, (744, 1004))

    values, counts = np.unique(floorplan, return_counts=True)
    value_counts = dict(zip(values, counts))
    self.assertEqual(value_counts, {0.0: 436332, 1.0: 60204, 2.0: 250440})

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_display_floorplan(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      filepath = os.path.join(temp_dir, 'tmp_floorplan.png')
      self.assertFalse(os.path.isfile(filepath))

      self.ds.display_floorplan(show=False, save=True, image_filepath=filepath)

      # it saves the image to disk, at the specified location:
      self.assertTrue(os.path.isfile(filepath))

    # testing note: the temp dir gets deleted automatically :-)
    self.assertFalse(os.path.isfile(filepath))

  # DEVICES

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_device_layout_map(self):
    device_layout_map = self.ds.device_layout_map

    self.assertIsInstance(device_layout_map, dict)
    self.assertEqual(sorted(list(device_layout_map.keys())), _DEVICE_LAYOUT_IDS)

    # each device layout is a list of lists containing two integer coordinates.
    # layouts may differ in length:

    example_layout_map = device_layout_map['VAV CO 1-1-06']
    self.assertIsInstance(example_layout_map, list)
    self.assertEqual(np.array(example_layout_map).shape, (1021, 2))
    self.assertEqual(example_layout_map[0], [79, 35])
    self.assertEqual(example_layout_map[-1], [80, 64])

    another_layout_map = device_layout_map['VAV RH 1-1-55']
    self.assertIsInstance(another_layout_map, list)
    self.assertEqual(np.array(another_layout_map).shape, (935, 2))
    self.assertEqual(another_layout_map[0], [145, 126])
    self.assertEqual(another_layout_map[-1], [149, 160])

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_device_infos(self):
    device_infos = self.ds.device_infos
    self.assertIsInstance(device_infos, list)
    self.assertEqual(len(device_infos), 173)
    self.assertEqual(device_infos[0], _FIRST_DEVICE_INFO)
    self.assertEqual(device_infos[-1], _LAST_DEVICE_INFO)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_devices_df(self):
    devices_df = self.ds.devices_df

    self.assertIsInstance(devices_df, pd.DataFrame)
    self.assertEqual(len(devices_df), 173)

    # each row uniquely identified by the device identifier:
    self.assertEqual(devices_df['device_id'].nunique(), len(devices_df))

    self.assertEqual(
        devices_df.columns.tolist(),
        [
            'device_id',
            'namespace',
            'code',
            'device_type',
            'observable_fields',
            'actionable_fields',
        ],
    )

    first_row = _FIRST_DEVICE_INFO.copy()
    del first_row['zone_id']  # we removed "zone_id" from the df
    self.assertEqual(devices_df.iloc[0].to_dict(), first_row)

    # each device belongs to a namespace:
    self.assertEqual(
        devices_df['namespace'].value_counts().to_dict(),
        {'CDM': 155, 'PHRED': 18},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observable_fields(self):
    observable_fields = self.ds.observable_fields
    self.assertIsInstance(observable_fields, list)
    self.assertEqual(len(observable_fields), 51)
    self.assertEqual(observable_fields, _DEVICE_OBSERVABLE_FIELD_NAMES)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observable_field_counts(self):
    value_counts = self.ds.observable_field_counts
    self.assertIsInstance(value_counts, pd.Series)
    self.assertEqual(len(value_counts), 51)
    self.assertEqual(
        sorted(list(value_counts.keys())), _DEVICE_OBSERVABLE_FIELD_NAMES
    )
    # counts how many devices support each field:
    self.assertEqual(
        value_counts.head(5).to_dict(),
        {
            'supply_air_damper_percentage_command': 123,
            'zone_air_cooling_temperature_setpoint': 123,
            'supply_air_flowrate_setpoint': 123,
            'zone_air_temperature_sensor': 123,
            'zone_air_heating_temperature_setpoint': 122,
        },
    )
    self.assertEqual(
        value_counts.tail(1).to_dict(),
        {'supervisor_supply_water_temperature_setpoint': 1},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_actionable_fields(self):
    actionable_fields = self.ds.actionable_fields
    self.assertIsInstance(actionable_fields, list)
    self.assertEqual(len(actionable_fields), 27)
    self.assertEqual(actionable_fields, _DEVICE_ACTIONABLE_FIELD_NAMES)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_actionable_field_counts(self):
    value_counts = self.ds.actionable_field_counts
    self.assertIsInstance(value_counts, pd.Series)
    self.assertEqual(len(value_counts), 27)
    self.assertEqual(
        sorted(list(value_counts.keys())), _DEVICE_ACTIONABLE_FIELD_NAMES
    )
    # counts how many devices support each field:
    self.assertEqual(
        value_counts.head(5).to_dict(),
        {
            'zone_air_cooling_temperature_setpoint': 123,
            'supply_air_damper_percentage_command': 123,
            'supply_air_flowrate_setpoint': 123,
            'zone_air_heating_temperature_setpoint': 122,
            'heating_water_valve_percentage_command': 99,
        },
    )
    self.assertEqual(
        value_counts.tail(1).to_dict(),
        {'program_supply_water_temperature_setpoint': 1},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_fields_df(self):
    fields_df = self.ds.fields_df
    self.assertIsInstance(fields_df, pd.DataFrame)
    self.assertEqual(len(fields_df), 51)
    self.assertEqual(
        fields_df.columns.tolist(),
        [
            'field_name',
            'is_actionable',
            'is_observable',
            'n_devices_actionable',
            'n_devices_observable',
        ],
    )
    # some are observable:
    self.assertEqual(
        fields_df.iloc[0].to_dict(),
        {
            'field_name': 'building_air_static_pressure_sensor',
            'is_actionable': False,
            'is_observable': True,
            'n_devices_actionable': 0,
            'n_devices_observable': 3,
        },
    )
    # some are actionable:
    self.assertEqual(
        fields_df.iloc[-1].to_dict(),
        {
            'field_name': 'zone_air_temperature_sensor',
            'is_actionable': False,
            'is_observable': True,
            'n_devices_actionable': 0,
            'n_devices_observable': 123,
        },
    )
    # some are both actionable and observable:
    self.assertEqual(
        len(fields_df[(fields_df.is_actionable & fields_df.is_observable)]), 27
    )

  # ZONES

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_zone_infos(self):
    zone_infos = self.ds.zone_infos

    self.assertIsInstance(zone_infos, list)
    self.assertEqual(len(zone_infos), 563)

    self.assertEqual(zone_infos[0], _FIRST_ZONE_INFO)
    self.assertEqual(zone_infos[-1], _LAST_ZONE_INFO)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_zones_df(self):
    zones_df = self.ds.zones_df

    self.assertIsInstance(zones_df, pd.DataFrame)
    self.assertEqual(len(zones_df), 563)

    # each row uniquely identified by the zone identifier:
    self.assertEqual(zones_df['zone_id'].nunique(), len(zones_df))

    self.assertEqual(
        zones_df.columns.tolist(),
        [
            'zone_id',
            'building_id',
            'zone_description',
            'area',
            'zone_type',
            'floor',
            'devices',
            'n_devices',
        ],
    )

    first_row = _FIRST_ZONE_INFO.copy()
    first_row['n_devices'] = 2  # we added this column to the df
    self.assertEqual(zones_df.iloc[0].to_dict(), first_row)


if __name__ == '__main__':
  absltest.main()
