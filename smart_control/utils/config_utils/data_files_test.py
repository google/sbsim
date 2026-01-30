"""Test for data files used within gin configs."""

from absl.testing import absltest
import numpy as np
import pandas as pd

from smart_buildings.smart_control.utils.config_utils import conftest
from smart_buildings.smart_control.utils.config_utils import data_files


class WeatherDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.weather_df = data_files.get_weather_data()

  def test_columns(self):
    self.assertEqual(list(self.weather_df.columns), conftest.WEATHER_COLUMNS)

  def test_time_range(self):
    self.assertEqual(self.weather_df["Time"].min(), "20240101-0100")
    self.assertEqual(self.weather_df["Time"].max(), "20241230-0900")


class FloorPlanTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.floorplan = data_files.get_floor_plan()
    self.df = pd.DataFrame(self.floorplan)

  def test_shape(self):
    self.assertEqual(self.df.shape, (744, 1004))

  def test_values(self):
    value_counts = self.df.stack().value_counts().to_dict()
    value_counts = {int(k): v for k, v in value_counts.items()}
    self.assertEqual(value_counts, {0: 436332, 1: 60204, 2: 250440})


class ZoneTempsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.zone_temps = data_files.get_zone_temps()

  def test_shape(self):
    self.assertEqual(self.zone_temps.shape, (744, 1004))

  def test_values(self):
    self.assertAlmostEqual(np.min(self.zone_temps), 288.287, places=3)
    self.assertAlmostEqual(np.max(self.zone_temps), 297.372, places=3)


if __name__ == "__main__":
  absltest.main()
