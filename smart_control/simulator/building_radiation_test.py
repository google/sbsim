"""Building radiation tests."""

import os

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from smart_control.simulator import conftest

TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "building_radiation_test_data"
)

INDEXED_FLOOR_PLAN = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
])


def assert_data_file_values_equal(
    data: np.array, csv_filename: str, precision: int = 3
):
  data_filepath = os.path.join(TEST_DATA_DIR, csv_filename)
  # temporarily uncomment to update the test data file, when applicable:
  # df = pd.DataFrame(data)
  # df.to_csv(data_filepath, index=False, header=False)

  df = pd.read_csv(data_filepath, header=None)
  expected = df.to_numpy()
  assert_array_almost_equal(
      np.round(data, precision),
      np.round(expected, precision),
  )


class BuildingRadiationScriptFTest(absltest.TestCase):

  def setUp(self):
    self.building = conftest.create_building_with_radiative_properties(
        view_factor_method="ScriptF"
    )
    self.building_with_interior_mass = (
        conftest.create_building_with_radiative_properties(
            view_factor_method="ScriptF",
            include_interior_mass=True,
        )
    )

  def test_interior_radiative_heat_transfer(self):
    with self.subTest("indexed_floor_plan"):
      assert_array_almost_equal(
          np.round(self.building.indexed_floor_plan, 3),
          np.round(INDEXED_FLOOR_PLAN, 3),
      )

    with self.subTest("view factor"):
      props = self.building.interior_wall_vf
      assert_data_file_values_equal(props, "expected_interior_wall_vf.csv")

    with self.subTest("alpha"):
      props = self.building._alpha
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "alpha.csv")

    with self.subTest("epsilon"):
      props = self.building._epsilon
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "epsilon.csv")

    with self.subTest("tau"):
      props = self.building._tau
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "tau.csv")

    with self.subTest("ifa_inv"):
      results = self.building.ifa_inv
      self.assertEqual(results.shape, (50, 50))
      assert_data_file_values_equal(results, "ifa_inv.csv", precision=5)

    with self.subTest("view factor with interior mass"):
      props = self.building_with_interior_mass.interior_wall_vf
      assert_data_file_values_equal(
          props, "expected_interior_wall_vf_interior_mass.csv"
      )


if __name__ == "__main__":
  absltest.main()
