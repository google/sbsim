"""Tests for building_utils.

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

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from smart_control.simulator import building_utils


class BuildingUtilsTest(parameterized.TestCase):

  def test_read_floor_plan_from_filepath_raises_error(self):
    with self.assertRaises(FileNotFoundError):
      building_utils.read_floor_plan_from_filepath(
          "not/a/filepath", save_debugging_image=False
      )

  def test_read_floor_plan_from_filepath_does_not_raise_error(self):
    tempdir = self.create_tempdir()
    filename = os.path.join(tempdir, str("test.wow") + ".csv")
    sample_floorplan = np.array(
        [[2.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 2.0]]
    )
    np.savetxt(filename, sample_floorplan, delimiter=",")
    f = building_utils.read_floor_plan_from_filepath(
        filename, save_debugging_image=False
    )
    np.testing.assert_array_equal(f, sample_floorplan)

  def test_read_floor_plan_from_filepath_npy(self):
    tempdir = self.create_tempdir()
    filename = os.path.join(tempdir, str("test.wow") + ".npy")
    sample_floorplan = np.array(
        [[2.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 2.0]]
    )
    np.save(filename, sample_floorplan)
    f = building_utils.read_floor_plan_from_filepath(
        filename, save_debugging_image=False
    )
    np.testing.assert_array_equal(f, sample_floorplan)

  @parameterized.named_parameters(
      (
          "building_against_right_edge",
          np.array([[2, 2, 2, 2, 2], [2, 1, 1, 1, 1], [2, 2, 2, 2, 2]]),
          np.array(
              [[2, 2, 2, 2, 2, 2], [2, 1, 1, 1, 1, 2], [2, 2, 2, 2, 2, 2]]
          ),
      ),
      (
          "placebo",
          np.array([[2, 2, 2, 2, 2], [2, 1, 1, 1, 2], [2, 2, 2, 2, 2]]),
          np.array([[2, 2, 2, 2, 2], [2, 1, 1, 1, 2], [2, 2, 2, 2, 2]]),
      ),
      (
          "no_air_at_all",
          np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
          np.array([
              [2, 2, 2, 2, 2],
              [2, 1, 1, 1, 2],
              [2, 1, 1, 1, 2],
              [2, 1, 1, 1, 2],
              [2, 2, 2, 2, 2],
          ]),
      ),
  )
  def test_guarantee_air_padding_in_frame(self, floor_plan, expected_output):
    np.testing.assert_array_equal(
        building_utils.guarantee_air_padding_in_frame(floor_plan),
        expected_output,
    )

  @parameterized.named_parameters(
      (
          "rectangular_fp",
          np.array([
              [2, 2, 2, 2, 2],
              [2, 1, 1, 1, 2],
              [2, 1, 0, 1, 2],
              [2, 1, 1, 1, 2],
              [2, 2, 2, 2, 2],
          ]),
          np.array([
              [-1, -1, -1, -1, -1],
              [-1, 0, 0, 0, -1],
              [-1, 0, 1, 0, -1],
              [-1, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1],
          ]),
      ),
      (
          "rectangular_fp_without_padded_air",
          np.array([
              [2, 2, 2, 2],
              [2, 1, 1, 1],
              [2, 1, 0, 1],
              [2, 1, 1, 1],
              [2, 2, 2, 2],
          ]),
          np.array([
              [-1, -1, -1, -1],
              [-1, 0, 0, 0],
              [-1, 0, 1, 0],
              [-1, 0, 0, 0],
              [-1, -1, -1, -1],
          ]),
      ),
      (
          "three_connections",
          np.array([
              [2, 2, 2, 2, 2, 2, 2, 2],
              [2, 1, 1, 1, 1, 1, 1, 1],
              [2, 1, 0, 0, 0, 0, 0, 1],
              [2, 1, 1, 1, 1, 1, 1, 1],
              [2, 1, 0, 0, 0, 0, 0, 1],
              [2, 1, 0, 0, 0, 0, 0, 1],
              [2, 1, 1, 1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2, 2, 2, 2],
          ]),
          np.array([
              [-1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0],
              [-1, 0, 1, 1, 1, 1, 1, 0],
              [-1, 0, 0, 0, 0, 0, 0, 0],
              [-1, 0, 2, 2, 2, 2, 2, 0],
              [-1, 0, 2, 2, 2, 2, 2, 0],
              [-1, 0, 0, 0, 0, 0, 0, 0],
              [-1, -1, -1, -1, -1, -1, -1, -1],
          ]),
      ),
  )
  def test_process_and_run_connected_components(
      self, floor_plan, expected_result
  ):
    outcome = building_utils.process_and_run_connected_components(floor_plan)
    np.testing.assert_array_equal(expected_result, outcome)

  @parameterized.named_parameters(
      (
          "three_connections",
          np.uint8(
              1
              - np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
              ])
          ),
          np.array([
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 2, 2, 2, 2, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 3, 3, 3, 3, 0, 1],
              [1, 0, 3, 3, 3, 3, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
          ]),
      ),
      (
          "one_connection",
          np.uint8(1 - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
          np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
      ),
      (
          "one_connection_float",
          (1.0 - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)),
          np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
      ),
  )
  def test_run_connected_components(self, floor_plan, expected_components):
    np.testing.assert_array_equal(
        building_utils._run_connected_components(
            floor_plan, save_debugging_image=False
        ),
        expected_components,
    )

  def test_add_row_or_column_of_air_raises_error(self):
    with self.assertRaises(IndexError):
      building_utils.guarantee_air_padding_in_frame(np.array([1, 1]))

  def test_determine_exterior_space(self):
    floor_plan = np.array([
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2],
    ])
    expected_exterior_space = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 0, 0, 0, -1],
        [-1, 0, 0, 0, -1],
        [-1, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1],
    ])
    expected_connection_floor_plan = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    outcome_connection_floor_plan, outcome_exterior_space = (
        building_utils._determine_exterior_space(floor_plan)
    )

    with self.subTest("expected_exterior_space"):
      np.testing.assert_array_equal(
          outcome_exterior_space, expected_exterior_space
      )
    with self.subTest("expected_connection_floor_plan"):
      np.testing.assert_array_equal(
          outcome_connection_floor_plan, expected_connection_floor_plan
      )

  def test_set_exterior_space_neg(self):
    # Note: this function is extremely simple, so one test should suffice.

    connected_components = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 2, 2, 2, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 3, 3, 3, 3, 0, 1],
        [1, 0, 3, 3, 3, 3, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    exterior_space = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])
    expected_result = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 2, 2, 2, 2, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 3, 3, 3, 3, 0, -1],
        [-1, 0, 3, 3, 3, 3, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])

    np.testing.assert_array_equal(
        building_utils._set_exterior_space_neg(
            connected_components, exterior_space
        ),
        expected_result,
    )

  @parameterized.named_parameters(
      (
          "larger_building",
          np.array([
              [-1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1],
          ]),
          np.array([
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],
              [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
              [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
              [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
              [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
              [0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          ]),
      ),
      (
          "no_building",
          np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
          np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
      ),
      (
          "point_building",
          np.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]]),
          np.array([[0.0, 0.0, 0.0], [0.0, -2, 0.0], [0.0, 0.0, 0.0]]),
      ),
  )
  def test_label_exterior_wall_shell(
      self, outside_air, expected_exterior_walls
  ):
    np.testing.assert_array_equal(
        building_utils._label_exterior_wall_shell(outside_air),
        expected_exterior_walls,
    )

  def test_label_interior_walls(self):
    exterior_walls_1 = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],
        [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
        [0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    original_floor_plan_1 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 0, 1, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 1, 1, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2],
    ])

    expected_interior_walls_1 = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -3.0, -3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    np.testing.assert_array_equal(
        building_utils._label_interior_walls(
            exterior_walls_1, original_floor_plan_1
        ),
        expected_interior_walls_1,
    )

  def test_construct_room_dict(self):
    rooms_with_outside_air_1 = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 1, 1, 1, 1, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 2, 2, 2, 2, 0, -1],
        [-1, 0, 2, 2, 2, 2, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])

    expected_room_dict = {
        "interior_wall": [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 1),
            (2, 6),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 1),
            (4, 6),
            (5, 1),
            (5, 6),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
        ],
        "exterior_space": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 7),
            (2, 0),
            (2, 7),
            (3, 0),
            (3, 7),
            (4, 0),
            (4, 7),
            (5, 0),
            (5, 7),
            (6, 0),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
        ],
        "room_1": [(2, 2), (2, 3), (2, 4), (2, 5)],
        "room_2": [
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
        ],
    }

    self.assertEqual(
        building_utils._construct_room_dict(rooms_with_outside_air_1),
        expected_room_dict,
    )

  def test_enlarge_component(self):
    floor_plan_1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    expected_enlargement_1 = np.array([
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
    ])

    np.testing.assert_array_equal(
        building_utils.enlarge_component(
            array_with_component_nonzero=floor_plan_1, distance_to_augment=1
        ),
        expected_enlargement_1,
    )


if __name__ == "__main__":
  absltest.main()
