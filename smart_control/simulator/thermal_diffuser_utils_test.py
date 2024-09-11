"""Tests for thermal_diffuser_utils.

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
from absl.testing import parameterized
import numpy as np
from smart_control.simulator import thermal_diffuser_utils


def _create_small_room():
  return [(1, 1), (1, 2), (2, 1), (2, 2)]


def _create_non_rectangular_room():
  return [(1, 1), (3, 3), (4, 4), (2, 2)]


def _create_medium_room():
  return [(2, 3), (3, 3), (3, 4), (3, 5), (4, 5), (4, 6)]


def _create_large_room():
  return [(x, y) for x in range(20) for y in range(20)]  # pylint: disable=g-complex-comprehension


class ThermalDiffuserUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("test_medium_list", 0, 24, [8, 16], 10),
      (
          "test_long_list",
          0,
          124,
          [10, 20, 29, 39, 48, 58, 67, 77, 86, 96, 105, 115],
          10,
      ),
      ("test_short_list", 0, 1, [1], 10),
      ("test_exact_case", 1, 4, [2, 3, 4], 1),
      ("end_points_not_zero", 5, 29, [13, 21], 10),
      ("end_points_not_zero_2", 1, 5, [3, 4], 2),
      ("test_exact", 2, 8, [4, 5, 7], 2),
  )
  def test_evenly_spaced_inds_from_domain(
      self, start, end, expected_output, spacing
  ):
    self.assertEqual(
        thermal_diffuser_utils._evenly_spaced_inds_from_domain(
            start=start, end=end, spacing=spacing
        ),
        expected_output,
    )

  @parameterized.named_parameters(
      ("rectangular_room", _create_small_room, True, 0.5),
      ("non_rectangular_room", _create_non_rectangular_room, False, 0.5),
      ("high_threshold", _create_medium_room, True, 0.99),
  )
  def test_rectangularity_test_by_function(
      self, room_generating_func, expected_output, threshold
  ):
    room_inds = room_generating_func()

    self.assertEqual(
        thermal_diffuser_utils._rectangularity_test(
            room_cv_indices=room_inds, threshold=threshold
        ),
        expected_output,
    )

  @parameterized.named_parameters(
      ("small_room", _create_small_room, np.array([[1, 1]])),
      (
          "non_rectangular_room",
          _create_non_rectangular_room,
          np.array([[1, 1]]),
      ),
      ("medium_room", _create_medium_room, np.array([[2, 3]])),
      (
          "larger_room",
          _create_large_room,
          np.array([[8, 8], [12, 16], [0, 14], [13, 16]]),
      ),
  )
  def test_determine_random_inds_for_thermal_diffusers(
      self, room_generating_func, expected_output
  ):
    room_inds = room_generating_func()

    np.testing.assert_array_equal(
        thermal_diffuser_utils._determine_random_inds_for_thermal_diffusers(
            room_cv_indices=room_inds, random_seed=23
        ),
        expected_output,
    )

  @parameterized.named_parameters(
      ("rectangular_room", _create_small_room, np.array([[2, 2]])),
      (
          "non_rectangular_room",
          _create_non_rectangular_room,
          np.array([[3, 3]]),
      ),
      ("medium_room", _create_medium_room, np.array([[3, 5]])),
  )
  def test_diffuser_allocation_switch(
      self, room_generating_func, expected_output
  ):
    room_inds = room_generating_func()
    output = thermal_diffuser_utils.diffuser_allocation_switch(room_inds)
    np.testing.assert_array_equal(output, expected_output)


if __name__ == "__main__":
  absltest.main()
