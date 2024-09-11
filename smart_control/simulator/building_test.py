"""Tests for building.

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

import random
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from smart_control.simulator import building
from smart_control.simulator import building_utils
from smart_control.simulator import constants
from smart_control.simulator import stochastic_convection_simulator


def _create_dummy_floor_plan():
  plan = np.array([
      [2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2],
  ])

  return plan


def _create_dummy_floor_plan_matching_deprecation():
  plan = np.array([
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  ])

  return plan


def _create_dummy_building_deprecated_matching_post_refactor():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  room_shape = (2, 2)
  building_shape = (3, 2)
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  b = building.Building(
      cv_size_cm,
      floor_height_cm,
      room_shape,
      building_shape,
      initial_temp,
      inside_air_properties,
      inside_wall_properties,
      building_exterior_properties,
  )

  return b


def _create_dummy_post_refactor_building_matching_deprecation():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  floor_plan = _create_dummy_floor_plan_matching_deprecation()
  zone_map = _create_dummy_floor_plan_matching_deprecation()

  b = building.FloorPlanBasedBuilding(
      cv_size_cm=cv_size_cm,
      floor_height_cm=floor_height_cm,
      initial_temp=initial_temp,
      inside_air_properties=inside_air_properties,
      inside_wall_properties=inside_wall_properties,
      building_exterior_properties=building_exterior_properties,
      zone_map=zone_map,
      floor_plan=floor_plan,
      buffer_from_walls=0,
  )

  return b


def _create_dummy_building_deprecated_1():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  room_shape = (3, 2)
  building_shape = (2, 3)
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  b = building.Building(
      cv_size_cm,
      floor_height_cm,
      room_shape,
      building_shape,
      initial_temp,
      inside_air_properties,
      inside_wall_properties,
      building_exterior_properties,
  )

  return b


def _create_dummy_building_deprecated_2():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  room_shape = (20, 10)
  building_shape = (6, 3)
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  b = building.Building(
      cv_size_cm,
      floor_height_cm,
      room_shape,
      building_shape,
      initial_temp,
      inside_air_properties,
      inside_wall_properties,
      building_exterior_properties,
  )
  return b


def _create_dummy_building_post_refactor():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  floor_plan = _create_dummy_floor_plan()
  zone_map = _create_dummy_floor_plan()

  b = building.FloorPlanBasedBuilding(
      cv_size_cm=cv_size_cm,
      floor_height_cm=floor_height_cm,
      initial_temp=initial_temp,
      inside_air_properties=inside_air_properties,
      inside_wall_properties=inside_wall_properties,
      building_exterior_properties=building_exterior_properties,
      floor_plan=floor_plan,
      floor_plan_filepath=None,
      zone_map=zone_map,
      zone_map_filepath=None,
      buffer_from_walls=0,
  )

  return b


def _create_dummy_floor_plan_weird_shape():
  """Creates a weirdly shaped dummy floor plan."""

  plan = np.array([
      [2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 1, 1, 1, 2, 1, 1, 1, 2],
      [2, 1, 0, 1, 1, 1, 0, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 0, 0, 1, 0, 0, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2],
  ])
  return plan


def _create_dummy_building_weird_shape():
  cv_size_cm = 20.0
  floor_height_cm = 300.0
  initial_temp = 292.0
  inside_air_properties = building.MaterialProperties(
      conductivity=50.0, heat_capacity=700.0, density=1.0
  )
  inside_wall_properties = building.MaterialProperties(
      conductivity=2.0, heat_capacity=1000.0, density=1800.0
  )
  building_exterior_properties = building.MaterialProperties(
      conductivity=0.05, heat_capacity=1000.0, density=3000.0
  )

  floor_plan = _create_dummy_floor_plan_weird_shape()
  zone_map = _create_dummy_floor_plan_weird_shape()

  b = building.FloorPlanBasedBuilding(
      cv_size_cm=cv_size_cm,
      floor_height_cm=floor_height_cm,
      initial_temp=initial_temp,
      inside_air_properties=inside_air_properties,
      inside_wall_properties=inside_wall_properties,
      building_exterior_properties=building_exterior_properties,
      floor_plan=floor_plan,
      floor_plan_filepath=None,
      zone_map=zone_map,
      zone_map_filepath=None,
      buffer_from_walls=0,
  )

  return b


def _create_dummy_room_dict():
  room_dict = {
      "exterior_space": [
          (0, 0),
          (0, 1),
          (0, 2),
          (0, 3),
          (0, 4),
          (0, 5),
          (0, 6),
          (0, 7),
          (0, 8),
          (1, 0),
          (1, 8),
          (2, 0),
          (2, 8),
          (3, 0),
          (3, 8),
          (4, 0),
          (4, 8),
          (5, 0),
          (5, 8),
          (6, 0),
          (6, 8),
          (7, 0),
          (7, 8),
          (8, 0),
          (8, 1),
          (8, 2),
          (8, 3),
          (8, 4),
          (8, 5),
          (8, 6),
          (8, 7),
          (8, 8),
      ],
      "interior_wall": [
          (1, 1),
          (1, 2),
          (1, 3),
          (1, 4),
          (1, 5),
          (1, 6),
          (1, 7),
          (2, 1),
          (2, 4),
          (2, 7),
          (3, 1),
          (3, 4),
          (3, 7),
          (4, 1),
          (4, 2),
          (4, 3),
          (4, 4),
          (4, 5),
          (4, 6),
          (4, 7),
          (5, 1),
          (5, 4),
          (5, 7),
          (6, 1),
          (6, 4),
          (6, 7),
          (7, 1),
          (7, 2),
          (7, 3),
          (7, 4),
          (7, 5),
          (7, 6),
          (7, 7),
      ],
      "room_1": [(2, 2), (2, 3), (3, 2), (3, 3)],
      "room_2": [(2, 5), (2, 6), (3, 5), (3, 6)],
      "room_3": [(5, 2), (5, 3), (6, 2), (6, 3)],
      "room_4": [(5, 5), (5, 6), (6, 5), (6, 6)],
  }

  return room_dict


class BuildingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Does not fit x 1", (6, 6), (3, 2)),
      ("Does not fit x 2", (6, 9), (3, 2)),
      ("Does not fit x 3", (12, 9), (3, 2)),
      ("Does not fit y 1", (6, 7), (2, 4)),
      ("Does not fit y 2", (12, 14), (2, 4)),
      ("Does not fit y 3", (6, 12), (2, 4)),
  )
  def test_check_room_sizes_raises_error(self, matrix_shape, room_shape):
    with self.assertRaises(ValueError):
      building._check_room_sizes(matrix_shape, room_shape)

  @parameterized.named_parameters(
      ("Does fit 1", (11, 6), (3, 2)),
      ("Does fit 2", (11, 9), (3, 2)),
      ("Does fit 3", (9, 13), (2, 4)),
  )
  def test_check_room_sizes_does_not_raise_error(
      self, matrix_shape, room_shape
  ):
    building._check_room_sizes(matrix_shape, room_shape)

  def test_init_flexible_floor_plan_direct_attributes(self):
    floor_plan = _create_dummy_floor_plan()

    cv_size_cm = 20.0
    floor_height_cm = 300.0
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    i = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
    e = constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    o = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION

    expected_exterior_space = np.array([
        [o, o, o, o, o, o, o, o, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, o],
        [o, o, o, o, o, o, o, o, o],
    ])

    expected_exterior_walls = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, e, e, e, e, e, e, e, 0],
        [0, e, 0, 0, e, 0, 0, e, 0],
        [0, e, 0, 0, e, 0, 0, e, 0],
        [0, e, e, e, 0, e, e, e, 0],
        [0, e, 0, 0, e, 0, 0, e, 0],
        [0, e, 0, 0, e, 0, 0, e, 0],
        [0, e, e, e, e, e, e, e, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    expected_interior_walls = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, i, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    expected_room_dict = {
        "exterior_space": [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 0),
            (1, 8),
            (2, 0),
            (2, 8),
            (3, 0),
            (3, 8),
            (4, 0),
            (4, 8),
            (5, 0),
            (5, 8),
            (6, 0),
            (6, 8),
            (7, 0),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        ],
        "interior_wall": [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 1),
            (2, 4),
            (2, 7),
            (3, 1),
            (3, 4),
            (3, 7),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 1),
            (5, 4),
            (5, 7),
            (6, 1),
            (6, 4),
            (6, 7),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
        ],
        "room_1": [(2, 2), (2, 3), (3, 2), (3, 3)],
        "room_2": [(2, 5), (2, 6), (3, 5), (3, 6)],
        "room_3": [(5, 2), (5, 3), (6, 2), (6, 3)],
        "room_4": [(5, 5), (5, 6), (6, 5), (6, 6)],
    }

    b = building.FloorPlanBasedBuilding(
        cv_size_cm,
        floor_height_cm,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
        floor_plan=floor_plan,
        zone_map=floor_plan,
        floor_plan_filepath=None,
        buffer_from_walls=0,
    )

    with self.subTest("floor_plans"):
      np.testing.assert_array_equal(b._floor_plan, floor_plan)
    with self.subTest("exterior_wall"):
      np.testing.assert_array_equal(b._exterior_walls, expected_exterior_walls)
    with self.subTest("interior_wall"):
      np.testing.assert_array_equal(b._interior_walls, expected_interior_walls)
    with self.subTest("exterior_space"):
      np.testing.assert_array_equal(b._exterior_space, expected_exterior_space)
    with self.subTest("room_dict"):
      self.assertEqual(b._room_dict, expected_room_dict)

  def test_assign_exterior_and_interior_attributes(self):
    e = constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    i = constants.INTERIOR_WALL_VALUE_IN_FUNCTION

    exterior_walls = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, e, e, e, e, 0],
        [0, e, 0, 0, e, 0],
        [0, e, 0, 0, e, 0],
        [0, e, e, e, e, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    interior_walls = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, i, 0, 0],
        [0, 0, 0, i, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    interior_value = 10.5
    exterior_value = 3.14
    interior_and_exterior_space_value = 0

    expected_output = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 3.14, 3.14, 3.14, 3.14, 0],
        [0, 3.14, 0, 10.5, 3.14, 0],
        [0, 3.14, 0, 10.5, 3.14, 0],
        [0, 3.14, 3.14, 3.14, 3.14, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    np.testing.assert_array_equal(
        building._assign_interior_and_exterior_values(
            exterior_walls=exterior_walls,
            interior_walls=interior_walls,
            interior_wall_value=interior_value,
            exterior_wall_value=exterior_value,
            interior_and_exterior_space_value=interior_and_exterior_space_value,
        ),
        expected_output,
    )

  @parameterized.named_parameters((
      "larger_spacing",
      10,
      np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ]),
  ))
  def test_assign_thermal_diffusers(self, diffuser_spacing, expected_output):
    room_dict = _create_dummy_room_dict()
    array_to_fill = np.zeros(shape=(9, 10))
    outcome = building._assign_thermal_diffusers(
        room_dict=room_dict,
        array_to_fill=array_to_fill,
        diffuser_spacing=diffuser_spacing,
        buffer_from_walls=0,
        interior_walls=None,
    )
    np.testing.assert_array_equal(outcome, expected_output)

  def test_init_direct_attributes(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (20, 10)
    building_shape = (6, 3)

    b = _create_dummy_building_deprecated_2()

    self.assertEqual(b.cv_size_cm, cv_size_cm)
    self.assertEqual(b.floor_height_cm, floor_height_cm)
    self.assertEqual(b.room_shape, room_shape)
    self.assertEqual(b.building_shape, building_shape)

  def test_init_matrix_shapes(self):
    # length = 4 + room_size * rooms + (rooms - 1)
    expected_width = 129  # 4 + 20 * 6 + 5
    expected_height = 36  # 4 + 10 * 3 + 2
    expected_shape = (expected_width, expected_height)

    b = _create_dummy_building_deprecated_2()

    self.assertEqual(b.temp.shape, expected_shape)
    self.assertEqual(b.conductivity.shape, expected_shape)
    self.assertEqual(b.heat_capacity.shape, expected_shape)
    self.assertEqual(b.density.shape, expected_shape)
    self.assertEqual(b.input_q.shape, expected_shape)
    self.assertEqual(b.diffusers.shape, expected_shape)

    self.assertLen(b.neighbors, expected_width)
    for i in range(expected_width):
      self.assertLen(b.neighbors[i], expected_height)

  def test_compare_rectangular_to_floor_plan_based(self):
    b_new = _create_dummy_post_refactor_building_matching_deprecation()
    b_old = _create_dummy_building_deprecated_matching_post_refactor()

    with self.subTest("cv_size"):
      self.assertEqual(b_new.cv_size_cm, b_old.cv_size_cm)
    with self.subTest("floor_height"):
      self.assertEqual(b_new.floor_height_cm, b_old.floor_height_cm)

  def test_init_matrix_shapes_compare_rect_to_floor_plan_based(self):
    b_new = _create_dummy_post_refactor_building_matching_deprecation()
    b_old = _create_dummy_building_deprecated_matching_post_refactor()

    self.assertEqual(
        b_new.temp.shape, (b_old.temp.shape[0] + 2, b_old.temp.shape[1] + 2)
    )
    self.assertEqual(
        b_new.conductivity.shape,
        (b_old.conductivity.shape[0] + 2, b_old.conductivity.shape[1] + 2),
    )
    self.assertEqual(
        b_new.heat_capacity.shape,
        (b_old.heat_capacity.shape[0] + 2, b_old.heat_capacity.shape[1] + 2),
    )
    self.assertEqual(
        b_new.density.shape,
        (b_old.density.shape[0] + 2, b_old.density.shape[1] + 2),
    )
    self.assertEqual(
        b_new.input_q.shape,
        (b_old.input_q.shape[0] + 2, b_old.input_q.shape[1] + 2),
    )
    self.assertEqual(
        b_new.diffusers.shape,
        (b_old.diffusers.shape[0] + 2, b_old.diffusers.shape[1] + 2),
    )

  def test_init_neighbors(self):
    # TODO(spangher): upon deprecation, delete this test. Reviewers, there's
    # no need to look at this test, it's temporary!
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (2, 1)
    building_shape = (1, 1)
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    # Shape is 6x5
    b = building.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    # Corners
    self.assertSameElements([(0, 1), (1, 0)], b.neighbors[0][0])
    self.assertSameElements([(0, 3), (1, 4)], b.neighbors[0][4])
    self.assertSameElements([(5, 1), (4, 0)], b.neighbors[5][0])
    self.assertSameElements([(4, 4), (5, 3)], b.neighbors[5][4])

    # Sides
    self.assertSameElements([(0, 1), (0, 3), (1, 2)], b.neighbors[0][2])
    self.assertSameElements([(1, 0), (3, 0), (2, 1)], b.neighbors[2][0])
    self.assertSameElements([(4, 3), (5, 4), (5, 2)], b.neighbors[5][3])
    self.assertSameElements([(3, 3), (2, 4), (4, 4)], b.neighbors[3][4])

    # Center
    self.assertSameElements([(0, 1), (1, 0), (1, 2), (2, 1)], b.neighbors[1][1])
    self.assertSameElements([(5, 2), (3, 2), (4, 1), (4, 3)], b.neighbors[4][2])
    self.assertSameElements([(3, 2), (2, 3), (4, 3), (3, 4)], b.neighbors[3][3])

  def test_init_neighbors_post_refactor(self):
    # TODO(spangher): upon deprecation, rename this test.
    b = _create_dummy_building_weird_shape()

    # exterior space
    with self.subTest("exterior_space_1"):
      self.assertSameElements([], b.neighbors[0][0])

    with self.subTest("exterior_space_2"):
      self.assertSameElements([], b.neighbors[0][5])

    # corner
    with self.subTest("corner_1"):
      self.assertSameElements([(1, 2), (2, 1)], b.neighbors[1][1])

    with self.subTest("corner_2"):
      self.assertSameElements([(2, 1), (4, 1), (3, 2)], b.neighbors[3][1])

    # Sides
    with self.subTest("sides_1"):
      self.assertSameElements([(1, 1), (1, 3), (2, 2)], b.neighbors[1][2])
    with self.subTest("sides_2"):
      self.assertSameElements([(1, 1), (3, 1), (2, 2)], b.neighbors[2][1])

    # Center
    with self.subTest("center_1"):
      self.assertSameElements(
          [(1, 2), (2, 1), (2, 3), (3, 2)], b.neighbors[2][2]
      )
      self.assertSameElements(
          [(5, 3), (3, 3), (4, 2), (4, 4)], b.neighbors[4][3]
      )

  # The following tests test values at a single specific location.
  # Later tests will check consistency across all edge/wall/air spaces.

  def test_building_exterior_values(self):
    initial_temp = 292.0
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    b = _create_dummy_building_deprecated_1()

    self.assertEqual(b.temp[0][0], initial_temp)
    self.assertEqual(
        b.conductivity[0][0], building_exterior_properties.conductivity
    )
    self.assertEqual(
        b.heat_capacity[0][0], building_exterior_properties.heat_capacity
    )
    self.assertEqual(b.density[0][0], building_exterior_properties.density)
    self.assertEqual(b.input_q[0][0], 0.0)

  def test_interior_wall_values(self):
    initial_temp = 292.0
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )

    b = _create_dummy_building_deprecated_2()

    self.assertEqual(b.temp[22][12], initial_temp)
    self.assertEqual(
        b.conductivity[22][12], inside_wall_properties.conductivity
    )
    self.assertEqual(
        b.heat_capacity[22][12], inside_wall_properties.heat_capacity
    )
    self.assertEqual(b.density[22][12], inside_wall_properties.density)
    self.assertEqual(b.input_q[22][12], 0.0)

  def test_enlarge_exterior_walls(self):
    e = -2
    i = -3

    ex = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, e, e, e, e, e, 0],
        [0, e, 0, 0, 0, e, 0],
        [0, e, 0, 0, 0, e, 0],
        [0, e, e, e, e, e, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    interior = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, i, 0, 0, 0],
        [0, 0, 0, i, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    expected_exterior_output = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, e, e, e, e, e, 0],
        [0, e, 0, e, 0, e, 0],
        [0, e, 0, e, 0, e, 0],
        [0, e, e, e, e, e, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    expexted_interior_output = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    exterior_output, interior_output = building.enlarge_exterior_walls(
        building_utils.ExteriorWalls(ex), building_utils.InteriorWalls(interior)
    )

    with self.subTest("exterior_output"):
      np.testing.assert_array_equal(exterior_output, expected_exterior_output)
    with self.subTest("interior_output"):
      np.testing.assert_array_equal(interior_output, expexted_interior_output)

  def test_interior_air_values(self):
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )

    b = _create_dummy_building_deprecated_1()

    self.assertEqual(b.temp[2][2], initial_temp)
    self.assertEqual(b.conductivity[2][2], inside_air_properties.conductivity)
    self.assertEqual(b.heat_capacity[2][2], inside_air_properties.heat_capacity)
    self.assertEqual(b.density[2][2], inside_air_properties.density)
    self.assertEqual(b.input_q[2][2], 0.0)

  def test_reset(self):
    initial_temp = 292.0

    b = _create_dummy_building_deprecated_1()

    b.temp[2][2] += 10.0
    b.temp[0][3] += 10.0
    b.input_q[2][2] = 1000.0
    b.input_q[0][3] = 1000.0

    b.reset()

    self.assertEqual(b.temp[2][2], initial_temp)
    self.assertEqual(b.temp[0][0], initial_temp)
    self.assertEqual(b.input_q[2][2], 0.0)
    self.assertEqual(b.input_q[0][3], 0.0)

  # The following tests are for the functions that distribute the initial values
  # to the correct locations.
  def test_assign_building_exterior_values(self):
    array = np.zeros(shape=(5, 5), dtype=np.float32)

    # Used to keep the array below cleanly formatted.
    a = 1.0  # Exterior

    expected_array = np.array(
        [
            [a, a, a, a, a],
            [a, a, a, a, a],
            [a, a, 0, a, a],
            [a, a, a, a, a],
            [a, a, a, a, a],
        ],
        dtype=np.float32,
    )

    building.assign_building_exterior_values(array, 1.0)

    np.testing.assert_array_equal(array, expected_array)

  def test_assign_interior_wall_values(self):
    room_shape = (3, 2)

    # This building shape is implied by the matrix
    # size and the room_shape
    # building_shape = (2, 3)initial

    array = np.zeros(shape=(11, 12), dtype=np.float32)

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Exterior
    i = 1.0  # Interior wall

    expected_array = np.array(
        [
            [w, w, w, w, w, w, w, w, w, w, w, w],
            [w, w, w, w, w, w, w, w, w, w, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, i, i, i, i, i, i, i, i, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
            [w, w, w, w, w, w, w, w, w, w, w, w],
            [w, w, w, w, w, w, w, w, w, w, w, w],
        ],
        dtype=np.float32,
    )

    building.assign_interior_wall_values(array, 1.0, room_shape)

    np.testing.assert_array_equal(array, expected_array)

  # Below is post-refactor of direct attributes (excluding thermal diffusers):

  def test_init_direct_attributes_post_refactor(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    b = _create_dummy_building_post_refactor()

    with self.subTest("air"):
      self.assertEqual(b.cv_size_cm, cv_size_cm)

    with self.subTest("floor_height"):
      self.assertEqual(b.floor_height_cm, floor_height_cm)

  def test_building_exterior_values_flexible_floor_plan(self):
    initial_temp = 292.0

    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    b = _create_dummy_building_post_refactor()

    with self.subTest("temp"):
      self.assertEqual(b.temp[1][1], initial_temp)

    with self.subTest("properties"):
      self.assertEqual(
          b.conductivity[1][1], building_exterior_properties.conductivity
      )

    with self.subTest("heat_capacity"):
      self.assertEqual(
          b.heat_capacity[1][1], building_exterior_properties.heat_capacity
      )

    with self.subTest("density"):
      self.assertEqual(b.density[1][1], building_exterior_properties.density)

    with self.subTest("input_q"):
      self.assertEqual(b.input_q[1][1], 0.0)

  def test_interior_wall_values_flexible_floor_plan(self):
    initial_temp = 292.0
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )

    b = _create_dummy_building_post_refactor()

    self.assertEqual(b.temp[4][4], initial_temp)
    self.assertEqual(b.conductivity[4][4], inside_wall_properties.conductivity)
    self.assertEqual(
        b.heat_capacity[4][4], inside_wall_properties.heat_capacity
    )
    self.assertEqual(b.density[4][4], inside_wall_properties.density)
    self.assertEqual(b.input_q[4][4], 0.0)

  def test_interior_air_values_flexible_floor_plan(self):
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )

    b = _create_dummy_building_post_refactor()

    with self.subTest("temp"):
      self.assertEqual(b.temp[2][2], initial_temp)

    with self.subTest("properties"):
      self.assertEqual(b.conductivity[2][2], inside_air_properties.conductivity)

    with self.subTest("heat_capacity"):
      self.assertEqual(
          b.heat_capacity[2][2], inside_air_properties.heat_capacity
      )

    with self.subTest("density"):
      self.assertEqual(b.density[2][2], inside_air_properties.density)

    with self.subTest("input_q"):
      self.assertEqual(b.input_q[2][2], 0.0)

  def test_reset_flexible_floor_plan(self):
    initial_temp = 292.0

    b = _create_dummy_building_post_refactor()

    b.temp[2][2] += 10.0
    b.temp[0][3] += 10.0
    b.input_q[2][2] = 1000.0
    b.input_q[0][3] = 1000.0

    b.reset()

    self.assertEqual(b.temp[2][2], initial_temp)
    self.assertEqual(b.temp[0][0], initial_temp)
    self.assertEqual(b.input_q[2][2], 0.0)
    self.assertEqual(b.input_q[0][3], 0.0)

  # The following tests are for the functions that distribute the initial values
  # to the correct locations.

  def test_assign_building_values_flexible_floor_plan(self):
    # Used to keep the array below cleanly formatted.
    e = constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    i = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
    exterior_walls = np.array(
        [
            [e, e, e, e, e],
            [e, 0, 0, 0, e],
            [e, 0, 0, 0, e],
            [e, 0, 0, 0, e],
            [e, e, e, e, e],
        ],
        dtype=np.float32,
    )
    interior_walls = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, i, i, i, 0],
            [0, i, 0, i, 0],
            [0, i, i, i, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    e_to_fill = 5
    i_to_fill = 3
    i_and_e_to_fill = 0

    expected_array = np.array(
        [
            [5, 5, 5, 5, 5],
            [5, 3, 3, 3, 5],
            [5, 3, 0, 3, 5],
            [5, 3, 3, 3, 5],
            [5, 5, 5, 5, 5],
        ],
        dtype=np.float32,
    )

    outcome = building._assign_interior_and_exterior_values(
        exterior_walls, interior_walls, i_to_fill, e_to_fill, i_and_e_to_fill
    )

    np.testing.assert_array_equal(outcome, expected_array)

  def test_assign_interior_wall_values_flexible_floor_plan(self):
    # Used to keep the array below cleanly formatted.
    w = 5.0  # Exterior
    i = 1.0  # Interior wall

    expected_array = np.array([
        [w, w, w, w, w, w, w, w, w, w, w, w],
        [w, w, w, w, w, w, w, w, w, w, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, i, i, i, i, i, i, i, i, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, 0, 0, i, 0, 0, i, 0, 0, w, w],
        [w, w, w, w, w, w, w, w, w, w, w, w],
        [w, w, w, w, w, w, w, w, w, w, w, w],
    ])

    interior_walls = np.zeros(shape=(11, 12), dtype=np.float32)
    interior_walls[expected_array == 1.0] = (
        constants.INTERIOR_WALL_VALUE_IN_FUNCTION
    )
    exterior_walls = np.zeros(shape=(11, 12), dtype=np.float32)
    exterior_walls[expected_array == 5.0] = (
        constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
    )

    array_to_fill = building._assign_interior_and_exterior_values(
        exterior_walls=exterior_walls,
        interior_walls=interior_walls,
        interior_wall_value=1.0,
        exterior_wall_value=5.0,
        interior_and_exterior_space_value=0,
    )

    np.testing.assert_array_equal(array_to_fill, expected_array)

  # Below is pre-refactor generation of thermal diffusers:

  def test_generate_thermal_diffusers_4x5(self):
    matrix_shape = (13, 15)
    # building_shape = (2, 2)
    room_shape = (4, 5)

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    d = 0.25  # Diffuser (four diffusers in each zone so value is 1/4)

    expected_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, d, 0, d, 0, w, 0, d, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, d, 0, w, 0, d, 0, d, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, d, 0, d, 0, w, 0, d, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, d, 0, w, 0, d, 0, d, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    diffusers = building.generate_thermal_diffusers(matrix_shape, room_shape)

    np.testing.assert_array_equal(diffusers, expected_array)

  def test_generate_thermal_diffusers_6x7(self):
    matrix_shape = (17, 19)
    # building_shape = (2, 2)
    room_shape = (6, 7)

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    d = 0.25  # Diffuser (four diffusers in each zone so value is 1/4)

    expected_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, 0, 0, d, 0, w, 0, d, 0, 0, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, 0, 0, d, 0, w, 0, d, 0, 0, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, 0, 0, d, 0, w, 0, d, 0, 0, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, 0, d, 0, 0, 0, d, 0, w, 0, d, 0, 0, 0, d, 0, w, 0],
            [0, w, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    diffusers = building.generate_thermal_diffusers(matrix_shape, room_shape)

    np.testing.assert_array_equal(diffusers, expected_array)

  @parameterized.named_parameters(
      ("2x2 room 0,0", (0, 0), (2, 2), (2, 3, 2, 3)),
      ("2x2 room 1,0", (1, 0), (2, 2), (5, 6, 2, 3)),
      ("2x2 room 0,1", (0, 1), (2, 2), (2, 3, 5, 6)),
      ("3x8 room 4,7", (4, 7), (3, 8), (18, 20, 65, 72)),
  )
  def test_get_zone_bounds(self, zone_coordinates, room_shape, expected):
    zone_bounds = building.get_zone_bounds(zone_coordinates, room_shape)

    self.assertEqual(zone_bounds, expected)

  ## Below are old tests (pre-refactor) for generating thermal energy etc.

  def test_get_zone_thermal_energy_rate(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (3, 2)
    building_shape = (2, 3)
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    expected_zone_0_0_rate = 16.5
    expected_zone_1_1_rate = -9.0

    badlands = building.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    a = 1.5  # Room 0,0 temp 1
    b = 4.0  # Room 0,0 temp 2
    c = -1.0  # Room 1,1 temp 1
    d = -2.0  # Room 1,1 temp 2

    badlands.input_q = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, a, b, w, 0, 0, w, 0, 0, w, 0],
            [0, w, b, a, w, 0, 0, w, 0, 0, w, 0],
            [0, w, a, b, w, 0, 0, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, w, c, d, w, 0, 0, w, 0],
            [0, w, 0, 0, w, d, c, w, 0, 0, w, 0],
            [0, w, 0, 0, w, c, d, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    zone_0_0_rate = badlands.get_zone_thermal_energy_rate((0, 0))
    zone_1_1_rate = badlands.get_zone_thermal_energy_rate((1, 1))

    self.assertEqual(zone_0_0_rate, expected_zone_0_0_rate)
    self.assertEqual(zone_1_1_rate, expected_zone_1_1_rate)

  def test_get_zone_temp_stats(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (3, 2)
    building_shape = (2, 3)
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    # Min, max, mean
    expected_zone_0_0_temp_stats = (1.5, 4.0, 2.75)
    expected_zone_1_1_temp_stats = (-2.0, -1.0, -1.5)

    grand_central = building.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    a = 1.5  # Room 0,0 temp 1
    b = 4.0  # Room 0,0 temp 2
    c = -1.0  # Room 1,1 temp 1
    d = -2.0  # Room 1,1 temp 2

    grand_central.temp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, a, b, w, 0, 0, w, 0, 0, w, 0],
            [0, w, b, a, w, 0, 0, w, 0, 0, w, 0],
            [0, w, a, b, w, 0, 0, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, w, c, d, w, 0, 0, w, 0],
            [0, w, 0, 0, w, d, c, w, 0, 0, w, 0],
            [0, w, 0, 0, w, c, d, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    zone_0_0_temp_stats = grand_central.get_zone_temp_stats((0, 0))
    zone_1_1_temp_stats = grand_central.get_zone_temp_stats((1, 1))

    self.assertEqual(zone_0_0_temp_stats, expected_zone_0_0_temp_stats)
    self.assertEqual(zone_1_1_temp_stats, expected_zone_1_1_temp_stats)

  def test_get_zone_average_temps(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (3, 2)
    building_shape = (2, 3)
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    a = 1.0  # Room 0,0 temp
    b = 2.0  # Room 0,1 temp
    c = 3.0  # Room 0,2 temp
    d = 4.0  # Room 1,0 temp
    f = 5.0  # Room 1,1 temp
    g = 6.0  # Room 1,2 temp

    expected_avg_temps = {
        (0, 0): a,
        (0, 1): b,
        (0, 2): c,
        (1, 0): d,
        (1, 1): f,
        (1, 2): g,
    }

    spear = building.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall

    spear.temp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, a, a, w, b, b, w, c, c, w, 0],
            [0, w, a, a, w, b, b, w, c, c, w, 0],
            [0, w, a, a, w, b, b, w, c, c, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, d, d, w, f, f, w, g, g, w, 0],
            [0, w, d, d, w, f, f, w, g, g, w, 0],
            [0, w, d, d, w, f, f, w, g, g, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    avg_temps = spear.get_zone_average_temps()

    self.assertDictEqual(avg_temps, expected_avg_temps)

  def test_apply_thermal_power_zone(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (3, 2)
    building_shape = (2, 3)
    initial_temp = 292.0
    inside_air_properties = building.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    b = building.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )

    input_power = 10.0

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    h = input_power / 4.0  # Power split between four diffusers

    expected_input_q = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, w, 0, 0, w, 0, 0, w, 0],
            [0, w, 0, 0, w, 0, 0, w, 0, 0, w, 0],
            [0, w, 0, 0, w, 0, 0, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, w, h, h, w, 0, 0, w, 0, 0, w, 0],
            [0, w, 0, 0, w, 0, 0, w, 0, 0, w, 0],
            [0, w, h, h, w, 0, 0, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    b.apply_thermal_power_zone((1, 0), 10.0)

    np.testing.assert_array_equal(b.input_q, expected_input_q)

  # Below are new tests for applying thermal energy rates, post refactor

  def test_assign_diffusers_post_refactor(self):
    b = _create_dummy_building_post_refactor()
    expected_diffuser_array_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    np.testing.assert_array_equal(b.diffusers, expected_diffuser_array_1)

  @parameterized.named_parameters(
      ("ex_space", "exterior_space", 0.0),
      ("1", "room_1", 11.0),
      ("2", "room_2", 0.0),
      ("3", "room_3", 0.0),
      ("4", "room_4", -6.0),
      ("i_wall", "interior_wall", 0.0),
  )
  def test_get_zone_thermal_enery_rate_post_refactor(
      self, zone_name, expected_outcome
  ):
    b = _create_dummy_building_post_refactor()

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    a = 1.5  # Room 0,0 temp 1
    e = 4.0  # Room 0,0 temp 2
    c = -1.0  # Room 1,1 temp 1
    d = -2.0  # Room 1,1 temp 2

    b.input_q = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, a, e, w, 0, 0, w, 0],
            [0, w, a, e, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, w, c, c, w, 0],
            [0, w, 0, 0, w, d, d, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        dtype=np.float32,
    )

    outcome = b.get_zone_thermal_energy_rate(zone_name)

    self.assertEqual(outcome, expected_outcome)

  @parameterized.named_parameters(
      ("ex_space", "exterior_space", (0.0, 0.0, 0.0)),
      ("1", "room_1", (1.5, 4.0, 2.75)),
      ("2", "room_2", (0.0, 0.0, 0.0)),
      ("3", "room_3", (0.0, 0.0, 0.0)),
      ("int_wall", "interior_wall", (0.0, 0.0, 0.0)),
      ("4", "room_4", (-2.0, -1.0, -1.5)),
  )
  def test_get_zone_temp_stats_post_refactor(self, zone_name, expected_outcome):
    b = _create_dummy_building_post_refactor()

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    a = 1.5  # Room 0,0 temp 1
    e = 4.0  # Room 0,0 temp 2
    c = -1.0  # Room 1,1 temp 1
    d = -2.0  # Room 1,1 temp 2

    b.temp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, a, e, w, 0, 0, w, 0],
            [0, w, a, e, w, 0, 0, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, 0, 0, w, c, c, w, 0],
            [0, w, 0, 0, w, d, d, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    self.assertEqual(b.get_zone_temp_stats(zone_name), expected_outcome)

  def test_get_zone_average_temps_post_refactor(self):
    b = _create_dummy_building_post_refactor()

    # Used to keep the array below cleanly formatted.
    w = 0.0  # Wall
    a = 1.5  # Room 0,0 temp 1
    e = 4.0  # Room 0,0 temp 2
    c = -1.0  # Room 1,1 temp 1
    d = -2.0  # Room 1,1 temp 2

    b.temp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, a, a, w, e, e, w, 0],
            [0, w, a, a, w, e, e, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, w, c, c, w, d, d, w, 0],
            [0, w, c, c, w, d, d, w, 0],
            [0, w, w, w, w, w, w, w, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    expected_average_temps = {
        "room_1": 1.5,
        "room_2": 4.0,
        "room_3": -1.0,
        "room_4": -2.0,
    }

    outcome = b.get_zone_average_temps()

    self.assertEqual(outcome, expected_average_temps)

  def test_apply_thermal_power_zone_post_refactor(self):
    b = _create_dummy_building_post_refactor()
    input_power = 10.0

    expected_diffuser_array_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    for zone in b._room_dict.keys():
      b.apply_thermal_power_zone(zone, input_power)

    expected_input_1 = expected_diffuser_array_1 * input_power
    np.testing.assert_array_equal(b.input_q, expected_input_1)

  @parameterized.named_parameters(
      ("shuffle prob 1 seed 10", 1, [4, 3, 2, 1], 10),
      ("shuffle prob 0.5 seed 10", 0.5, [2, 1, 4, 3], 10),
      ("shuffle prob 0.5 seed 20", 0.5, [1, 2, 3, 4], 20),
      ("shuffle prob 0.5 seed 30", 0.5, [2, 1, 3, 4], 30),
      ("shuffle prob 0.0 seed 20", 0.0, [1, 2, 3, 4], 20),
      ("shuffle prob 0.0 seed 30", 0.0, [1, 2, 3, 4], 30),
      ("shuffle prob 0.0 seed 40", 0.0, [1, 2, 3, 4], 40),
  )
  def test_stochastic_convection_simulator_shuffle_no_max_dist(
      self, p, vals, seed
  ):
    b = _create_dummy_building_post_refactor()

    # add convection simulator
    b._convection_simulator = (
        stochastic_convection_simulator.StochasticConvectionSimulator(
            p=p, distance=-1, seed=seed
        )
    )
    # assert all temps start at 292.0, as expected
    for i in range(b.temp.shape[0]):
      for j in range(b.temp.shape[1]):
        self.assertEqual(b.temp[i][j], 292.0)

    # now lets change the temps in a room
    b.temp[2][2] = 1
    b.temp[2][3] = 2
    b.temp[3][2] = 3
    b.temp[3][3] = 4

    self.assertEqual(b.temp[2][2], 1)
    self.assertEqual(b.temp[2][3], 2)
    self.assertEqual(b.temp[3][2], 3)
    self.assertEqual(b.temp[3][3], 4)

    b.apply_convection()

    # since we seeded the random number generator, we can assert exact vals
    self.assertEqual(b.temp[2][2], vals[0])
    self.assertEqual(b.temp[2][3], vals[1])
    self.assertEqual(b.temp[3][2], vals[2])
    self.assertEqual(b.temp[3][3], vals[3])

    # assert all temps in other rooms are not affected
    for i in range(b.temp.shape[0]):
      for j in range(b.temp.shape[1]):
        if (i, j) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
          continue
        self.assertEqual(b.temp[i][j], 292.0)

  @parameterized.named_parameters(
      ("shuffle prob 1 seed 10 dist 0", 1, [1, 2, 3, 4], 10, 0),
      ("shuffle prob 1 seed 20 dist 0", 1, [1, 2, 3, 4], 20, 0),
      ("shuffle prob 0 seed 10 dist 5", 0, [1, 2, 3, 4], 10, 5),
      ("shuffle prob 0 seed 20 dist 5", 0, [1, 2, 3, 4], 20, 5),
      ("shuffle prob 1 seed 10 dist 1", 1, [2, 1, 3, 4], 10, 1),
      ("shuffle prob 1 seed 20 dist 1", 1, [3, 1, 4, 2], 20, 1),
      ("shuffle prob 1 seed 30 dist 1", 1, [2, 3, 1, 4], 30, 1),
      ("shuffle prob 1 seed 40 dist 1", 1, [2, 1, 3, 4], 40, 1),
      ("shuffle prob 1 seed 50 dist 1", 1, [2, 1, 4, 3], 50, 1),
      ("shuffle prob 1 seed 60 dist 1", 1, [1, 4, 3, 2], 60, 1),
      ("shuffle prob 1 seed 50 dist 2", 1, [4, 2, 1, 3], 50, 2),
  )
  def test_stochastic_convection_simulator_shuffle_max_dist(
      self, p, vals, seed, distance
  ):
    b = _create_dummy_building_post_refactor()

    # add convection simulator
    b._convection_simulator = (
        stochastic_convection_simulator.StochasticConvectionSimulator(
            p=p, distance=distance, seed=seed
        )
    )
    # assert all temps start at 292.0, as expected
    for i in range(b.temp.shape[0]):
      for j in range(b.temp.shape[1]):
        self.assertEqual(b.temp[i][j], 292.0)

    # now lets change the temps in a room
    b.temp[2][2] = 1
    b.temp[2][3] = 2
    b.temp[3][2] = 3
    b.temp[3][3] = 4

    self.assertEqual(b.temp[2][2], 1)
    self.assertEqual(b.temp[2][3], 2)
    self.assertEqual(b.temp[3][2], 3)
    self.assertEqual(b.temp[3][3], 4)

    b.apply_convection()

    # since we seeded the random number generator, we can assert exact vals
    # when max dist is 1, [2][2] cannot swap with [3][3], nor [2][3] with [3][2]
    self.assertEqual(b.temp[2][2], vals[0])
    self.assertEqual(b.temp[2][3], vals[1])
    self.assertEqual(b.temp[3][2], vals[2])
    self.assertEqual(b.temp[3][3], vals[3])

    # assert all temps in other rooms are not affected
    for i in range(b.temp.shape[0]):
      for j in range(b.temp.shape[1]):
        if (i, j) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
          continue
        self.assertEqual(b.temp[i][j], 292.0)

    # lets reset and try again, to make sure the chache works
    # now lets change the temps in a room
    b.temp[2][2] = 1
    b.temp[2][3] = 2
    b.temp[3][2] = 3
    b.temp[3][3] = 4

    self.assertEqual(b.temp[2][2], 1)
    self.assertEqual(b.temp[2][3], 2)
    self.assertEqual(b.temp[3][2], 3)
    self.assertEqual(b.temp[3][3], 4)
    random.seed(seed)

    b.apply_convection()

    # since we seeded the random number generator, we can assert exact vals
    # when max dist is 1, [2][2] cannot swap with [3][3], nor [2][3] with [3][2]
    # this time, we are using cache for efficiency
    self.assertEqual(b.temp[2][2], vals[0])
    self.assertEqual(b.temp[2][3], vals[1])
    self.assertEqual(b.temp[3][2], vals[2])
    self.assertEqual(b.temp[3][3], vals[3])


if __name__ == "__main__":
  absltest.main()
