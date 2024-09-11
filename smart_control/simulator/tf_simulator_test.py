"""Tests for Tensorflow-enabled Finite Difference calculator.

Copyright 2024 Google LLC

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

from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import tf_simulator as tf_simulator_py
from smart_control.simulator import weather_controller as weather_controller_py
import tensorflow as tf


class TFSimulatorTest(absltest.TestCase):

  ## Part 1: Tests for matrix utilities used by TFSimulator
  def _create_test_floor_plan(self):
    """Creates a test floor plan."""

    plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 2, 2, 2, 1, 1, 2],
        [2, 1, 0, 1, 2, 1, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])

    return plan

  def _get_boundary_cv_mapping_5x6(self):
    return {
        (1, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_LEFT,
        ),
        (1, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.TOP,
        ),
        (1, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.TOP,
        ),
        (1, 4): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_RIGHT,
        ),
        (2, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.LEFT,
        ),
        (2, 4): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.RIGHT,
        ),
        (3, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_LEFT,
        ),
        (3, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.BOTTOM,
        ),
        (3, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.BOTTOM,
        ),
        (3, 4): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_RIGHT,
        ),
    }

  def _create_test_building(self):
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    initial_temp = 292.0
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=700.0, density=1.0
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=1000.0, density=1800.0
    )
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=1000.0, density=3000.0
    )

    floor_plan = self._create_test_floor_plan()
    zone_map = self._create_test_floor_plan()

    b = building_py.FloorPlanBasedBuilding(
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

  def _create_small_hvac(self):
    """Returns hvac matching zones for small test building."""
    reheat_water_setpoint = 260
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    boiler = boiler_py.Boiler(
        reheat_water_setpoint,
        water_pump_differential_head,
        water_pump_efficiency,
        "boiler_id",
    )

    recirculation = 0.3
    heating_air_temp_setpoint = 270
    cooling_air_temp_setpoint = 288
    fan_differential_pressure = 20000.0
    fan_efficiency = 0.8

    air_handler = air_handler_py.AirHandler(
        recirculation,
        heating_air_temp_setpoint,
        cooling_air_temp_setpoint,
        fan_differential_pressure,
        fan_efficiency,
    )

    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 223, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    zone_identifier = ["room_1", "room_2"]

    hvac = floorplan_hvac_py.FloorPlanBasedHvac(
        zone_identifier=zone_identifier,
        air_handler=air_handler,
        boiler=boiler,
        schedule=schedule,
        vav_max_air_flow_rate=0.45,
        vav_reheat_max_water_flow_rate=0.02,
    )
    return hvac

  def test_get_cv_mapping_boundary(self):
    test_neighbors = [
        [[], [], [(1, 2)], []],
        [
            [],
            [(0, 1), (2, 1), (1, 0), (1, 2)],
            [(2, 2), (1, 3)],
            [(2, 3), (1, 2)],
        ],
        [[], [(1, 1), (2, 2)], [(2, 3), (2, 1), (3, 2)], [(1, 3), (2, 2)]],
        [
            [],
            [(3, 2), (3, 0), (4, 1), (4, 2)],
            [(3, 1), (3, 3), (2, 2)],
            [(2, 3), (3, 4), (4, 3)],
        ],
        [[], [], [(3, 2), (4, 1), (5, 2)], []],
    ]
    expected_cv_mapping = {
        (1, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_LEFT,
        ),
        (1, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_RIGHT,
        ),
        (2, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_LEFT,
        ),
        (2, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_RIGHT,
        ),
        (2, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.TOP,
        ),
        (3, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.BOTTOM,
        ),
        (3, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.LEFT,
        ),
        (4, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.RIGHT,
        ),
    }
    cv_mapping = tf_simulator_py.get_cv_mapping(
        test_neighbors, tf_simulator_py.CVPositionType.BOUNDARY
    )

    self.assertEqual(
        cv_mapping,
        expected_cv_mapping,
    )

  def test_cv_dimension_tensors(self):

    expected_horizontal_dims = np.array([
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 50.0, 100.0, 100.0, 50.0, 100.0],
        [100.0, 50.0, 100.0, 100.0, 50.0, 100.0],
        [100.0, 50.0, 100.0, 100.0, 50.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    ])

    expected_vertical_dims = np.array([
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 50.0, 50.0, 50.0, 50.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 50.0, 50.0, 50.0, 50.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    ])

    control_volume_cm = 100.0
    shape = (5, 6)

    boundary_cv_mapping = self._get_boundary_cv_mapping_5x6()

    t_horizontal_cv_dimension, t_vertical_cv_dimension = (
        tf_simulator_py.get_cv_dimension_tensors(
            control_volume_cm, boundary_cv_mapping, shape
        )
    )
    horizontal_cv_dimension = t_horizontal_cv_dimension.numpy()
    vertical_cv_dimension = t_vertical_cv_dimension.numpy()

    np.testing.assert_array_equal(
        horizontal_cv_dimension, expected_horizontal_dims
    )
    np.testing.assert_array_equal(vertical_cv_dimension, expected_vertical_dims)

  def test_get_cv_mapping_interior(self):
    test_neighbors = [
        [[], [], [(1, 2)], []],
        [
            [],
            [(0, 1), (2, 1), (1, 0), (1, 2)],
            [(2, 2), (1, 3)],
            [(2, 3), (1, 2)],
        ],
        [[], [(1, 1), (2, 2)], [(2, 3), (2, 1), (3, 2)], [(1, 3), (2, 2)]],
        [
            [],
            [(3, 2), (3, 0), (4, 1), (4, 2)],
            [(3, 1), (3, 3), (2, 2)],
            [(2, 3), (3, 4), (4, 3)],
        ],
        [[], [], [(3, 2), (4, 1), (5, 2)], []],
    ]

    expected_cv_mapping = {
        (1, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.INTERIOR,
        ),
        (3, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.INTERIOR,
        ),
    }
    cv_mapping = tf_simulator_py.get_cv_mapping(
        test_neighbors, tf_simulator_py.CVPositionType.INTERIOR
    )

    self.assertEqual(
        cv_mapping,
        expected_cv_mapping,
    )

  def test_get_cv_mapping_exterior(self):
    test_neighbors = [
        [[], [], [(1, 2)], []],
        [
            [],
            [(0, 1), (2, 1), (1, 0), (1, 2)],
            [(2, 2), (1, 3)],
            [(2, 3), (1, 2)],
        ],
        [[], [(1, 1), (2, 2)], [(2, 3), (2, 1), (3, 2)], [(1, 3), (2, 2)]],
        [
            [],
            [(3, 2), (3, 0), (4, 1), (4, 2)],
            [(3, 1), (3, 3), (2, 2)],
            [(2, 3), (3, 4), (4, 3)],
        ],
        [[], [], [(3, 2), (4, 1), (5, 2)], []],
    ]
    expected_cv_mapping = {
        (1, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_LEFT,
        ),
        (1, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.TOP_RIGHT,
        ),
        (2, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_LEFT,
        ),
        (2, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.CORNER,
            corner=tf_simulator_py.CVCornerOrientationType.BOTTOM_RIGHT,
        ),
        (2, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.TOP,
        ),
        (3, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.BOTTOM,
        ),
        (3, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.LEFT,
        ),
        (4, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.BOUNDARY,
            boundary=tf_simulator_py.CVBoundaryType.EDGE,
            edge=tf_simulator_py.CVEdgeOrientationType.RIGHT,
        ),
    }
    cv_mapping = tf_simulator_py.get_cv_mapping(
        test_neighbors, tf_simulator_py.CVPositionType.BOUNDARY
    )

    self.assertEqual(
        cv_mapping,
        expected_cv_mapping,
    )
    expected_cv_mapping = {
        (1, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.INTERIOR,
        ),
        (3, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.INTERIOR,
        ),
    }
    cv_mapping = tf_simulator_py.get_cv_mapping(
        test_neighbors, tf_simulator_py.CVPositionType.INTERIOR
    )

    self.assertEqual(
        cv_mapping,
        expected_cv_mapping,
    )
    expected_cv_mapping = {
        (0, 0): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (0, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (0, 2): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (0, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (1, 0): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (2, 0): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (3, 0): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (4, 0): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (4, 1): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
        (4, 3): tf_simulator_py.CVType(
            position=tf_simulator_py.CVPositionType.EXTERIOR,
        ),
    }
    cv_mapping = tf_simulator_py.get_cv_mapping(
        test_neighbors, tf_simulator_py.CVPositionType.EXTERIOR
    )

    self.assertEqual(
        cv_mapping,
        expected_cv_mapping,
    )

  def test_get_oriented_convection_coefficient_tensors(self):

    expected_h_left_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_h_right_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_h_top_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_h_bottom_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    convection_coefficient = 1.0
    boundary_cv_mapping = self._get_boundary_cv_mapping_5x6()
    shape = (5, 6)

    t_h_left_edge, t_h_right_edge, t_h_top_edge, t_h_bottom_edge = (
        tf_simulator_py.get_oriented_convection_coefficient_tensors(
            convection_coefficient, shape, boundary_cv_mapping
        )
    )
    h_left_edge = t_h_left_edge.numpy()
    h_right_edge = t_h_right_edge.numpy()
    h_top_edge = t_h_top_edge.numpy()
    h_bottom_edge = t_h_bottom_edge.numpy()

    np.testing.assert_array_equal(h_left_edge, expected_h_left_edge)
    np.testing.assert_array_equal(h_right_edge, expected_h_right_edge)
    np.testing.assert_array_equal(h_top_edge, expected_h_top_edge)
    np.testing.assert_array_equal(h_bottom_edge, expected_h_bottom_edge)

  def test_get_oriented_conductivity_tensors(self):

    boundary_cv_mapping = self._get_boundary_cv_mapping_5x6()

    conductivity = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_k_left_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_k_right_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_k_top_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_k_bottom_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    t_k_left_edge, t_k_right_edge, t_k_top_edge, t_k_bottom_edge = (
        tf_simulator_py.get_oriented_conductivity_tensors(
            conductivity, boundary_cv_mapping
        )
    )
    k_left_edge = t_k_left_edge.numpy()
    k_right_edge = t_k_right_edge.numpy()
    k_top_edge = t_k_top_edge.numpy()
    k_bottom_edge = t_k_bottom_edge.numpy()
    np.testing.assert_array_equal(k_left_edge, expected_k_left_edge)
    np.testing.assert_array_equal(k_right_edge, expected_k_right_edge)
    np.testing.assert_array_equal(k_top_edge, expected_k_top_edge)
    np.testing.assert_array_equal(k_bottom_edge, expected_k_bottom_edge)

  def test_shift_tensors(self):
    padding_value = 7
    x = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
    t_x = tf.convert_to_tensor(x)

    expected_x_left = np.array([[1, 1, 7], [1, 1, 7]], dtype=np.float32)
    expected_x_right = np.array([[7, 1, 1], [7, 1, 1]], dtype=np.float32)
    expected_x_up = np.array([[1, 1, 1], [7, 7, 7]], dtype=np.float32)
    expected_x_down = np.array([[7, 7, 7], [1, 1, 1]], dtype=np.float32)

    x_left = tf_simulator_py.shift_tensor_left(
        t_x,
        padding_value=padding_value,
    ).numpy()
    x_right = tf_simulator_py.shift_tensor_right(
        t_x,
        padding_value=padding_value,
    ).numpy()
    x_up = tf_simulator_py.shift_tensor_up(
        t_x,
        padding_value=padding_value,
    ).numpy()
    x_down = tf_simulator_py.shift_tensor_down(
        t_x,
        padding_value=padding_value,
    ).numpy()
    np.testing.assert_array_equal(x_left, expected_x_left)
    np.testing.assert_array_equal(x_right, expected_x_right)
    np.testing.assert_array_equal(x_up, expected_x_up)
    np.testing.assert_array_equal(x_down, expected_x_down)

  def test_apply_exterior_temps(self):
    temps_in = np.array(
        [
            [285, 285, 285, 285, 285, 285],
            [285, 285, 285, 285, 285, 285],
            [285, 285, 285, 285, 285, 285],
            [285, 285, 285, 285, 285, 285],
            [285, 285, 285, 285, 285, 285],
        ],
        dtype=np.float32,
    )
    t_temps_in = tf.convert_to_tensor(temps_in)
    expected_temps = np.array(
        [
            [270, 270, 270, 270, 270, 270],
            [270, 285, 285, 285, 285, 270],
            [270, 285, 285, 285, 285, 270],
            [270, 285, 285, 285, 285, 270],
            [270, 270, 270, 270, 270, 270],
        ],
        dtype=np.float32,
    )

    exterior_temps_mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, False, False, False, False, True],
            [True, True, True, True, True, True],
        ],
        dtype=np.bool_,
    )
    t_exterior_temps_mask = tf.convert_to_tensor(exterior_temps_mask)

    temps_out = tf_simulator_py.apply_exterior_temps(
        t_temps_in, 270.0, t_exterior_temps_mask
    ).numpy()
    np.testing.assert_array_equal(temps_out, expected_temps)

  ## Part2: Tests of the FD calculator
  def test_finite_difference_one_step(self):
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 4
    iteration_warning = 2
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_test_building()

    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    temps_estimate_in = building.temp.copy()

    _, max_delta = tf_simulator.update_temperature_estimates(
        temps_estimate_in,
        ambient_temperature=285.0,
        convection_coefficient=12.0,
    )

    self.assertLessEqual(7.0, max_delta)

  def test_finite_difference_convergence(self):
    """Tests that the FD problem converges within a fixed numbe of steps."""
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 4
    iteration_warning = 2
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_test_building()

    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    result = tf_simulator.finite_differences_timestep(
        ambient_temperature=285.0, convection_coefficient=12.0
    )
    self.assertTrue(result)


if __name__ == "__main__":
  absltest.main()
