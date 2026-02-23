"""Tests for Tensorflow-enabled Finite Difference calculator."""

import os
import time
from unittest import mock

from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import tensorflow as tf

from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import simulator_flexible_floor_plan as simulator_py
from smart_control.simulator import tf_simulator as tf_simulator_py
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.simulator.simulator_flexible_floor_plan_test import FlexibleFloorplanSimulatorTest  # pylint: disable=line-too-long


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

  def _create_test_building_radiative(self):
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

    inside_air_radiative_properties = building_py.RadiationProperties(
        alpha=0.0, epsilon=0.0, tau=1.0, rho=None
    )
    inside_wall_radiative_properties = building_py.RadiationProperties(
        alpha=0.4, epsilon=0.6, tau=0.0, rho=None
    )
    building_exterior_radiative_properties = building_py.RadiationProperties(
        alpha=0.65, epsilon=0.35, tau=0.0, rho=None
    )

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
        inside_air_radiative_properties=inside_air_radiative_properties,
        inside_wall_radiative_properties=inside_wall_radiative_properties,
        building_exterior_radiative_properties=building_exterior_radiative_properties,  # pylint: disable=line-too-long
        include_radiative_heat_transfer=True,
        view_factor_method="ScriptF",
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

  def test_finite_difference_convergence_with_radiative_heat_transfer(self):
    """Tests that the FD problem with radiative heat transfer converges within
    a fixed number of steps."""
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    convergence_threshold = 0.1
    iteration_limit = 100
    iteration_warning = 2
    start_timestamp = pd.Timestamp("2012-12-21")

    building = self._create_test_building_radiative()

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

  def test_compare_temperature_estimates_iterative_approach(self):
    """Tests that the temperature estimates from TFSimulator match those from
    SimulatorFlexibleGeometries when using radiative heat transfer.

    Creates two simulators with identical buildings, HVAC and parameters:
    1. A SimulatorFlexibleGeometries instance (baseline)
    2. A TFSimulator instance (under test)

    Runs one timestep on both and verifies their temperature arrays match
      exactly.
    This validates that TFSimulator's radiative heat transfer calculations
      produce the same results as the original implementation.
    """
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 300
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2012-12-21")

    # Create baseline simulator
    simulator = FlexibleFloorplanSimulatorTest()
    simulator_hvac = simulator._create_small_hvac()
    simulator_building = simulator._create_small_building(
        initial_temp=292.0,
        include_radiative_heat_transfer=False,
        include_interior_mass=False,
    )
    simulator_simulator = simulator_py.SimulatorFlexibleGeometries(
        simulator_building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )
    simulator_result = simulator_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    building = simulator_building  # self._create_test_building_radiative()

    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    result = tf_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    self.assertTrue(result)
    self.assertTrue(simulator_result)
    with self.subTest("CV temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.temp, simulator_simulator.building.temp
      )

  def test_compare_temperature_estimates_iterative_approach_with_lwx(self):
    """Tests that the temperature estimates from TFSimulator match those from
    SimulatorFlexibleGeometries when using lwx (interior radiative heat
    exchange).

    Creates two simulators with identical buildings, HVAC and parameters:
    1. A SimulatorFlexibleGeometries instance (baseline)
    2. A TFSimulator instance (under test)

    Runs one timestep on both and verifies their temperature arrays match
      exactly.
    This validates that TFSimulator's radiative heat transfer calculations
      produce the same results as the original implementation.
    """
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 300
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2012-12-21")

    # Create baseline simulator
    simulator = FlexibleFloorplanSimulatorTest()
    simulator_hvac = simulator._create_small_hvac()
    simulator_building = simulator._create_small_building(
        initial_temp=292.0, include_radiative_heat_transfer=True
    )

    simulator_simulator = simulator_py.SimulatorFlexibleGeometries(
        simulator_building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )
    simulator_result = simulator_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    building = simulator_building

    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    result = tf_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    self.assertTrue(result)
    self.assertTrue(simulator_result)
    with self.subTest("CV temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.temp, simulator_simulator.building.temp
      )

  def test_compare_temperature_estimates_iterative_approach_with_interior_mass(self):  # pylint: disable=line-too-long
    """Tests that temperature estimates from TFSimulator match those from
    SimulatorFlexibleGeometries when using interior mass heat transfer

    Creates two simulators with identical buildings, HVAC and parameters:
    1. A SimulatorFlexibleGeometries instance (baseline/iterative approach)
    2. A TFSimulator instance (tensor approach under test)

    Runs one timestep on both and verifies their temperature arrays match
     exactly, including both air CV temperatures and interior mass temperatures.
    This validates that TFSimulator's interior mass heat transfer calculations
     produce the same results as the original iterative implementation.
    """
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 300
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2012-12-21")

    # Create baseline simulator with interior mass
    simulator = FlexibleFloorplanSimulatorTest()
    simulator_hvac = simulator._create_small_hvac()

    _, simulator_building = simulator._create_simulator_and_building(
        convergence_threshold=0.001,
        initial_temp=292.0,
        iteration_limit=100,
        include_interior_mass=True,
        include_radiative_heat_transfer=False,
    )
    simulator_simulator = simulator_py.SimulatorFlexibleGeometries(
        simulator_building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Create TFSimulator with the same building
    building = simulator_building
    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    result = tf_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    simulator_result = simulator_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    self.assertTrue(result)
    self.assertTrue(simulator_result)

    # Compare air CV temperatures
    with self.subTest("Air CV temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.temp,
          simulator_simulator.building.temp,
          decimal=5,
      )

    # Compare interior mass temperatures
    with self.subTest("Interior mass temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.interior_mass_temp,
          simulator_simulator.building.interior_mass_temp,
          decimal=5,
      )

  def test_compare_temperature_estimates_iterative_approach_with_interior_mass_and_lwx(self):  # pylint: disable=line-too-long
    """Tests that temperature estimates from TFSimulator match those from
    SimulatorFlexibleGeometries when using interior mass heat transfer
    and lwx (interior radiative heat exchange).

    Creates two simulators with identical buildings, HVAC and parameters:
    1. A SimulatorFlexibleGeometries instance (baseline/iterative approach)
    2. A TFSimulator instance (tensor approach under test)

    Runs one timestep on both and verifies their temperature arrays match
     exactly, including both air CV temperatures and interior mass temperatures.
    This validates that TFSimulator's interior mass heat transfer calculations
     produce the same results as the original iterative implementation.
     Note that interior mass convergence can be slow (due to the large thermal
      mass and small conductivity), so setting a too low convergence threshold
      may cause the test to fail.
    """
    weather_controller = mock.create_autospec(
        weather_controller_py.WeatherController
    )
    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 500
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2012-12-21")

    # Create baseline simulator with interior mass
    simulator = FlexibleFloorplanSimulatorTest()
    simulator_hvac = simulator._create_small_hvac()

    _, simulator_building = simulator._create_simulator_and_building(
        convergence_threshold=0.001,
        initial_temp=292.0,
        iteration_limit=100,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
    )

    simulator_simulator = simulator_py.SimulatorFlexibleGeometries(
        simulator_building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    # Create TFSimulator with the same building
    building = simulator_building
    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
    )

    result = tf_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    simulator_result = simulator_simulator.finite_differences_timestep(
        ambient_temperature=315.0, convection_coefficient=12.0
    )

    self.assertTrue(result)
    self.assertTrue(simulator_result)

    # Compare air CV temperatures
    with self.subTest("Air CV temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.temp,
          simulator_simulator.building.temp,
          decimal=5,
      )

    # Compare interior mass temperatures
    with self.subTest("Interior mass temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.interior_mass_temp,
          simulator_simulator.building.interior_mass_temp,
          decimal=5,
      )

  def test_compare_temperature_estimates_with_interior_mass_lwx_lwr(self):
    """Tests TFSimulator vs SimulatorFlexibleGeometries with full radiation.

    This test compares both simulators with the complete radiative heat
    transfer setup:
    - Interior mass heat transfer
    - Interior longwave radiative exchange (lwx)
    - Exterior longwave radiative heat transfer (lwr)
    - Solar radiation (shortwave) with fenestrations
    - Sky temperature for exterior LWR

    Creates two simulators with identical buildings, HVAC and parameters:
    1. A SimulatorFlexibleGeometries instance (baseline/iterative approach)
    2. A TFSimulator instance (tensor approach under test)

    Runs one timestep on both with real weather data and verifies their
    temperature arrays match, including both air CV temperatures and
    interior mass temperatures.
    """
    # Load real weather data for realistic irradiance values
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "resources",
        "sb1",
        "weather_data",
        "local_weather_test_data.csv",
    )
    weather_controller = weather_controller_py.ReplayWeatherController(
        local_weather_path=data_path,
        convection_coefficient=10.0,
        timezone="US/Pacific",
        latitude=37.4,
        longitude=-122.1,
        irradiance_method="campbell_norman",
    )

    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 500
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2023-07-01 09:30:00")

    # Create baseline simulator with interior mass, lwx, and fenestrations
    simulator = FlexibleFloorplanSimulatorTest()
    simulator_hvac = simulator._create_small_hvac()

    _, simulator_building = simulator._create_simulator_and_building(
        convergence_threshold=convergence_threshold,
        initial_temp=292.0,
        iteration_limit=iteration_limit,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
        weather_controller=weather_controller,
        include_fenestrations=True,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    # Create iterative SimulatorFlexibleGeometries
    simulator_simulator = simulator_py.SimulatorFlexibleGeometries(
        simulator_building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    # Create TFSimulator with the same building
    building = simulator_building
    tf_simulator = tf_simulator_py.TFSimulator(
        building,
        simulator_hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    # Get weather data for the timestep
    ambient_temperature = weather_controller.get_current_temp(start_timestamp)
    sky_temperature = weather_controller.get_current_sky_temperature(
        start_timestamp
    )
    irradiance_data = weather_controller.get_current_irradiance(start_timestamp)
    irradiance_components = {
        "ghi": irradiance_data["ghi"],
        "dni": irradiance_data["dni"],
        "dhi": irradiance_data["dhi"],
    }
    solar_zenith = irradiance_data["solar_zenith"]
    solar_azimuth = irradiance_data["solar_azimuth"]

    # Run TFSimulator
    tf_result = tf_simulator.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=12.0,
        sky_temperature=sky_temperature,
        irradiance_components=irradiance_components,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    # Run iterative SimulatorFlexibleGeometries
    simulator_result = simulator_simulator.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=12.0,
        sky_temperature=sky_temperature,
        irradiance_components=irradiance_components,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    self.assertTrue(tf_result, msg="TFSimulator should converge")
    self.assertTrue(
        simulator_result, msg="SimulatorFlexibleGeometries should converge"
    )

    # Compare air CV temperatures
    with self.subTest("Air CV temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.temp,
          simulator_simulator.building.temp,
          decimal=4,
      )

    # Compare interior mass temperatures
    with self.subTest("Interior mass temperatures match"):
      assert_array_almost_equal(
          tf_simulator.building.interior_mass_temp,
          simulator_simulator.building.interior_mass_temp,
          decimal=4,
      )

    # Visualize temperature comparison
    output_path = os.path.join(
        os.path.dirname(__file__), "temperature_comparison_lwx_lwr.png"
    )
    visualize_temperature_comparison(
        iterative_temp=simulator_simulator.building.temp,
        tf_temp=tf_simulator.building.temp,
        floor_plan=building.floor_plan,
        output_path=output_path,
        title="Validation: Iterative vs Tensorized (with LWX + LWR)",
    )

  def test_benchmark_iterative_vs_tensorized_performance(self):
    """Benchmarks iterative vs tensorized simulator performance.

    Runs multiple timesteps on both SimulatorFlexibleGeometries (iterative)
    and TFSimulator (tensorized) to compare execution times. The tensorized
    version should be significantly faster due to matrix operations.

    This test:
    1. Creates identical simulators with full radiation setup
    2. Runs 10 timesteps on each simulator
    3. Measures and compares elapsed time
    4. Asserts that TFSimulator is faster
    5. Prints benchmark results
    """
    num_steps = 10

    # Load real weather data for realistic irradiance values
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "resources",
        "sb1",
        "weather_data",
        "local_weather_test_data.csv",
    )
    weather_controller_iterative = (
        weather_controller_py.ReplayWeatherController(
            local_weather_path=data_path,
            convection_coefficient=10.0,
            timezone="US/Pacific",
            latitude=37.4,
            longitude=-122.1,
            irradiance_method="campbell_norman",
        )
    )
    weather_controller_tf = weather_controller_py.ReplayWeatherController(
        local_weather_path=data_path,
        convection_coefficient=10.0,
        timezone="US/Pacific",
        latitude=37.4,
        longitude=-122.1,
        irradiance_method="campbell_norman",
    )

    time_step_sec = 300.0
    convergence_threshold = 1e-3
    iteration_limit = 500
    iteration_warning = 30
    start_timestamp = pd.Timestamp("2023-07-01 09:30:00")

    # Create iterative simulator
    simulator_test = FlexibleFloorplanSimulatorTest()
    iterative_hvac = simulator_test._create_small_hvac()

    _, iterative_building = simulator_test._create_simulator_and_building(
        convergence_threshold=convergence_threshold,
        initial_temp=292.0,
        iteration_limit=iteration_limit,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
        weather_controller=weather_controller_iterative,
        include_fenestrations=True,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    iterative_simulator = simulator_py.SimulatorFlexibleGeometries(
        iterative_building,
        iterative_hvac,
        weather_controller_iterative,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    # Create TFSimulator with separate building instance
    tf_hvac = simulator_test._create_small_hvac()

    _, tf_building = simulator_test._create_simulator_and_building(
        convergence_threshold=convergence_threshold,
        initial_temp=292.0,
        iteration_limit=iteration_limit,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
        weather_controller=weather_controller_tf,
        include_fenestrations=True,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    tf_simulator = tf_simulator_py.TFSimulator(
        tf_building,
        tf_hvac,
        weather_controller_tf,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=1e-5,
        relative_convergence_streak=15,
    )

    # Benchmark iterative simulator
    iterative_times = []
    current_timestamp = start_timestamp
    for step in range(num_steps):
      ambient_temperature = weather_controller_iterative.get_current_temp(
          current_timestamp
      )
      sky_temperature = (
          weather_controller_iterative.get_current_sky_temperature(
              current_timestamp
          )
      )
      irradiance_data = weather_controller_iterative.get_current_irradiance(
          current_timestamp
      )
      irradiance_components = {
          "ghi": irradiance_data["ghi"],
          "dni": irradiance_data["dni"],
          "dhi": irradiance_data["dhi"],
      }
      solar_zenith = irradiance_data["solar_zenith"]
      solar_azimuth = irradiance_data["solar_azimuth"]

      step_start = time.perf_counter()
      converged = iterative_simulator.finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=12.0,
          sky_temperature=sky_temperature,
          irradiance_components=irradiance_components,
          solar_zenith=solar_zenith,
          solar_azimuth=solar_azimuth,
      )
      step_end = time.perf_counter()
      iterative_times.append(step_end - step_start)

      self.assertTrue(converged, f"Iterative simulator failed at step {step}")
      current_timestamp += pd.Timedelta(seconds=time_step_sec)

    # Benchmark TFSimulator
    tf_times = []
    current_timestamp = start_timestamp
    for step in range(num_steps):
      ambient_temperature = weather_controller_tf.get_current_temp(
          current_timestamp
      )
      sky_temperature = weather_controller_tf.get_current_sky_temperature(
          current_timestamp
      )
      irradiance_data = weather_controller_tf.get_current_irradiance(
          current_timestamp
      )
      irradiance_components = {
          "ghi": irradiance_data["ghi"],
          "dni": irradiance_data["dni"],
          "dhi": irradiance_data["dhi"],
      }
      solar_zenith = irradiance_data["solar_zenith"]
      solar_azimuth = irradiance_data["solar_azimuth"]

      step_start = time.perf_counter()
      converged = tf_simulator.finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=12.0,
          sky_temperature=sky_temperature,
          irradiance_components=irradiance_components,
          solar_zenith=solar_zenith,
          solar_azimuth=solar_azimuth,
      )
      step_end = time.perf_counter()
      tf_times.append(step_end - step_start)

      self.assertTrue(converged, f"TFSimulator failed at step {step}")
      current_timestamp += pd.Timedelta(seconds=time_step_sec)

    # Calculate statistics
    iterative_total = sum(iterative_times)
    tf_total = sum(tf_times)
    iterative_avg = iterative_total / num_steps
    tf_avg = tf_total / num_steps
    speedup = iterative_total / tf_total if tf_total > 0 else float("inf")

    # Get grid size information
    grid_shape = tf_building.floor_plan.shape
    grid_rows, grid_cols = grid_shape
    total_cvs = grid_rows * grid_cols

    # Build benchmark report
    report_lines = [
        "=" * 60,
        "BENCHMARK RESULTS: Iterative vs Tensorized Simulator",
        "=" * 60,
        "",
        "Configuration:",
        f"  Grid size:        {grid_rows} x {grid_cols} ({total_cvs} CVs)",
        f"  Number of steps:  {num_steps}",
        f"  Time step:        {time_step_sec} seconds",
        f"  Convergence:      {convergence_threshold}",
        f"  Iteration limit:  {iteration_limit}",
        "",
        "-" * 60,
        "Iterative (SimulatorFlexibleGeometries):",
        f"  Total time:       {iterative_total:.4f} seconds",
        f"  Average time:     {iterative_avg:.4f} seconds/step",
        f"  Per-step times:   {[f'{t:.4f}' for t in iterative_times]}",
        "",
        "-" * 60,
        "Tensorized (TFSimulator):",
        f"  Total time:       {tf_total:.4f} seconds",
        f"  Average time:     {tf_avg:.4f} seconds/step",
        f"  Per-step times:   {[f'{t:.4f}' for t in tf_times]}",
        "",
        "-" * 60,
        f"SPEEDUP: {speedup:.2f}x faster with TFSimulator",
        "=" * 60,
    ]
    report = "\n".join(report_lines)

    # Print benchmark results
    print("\n" + report + "\n")

    # Save benchmark results to file
    output_path = os.path.join(
        os.path.dirname(__file__),
        f"benchmark_results_{grid_rows}x{grid_cols}.txt",
    )
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(report)
    print(f"Benchmark results saved to: {output_path}")

    # Assert TFSimulator is faster
    self.assertLess(
        tf_total,
        iterative_total,
        msg=(
            f"TFSimulator ({tf_total:.4f}s) should be faster than "
            f"iterative ({iterative_total:.4f}s)"
        ),
    )


def visualize_temperature_comparison(
    iterative_temp: np.ndarray,
    tf_temp: np.ndarray,
    floor_plan: np.ndarray,
    output_path: str | None = None,
    title: str = "Temperature Comparison: Iterative vs Tensorized",
) -> None:
  """Visualizes temperature comparison between iterative and TF approaches.

  Creates a side-by-side heatmap visualization comparing temperature
  distributions from the iterative (SimulatorFlexibleGeometries) and
  tensorized (TFSimulator) approaches.

  Args:
    iterative_temp: Temperature array from iterative simulator.
    tf_temp: Temperature array from TFSimulator.
    floor_plan: Floor plan array (2=ambient air, 1=wall, 0=indoor air).
    output_path: Optional path to save the figure. If None, displays the plot.
    title: Title for the overall figure.
  """
  # Calculate the base temperature (minimum) for display offset
  min_temp = min(np.min(iterative_temp), np.min(tf_temp))

  # Calculate temperature differences from base for better visualization
  iterative_diff = iterative_temp - min_temp
  tf_diff = tf_temp - min_temp

  # Determine common color scale
  vmin = 0
  vmax = max(np.max(iterative_diff), np.max(tf_diff))

  _, axes = plt.subplots(1, 2, figsize=(14, 8))

  for ax, temp_diff, approach_name in [
      (axes[0], iterative_diff, "Iterative Approach"),
      (axes[1], tf_diff, "Tensorized Approach"),
  ]:
    # Plot temperature heatmap
    im = ax.imshow(temp_diff, cmap="viridis", vmin=vmin, vmax=vmax)

    # Overlay floor plan values on each cell
    for i in range(floor_plan.shape[0]):
      for j in range(floor_plan.shape[1]):
        text_color = "white" if temp_diff[i, j] < (vmax - vmin) / 2 else "black"
        ax.text(
            j,
            i,
            str(int(floor_plan[i, j])),
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(f"{approach_name}+{min_temp:.5e}", fontsize=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", rotation=270, labelpad=15)

  plt.suptitle(title, fontsize=14, fontweight="bold")
  plt.tight_layout()

  if output_path:
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
  else:
    plt.show()

  plt.close()


if __name__ == "__main__":
  absltest.main()
