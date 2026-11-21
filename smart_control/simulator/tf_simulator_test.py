"""Tests for Tensorflow-enabled Finite Difference calculator."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import tensorflow as tf

from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import scipy_solver as scipy_solver_py
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

  def _run_simulators_for_comparison(
      self,
      include_radiative_heat_transfer,
      include_interior_mass,
      convergence_threshold,
      iteration_limit,
  ):
    """Runs one time step with each solver, each on its own building.

    Every solver gets a fresh, identical building: sharing one would let the
    second solver start from the first one's result and compare an array with
    itself. ScipySolverSimulator solves each linearized balance directly, so
    with a tight threshold it is the reference the two relaxation solvers are
    checked against; its spread from them also tells whether
    `convergence_threshold` is tight enough.

    Args:
      include_radiative_heat_transfer: Whether to enable interior longwave.
      include_interior_mass: Whether to enable interior mass.
      convergence_threshold: Threshold for the two relaxation solvers.
      iteration_limit: Iteration limit for the two relaxation solvers.

    Returns:
      Dict from solver name to its converged simulator.
    """
    fixture = FlexibleFloorplanSimulatorTest()
    simulators = {}
    for name, simulator_class, threshold in (
        (
            "simulator",
            simulator_py.SimulatorFlexibleGeometries,
            convergence_threshold,
        ),
        ("tf", tf_simulator_py.TFSimulator, convergence_threshold),
        ("scipy", scipy_solver_py.ScipySolverSimulator, 1e-8),
    ):
      if include_interior_mass:
        _, building = fixture._create_simulator_and_building(
            convergence_threshold=threshold,
            initial_temp=292.0,
            iteration_limit=iteration_limit,
            include_interior_mass=True,
            include_radiative_heat_transfer=include_radiative_heat_transfer,
        )
      else:
        building = fixture._create_small_building(
            initial_temp=292.0,
            include_radiative_heat_transfer=include_radiative_heat_transfer,
            include_interior_mass=False,
        )
      simulator = simulator_class(
          building,
          fixture._create_small_hvac(),
          mock.create_autospec(weather_controller_py.WeatherController),
          300.0,
          threshold,
          iteration_limit,
          iteration_limit,
          pd.Timestamp("2012-12-21"),
      )
      converged = simulator.finite_differences_timestep(
          ambient_temperature=315.0, convection_coefficient=12.0
      )
      self.assertTrue(converged, msg=f"{name} did not converge")
      simulators[name] = simulator
    return simulators

  def _assert_temperatures_match(self, simulators, air_atol, mass_atol=None):
    """Checks both relaxation solvers against the direct solve."""
    reference = simulators["scipy"].building
    for name in ("simulator", "tf"):
      building = simulators[name].building
      with self.subTest(f"Air CV temperatures match ({name} vs scipy)"):
        np.testing.assert_allclose(
            building.temp, reference.temp, rtol=0, atol=air_atol
        )
      if mass_atol is not None:
        with self.subTest(
            f"Interior mass temperatures match ({name} vs scipy)"
        ):
          np.testing.assert_allclose(
              building.interior_mass_temp,
              reference.interior_mass_temp,
              rtol=0,
              atol=mass_atol,
          )

  def test_compare_temperature_estimates_iterative_approach(self):
    """Tests that SimulatorFlexibleGeometries, TFSimulator and
    ScipySolverSimulator agree on one time step without radiation or mass.

    All three solve the same finite volume balance with the same series
    half-cell conductances. The direct sparse solve is the reference; the
    tolerance covers the convergence threshold and TFSimulator's float32.
    """
    simulators = self._run_simulators_for_comparison(
        include_radiative_heat_transfer=False,
        include_interior_mass=False,
        convergence_threshold=1e-3,
        iteration_limit=300,
    )
    self._assert_temperatures_match(simulators, air_atol=5e-3)

    # Interior faces cancel in pairs only if both CVs use the same
    # conductance, so over the whole building the heat stored in the step
    # must equal the heat taken from the ambient.
    with self.subTest("Energy is conserved (scipy)"):
      simulator = simulators["scipy"]
      building = simulator.building
      z = building.floor_height_cm / 100.0
      stored = 0.0
      from_ambient = 0.0
      for i in range(building.temp.shape[0]):
        for j in range(building.temp.shape[1]):
          neighbors = building.neighbors[i][j]
          if len(neighbors) <= 1:
            continue
          u, v = simulator._get_cv_dimensions((i, j))  # pylint: disable=protected-access
          stored += (
              building.density[i][j]
              * building.heat_capacity[i][j]
              * u
              * v
              * z
              * (building.temp[i][j] - 292.0)
              / 300.0
          )
          conductance = simulator._get_ambient_conductance((i, j), 12.0)  # pylint: disable=protected-access
          for ni, nj in neighbors:
            if len(building.neighbors[ni][nj]) <= 1:
              conductance += simulator._get_face_conductance((i, j), (ni, nj))  # pylint: disable=protected-access
          from_ambient += conductance * (315.0 - building.temp[i][j])
      self.assertGreater(stored, 0.0)
      self.assertAlmostEqual(stored, from_ambient, delta=abs(stored) * 1e-6)

  def test_compare_temperature_estimates_iterative_approach_with_lwx(self):
    """Tests that the three solvers agree with lwx (interior radiative heat
    exchange).

    The radiative exchange is lagged by one iteration in every solver, so the
    relaxation solvers need a tighter threshold here to reach the direct
    solve; at 1e-3 they stop about 0.03 K short.
    """
    simulators = self._run_simulators_for_comparison(
        include_radiative_heat_transfer=True,
        include_interior_mass=False,
        convergence_threshold=1e-5,
        iteration_limit=1000,
    )
    self._assert_temperatures_match(simulators, air_atol=1e-2)

  def test_compare_temperature_estimates_iterative_approach_with_interior_mass(self):  # pylint: disable=line-too-long
    """Tests that the three solvers agree with interior mass heat transfer.

    The interior mass node is re-solved every iteration from the previous
    time step's mass temperature, so all three converge to the same coupled
    air/mass balance.
    """
    simulators = self._run_simulators_for_comparison(
        include_radiative_heat_transfer=False,
        include_interior_mass=True,
        convergence_threshold=1e-3,
        iteration_limit=300,
    )
    self._assert_temperatures_match(simulators, air_atol=1e-4, mass_atol=1e-6)

  def test_compare_temperature_estimates_iterative_approach_with_interior_mass_and_lwx(self):  # pylint: disable=line-too-long
    """Tests that the three solvers agree with interior mass and lwx.

    Point relaxation converges slowly here: the room air is coupled to the
    walls only through half a cell of insulation, and the lagged radiative
    exchange adds an outer iteration on top. The threshold is relaxed to keep
    the test fast, and the tolerance is set from the measured distance to the
    direct solve at that threshold (about 0.13 K for TFSimulator).
    """
    simulators = self._run_simulators_for_comparison(
        include_radiative_heat_transfer=True,
        include_interior_mass=True,
        convergence_threshold=3e-4,
        iteration_limit=6000,
    )
    self._assert_temperatures_match(simulators, air_atol=0.2, mass_atol=1e-4)

  ## Part 3: Tests that pin the conduction stencil rather than the
  ## implementation.
  ##
  ## The defect these cover was invisible to every other test in this file:
  ## an air-to-wall face whose two sides computed different fluxes, which
  ## destroys energy at every such interface while still producing temperature
  ## fields that look plausible.

  def _create_square_zone_building(self, cv_cm: float, wall_cv: int):
    """Returns one square zone wrapped in `wall_cv` control volumes of wall."""
    room_m = 6.0
    rows = int(round(room_m / (cv_cm / 100.0)))
    pad = wall_cv + 1
    n = rows + 2 * pad
    plan = np.full((n, n), 2, dtype=int)
    plan[1:-1, 1:-1] = 1
    plan[pad:-pad, pad:-pad] = 0

    wall = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=700.0, density=1.0
    )
    return building_py.FloorPlanBasedBuilding(
        cv_size_cm=cv_cm,
        floor_height_cm=300.0,
        initial_temp=293.15,
        inside_air_properties=building_py.MaterialProperties(
            conductivity=50.0, heat_capacity=1006.0, density=1.2
        ),
        inside_wall_properties=wall,
        building_exterior_properties=wall,
        floor_plan=plan,
        zone_map=plan.copy(),
        buffer_from_walls=0,
    )

  def _create_square_zone_simulator(self, building, **kwargs):
    """Returns a simulator for `building` with the HVAC effectively off."""
    zones = [
        zone
        for zone in building._room_dict  # pylint: disable=protected-access
        if isinstance(zone, str) and zone.startswith("room_")
    ]
    hvac = floorplan_hvac_py.FloorPlanBasedHvac(
        air_handler=air_handler_py.AirHandler(0.3, 270.0, 288.0, 20000.0, 0.8),
        boiler=boiler_py.Boiler(350.0, 3.0, 0.6, "boiler_id"),
        schedule=setpoint_schedule.SetpointSchedule(
            morning_start_hour=9,
            evening_start_hour=18,
            comfort_temp_window=(200.0, 400.0),
            eco_temp_window=(200.0, 400.0),
            holidays=set(),
        ),
        vav_max_air_flow_rate=1e-9,
        vav_reheat_max_water_flow_rate=0.02,
        zone_identifier=zones,
    )
    return tf_simulator_py.TFSimulator(
        building=building,
        hvac=hvac,
        weather_controller=weather_controller_py.WeatherController(
            273.15, 273.15, convection_coefficient=100.0
        ),
        time_step_sec=60.0,
        convergence_threshold=0.01,
        iteration_limit=200,
        iteration_warning=10**9,
        start_timestamp=pd.Timestamp("2012-12-21"),
        **kwargs,
    )

  def test_face_conductances_are_flux_conservative(self):
    """Both CVs sharing a face have to agree on the conductance across it."""
    building = self._create_square_zone_building(cv_cm=10.0, wall_cv=2)
    simulator = self._create_square_zone_simulator(building)

    # Faces onto an exterior CV are carried by the surface film and those CVs
    # are pinned at the ambient temperature, so only interior faces have two
    # sides that have to agree.
    interior = ~simulator._t_exerior_temps_mask.numpy()  # pylint: disable=protected-access

    # The right-edge tensor of a CV and the left-edge tensor of the CV at
    # j + 1 describe the same face, so they have to carry the same value.
    left = simulator._t_face_conductance_left.numpy()  # pylint: disable=protected-access
    right = simulator._t_face_conductance_right.numpy()  # pylint: disable=protected-access
    horizontal = interior[:, :-1] & interior[:, 1:]
    assert_array_almost_equal(
        right[:, :-1][horizontal], left[:, 1:][horizontal], decimal=6
    )

    top = simulator._t_face_conductance_top.numpy()  # pylint: disable=protected-access
    bottom = simulator._t_face_conductance_bottom.numpy()  # pylint: disable=protected-access
    vertical = interior[1:, :] & interior[:-1, :]
    assert_array_almost_equal(
        top[1:, :][vertical], bottom[:-1, :][vertical], decimal=6
    )

  def test_face_conductance_matches_series_resistance(self):
    """An air-to-wall face is the series of the two half cells, not either k."""
    building = self._create_square_zone_building(cv_cm=10.0, wall_cv=2)
    simulator = self._create_square_zone_simulator(building)
    plan = np.asarray(building.floor_plan)

    # Walk one row to the first air CV that has a wall CV on its left.
    row = plan.shape[0] // 2
    air_columns = np.nonzero(plan[row] == 0)[0]
    column = int(air_columns[0])
    self.assertEqual(plan[row][column - 1], 1)

    half_cm = building.cv_size_cm / 100.0 / 2.0
    expected = 1.0 / (half_cm / 50.0 + half_cm / 0.05)
    conductance = simulator._t_face_conductance_left.numpy()[row][column]  # pylint: disable=protected-access
    self.assertAlmostEqual(conductance, expected, delta=expected * 1e-4)

    # The shipped formula, this CV's own k over its own width, is three orders
    # of magnitude away from that.
    self.assertGreater(50.0 / (2.0 * half_cm) / conductance, 100.0)

  def test_face_conductances_match_iterative_simulator(self):
    """The tensor and per-CV paths build the same conductance for every face.

    The tensors carry the face conductance per unit of the owner's own face
    area, so multiplying by Vz (left/right) or Uz (top/bottom) gives the W/K
    that `Simulator._get_face_conductance` returns. The floor plan is L-shaped
    so that it has a reentrant corner, where a half-width edge CV meets a full
    interior CV and only the narrower width is shared.
    """
    wall_cv = 2
    square = self._create_square_zone_building(cv_cm=20.0, wall_cv=wall_cv)
    plan = np.asarray(square.floor_plan).copy()
    # Cut a notch out of the top-left corner and wrap it in wall.
    notch = wall_cv + 4
    region = plan[: notch + wall_cv, : notch + wall_cv]
    region[region == 0] = 1
    plan[:notch, :notch] = 2
    wall = building_py.MaterialProperties(
        conductivity=0.05, heat_capacity=700.0, density=1.0
    )
    building = building_py.FloorPlanBasedBuilding(
        cv_size_cm=20.0,
        floor_height_cm=300.0,
        initial_temp=293.15,
        inside_air_properties=building_py.MaterialProperties(
            conductivity=50.0, heat_capacity=1006.0, density=1.2
        ),
        inside_wall_properties=wall,
        building_exterior_properties=wall,
        floor_plan=plan,
        zone_map=plan.copy(),
        buffer_from_walls=0,
    )
    simulator = self._create_square_zone_simulator(building)
    # The notch puts a full-width CV next to half-width edge CVs.
    self.assertEqual(len(building.neighbors[notch][notch]), 4)
    self.assertEqual(len(building.neighbors[notch - 1][notch]), 3)
    z = building.floor_height_cm / 100.0
    u = simulator._t_u.numpy()  # pylint: disable=protected-access
    v = simulator._t_v.numpy()  # pylint: disable=protected-access
    face_tensors = {
        (0, -1): simulator._t_face_conductance_left.numpy() * v * z,  # pylint: disable=protected-access
        (0, 1): simulator._t_face_conductance_right.numpy() * v * z,  # pylint: disable=protected-access
        (-1, 0): simulator._t_face_conductance_top.numpy() * u * z,  # pylint: disable=protected-access
        (1, 0): simulator._t_face_conductance_bottom.numpy() * u * z,  # pylint: disable=protected-access
    }

    rows, cols = building.temp.shape
    for i in range(rows):
      for j in range(cols):
        if len(building.neighbors[i][j]) <= 1:
          continue
        for ni, nj in building.neighbors[i][j]:
          if len(building.neighbors[ni][nj]) <= 1:
            continue
          expected = simulator._get_face_conductance((i, j), (ni, nj))  # pylint: disable=protected-access
          actual = face_tensors[(ni - i, nj - j)][i][j]
          self.assertAlmostEqual(actual, expected, delta=expected * 1e-5)


if __name__ == "__main__":
  absltest.main()
