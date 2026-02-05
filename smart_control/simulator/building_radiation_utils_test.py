"""Tests for radiation utility functions."""

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal

from smart_control.simulator import building_radiation_utils as utils
from smart_control.simulator import constants

# Import the constant for air in line of sight
AIR_IN_LINE_OF_SIGHT = utils.AIR_IN_LINE_OF_SIGHT
TEMPORARY_MARKED_VALUE = utils.TEMPORARY_MARKED_VALUE
TEMPORARY_BLOCKED_VALUE = utils.TEMPORARY_BLOCKED_VALUE

# we are choosing to keep the mathematical notation names
# pylint: disable=invalid-name


class BuildingRadiationUtilsTest(absltest.TestCase):

  def test_calculate_A_tilde_inv_and_ifa_inv(self):
    """Test calculation of A-tilde inverse and IFA inverse matrices.

    Tests the core matrix calculations used in radiative heat transfer:
    - a_tilde_inv: Matrix relating radiosity to blackbody emissive power
    - ifa_inv: Matrix used to calculate net radiative heat flux

    Uses a 3-surface system with different emissivities (0.8, 0.4, 0.8)
    and symmetric view factors.
    """
    epsilon = np.array([0.8, 0.4, 0.8])
    F = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    expected_a_tilde_inv = np.array([
        [0.83982684, 0.04761905, 0.11255411],
        [0.28571429, 0.42857143, 0.28571429],
        [0.11255411, 0.04761905, 0.83982684],
    ])

    expected_ifa_inv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])

    a_tilde_inv = utils.calculate_a_tilde_inv(epsilon, F)
    ifa_inv = utils.calculate_ifa_inv(F, a_tilde_inv)
    with self.subTest("a_tilde_inv shape"):
      self.assertEqual(a_tilde_inv.shape, F.shape)
    with self.subTest("ifa_inv shape"):
      self.assertEqual(ifa_inv.shape, F.shape)

    with self.subTest("a_tilde_inv"):
      assert_array_almost_equal(a_tilde_inv, expected_a_tilde_inv, decimal=3)
    with self.subTest("ifa_inv"):
      assert_array_almost_equal(ifa_inv, expected_ifa_inv, decimal=3)

  def test_net_radiative_heatflux_function_of_t(self):
    """Test calculation of net radiative heat flux from surface temperatures.

    Tests the main radiative heat transfer equation that calculates net heat
    flux for each surface given their temperatures and the IFA inverse matrix.

    Uses a 3-surface system with temperatures [1200, 500, 1102] K and
    the IFA inverse matrix from the previous test.
    """
    # fmt: off
    #pylint:disable=line-too-long
    temperatures=np.array([1200,500,1102])#  [K]
    ifa_inv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])
    # fmt: on
    # pylint:enable=line-too-long
    expected_q = np.array([3.70061961e04, -3.69724724e04, -3.37237040e01])

    q = utils.net_radiative_heatflux_function_of_t(temperatures, ifa_inv)

    with self.subTest("q results as expected"):
      assert_array_almost_equal(
          np.round(q, 4), np.round(expected_q, 4), decimal=4
      )

  def test_mark_air_connected_interior_walls(self):
    """Test identification of interior walls connected through air spaces.

    This test verifies that interior wall nodes connected to the same air space
    are correctly identified and marked. This is the first step in radiative
    heat transfer calculations to determine which walls can potentially
    exchange heat through radiation.

    Test case:
    - Starting node at (2,3) - tests connectivity from a top-left corner
      position.

    Value meanings:
    - -33: Interior wall nodes that are connected to the same air space through
          4-directional connectivity (can potentially participate in radiative
          transfer)
    - 0: Air spaces that connect the interior walls
    - -3: Interior wall nodes that are not connected to the starting air space
    - -2: Exterior wall nodes (not part of the interior space)
    - -1: Exterior space (outside the building)

    The function uses 4-directional connectivity to find all air cells connected
    to the starting wall, then marks all interior walls adjacent to those air
    cells.
    """
    # fmt: off
    #pylint:disable=line-too-long

    indexed_floor_plan =\
      np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3, -3, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0, -3,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0, -3,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -3,  0, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3,  0,  0,  0,  0,  0,  0, -3, -2, -1],
                [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
                [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])


    expected_result = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -33, -33,  -3, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])
    # fmt: on
    # pylint:enable=line-too-long
    # Test case: Starting node at (2,3) - top-left corner
    # Tests 4-directional connectivity to find all interior walls connected
    #  to the same air space
    result, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 3),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )
    # setup (temperatures and ifa_inv have same number of rows):
    self.assertEqual(result.shape, indexed_floor_plan.shape)

    # result has same shape as the temperatures array:
    with self.subTest("air-connected interior walls correctly marked"):
      assert_array_almost_equal(result, expected_result)

  def test_mark_interior_surface_adjacent_to_air_with_fenestration(self):
    """Test that interior surfaces (walls and fenestration) are marked correctly

    This test verifies that both interior walls (-3) and interior fenestration
    (-43) that are adjacent to air spaces (0) are correctly identified.
    """
    # Floor plan with interior walls and interior fenestration
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -42, -42, -42, -42, -42, -1],  # Exterior fenestration (not marked)
        [
            -1,
            -43,
            -43,
            -43,
            -43,
            -43,
            -1,
        ],  # Interior fenestration (adjacent to air)
        [-1, 0, 0, 0, 0, 0, -1],  # Air
        [-1, -3, -3, -3, -3, -3, -1],  # Interior wall (adjacent to air)
        [-1, -1, -1, -1, -1, -1, -1],
    ])

    result = utils.mark_interior_surface_adjacent_to_air(floor_plan)

    # Interior fenestration at row 2 should be marked (adjacent to air at row 3)
    for col in range(1, 6):
      self.assertTrue(
          result[2, col],
          f"Interior fenestration at (2, {col}) should be marked",
      )

    # Interior wall at row 4 should be marked (adjacent to air at row 3)
    for col in range(1, 6):
      self.assertTrue(
          result[4, col],
          f"Interior wall at (4, {col}) should be marked",
      )

    # Exterior fenestration at row 1 should NOT be marked
    for col in range(1, 6):
      self.assertFalse(
          result[1, col],
          f"Exterior fenestration at (1, {col}) should NOT be marked",
      )

    # Air nodes should NOT be marked
    for col in range(1, 6):
      self.assertFalse(
          result[3, col],
          f"Air node at (3, {col}) should NOT be marked",
      )

  def test_mark_directly_seeing_nodes(self):
    """Test line-of-sight calculations for radiative heat transfer.

    This test verifies that wall nodes are correctly classified based on their
    visibility to a starting node for radiative heat transfer calculations.

    Test cases:
    - case_23: Starting node at (2,3) - tests visibility from top-left corner
    - case_27: Starting node at (2,7) - tests visibility from top-right corner
    - case_116: Starting node at (11,6) - tests visibility from bottom-center
    - case_33: Starting node at (3,3) - tests visibility from air node
    - case_128: Starting node at (12,8) - tests visibility from air node
    Value meanings:
    - -33: Interior wall nodes connected to the same air space
           (can participate in radiative transfer)
    - -34: Interior wall nodes that cannot see the starting node
           (blocked from radiative transfer)
    - -67: The starting node itself (marked_value + blocked_value)
    -   9: Air nodes along line of sight between wall nodes
    """
    # fmt: off
    #pylint:disable=line-too-long

    indexed_floor_plan =\
       np.array(
        [[ -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
        [ -1,  -2, -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,  -3,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,  -3,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
        [ -1,  -2, -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
        [ -1,  -2, -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
        [ -1,  -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]]
      )

    # Expected results for cases
    expected_result_23 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -67, -34, -34, -34, -34, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33, -33, -33,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33,   0, -34,   0, -34,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33,   0, -34,   0, -34,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33,   0,   0,   0, -34,  -2,  -1],
       [ -1,  -2,  -3, -33, -33,  -3, -34, -34, -34,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    expected_result_27 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -34, -34, -34, -34, -67, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -33, -33, -33,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -34,   0, -34,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   9, -34,   0, -34,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   9,   0, -34,   0, -34,   9, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0,   9,   9, -33,  -2,  -1],
       [ -1,  -2,  -3, -34, -34,  -3, -34, -34, -33,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
       [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    expected_result_116 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3, -34, -34, -34, -34, -34, -34,  -3,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   9, -33,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   9, -33,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   9, -33,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -34, -34, -67, -34, -34,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -33,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2, -34,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2, -33,   9,   9,   9,   9,   9,   9, -33,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    # Additional test cases for (3,3) and (12,8)
    expected_result_33 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2, -33, -67,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -34,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0, -34,   0, -34,  -2,  -1],
                [ -1,  -2, -33,   0,   0, -33,   0,   0,   0, -34,  -2,  -1],
                [ -1,  -2,  -3, -33, -33,  -3, -34, -34, -34,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    expected_result_128 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,   0,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,  -3,  -3,  -3,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,  -3,   0,  -3,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0,   0,  -3,   0,   0,   0,  -3,  -2,  -1],
                [ -1,  -2,  -3,  -3,  -3, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0, -67, -33,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -33, -33, -33, -33, -33, -33,  -3,  -2,  -1],
                [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
                [ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])

    # fmt: on
    # pylint:enable=line-too-long
    # Test case 1: Starting node at (2,3) - top-left corner
    # Tests visibility from a corner position with clear line of sight to
    # some walls
    result23_, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 3),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=utils.TEMPORARY_MARKED_VALUE,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result23 = utils.mark_directly_seeing_nodes(
        floor_plan=result23_, base_node=(2, 3)
    )

    # result has same shape as the temperatures array:
    with self.subTest("case_23 - top-left corner visibility"):
      # Check that wall visibility markings match (excluding air nodes)
      # Air nodes may be marked as TEMPORARY_MARKED_VALUE
      wall_mask = (
          (result23 == TEMPORARY_MARKED_VALUE)
          | (result23 == TEMPORARY_BLOCKED_VALUE)
          | (result23 == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE)
      )
      expected_wall_mask = (
          (expected_result_23 == TEMPORARY_MARKED_VALUE)
          | (expected_result_23 == TEMPORARY_BLOCKED_VALUE)
          | (
              expected_result_23
              == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE
          )
      )
      assert_array_almost_equal(
          result23[wall_mask], expected_result_23[expected_wall_mask]
      )
      # Verify that some air nodes along lines are marked
      air_in_line = np.sum(result23 == AIR_IN_LINE_OF_SIGHT)
      self.assertGreater(air_in_line, 0, "Some air nodes should be marked")

    # Test case 2: Starting node at (2,7) - top-right corner
    # Tests visibility from another corner position with different line of
    # sight patterns
    result27_, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(2, 7),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result27 = utils.mark_directly_seeing_nodes(
        floor_plan=result27_, base_node=(2, 7)
    )

    # result has same shape as the temperatures array:
    with self.subTest("case_27 - top-right corner visibility"):
      # Check that wall visibility markings match (excluding air nodes)
      wall_mask = (
          (result27 == TEMPORARY_MARKED_VALUE)
          | (result27 == TEMPORARY_BLOCKED_VALUE)
          | (result27 == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE)
      )
      expected_wall_mask = (
          (expected_result_27 == TEMPORARY_MARKED_VALUE)
          | (expected_result_27 == TEMPORARY_BLOCKED_VALUE)
          | (
              expected_result_27
              == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE
          )
      )
      assert_array_almost_equal(
          result27[wall_mask], expected_result_27[expected_wall_mask]
      )
      # Verify that some air nodes along lines are marked
      air_in_line = np.sum(result27 == AIR_IN_LINE_OF_SIGHT)
      self.assertGreater(air_in_line, 0, "Some air nodes should be marked")

    # Test case 3: Starting node at (11,6) - bottom-center
    # Tests visibility from a center position with complex line of sight
    # through interior walls
    result116_, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(11, 6),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result116 = utils.mark_directly_seeing_nodes(
        floor_plan=result116_, base_node=(11, 6)
    )

    # result has same shape as the temperatures array:
    with self.subTest("case_116 - bottom-center visibility"):
      # Check that wall visibility markings match (excluding air nodes)
      wall_mask = (
          (result116 == TEMPORARY_MARKED_VALUE)
          | (result116 == TEMPORARY_BLOCKED_VALUE)
          | (result116 == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE)
      )
      expected_wall_mask = (
          (expected_result_116 == TEMPORARY_MARKED_VALUE)
          | (expected_result_116 == TEMPORARY_BLOCKED_VALUE)
          | (
              expected_result_116
              == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE
          )
      )
      assert_array_almost_equal(
          result116[wall_mask], expected_result_116[expected_wall_mask]
      )
      # Verify that some air nodes along lines are marked
      air_in_line = np.sum(result116 == AIR_IN_LINE_OF_SIGHT)
      self.assertGreater(air_in_line, 0, "Some air nodes should be marked")

    # Test case 4: Starting node at (3,3) - mid air node
    # Tests visibility from an air node to surrounding walls
    result33_, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(3, 3),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=utils.TEMPORARY_MARKED_VALUE,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )
    result33 = utils.mark_directly_seeing_nodes(
        floor_plan=result33_, base_node=(3, 3)
    )
    with self.subTest("case_33 - air node visibility"):
      wall_mask = (
          (result33 == TEMPORARY_MARKED_VALUE)
          | (result33 == TEMPORARY_BLOCKED_VALUE)
          | (result33 == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE)
      )
      expected_wall_mask = (
          (expected_result_33 == TEMPORARY_MARKED_VALUE)
          | (expected_result_33 == TEMPORARY_BLOCKED_VALUE)
          | (
              expected_result_33
              == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE
          )
      )
      assert_array_almost_equal(
          result33[wall_mask], expected_result_33[expected_wall_mask]
      )
      air_in_line = np.sum(result33 == AIR_IN_LINE_OF_SIGHT)
      self.assertEqual(air_in_line, 0, "No air in line of sight.")

    # Test case 5: Starting node at (12,8) - air node
    # Tests visibility from another air node deeper inside
    result128_, _ = utils.mark_air_connected_interior_walls(
        indexed_floor_plan=indexed_floor_plan,
        start_pos=(12, 8),
        interior_wall_value=constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
        marked_value=utils.TEMPORARY_MARKED_VALUE,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )
    result128 = utils.mark_directly_seeing_nodes(
        floor_plan=result128_, base_node=(12, 8)
    )
    with self.subTest("case_128 - air node (deeper) visibility"):
      wall_mask = (
          (result128 == TEMPORARY_MARKED_VALUE)
          | (result128 == TEMPORARY_BLOCKED_VALUE)
          | (result128 == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE)
      )
      expected_wall_mask = (
          (expected_result_128 == TEMPORARY_MARKED_VALUE)
          | (expected_result_128 == TEMPORARY_BLOCKED_VALUE)
          | (
              expected_result_128
              == TEMPORARY_MARKED_VALUE + TEMPORARY_BLOCKED_VALUE
          )
      )
      assert_array_almost_equal(
          result128[wall_mask], expected_result_128[expected_wall_mask]
      )
      air_in_line = np.sum(result128 == AIR_IN_LINE_OF_SIGHT)
      self.assertEqual(air_in_line, 0, "No air in line of sight.")

  def test_fenestration_validation_passes_with_connected_fenestration(self):
    """Test that fenestration validation passes when connected to air.

    Outermost layer is all 2 (exterior space); fenestration (4) connects
    exterior to interior air (0) in the wall layer.
    """
    # Create a floor plan with fenestration connected to air
    # Most exterior cells are 2; fenestration sits inside, adjacent to 2 and 0
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 4, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # Should not raise
    result = utils.validate_fenestration_connectivity(
        floor_plan,
        fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    )
    self.assertTrue(result)

  def test_fenestration_validation_fails_without_air_connection(self):
    """Test that fenestration validation fails when not connected to air."""
    # Create a floor plan with fenestration not connected to air
    floor_plan = np.array([
        [2, 2, 2, 2, 2],
        [2, 4, 4, 4, 2],
        [2, 4, 1, 4, 2],  # Fenestration only connected to wall (1)
        [2, 4, 4, 4, 2],
        [2, 2, 2, 2, 2],
    ])

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          floor_plan,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    self.assertIn("not connected to indoor air", str(context.exception))

  def test_fenestration_position_marking(self):
    """Test that fenestration nodes are correctly marked by position."""
    # Create indexed floor plan with fenestration at value -4
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1],  # Exterior space
        [-1, -4, -4, -4, -4, -4, -1],  # Fenestration row
        [-1, 0, 0, 0, 0, 0, -1],  # Air row
        [-1, 0, 0, 0, 0, 0, -1],  # Air row
        [-1, -1, -1, -1, -1, -1, -1],  # Exterior space
    ])

    result = utils.mark_fenestration_positions(
        floor_plan,
        fenestration_value=constants.FENESTRATION_VALUE_IN_FUNCTION,
        exterior_space_value=constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    # Fenestration at row 1 should be:
    # - Adjacent to exterior (row 0) => exterior fenestration (-42)
    # - Adjacent to air (row 2) => interior fenestration (-43)
    # Since both are adjacent, they should be interior (-43) per logic
    for col in range(1, 6):
      self.assertEqual(
          result[1, col],
          constants.INTERIOR_FENESTRATION_VALUE,
          f"Position (1, {col}) should be interior fenestration",
      )

  def test_fenestration_grouping_with_view_factors(self):
    """Test that fenestration groups are created with correct view factors."""
    import math  # pylint: disable=import-outside-toplevel

    # Create indexed floor plan with marked fenestration positions
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -42, -42, -42, -42, -42, -1],  # Exterior fenestration
        [-1, -43, -43, -43, -43, -43, -1],  # Interior fenestration
        [-1, 0, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ])

    groups = utils.group_fenestrations(floor_plan)

    # Should have one fenestration group
    self.assertEqual(len(groups), 1)

    group_name = list(groups.keys())[0]
    group = groups[group_name]

    # Check phi is 90 degrees
    self.assertEqual(group["phi"], 90.0)

    # Check azimuth is 0 (top/north) since adjacent to exterior at top
    self.assertEqual(group["azimuth"], 0.0)

    # Check view factors using the formulas
    phi_rad = math.radians(90)
    cos_phi = math.cos(phi_rad)  # ~0

    expected_F_gnd = 0.5 * (1 - cos_phi)  # ~0.5
    expected_factor = 0.5 * (1 + cos_phi)  # ~0.5
    expected_F_sky = expected_factor * math.sqrt(expected_factor)  # ~0.354
    expected_F_air = expected_factor * (
        1 - math.sqrt(expected_factor)
    )  # ~0.146

    self.assertAlmostEqual(group["F_gnd"], expected_F_gnd, places=6)
    self.assertAlmostEqual(group["F_sky"], expected_F_sky, places=6)
    self.assertAlmostEqual(group["F_air"], expected_F_air, places=6)

    # Check that count and indices are correct
    self.assertEqual(group["count"], 10)  # 5 exterior + 5 interior
    self.assertEqual(len(group["indices"]), 10)

    # Check exterior_count (5 exterior fenestration nodes at row 1)
    self.assertEqual(group["exterior_count"], 5)

    # Check indices_array
    self.assertEqual(group["indices_array"].shape, floor_plan.shape)
    self.assertEqual(group["indices_array"].dtype, bool)
    # All fenestration positions should be True
    for r, c in group["indices"]:
      self.assertTrue(group["indices_array"][r, c])
    # Total True values should equal count
    self.assertEqual(np.sum(group["indices_array"]), group["count"])

  def test_air_node_grouping(self):
    """Test that air nodes are correctly grouped."""
    # Create indexed floor plan with two separate air spaces
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, -3, 0, 0, -1],  # Two air groups separated by wall
        [-1, 0, 0, -3, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ])

    groups = utils.group_air_nodes(floor_plan)

    # Should have two air groups
    self.assertEqual(len(groups), 2)

    # Check total air nodes
    total_nodes = sum(g["count"] for g in groups.values())
    self.assertEqual(total_nodes, 8)

    # Check indices_array for each group
    for group in groups.values():
      self.assertEqual(group["indices_array"].shape, floor_plan.shape)
      self.assertEqual(group["indices_array"].dtype, bool)
      # All air positions should be True
      for r, c in group["indices"]:
        self.assertTrue(group["indices_array"][r, c])
      # Total True values should equal count
      self.assertEqual(np.sum(group["indices_array"]), group["count"])

  def test_air_node_grouping_with_fenestration_links(self):
    """Test that air groups are linked to adjacent fenestration groups."""
    # Create indexed floor plan with fenestration adjacent to air
    floor_plan = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -42, -42, -42, -1],  # Exterior fenestration
        [-1, -43, -43, -43, -1],  # Interior fenestration
        [-1, 0, 0, 0, -1],  # Air
        [-1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    air_groups = utils.group_air_nodes(
        floor_plan, fenestration_groups=fenestration_groups
    )

    # Should have one air group with one fenestration group adjacent
    self.assertEqual(len(air_groups), 1)

    air_group = list(air_groups.values())[0]
    self.assertGreater(len(air_group["fenestration_groups"]), 0)

  def test_large_dummy_floor_plan_fenestration_groups(self):
    """Test that a floor plan with fenestration creates multiple groups."""
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [4, 4, 4, 4, 0, 0, 1, 0, 0, 4, 4],
        [4, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2],
        [4, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2],
    ])

    # First convert to indexed format
    indexed_floor_plan = floor_plan.copy()
    indexed_floor_plan[
        indexed_floor_plan == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
    ] = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
    indexed_floor_plan[
        indexed_floor_plan == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
    ] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
    indexed_floor_plan[
        indexed_floor_plan == constants.FENESTRATION_VALUE_IN_FILE_INPUT
    ] = constants.FENESTRATION_VALUE_IN_FUNCTION

    # Mark fenestration positions
    indexed_floor_plan = utils.mark_fenestration_positions(indexed_floor_plan)

    # Group fenestrations
    fenestration_groups = utils.group_fenestrations(indexed_floor_plan)

    # The large floor plan has fenestration on multiple sides
    # Should have multiple fenestration groups
    self.assertGreater(
        len(fenestration_groups),
        0,
        "Should have at least one fenestration group",
    )
    # TODO
    # need to add test validation based on manual calculations.

    # Check each group has required properties
    for group_name, group in fenestration_groups.items():
      with self.subTest(group=group_name):
        self.assertIn("count", group)
        self.assertIn("indices", group)
        self.assertIn("phi", group)
        self.assertIn("azimuth", group)
        self.assertIn("F_gnd", group)
        self.assertIn("F_sky", group)
        self.assertIn("F_air", group)

        # Check view factors sum to ~1 (they should for vertical surfaces)
        vf_sum = group["F_gnd"] + group["F_sky"] + group["F_air"]
        self.assertAlmostEqual(
            vf_sum,
            1.0,
            places=5,
            msg=f"View factors should sum to 1, got {vf_sum}",
        )

  def test_fenestration_azimuth_detection(self):
    # pylint: disable=line-too-long
    """Test that fenestration azimuth is correctly detected based on exterior.

    Floor plan structure:
    - Array boundaries must have exterior space(-1) | exterior fenestration(-42)
    - Exterior fenestration (-42) is at the building boundary facing outward
    - Interior fenestration (-43) is between exterior fenestration and indoor air
    - Indoor air (0) is the interior space

    The azimuth is determined by which boundary the exterior fenestration is on:
    - Row 0: TOP (azimuth 0)
    - Last row: BOTTOM (azimuth 180)
    - Col 0: LEFT (azimuth 270)
    - Last col: RIGHT (azimuth 90)
    """
    # pylint: enable=line-too-long
    # Test fenestration on left side (should be azimuth 270)
    # Exterior fenestration (-42) at col 0 boundary
    floor_plan_left = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -3, -3, -3, -3, -1],
        [-42, -43, 0, 0, -3, -1],  # Ext fen at col 0 boundary
        [-42, -43, 0, 0, -3, -1],
        [-1, -3, -3, -3, -3, -1],
        [-1, -1, -1, -1, -1, -1],
    ])
    groups = utils.group_fenestrations(floor_plan_left)
    self.assertEqual(
        list(groups.values())[0]["azimuth"],
        constants.FENESTRATION_AZIMUTH_LEFT,
    )

    # Test fenestration on right side (should be azimuth 90)
    # Exterior fenestration (-42) at last col boundary
    floor_plan_right = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -3, -3, -3, -3, -1],
        [-1, -3, 0, 0, -43, -42],  # Ext fen at last col boundary
        [-1, -3, 0, 0, -43, -42],
        [-1, -3, -3, -3, -3, -1],
        [-1, -1, -1, -1, -1, -1],
    ])
    groups = utils.group_fenestrations(floor_plan_right)
    self.assertEqual(
        list(groups.values())[0]["azimuth"],
        constants.FENESTRATION_AZIMUTH_RIGHT,
    )

    # Test fenestration on bottom side (should be azimuth 180)
    # Exterior fenestration (-42) at last row boundary
    floor_plan_bottom = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -3, -3, -3, -3, -3, -1],
        [-1, -3, 0, 0, 0, -3, -1],
        [-1, -3, -43, -43, -43, -3, -1],  # Interior fenestration
        [-1, -1, -42, -42, -42, -1, -1],  # Ext fen at last row boundary
    ])
    groups = utils.group_fenestrations(floor_plan_bottom)
    self.assertEqual(
        list(groups.values())[0]["azimuth"],
        constants.FENESTRATION_AZIMUTH_BOTTOM,
    )

    # Test fenestration on top side (should be azimuth 0)
    # Exterior fenestration (-42) at row 0 boundary
    floor_plan_top = np.array([
        [-1, -1, -42, -42, -42, -1, -1],  # Ext fen at row 0 boundary
        [-1, -3, -43, -43, -43, -3, -1],  # Interior fenestration
        [-1, -3, 0, 0, 0, -3, -1],
        [-1, -3, -3, -3, -3, -3, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ])
    groups = utils.group_fenestrations(floor_plan_top)
    self.assertEqual(
        list(groups.values())[0]["azimuth"],
        constants.FENESTRATION_AZIMUTH_TOP,
    )

  def test_calculate_poa_irradiance(self):
    """Test POA irradiance calculation from horizontal irradiance components."""
    # Test case: South-facing surface at 30 degree tilt, sun at 30 degree zenith
    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }
    surface_tilt = 30.0
    surface_azimuth = 180.0  # South-facing
    solar_zenith = 30.0
    solar_azimuth = 180.0  # Sun due south

    poa = utils.calculate_poa_irradiance(
        irradiance_components=irradiance_components,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    # POA should be positive
    self.assertGreater(poa, 0)
    # POA should be reasonable (not exceed theoretical max)
    self.assertLess(poa, 1500)

    # Test with sun below horizon (zenith > 90)
    poa_night = utils.calculate_poa_irradiance(
        irradiance_components={"ghi": 0.0, "dni": 0.0, "dhi": 0.0},
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=100.0,  # Sun below horizon
        solar_azimuth=180.0,
    )
    self.assertEqual(poa_night, 0.0)

  def test_calculate_poa_irradiance_different_orientations(self):
    """Test POA irradiance for different surface orientations."""
    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }
    solar_zenith = 30.0
    solar_azimuth = 180.0  # Sun due south

    # South-facing surface (should receive most direct radiation)
    poa_south = utils.calculate_poa_irradiance(
        irradiance_components=irradiance_components,
        surface_tilt=30.0,
        surface_azimuth=180.0,  # South
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    # North-facing surface (should receive less direct radiation)
    poa_north = utils.calculate_poa_irradiance(
        irradiance_components=irradiance_components,
        surface_tilt=30.0,
        surface_azimuth=0.0,  # North
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
    )

    # South-facing should receive more radiation than north-facing
    # when sun is in the south
    self.assertGreater(poa_south, poa_north)

  def test_calculate_exterior_lwr_for_fenestration_group(self):
    """Test exterior LWR calculation for a fenestration group."""
    # Create a simple fenestration group
    floor_plan = np.array([
        [-1, -1, -42, -42, -1, -1],  # Exterior fenestration at boundary
        [-1, -3, -43, -43, -3, -1],  # Interior fenestration
        [-1, -3, 0, 0, -3, -1],
        [-1, -1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    self.assertEqual(len(fenestration_groups), 1)

    group = list(fenestration_groups.values())[0]

    # Verify beta and exterior_indices_array are present
    self.assertIn("beta", group)
    self.assertIn("exterior_indices_array", group)
    self.assertEqual(group["exterior_count"], 2)

    # Create temperature and emissivity arrays
    surface_temps = np.full_like(floor_plan, 300.0, dtype=float)
    emissivity = np.full_like(floor_plan, 0.9, dtype=float)

    # Test with T_air > T_surf (surface should gain heat)
    q_lwr = utils.calculate_exterior_lwr_for_fenestration_group(
        fenestration_group=group,
        surface_temperatures=surface_temps,
        emissivity_array=emissivity,
        ambient_temperature=310.0,  # Warmer ambient
        sky_temperature=280.0,  # Cold sky
    )

    # With warmer ambient and cold sky, net heat flux depends on balance
    # The result should be a float
    self.assertIsInstance(q_lwr, float)

  def test_net_exterior_radiative_heatflux(self):
    """Test net exterior radiative heat flux for all fenestrations."""  # pylint: disable=line-too-long
    # Create floor plan with fenestrations
    floor_plan = np.array([
        [-1, -1, -42, -42, -1, -1],
        [-1, -3, -43, -43, -3, -1],
        [-1, -3, 0, 0, -3, -1],
        [-1, -1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)

    # Create temperature and emissivity arrays
    surface_temps = np.full_like(floor_plan, 300.0, dtype=float)
    emissivity = np.full_like(floor_plan, 0.9, dtype=float)

    # Calculate q_lwr
    q_lwr = utils.net_exterior_radiative_heatflux(
        floor_plan=floor_plan,
        fenestration_groups=fenestration_groups,
        surface_temperatures=surface_temps,
        emissivity_array=emissivity,
        ambient_temperature=310.0,
        sky_temperature=280.0,
    )

    # Output should have same shape as floor_plan
    self.assertEqual(q_lwr.shape, floor_plan.shape)

    # q_lwr should be zero at non-fenestration positions
    non_fenestration_mask = (floor_plan != -42) & (floor_plan != -43)
    self.assertTrue(np.all(q_lwr[non_fenestration_mask] == 0.0))

    # q_lwr should be non-zero at fenestration positions
    fenestration_mask = (floor_plan == -42) | (floor_plan == -43)
    # All fenestration nodes in a group should have the same q_lwr
    q_lwr_fenestrations = q_lwr[fenestration_mask]
    self.assertTrue(np.all(q_lwr_fenestrations == q_lwr_fenestrations[0]))

  def test_net_exterior_radiative_heatflux_no_fenestrations(self):
    """Test that q_lwr is zero when there are no fenestrations."""
    floor_plan = np.array([
        [-1, -1, -1, -1],
        [-1, -3, -3, -1],
        [-1, -3, 0, -1],
        [-1, -1, -1, -1],
    ])

    # No fenestrations
    fenestration_groups = utils.group_fenestrations(floor_plan)
    self.assertEqual(len(fenestration_groups), 0)

    surface_temps = np.full_like(floor_plan, 300.0, dtype=float)
    emissivity = np.full_like(floor_plan, 0.9, dtype=float)

    q_lwr = utils.net_exterior_radiative_heatflux(
        floor_plan=floor_plan,
        fenestration_groups=fenestration_groups,
        surface_temperatures=surface_temps,
        emissivity_array=emissivity,
        ambient_temperature=310.0,
        sky_temperature=280.0,
    )

    # All zeros when no fenestrations
    self.assertTrue(np.all(q_lwr == 0.0))

  def test_calculate_solar_absorbed_for_fenestration_group(self):
    """Test absorbed solar radiation calculation for a fenestration group."""
    # Create a floor plan with fenestration
    floor_plan = np.array([
        [-1, -1, -42, -42, -1],
        [-1, -3, -43, -43, -1],
        [-1, -3, 0, 0, -1],
        [-1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    self.assertEqual(len(fenestration_groups), 1)

    group = list(fenestration_groups.values())[0]

    # Test irradiance
    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }
    solar_zenith = 30.0
    solar_azimuth = 180.0
    alpha = 0.1  # absorptance

    q_sol_alpha_per_node = (
        utils.calculate_solar_absorbed_for_fenestration_group(
            group,
            irradiance_components,
            solar_zenith,
            solar_azimuth,
            alpha,
        )
    )

    # Should be positive
    self.assertGreater(q_sol_alpha_per_node, 0.0)

    # Calculate expected: G_Ts * alpha * exterior_count / total_count
    g_ts = utils.calculate_poa_irradiance(
        irradiance_components,
        group["phi"],
        group["azimuth"],
        solar_zenith,
        solar_azimuth,
    )
    expected = g_ts * alpha * group["exterior_count"] / group["count"]
    self.assertAlmostEqual(q_sol_alpha_per_node, expected, places=5)

  def test_net_solar_absorbed_heatflux_fenestration(self):
    """Test absorbed solar radiation array calculation for fenestrations."""
    floor_plan = np.array([
        [-1, -1, -42, -42, -1],
        [-1, -3, -43, -43, -1],
        [-1, -3, 0, 0, -1],
        [-1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)

    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }

    q_sol_alpha_array = utils.net_solar_absorbed_heatflux_fenestration(
        floor_plan,
        fenestration_groups,
        irradiance_components,
        solar_zenith=30.0,
        solar_azimuth=180.0,
        alpha=0.1,
    )

    # Check shape
    self.assertEqual(q_sol_alpha_array.shape, floor_plan.shape)

    # Fenestration positions should have non-zero values
    self.assertGreater(q_sol_alpha_array[0, 2], 0.0)  # exterior fenestration
    self.assertGreater(q_sol_alpha_array[0, 3], 0.0)  # exterior fenestration
    self.assertGreater(q_sol_alpha_array[1, 2], 0.0)  # interior fenestration
    self.assertGreater(q_sol_alpha_array[1, 3], 0.0)  # interior fenestration

    # Non-fenestration positions should be zero
    self.assertEqual(q_sol_alpha_array[2, 2], 0.0)  # air
    self.assertEqual(q_sol_alpha_array[1, 1], 0.0)  # wall

    # All fenestration nodes in same group should have same value
    self.assertAlmostEqual(
        q_sol_alpha_array[0, 2], q_sol_alpha_array[1, 3], places=5
    )

  def test_calculate_solar_transmitted_for_fenestration_group(self):
    """Test transmitted solar radiation calculation for a fenestration group."""
    floor_plan = np.array([
        [-1, -1, -42, -42, -1],
        [-1, -3, -43, -43, -1],
        [-1, -3, 0, 0, -1],
        [-1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    group = list(fenestration_groups.values())[0]

    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }
    tau = 0.8  # transmittance

    total_q_sol_tau = utils.calculate_solar_transmitted_for_fenestration_group(
        group,
        irradiance_components,
        solar_zenith=30.0,
        solar_azimuth=180.0,
        tau=tau,
    )

    # Should be positive
    self.assertGreater(total_q_sol_tau, 0.0)

    # Calculate expected: G_Ts * tau * exterior_count
    g_ts = utils.calculate_poa_irradiance(
        irradiance_components,
        group["phi"],
        group["azimuth"],
        30.0,
        180.0,
    )
    expected = g_ts * tau * group["exterior_count"]
    self.assertAlmostEqual(total_q_sol_tau, expected, places=5)

  def test_net_solar_transmitted_heatflux_fenestration(self):
    """Test transmitted solar radiation through fenestrations to air nodes."""
    floor_plan = np.array([
        [-1, -1, -42, -42, -1],
        [-1, -3, -43, -43, -1],
        [-1, -3, 0, 0, -1],
        [-1, -1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    air_groups = utils.group_air_nodes(
        floor_plan, fenestration_groups=fenestration_groups
    )

    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }

    q_sol_tau_array = utils.net_solar_transmitted_heatflux_fenestration(
        floor_plan,
        fenestration_groups,
        air_groups,
        irradiance_components,
        solar_zenith=30.0,
        solar_azimuth=180.0,
        tau=0.8,
    )

    # Check shape
    self.assertEqual(q_sol_tau_array.shape, floor_plan.shape)

    # Air positions should have non-zero values
    self.assertGreater(q_sol_tau_array[2, 2], 0.0)
    self.assertGreater(q_sol_tau_array[2, 3], 0.0)

    # All air nodes in same group should have same value
    self.assertAlmostEqual(
        q_sol_tau_array[2, 2], q_sol_tau_array[2, 3], places=5
    )

    # Non-air positions should be zero
    self.assertEqual(q_sol_tau_array[0, 2], 0.0)  # fenestration
    self.assertEqual(q_sol_tau_array[1, 1], 0.0)  # wall

  def test_net_solar_absorbed_heatflux_fenestration_no_fenestrations(self):
    """Test that q_sol_alpha is zero when there are no fenestrations."""
    floor_plan = np.array([
        [-1, -1, -1, -1],
        [-1, -3, -3, -1],
        [-1, -3, 0, -1],
        [-1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    self.assertEqual(len(fenestration_groups), 0)

    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }

    q_sol_alpha_array = utils.net_solar_absorbed_heatflux_fenestration(
        floor_plan,
        fenestration_groups,
        irradiance_components,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )

    # All zeros when no fenestrations
    self.assertTrue(np.all(q_sol_alpha_array == 0.0))

  def test_net_solar_transmitted_heatflux_fenestration_no_fenestrations(self):
    """Test that q_sol_tau is zero when there are no fenestrations."""
    floor_plan = np.array([
        [-1, -1, -1, -1],
        [-1, -3, -3, -1],
        [-1, -3, 0, -1],
        [-1, -1, -1, -1],
    ])

    fenestration_groups = utils.group_fenestrations(floor_plan)
    air_groups = utils.group_air_nodes(floor_plan)

    irradiance_components = {
        "ghi": 800.0,
        "dni": 700.0,
        "dhi": 100.0,
    }

    q_sol_tau_array = utils.net_solar_transmitted_heatflux_fenestration(
        floor_plan,
        fenestration_groups,
        air_groups,
        irradiance_components,
        solar_zenith=30.0,
        solar_azimuth=180.0,
    )

    # All zeros when no fenestrations
    self.assertTrue(np.all(q_sol_tau_array == 0.0))

  def test_wrong_fenestration_floor_plan_1_interior_wall_connection(self):
    """Test detection: fenestration connected to interior wall instead of air.

    In this floor plan, the left-side fenestration group at row 5 is connected
    to interior wall (1) instead of air (0), breaking the proper connection
    from exterior to indoor air.
    """
    # fmt: off
    wrong_fenestration_floor_plan_1 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 4, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # fmt: on

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          wrong_fenestration_floor_plan_1,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    # Should detect that fenestration at row 5 is blocked by interior wall
    self.assertIn("blocked", str(context.exception).lower())

  def test_wrong_fenestration_floor_plan_2_blocked_by_interior_wall(self):
    """Test detection: fenestration blocked by interior wall.

    In this floor plan, the fenestration at row 7 is blocked by an interior
    wall at position (7,3), preventing proper connection to air.
    """
    # fmt: off
    wrong_fenestration_floor_plan_2 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 4, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 4, 4, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # fmt: on

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          wrong_fenestration_floor_plan_2,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    # Should detect blocked fenestration (blocked from air or not connected)
    error_msg = str(context.exception).lower()
    self.assertTrue(
        "blocked" in error_msg or "not connected" in error_msg,
        "Expected 'blocked' or 'not connected' in error message, got:"
        f" {error_msg}",
    )

  def test_wrong_fenestration_floor_plan_3_not_exposed_to_outdoor(self):
    """Test detection: fenestration not exposed to outdoor.

    In this floor plan, the fenestration at row 5 (cols 3-5) is completely
    inside the building, surrounded by interior walls (1) and air (0),
    with no exposure to exterior space (2) or array boundary.
    """
    # fmt: off
    wrong_fenestration_floor_plan_3 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2],
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2],
        [2, 1, 4, 4, 4, 4, 0, 0, 1, 0, 2],  # fenestration inside building
        [2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # fmt: on

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          wrong_fenestration_floor_plan_3,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    # Should detect fenestration not exposed to exterior
    self.assertIn("exterior", str(context.exception).lower())

  def test_wrong_fenestration_floor_plan_4_partial_blockage(self):
    """Test detection: part of fenestration blocked by interior wall.

    In this floor plan, the top fenestration at column 4-5 has a partial
    blockage where part of the fenestration is connected to air while
    another part is blocked by interior wall at (2,4).
    """
    # fmt: off
    wrong_fenestration_floor_plan_4 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 4, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 4, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2],
        [2, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # fmt: on

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          wrong_fenestration_floor_plan_4,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    # Should detect blocked or disconnected fenestration node
    error_msg = str(context.exception).lower()
    self.assertTrue(
        "blocked" in error_msg or "disconnect" in error_msg,
        "Expected 'blocked' or 'disconnect' in error message, got:"
        f" {error_msg}",
    )

  def test_wrong_fenestration_floor_plan_5_fenestration_within_air(self):
    """Test detection: fenestration node within air (surrounded by air).

    In this floor plan, the fenestration at position (6,6) is surrounded
    by air nodes, meaning it's not properly connecting exterior to interior
    (it's floating in the middle of the air space).
    """
    # fmt: off
    wrong_fenestration_floor_plan_5 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 0, 0, 1, 0, 0, 4, 2],
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        [2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2],
        [2, 1, 1, 1, 0, 0, 4, 0, 0, 1, 2],
        [2, 4, 4, 4, 0, 0, 0, 0, 0, 1, 2],
        [2, 1, 1, 1, 4, 1, 1, 1, 4, 1, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    # fmt: on

    with self.assertRaises(ValueError) as context:
      utils.validate_fenestration_connectivity(
          wrong_fenestration_floor_plan_5,
          fenestration_value=constants.FENESTRATION_VALUE_IN_FILE_INPUT,
          air_value=constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      )
    # Should detect fenestration surrounded by air
    error_msg = str(context.exception).lower()
    self.assertTrue(
        "surrounded by air" in error_msg
        or "not exposed to exterior" in error_msg,
        "Expected 'surrounded by air' or 'not exposed to exterior' in error"
        f" message, got: {error_msg}",
    )

  def test_get_exterior_wall_boundary_mask_with_ambient_air(self):
    """Test exterior wall boundary mask with ambient air nodes.

    The algorithm identifies exterior walls NOT adjacent to enclosed interior
    AIR spaces. Interior walls (value 1) are treated as solid material.
    """
    # Test case: building with ambient air on the right, interior walls inside
    # fmt: off
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, -1],
        [2, 1, 1, 1, 1, 1, 1, -1],
        [2, 1, 1, 1, 1, 1, 2, -1],
        [2, 1, 1, 1, 1, 1, 2, -1],
        [2, 2, 2, 2, 2, 2, 2, -1],
    ])
    # fmt: on

    # Expected: all exterior walls are marked (no interior AIR to exclude)
    # Interior walls (1) are solid, not air; adjacent exterior walls marked
    # fmt: off
    expected = np.array([
        [True, True, True, True, True, True, True, False],
        [True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, True, False],
        [True, False, False, False, False, False, True, False],
        [True, True, True, True, True, True, True, False],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(
        result,
        expected,
        "Exterior wall boundary mask does not match expected with ambient air",
    )

  def test_get_exterior_wall_boundary_mask_thick_walls(self):
    """Test exterior wall boundary mask with thick walls.

    When walls are multiple cells thick with interior walls (value 1) inside,
    all exterior wall layers are marked since there's no interior AIR (0).
    """
    # Test case: thick exterior walls with interior walls (1), no interior air
    # fmt: off
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, -1],
        [2, 2, 2, 2, 2, 2, 2, -1],
        [2, 2, 1, 1, 1, 2, 2, -1],
        [2, 2, 2, 2, 2, 2, 2, -1],
        [2, 2, 2, 2, 2, 2, 2, -1],
    ])
    # fmt: on

    # Expected: all exterior walls marked (no interior air to exclude)
    # Interior walls (1) don't block the exterior wall marking
    # fmt: off
    expected = np.array([
        [True, True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True, False],
        [True, True, False, False, False, True, True, False],
        [True, True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True, False],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(
        result,
        expected,
        "Exterior wall boundary mask does not match expected for thick walls",
    )

  def test_get_exterior_wall_boundary_mask_with_fenestration(self):
    """Test exterior wall boundary mask with fenestration in walls.

    Fenestration (value 4) is not an exterior wall, so it's not marked.
    Exterior walls around fenestration are still marked normally.
    """
    # Test case: building with fenestration (4) in wall
    # fmt: off
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 2, 2, -1],
        [2, 1, 1, 1, 1, 1, 4, -1],
        [2, 1, 1, 1, 1, 1, 2, -1],
        [2, 1, 1, 1, 1, 1, 2, -1],
        [2, 2, 2, 2, 2, 2, 2, -1],
    ])
    # fmt: on

    # Expected: all exterior walls marked, fenestration (4) not marked
    # fmt: off
    expected = np.array([
        [True, True, True, True, True, True, True, False],
        [True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, True, False],
        [True, False, False, False, False, False, True, False],
        [True, True, True, True, True, True, True, False],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(
        result,
        expected,
        "Exterior wall boundary mask does not match with fenestration",
    )

  def test_get_exterior_wall_boundary_mask_enclosed_interior_air(self):
    """Test exterior wall boundary mask with enclosed interior AIR space.

    When interior AIR (value 0) is enclosed by walls, exterior walls
    adjacent to the enclosed air are NOT marked (they face inside).
    """
    # Test case: building with enclosed interior AIR (courtyard, value 0)
    # fmt: off
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 0, 0, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])
    # fmt: on

    # Expected: walls adjacent to enclosed air (0) are NOT marked
    # fmt: off
    expected = np.array([
        [False, False, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False],
        [False, True, True, False, False, True, True, False],
        [False, True, False, False, False, False, True, False],
        [False, True, True, False, False, True, True, False],
        [False, True, True, True, True, True, True, False],
        [False, False, False, False, False, False, False, False],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(
        result,
        expected,
        "Exterior wall boundary mask does not match for enclosed interior air",
    )

  def test_get_exterior_wall_boundary_mask_enclosed_interior_wall(self):
    """Test exterior wall boundary mask with enclosed interior WALL space.

    When interior WALLS (value 1) are enclosed, they're treated as solid
    material. All surrounding exterior walls are marked.
    """
    # Test case: building with enclosed interior WALLS (value 1)
    # fmt: off
    floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 1, 1, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, 2, 2, 2, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])
    # fmt: on

    # Expected: all exterior walls marked (interior walls are solid, not air)
    # fmt: off
    expected = np.array([
        [False, False, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False],
        [False, True, True, True, True, True, True, False],
        [False, True, True, False, False, True, True, False],
        [False, True, True, True, True, True, True, False],
        [False, True, True, True, True, True, True, False],
        [False, False, False, False, False, False, False, False],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(
        result,
        expected,
        "Exterior wall boundary mask does not match for enclosed interior"
        " walls",
    )

  def test_get_exterior_wall_boundary_mask_no_ambient_air(self):
    """Test when no ambient air nodes exist.

    When there are no ambient air nodes, exterior walls at array boundary
    are marked, and walls adjacent to enclosed interior air are excluded.
    """
    # Test case: no ambient air, exterior walls with interior walls
    # fmt: off
    floor_plan = np.array([
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2],
    ])
    # fmt: on

    # Expected: all exterior walls marked (no interior AIR to exclude)
    # Interior walls (1) don't cause exclusion
    # fmt: off
    expected = np.array([
        [True, True, True, True, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, True, True, True, True],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(result, expected)

  def test_get_exterior_wall_boundary_mask_no_ambient_with_interior_air(self):
    """Test when no ambient air but has interior air.

    When there's no ambient air but interior air doesn't touch array edges,
    exterior walls at array boundary are still marked.
    """
    # Test case: no ambient air, exterior walls with interior air (0) inside
    # Interior air doesn't touch array boundary
    # fmt: off
    floor_plan = np.array([
        [2, 2, 2, 2, 2],
        [2, 0, 0, 0, 2],
        [2, 0, 0, 0, 2],
        [2, 2, 2, 2, 2],
    ])
    # fmt: on

    # Expected: exterior walls at boundary are marked
    # Interior walls adjacent to enclosed interior air are NOT marked
    # But all these exterior walls are at array boundary, so all are marked
    # fmt: off
    expected = np.array([
        [True, True, True, True, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, True, True, True, True],
    ])
    # fmt: on

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    np.testing.assert_array_equal(result, expected)

  def test_get_exterior_wall_boundary_mask_no_exterior_walls(self):
    """Test with floor plan containing no exterior walls."""
    floor_plan = np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
    ])

    result = utils.get_exterior_wall_boundary_mask(
        floor_plan, wall_value=2, exterior_space_value=-1
    )

    # Should return all False
    self.assertFalse(np.any(result))

  def test_determine_exterior_wall_azimuth_array(self):
    """Test azimuth determination for exterior wall boundaries.

    Walls are adjacent to exterior space (-1), not at array boundary.
    Azimuth is based on which direction the exterior space is.
    """
    # Create floor plan: -1 = exterior space, -3 = wall, 0 = interior air
    # fmt: off
    indexed_floor_plan = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -3, -3, -3, -1],
        [-1, -3, 0, -3, -1],
        [-1, -3, -3, -3, -1],
        [-1, -1, -1, -1, -1],
    ])
    # fmt: on

    # Walls at (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)
    # are adjacent to exterior space
    exterior_wall_boundary_mask = indexed_floor_plan == -3

    result = utils.determine_exterior_wall_azimuth_array(
        exterior_wall_boundary_mask,
        indexed_floor_plan,
        exterior_space_value=-1,
    )

    # Check corners (intermediate angles)
    self.assertEqual(result[1, 1], 315.0, "Top-left corner should be 315°")
    self.assertEqual(result[1, 3], 45.0, "Top-right corner should be 45°")
    self.assertEqual(result[3, 1], 225.0, "Bottom-left corner should be 225°")
    self.assertEqual(result[3, 3], 135.0, "Bottom-right corner should be 135°")

    # Check edges (cardinal directions)
    self.assertEqual(result[1, 2], 0.0, "Top edge should be 0° (North)")
    self.assertEqual(result[3, 2], 180.0, "Bottom edge should be 180° (South)")
    self.assertEqual(result[2, 1], 270.0, "Left edge should be 270° (West)")
    self.assertEqual(result[2, 3], 90.0, "Right edge should be 90° (East)")

    # Check interior (should be 0, not a wall)
    self.assertEqual(result[2, 2], 0.0, "Interior air should be 0")

  def test_determine_exterior_wall_azimuth_array_l_shape(self):
    """Test azimuth determination for L-shaped building."""
    # L-shaped building: -1 = exterior space, -3 = wall, 0 = interior air
    # fmt: off
    indexed_floor_plan = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -3, -3, -3, -3, -1],
        [-1, -3, 0, 0, -3, -1],
        [-1, -3, 0, 0, -3, -3],
        [-1, -3, -3, -3, -3, -3],
        [-1, -1, -1, -1, -1, -1],
    ])
    # fmt: on

    exterior_wall_boundary_mask = indexed_floor_plan == -3

    result = utils.determine_exterior_wall_azimuth_array(
        exterior_wall_boundary_mask,
        indexed_floor_plan,
        exterior_space_value=-1,
    )

    # Check key positions
    self.assertEqual(result[1, 1], 315.0, "Top-left corner")
    self.assertEqual(result[1, 2], 0.0, "Top edge (not corner)")
    self.assertEqual(result[1, 4], 45.0, "Top-right corner (before notch)")
    self.assertEqual(result[2, 4], 90.0, "Right edge")
    self.assertEqual(result[4, 5], 135.0, "Bottom-right corner")
    self.assertEqual(result[4, 1], 225.0, "Bottom-left corner")


if __name__ == "__main__":
  absltest.main()
