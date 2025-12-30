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


if __name__ == "__main__":
  absltest.main()
