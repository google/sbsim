"""Tests for radiation utility functions."""

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal

from smart_control.simulator import building_radiation_utils as utils
from smart_control.simulator import constants

# we are choosing to keep the mathematical notation names
# pylint: disable=invalid-name


class BuildingRadiationUtilsTest(absltest.TestCase):

  def test_calculate_A_tilde_inv_and_IFAinv(self):
    """Test calculation of A-tilde inverse and IFA inverse matrices.

    Tests the core matrix calculations used in radiative heat transfer:
    - A_tilde_inv: Matrix relating radiosity to blackbody emissive power
    - IFAinv: Matrix used to calculate net radiative heat flux

    Uses a 3-surface system with different emissivities (0.8, 0.4, 0.8)
    and symmetric view factors.
    """
    epsilon = np.array([0.8, 0.4, 0.8])
    F = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    expected_A_tilde_inv = np.array([
        [0.83982684, 0.04761905, 0.11255411],
        [0.28571429, 0.42857143, 0.28571429],
        [0.11255411, 0.04761905, 0.83982684],
    ])

    expected_IFAinv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])

    A_tilde_inv = utils.calculate_A_tilde_inv(epsilon, F)
    IFAinv = utils.calculate_IFAinv(F, A_tilde_inv)
    with self.subTest("A_tilde_inv shape"):
      self.assertEqual(A_tilde_inv.shape, F.shape)
    with self.subTest("IFAinv shape"):
      self.assertEqual(IFAinv.shape, F.shape)

    with self.subTest("A_tilde_inv"):
      assert_array_almost_equal(A_tilde_inv, expected_A_tilde_inv, decimal=3)
    with self.subTest("IFAinv"):
      assert_array_almost_equal(IFAinv, expected_IFAinv, decimal=3)

  def test_net_radiative_heatflux_function_of_T(self):
    """Test calculation of net radiative heat flux from surface temperatures.

    Tests the main radiative heat transfer equation that calculates net heat
    flux for each surface given their temperatures and the IFA inverse matrix.

    Uses a 3-surface system with temperatures [1200, 500, 1102] K and
    the IFA inverse matrix from the previous test.
    """
    # fmt: off
    #pylint:disable=line-too-long
    temperatures=np.array([1200,500,1102])#  [K]
    IFAinv = np.array([
        [0.64069264, -0.19047619, -0.45021645],
        [-0.19047619, 0.38095238, -0.19047619],
        [-0.45021645, -0.19047619, 0.64069264],
    ])
    # fmt: on
    # pylint:enable=line-too-long
    expected_q = np.array([3.70061961e04, -3.69724724e04, -3.37237040e01])

    q = utils.net_radiative_heatflux_function_of_T(temperatures, IFAinv)

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
    # setup (temperatures and IFAinv have same number of rows):
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

    Value meanings:
    - -33: Interior wall nodes connected to the same air space
           (can participate in radiative transfer)
    - -34: Interior wall nodes that cannot see the starting node
           (blocked from radiative transfer)
    - -67: The starting node itself (marked_value + blocked_value)
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



    expected_result_23 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -67, -34, -34, -34, -34, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
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

    expected_result_27 = \
      np.array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       [ -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1],
       [ -1,  -2,  -3, -34, -34, -34, -34, -67, -34,  -3,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -33, -33, -33,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0, -34,   0, -33,  -2,  -1],
       [ -1,  -2, -33,   0,   0, -34,   0,   0,   0, -33,  -2,  -1],
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
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0, -33,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0, -33,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -34, -34, -67, -34, -34,  -3,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3,   0, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2,  -3, -34, -33,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
                [ -1,  -2, -34,   0,   0,   0,   0,   0,   0, -33,  -2,  -1],
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
        marked_value=-33,
        air_value=constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    )

    result23 = utils.mark_directly_seeing_nodes(
        floor_plan=result23_, base_node=(2, 3)
    )

    # result has same shape as the temperatures array:
    with self.subTest("case_23 - top-left corner visibility"):
      assert_array_almost_equal(result23, expected_result_23)

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
      assert_array_almost_equal(result27, expected_result_27)

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
      assert_array_almost_equal(result116, expected_result_116)


if __name__ == "__main__":
  absltest.main()
